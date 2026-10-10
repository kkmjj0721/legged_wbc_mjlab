"""Play saved LP-ACRL policies using their own task and robot configuration."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path
import threading

import torch

from ..config.env_cfgs import build_env_cfg
from ..config.experiment_cfg import compile_experiment, from_dict
from ..config.rl_cfg import ri_4438_him_runner_cfg
from ..mdp.curriculums import get_controller


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viewer", choices=("auto", "native", "viser", "none"), default="viser")
    p.add_argument("--task-id", type=int, action="append", help="Repeat to sample uniformly among selected tasks; defaults to all tasks")
    p.add_argument("--list-tasks", action="store_true", help="Print saved task IDs without starting simulation")
    p.add_argument("--steps", type=int, help="Stop after this many control steps; required with --viewer none")
    p.add_argument("--video", action="store_true", help="Record the first --steps (or 1000) steps to checkpoint_dir/videos/play_TIMESTAMP")
    return p


def run_viewer(viewer, *, native, steps):
    previous_threads = set(threading.enumerate())
    try:
        viewer.run(num_steps=steps)
    finally:
        if native:
            viewer.close()
            # launch_passive starts a daemon render thread. Its close() only
            # requests exit; wait for MuJoCo cleanup before Python's GLFW atexit
            # handler runs (otherwise GLX teardown can segfault on Linux).
            for thread in threading.enumerate():
                target = getattr(thread, "_target", None)
                if thread not in previous_threads and getattr(target, "__module__", None) == "mujoco.viewer":
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        raise RuntimeError("Native viewer did not finish closing; use --viewer viser")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.num_envs < 1 or (args.steps is not None and args.steps < 1):
        p.error("--num-envs and --steps must be positive")
    if args.viewer == "none" and args.steps is None and not args.list_tasks:
        p.error("--viewer none requires --steps")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = (checkpoint.get("infos") or {}).get("lpacrl")
    if state is None:
        raise ValueError("Playback requires an LPACRL checkpoint with its saved configuration")
    cfg = from_dict(state["resolved_config"])
    cfg.training.num_envs = args.num_envs
    compiled = compile_experiment(cfg)
    if compiled.fingerprints != state["fingerprints"]:
        raise ValueError("Saved task/protocol differs from this implementation")
    records = compiled.catalog.records
    selected = sorted(set(args.task_id)) if args.task_id is not None else list(range(len(records)))
    if any(i < 0 or i >= len(records) for i in selected):
        p.error(f"--task-id must be in [0, {len(records)-1}]; use --list-tasks")
    if args.list_tasks:
        for i, record in enumerate(records):
            print(f"{i:3d}  {record.key}")
        return

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.models.him_actor_model import HIMActorModel
    import warp as wp

    configure_torch_backends()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu if intended")
    wp.config.kernel_cache_dir = str((Path(cfg.training.output_dir) / "kernel_cache").resolve())
    viewer_name = args.viewer
    if viewer_name == "auto":
        viewer_name = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
        if args.device.startswith("cuda"):
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(torch.device(args.device).index or 0)

    env_cfg = build_env_cfg(compiled, "play")
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array" if args.video else None)
        controller = get_controller(env)
        # Playback covers the requested tasks equally instead of adopting the
        # potentially concentrated training curriculum. No training state loads.
        controller.state.probabilities.fill(0.0)
        controller.state.probabilities[selected] = 1.0 / len(selected)
        if args.video:
            from mjlab.utils.wrappers import VideoRecorder
            video_dir = args.checkpoint.parent / "videos" / f"play_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
            env = VideoRecorder(env, video_folder=video_dir, step_trigger=lambda step: step == 0,
                                video_length=args.steps or 1000, disable_logger=True)
            print(f"Video directory: {video_dir}")
        agent_cfg = ri_4438_him_runner_cfg(compiled.cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        obs = env.get_observations()
        actor_cfg = asdict(agent_cfg.actor)
        actor_cfg.pop("class_name")
        actor_cfg = {key: value for key, value in actor_cfg.items() if value is not None}
        actor = HIMActorModel(obs, {"actor": ["actor"]}, "actor", env.num_actions, **actor_cfg).to(args.device)
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        actor.eval().requires_grad_(False)
        print(f"Loaded {args.checkpoint.name}; update={checkpoint.get('iter')}; nominal dynamics; tasks={selected}")

        @torch.inference_mode()
        def policy(observations):
            if env.episode_length_buf[0].item() == 0:
                task = int(controller.episode_task_id[0])
                command = controller.commands[0].tolist()
                print(f"env 0: task={task} {records[task].key}; vx={command[0]:+.3f}, vy={command[1]:+.3f}, wz={command[2]:+.3f}")
            return actor(observations)

        if viewer_name == "none":
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, rewards, _, _ = env.step(policy(obs))
                    if not torch.isfinite(rewards).all() or any(not torch.isfinite(value).all() for value in obs.values()):
                        raise RuntimeError("Nonfinite playback transition")
            print(f"Finished {args.steps} playback steps")
        else:
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
            viewer_cls = NativeMujocoViewer if viewer_name == "native" else ViserPlayViewer
            if viewer_name == "native":
                print("Native controls: Space=pause, Enter=reset, ,/.=previous/next environment, P=reward plots, -/+=speed")
            viewer = viewer_cls(env, policy)
            run_viewer(viewer, native=viewer_name == "native", steps=args.steps)
            print("Viewer playback finished")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
