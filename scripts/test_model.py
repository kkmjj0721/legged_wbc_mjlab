"""Load the supplied RI-4438 LP-ACRL checkpoint and check policy inference.

Run from the repository root:
    uv run --no-sync python -B -m scripts.test_model
    uv run --no-sync python -B -m scripts.test_model --steps 200
    uv run --no-sync python -B -m scripts.test_model --viewer native
    uv run --no-sync python -B -m scripts.test_model --viewer viser

The default is a CPU loading check. Simulation uses the current LP-ACRL terrain
with the saved robot and policy settings; it does not restore training state.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time


DEFAULT_CHECKPOINT = Path(
    "/home/sunteng/Downloads/logs/rsl_rl/ri_4438_him_lpacrl/"
    "2026-10-15_22-01-33/model_5000.pt"
)


def make_replay_cfg(snapshot):
    """Keep saved robot/command settings and use the current Rough terrain."""
    from src.tasks.locomotion.ri_4438_him_lpacrl.config.curriculum_cfg import make_lpacrl_cfg
    from src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate import from_snapshot

    cfg = from_snapshot(snapshot)
    current = make_lpacrl_cfg()
    cfg.geometry = deepcopy(current.geometry)
    cfg.task_space.terrains = deepcopy(current.task_space.terrains)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", help="Default: cpu for loading, cuda:0 for simulation")
    parser.add_argument("--viewer", choices=("none", "native", "viser"), default="none")
    parser.add_argument("--steps", type=int, default=0, help="Headless simulation steps; 0 only checks loading")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if args.num_envs < 1 or args.steps < 0:
        parser.error("--num-envs must be positive and --steps must be nonnegative")
    if args.viewer != "none" and args.steps:
        parser.error("Use --steps for headless simulation, or --viewer for interactive playback")

    project_root = Path(__file__).resolve().parents[1]
    temporary_dir = project_root / ".artifacts" / "WF-030" / "tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary_dir)
    if args.viewer == "none":
        os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    from tensordict import TensorDict

    import src.tasks  # noqa: F401 -- register project tasks
    from rsl_rl.models.him_actor_model import HIMActorModel
    from src.config.ri_4438.him_layouts import PHASE47

    started = time.perf_counter()
    simulate = args.steps > 0 or args.viewer != "none"
    device = args.device or ("cuda:0" if simulate and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    infos = checkpoint["infos"]
    state = infos["lpacrl"]
    actor_cfg = deepcopy(infos["agent_config"]["actor"])
    actor_cfg.pop("class_name", None)
    actor_cfg = {key: value for key, value in actor_cfg.items() if value is not None}
    if (
        actor_cfg["num_one_step_obs"] != PHASE47.frame_size
        or actor_cfg["history_size"] != PHASE47.history_size
        or tuple(actor_cfg["history_term_dims"]) != PHASE47.term_dims
        or actor_cfg["history_order"] != "frame_major_oldest_first"
    ):
        raise ValueError("Checkpoint does not use the supported phase47 HIM history layout")
    contract = infos["him_numerics"]
    for key, expected in (
        ("frame_size", PHASE47.frame_size),
        ("history_size", PHASE47.history_size),
        ("action_observation_slice", list(PHASE47.action_slice)),
        ("action_clip", actor_cfg.get("action_clip")),
        ("observation_clip", actor_cfg.get("observation_clip")),
    ):
        if contract.get(key) != expected:
            raise ValueError(f"Checkpoint HIM numerical contract differs: {key}")

    observations = TensorDict(
        {"actor": torch.zeros(args.num_envs, PHASE47.history_size, PHASE47.frame_size, device=device)},
        batch_size=[args.num_envs],
    )
    actor = HIMActorModel(observations, {"actor": ["actor"]}, "actor", 12, **actor_cfg).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    for name, value in actor.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite policy tensor: {name}")

    @torch.inference_mode()
    def policy(obs):
        actions = actor(obs)
        if actions.shape != (args.num_envs, 12) or not torch.isfinite(actions).all():
            raise ValueError("Policy produced invalid actions")
        return actions

    actions = policy(observations)
    print(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_iteration": checkpoint.get("iter"),
        "curriculum_version": state["version"],
        "completed_policy_updates": state["completed_policy_updates"],
        "device": device,
        "seed": args.seed,
        "observation_shape": list(observations["actor"].shape),
        "action_shape": list(actions.shape),
        "actions_for_zero_observation": actions[0].tolist(),
        "load_and_inference": "passed",
    }, indent=2), flush=True)
    if not simulate:
        return

    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from src.envs import ManagerBasedRlEnv
    from src.tasks.locomotion.ri_4438_him_lpacrl.config.env_cfgs import build_env_cfg
    from src.tasks.locomotion.ri_4438_him_lpacrl.mdp.curriculums import get_controller

    configure_torch_backends()
    task_cfg = make_replay_cfg(state["task_config"])
    task_cfg.training.num_envs = args.num_envs
    task_cfg.training.seed = args.seed
    env_cfg = build_env_cfg(task_cfg, "play")
    env_cfg.seed = args.seed
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=actor.action_clip)
        controller = get_controller(raw_env)
        for index in range(args.num_envs):
            controller.play.request_command(index, args.command, heading_hold=False)
        obs, _ = env.reset()
        if tuple(obs["actor"].shape) != tuple(observations["actor"].shape):
            raise ValueError("Simulation observations differ from the checkpoint layout")
        if args.viewer != "none":
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

            viewer_cls = NativeMujocoViewer if args.viewer == "native" else ViserPlayViewer
            viewer_cls(env, policy).run()
        else:
            resets = 0
            reward_sum = 0.0
            with torch.inference_mode():
                for _ in range(args.steps):
                    obs, reward, done, _ = env.step(policy(obs))
                    if not torch.isfinite(obs["actor"]).all() or not torch.isfinite(reward).all():
                        raise ValueError("Simulation produced non-finite observations or rewards")
                    resets += int(done.sum())
                    reward_sum += float(reward.sum())
            print(json.dumps({
                "simulation": "passed",
                "steps": args.steps,
                "num_envs": args.num_envs,
                "command": args.command,
                "resets": resets,
                "mean_step_reward": reward_sum / (args.steps * args.num_envs),
                "elapsed_s": time.perf_counter() - started,
            }, indent=2), flush=True)
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
