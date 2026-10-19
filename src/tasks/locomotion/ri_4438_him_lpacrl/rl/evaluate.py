"""Offline fixed-trial evaluation and optional videos; never loads LP history."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

from ..config.curriculum_cfg import (
    build_curriculum,
    finalize_fingerprints,
    make_lpacrl_cfg,
)
from ..mdp.task_catalog import fingerprint, plain


COMMANDS = {
    "forward": (0.5, 0.0, 0.0),
    "backward": (-0.5, 0.0, 0.0),
    "left": (0.0, 0.3, 0.0),
    "right": (0.0, -0.3, 0.0),
    "turn_left": (0.0, 0.0, 0.5),
    "turn_right": (0.0, 0.0, -0.5),
}


def from_snapshot(snapshot):
    """Restore saved task/robot parameters, adding only missing online fields.

    Old evaluation-based LP state is deliberately not migrated. The saved
    configuration remains usable for standalone evaluation of policy weights.
    """
    cfg = make_lpacrl_cfg()
    if set(snapshot) != set(vars(cfg)):
        raise ValueError(
            "Unsupported checkpoint configuration; use --protocol current for a new common evaluation"
        )
    for name in ("training", "geometry", "evaluation", "curriculum"):
        values = {**vars(getattr(cfg, name)), **deepcopy(snapshot[name])}
        setattr(cfg, name, NS(**values))
    cfg.geometry.map_protocol = snapshot["geometry"].get("map_protocol", "legacy_v6")
    # Historical task/robot parameters remain usable with the current evaluator.
    # The evaluation fingerprint records this upgrade; teacher state is not loaded.
    cfg.evaluation.protocol_version = make_lpacrl_cfg().evaluation.protocol_version
    for name in ("robot", "learning"):
        setattr(
            cfg, name, NS(**{k: NS(**deepcopy(v)) for k, v in snapshot[name].items()})
        )
    task = deepcopy(snapshot["task_space"])
    task.setdefault("sample_bin_centers", False)
    task.setdefault("heading_command", False)
    task.setdefault("heading_control_stiffness", 0.5)
    task.setdefault("ang_vel_z_range", (-1.0, 1.0))
    for name in ("process_amplitudes", "process_heading_range", "process_hold_s"):
        task.setdefault(name, getattr(cfg.task_space, name))
    task.setdefault("random_process", False)
    task.setdefault("fixed_tasks", True)
    for name in ("command_axes", "terrains"):
        task[name] = {k: NS(**v) for k, v in task[name].items()}
    cfg.task_space = NS(**task)
    cfg.scan_size_m = tuple(snapshot["scan_size_m"])
    cfg.scan_resolution_m = snapshot["scan_resolution_m"]
    return cfg


def fixed_command(cfg, name):
    """Translations retain heading hold; turn suites explicitly request constant wz."""
    cfg = deepcopy(cfg)
    cfg.task_space.random_process = False
    cfg.task_space.fixed_tasks = True
    defaults = make_lpacrl_cfg().task_space.command_axes["vx"]
    values = COMMANDS[name]
    if values[2]:
        cfg.task_space.heading_command = False
    axes = ("vx", "vy", "heading" if cfg.task_space.heading_command else "wz")
    cfg.task_space.command_axes = {
        axis: NS(
            **{
                **vars(defaults),
                "mode": "fixed",
                "value": value,
                "edges": None,
                "bins": None,
                "range": None,
            }
        )
        for axis, value in zip(axes, values)
    }
    cfg.task_space.resampling_interval_s = None
    cfg.task_space.include = []
    cfg.task_space.exclude = []
    return cfg


def select_task_mode(cfg, mode):
    """Choose evaluation families independently of the saved training sampler."""
    cfg = deepcopy(cfg)
    if mode not in (None, "mixed", "fixed", "random_process"):
        raise ValueError(f"Unknown task mode: {mode}")
    if mode == "random_process" and not cfg.task_space.heading_command:
        raise ValueError("random_process requires heading mode")
    if mode is not None:
        cfg.task_space.fixed_tasks = mode != "random_process"
        cfg.task_space.random_process = mode != "fixed" and cfg.task_space.heading_command
        cfg.curriculum.forward_stairs_fraction = 0.0
    return cfg


class TrialVideo:
    """Stream one environment to disk without retaining a full video in RAM."""

    def __init__(self, directory):
        self.directory = directory
        self.writer = None

    def __call__(self, env, batch, active, step):
        if step == 0:
            self.close()
            import imageio.v2 as imageio

            self.directory.mkdir(parents=True, exist_ok=True)
            trial = batch[0]
            filename = f"task_{trial['task_id']:04d}_trial_{trial['trial_id']:02d}.mp4"
            self.writer = imageio.get_writer(
                self.directory / filename, fps=1 / (2 * env.step_dt)
            )
        if step % 2 == 0 and bool(active[0]):
            # Render the physical scene without ray/contact debug overlays.
            env._offline_renderer.update(env.sim.data)
            self.writer.append_data(env._offline_renderer.render())

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=("checkpoint", "current"), default="checkpoint"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--trials-per-task", type=int)
    parser.add_argument("--horizon-s", type=float)
    parser.add_argument("--command", choices=tuple(COMMANDS))
    parser.add_argument("--task-mode", choices=("mixed", "fixed", "random_process"),
                        help="Override the protocol's task families; default uses its configured sampler")
    parser.add_argument("--suite", choices=("task", "deployment", "both"), default="both",
                        help="Task-space evaluation, fixed flat command sequences, or both (default)")
    parser.add_argument("--terrain", action="append")
    parser.add_argument("--level")
    parser.add_argument(
        "--video",
        action="store_true",
        help="Requires --num-envs 1; saves every selected trial",
    )
    parser.add_argument(
        "--previous-result",
        type=Path,
        help="Paired offline return diagnostics under the identical protocol",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    if args.video and args.num_envs != 1:
        parser.error("video recording requires --num-envs 1")
    if args.video and args.suite != "task":
        parser.error("video recording requires --suite task")
    if args.suite == "deployment" and any((args.command, args.terrain, args.level, args.horizon_s)):
        parser.error("deployment uses fixed flat sequences; command/terrain/level/horizon apply to --suite task or both")
    os.environ.setdefault("MUJOCO_GL", "egl")

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from tensordict import TensorDict
    from rsl_rl.models.him_actor_model import HIMActorModel
    from ..config.rl_cfg import ri_4438_him_runner_cfg
    from .evaluator import TaskEvaluator, export_result
    from .metrics import annotate_progress
    from .deployment import make_deployment_setup

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    infos = saved.get("infos") or {}
    snapshot = (infos.get("lpacrl") or {}).get("task_config")
    if args.protocol == "checkpoint":
        if snapshot is None:
            parser.error(
                "No supported saved task_config; use the historical harness or --protocol current for common trials"
            )
        cfg = from_snapshot(snapshot)
    else:
        cfg = make_lpacrl_cfg()
    if args.command:
        cfg = fixed_command(cfg, args.command)
    if args.command and args.task_mode in ("mixed", "random_process"):
        parser.error("--command requires fixed tasks; omit --task-mode or select fixed")
    try:
        cfg = select_task_mode(cfg, args.task_mode)
    except ValueError as error:
        parser.error(str(error))
    if args.terrain or args.level:
        terrains = args.terrain or list(cfg.task_space.terrains)
        if set(terrains) - set(cfg.task_space.terrains):
            parser.error("Unknown terrain selection")
        cfg.task_space.terrains = {
            k: v for k, v in cfg.task_space.terrains.items() if k in terrains
        }
        if args.level:
            for terrain in cfg.task_space.terrains.values():
                if args.level not in terrain.levels:
                    parser.error(
                        "Selected level does not exist in every selected terrain"
                    )
                terrain.levels = {args.level: terrain.levels[args.level]}
    for name in ("num_envs", "trials_per_task", "horizon_s"):
        if getattr(args, name) is not None:
            setattr(cfg.evaluation, name, getattr(args, name))
    deployment_cfg = deepcopy(cfg)
    setup = make_deployment_setup(cfg) if args.suite == "deployment" else build_curriculum(cfg)
    cfg = setup.cfg
    # Evaluation settings and fixed trials have their own fingerprint, outside
    # the training resume contract. num_envs is included for reproducibility.
    setup.fingerprints["evaluation"] = fingerprint(
        [cfg.evaluation, setup.trials, "traversal_v7", setup.terrain_map.manifest,
         cfg.geometry.pass_margin_m, cfg.geometry.pass_confirm_s]
    )
    video = TrialVideo(args.output_dir / "videos") if args.video else None
    def env_factory(**kwargs):
        if video:
            viewer = kwargs["cfg"].viewer
            viewer.width, viewer.height = 640, 480
            viewer.distance, viewer.elevation = 2.5, -25.0
        return ManagerBasedRlEnv(**kwargs, render_mode="rgb_array" if video else None)

    evaluator = TaskEvaluator(
        setup,
        args.device,
        env_factory=env_factory,
        step_callback=video,
    )
    configure_torch_backends()
    try:
        env = evaluator._get_env()
        finalize_fingerprints(setup, env.cfg)
        obs, _ = env.reset()
        actor_cfg = deepcopy(
            infos.get("agent_config", {}).get("actor")
            or asdict(ri_4438_him_runner_cfg(cfg).actor)
        )
        actor_cfg.pop("class_name", None)
        actor_cfg = {k: v for k, v in actor_cfg.items() if v is not None}
        actor = HIMActorModel(
            TensorDict(obs, batch_size=[env.num_envs]),
            {"actor": ["actor"]},
            "actor",
            env.action_manager.total_action_dim,
            **actor_cfg,
        ).to(args.device)
        actor.load_state_dict(saved["actor_state_dict"], strict=True)
        result = evaluator.evaluate(actor)
        if args.previous_result:
            previous = json.loads(args.previous_result.read_text())
            for key in ("task", "geometry", "evaluation", "environment", "reward"):
                if previous["fingerprints"].get(key) != setup.fingerprints[key]:
                    raise ValueError(
                        "Paired evaluations must use identical tasks, geometry and trials"
                    )
            annotate_progress(result, previous["trial_returns"])
        result.update(
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_iteration=saved.get("iter"),
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            configuration=plain(cfg),
            command_suite=args.command,
            suite=args.suite,
        )
        if args.suite == "both":
            deployment_setup = make_deployment_setup(deployment_cfg)
            deployment_setup.fingerprints["evaluation"] = fingerprint(
                [deployment_setup.cfg.evaluation, deployment_setup.trials, "traversal_v1"]
            )
            # Release the large terrain evaluation scene before the small suite.
            evaluator.close()
            evaluator.env = None
            sequence_evaluator = TaskEvaluator(deployment_setup, args.device)
            try:
                sequence_env = sequence_evaluator._get_env()
                finalize_fingerprints(deployment_setup, sequence_env.cfg)
                sequence_result = sequence_evaluator.evaluate(actor)
                sequence_result.update(
                    checkpoint=result["checkpoint"], checkpoint_sha256=result["checkpoint_sha256"],
                    checkpoint_iteration=result["checkpoint_iteration"],
                    configuration=plain(deployment_setup.cfg), suite="deployment",
                )
                export_result(sequence_result, args.output_dir / "deployment", deployment_setup.catalog,
                              deployment_setup.fingerprints)
                result["deployment_suite"] = {
                    "result_path": "deployment/evaluation.json",
                    **sequence_result["deployment"], "by_profile": sequence_result["by_profile"],
                }
            finally:
                sequence_evaluator.close()
        export_result(result, args.output_dir, setup.catalog, setup.fingerprints)
        print(json.dumps(result["macro"], indent=2))
        print(json.dumps(result.get("deployment_suite", result["deployment"]), indent=2))
    finally:
        if video:
            video.close()
        evaluator.close()


if __name__ == "__main__":
    main()
