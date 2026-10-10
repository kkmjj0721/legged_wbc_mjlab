"""Compile first, then train. Run this module with --help or --describe."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import random
from zoneinfo import ZoneInfo
import numpy as np
import torch
import yaml

from ..config.experiment_cfg import compile_experiment, resolve_experiment
from ..config.presets import PRESETS
from ..config.rl_cfg import ri_4438_him_runner_cfg
from ..mdp.task_catalog import plain


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="ri_flat_smoke")
    p.add_argument("--config", help="YAML experiment overrides")
    p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="Repeatable dotted-path YAML value override")
    p.add_argument("--describe", action="store_true", help="Print compiled data without creating an environment or GPU tensors")
    for flag in ("device", "resume", "warm-start", "output-dir"):
        p.add_argument("--"+flag)
    for flag in ("num-envs", "max-iterations"):
        p.add_argument("--"+flag, type=int)
    p.add_argument("--video", action="store_true", default=None)
    return p


def run_training(compiled):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.wrappers import VideoRecorder
    from .runner import LPACRLRunner
    cfg = compiled.cfg
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("LPACRL v1 supports one process and one GPU")
    device = cfg.training.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA could not initialize. CPU --describe/unit tests remain available; GPU training has not run.")
    import warp as wp
    wp.config.kernel_cache_dir = str((Path(cfg.training.output_dir) / "kernel_cache").resolve())
    configure_torch_backends()
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    os.environ.setdefault("MUJOCO_GL", "egl")
    if device.startswith("cuda"):
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(torch.device(device).index or 0)
    run_name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    log_dir = Path(cfg.training.output_dir) / f"{run_name}_{cfg.curriculum.method}_seed{cfg.training.seed}"
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "experiment.yaml").write_text(yaml.safe_dump(plain(cfg), sort_keys=False))
    (log_dir / "compiled.json").write_text(json.dumps(compiled.describe(), indent=2, allow_nan=False))
    env, runner = None, None
    try:
        env = ManagerBasedRlEnv(cfg=compiled.env_cfg, device=device, render_mode="rgb_array" if cfg.training.video else None)
        if cfg.training.video:
            env = VideoRecorder(env, video_folder=log_dir / "videos", step_trigger=lambda s: s % cfg.training.video_interval == 0,
                                video_length=cfg.training.video_length, disable_logger=True)
        env = RslRlVecEnvWrapper(env, clip_actions=10.0)
        agent = ri_4438_him_runner_cfg(cfg)
        (log_dir / "agent.yaml").write_text(yaml.safe_dump(plain(agent), sort_keys=False))
        runner = LPACRLRunner(env, asdict(agent), str(log_dir), device)
        if cfg.training.resume:
            runner.load(cfg.training.resume)
        elif cfg.training.warm_start:
            runner.warm_start(cfg.training.warm_start)
        runner.learn(cfg.training.max_iterations, init_at_random_ep_len=False)
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            if env is not None:
                env.close()
    return log_dir


def main(argv=None):
    args = parser().parse_args(argv)
    overrides = list(args.set)
    for key in ("device", "resume", "warm_start", "output_dir", "num_envs", "max_iterations", "video"):
        value = getattr(args, key)
        if value is not None:
            overrides.append(f"training.{key}=" + json.dumps(value))
    cfg = resolve_experiment(args.preset, args.config, overrides)
    compiled = compile_experiment(cfg)
    if args.describe:
        print(json.dumps(compiled.describe(), indent=2, ensure_ascii=False, allow_nan=False))
        return
    print(f"LPACRL: {compiled.catalog.num_tasks} tasks; budget {compiled.describe()['evaluation_budget']}")
    print(f"Run saved at {run_training(compiled)}")


if __name__ == "__main__":
    main()
