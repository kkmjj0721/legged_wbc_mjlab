from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .policy import OnnxPolicy
from .runtime import DeploymentRuntime
from .task_registry import task_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generic sim2sim/sim2real-style deployment runner")
    parser.add_argument("--task", default="ri_4438_ppo", help="task folder under deploy/task")
    parser.add_argument("--config", type=str, default=None, help="override task config YAML")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--input", choices=("keyboard", "joystick", "both"), default=None)
    parser.add_argument("--auto-stand", action="store_true")
    args = parser.parse_args(argv)
    config_path = task_config(args.task) if args.config is None else Path(args.config)
    policy_path = None if args.no_policy else (None if args.policy is None else Path(args.policy))
    cfg = load_config(config_path, args.task, policy_path)
    if args.input is not None:
        cfg.input_backend = args.input
    if args.auto_stand:
        cfg.auto_stand = True
    if args.no_policy:
        cfg.policy = None
    try:
        policy = OnnxPolicy(cfg.policy, cfg.observation_dim, cfg.action_dim)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"[ERROR] Cannot initialize policy: {exc}\n"
            "Install deployment dependencies with `uv sync --extra sim2sim`, "
            "or pass --no-policy for a MuJoCo/FSM smoke test."
        ) from None
    runtime = DeploymentRuntime(cfg, policy)
    try:
        print(f"[INFO] task={cfg.task_name} backend=mujoco policy={'enabled' if policy.available else 'disabled'}")
        runtime.run(headless=args.headless or not cfg.render, max_steps=args.steps)
    finally:
        runtime.close()
    return 0
