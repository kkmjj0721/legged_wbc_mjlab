from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .policy import OnnxPolicy
from .runtime import DeploymentRuntime
from .task_registry import task_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generic sim2sim/sim2real-style deployment runner")
    parser.add_argument("--task", default="ri_4438_him", help="task folder under deploy/task")
    parser.add_argument("--config", type=str, default=None, help="override task config YAML")
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--input", choices=("keyboard", "joystick", "both"), default=None)
    parser.add_argument("--gamepad-name", type=str, default=None, help="substring used to select a gamepad")
    parser.add_argument("--gamepad-guid", type=str, default=None, help="SDL GUID used to select a gamepad")
    parser.add_argument("--gamepad-index", type=int, default=None, help="joystick index (0-based) to select")
    auto_stand = parser.add_mutually_exclusive_group()
    auto_stand.add_argument(
        "--auto-stand",
        dest="auto_stand",
        action="store_true",
        default=None,
        help="override the task config and auto-start the one-shot stand sequence",
    )
    auto_stand.add_argument(
        "--no-auto-stand",
        dest="auto_stand",
        action="store_false",
        help="override the task config and wait in PASSIVE for a manual stand command",
    )
    parser.add_argument(
        "--terrain",
        type=str,
        default=None,
        help="spawn preset from the task config (combined, checkerboard, stairs_5cm...stairs_14cm, heightfield)",
    )
    args = parser.parse_args(argv)
    config_path = task_config(args.task) if args.config is None else Path(args.config)
    policy_path = None if args.no_policy else (None if args.policy is None else Path(args.policy))
    cfg = load_config(config_path, args.task, policy_path, terrain_override=args.terrain)
    if args.input is not None:
        cfg.input_backend = args.input
    if args.gamepad_name is not None:
        cfg.gamepad_options["device_name"] = args.gamepad_name
    if args.gamepad_guid is not None:
        cfg.gamepad_options["device_guid"] = args.gamepad_guid
    if args.gamepad_index is not None:
        cfg.gamepad_options["device_index"] = args.gamepad_index
    if args.auto_stand is not None:
        cfg.auto_stand = args.auto_stand
    if args.no_policy:
        cfg.policy = None
    try:
        policy = OnnxPolicy(cfg.policy, cfg.observation_dim, cfg.action_dim, action_clip=cfg.action_clip)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"[ERROR] Cannot initialize policy: {exc}\n"
            "Install deployment dependencies with `uv sync --extra sim2sim`, "
            "or pass --no-policy for a MuJoCo/FSM smoke test."
        ) from None
    runtime = DeploymentRuntime(cfg, policy)
    try:
        print(
            f"[INFO] task={cfg.task_name} backend=mujoco terrain={cfg.terrain_name} "
            f"policy={'enabled' if policy.available else 'disabled'}"
        )
        print(f"[INFO] policy_path={cfg.policy if cfg.policy is not None else 'none'}")
        print(
            f"[INFO] history_order={cfg.history_order} history_size={cfg.history_size} "
            f"action_clip={cfg.action_clip if cfg.action_clip is not None else 'none'}"
        )
        print(
            f"[INFO] fsm_state={runtime.fsm.state.value} "
            f"flow=PASSIVE->GETUP->STAND_SETTLE->READY->STAND->RL->GETDOWN->PASSIVE "
            f"auto_stand={cfg.auto_stand} auto_enable_rl={cfg.auto_enable_rl} "
            f"stand_duration={cfg.stand_duration:.3f}s getdown_duration={cfg.getdown_duration:.3f}s"
        )
        if cfg.input_backend in ("joystick", "both"):
            options = cfg.gamepad_options
            print(
                "[INFO] gamepad="
                f"name={options.get('device_name') or 'any'} "
                f"guid={options.get('device_guid') or 'any'} "
                f"index={options.get('device_index') if options.get('device_index') is not None else 'any'} "
                f"deadzone={float(options.get('deadzone', 0.12)):.2f} "
                "mapping=A:stand X:RL B:disable Y:getdown Start:reset Back:estop+quit"
            )
        runtime.run(headless=args.headless or not cfg.render, max_steps=args.steps)
    finally:
        runtime.close()
    return 0
