"""Resume-training lab for the phase-agnostic foot-clearance "window" design.

Loads the RI-4438 HIM task, optionally swaps the ``foot_clearance`` term for
:class:`tests.exp.clearance_window.FeetClearanceWindow` **in memory only**
(nothing under ``src/`` is touched), resumes from an existing checkpoint and
runs continued training.

Examples
--------
# smoke: build 1024 envs, step twice with zero actions, print memory/time
python -m tests.exp.window_lab --group window --smoke --num-envs 1024

# continued training of the new design from the reference checkpoint
python -m tests.exp.window_lab --group window \
    --resume /home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/\
2026-09-30_10-05-21/model_10500.pt \
    --iters 150 --out logs/window/win_run1

# control: same budget with the untouched phase-gated clearance term
python -m tests.exp.window_lab --group control --resume <same> --iters 150 \
    --out logs/window/ctl_run1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from tests.exp.clearance_window import FeetClearanceWindow  # noqa: E402

# Design-doc defaults (docs/RI4438-保留相位的足端间隙奖励设计方案.md, section 4).
WINDOW_PARAMS: dict[str, Any] = {
  "height_range": (0.10, 0.20),
  "rest_height": 0.01573,
  "step_window": 1.0,
  "late_ramp": 0.4,
  "min_air_time": 0.04,
  "contact_confirm": 0.04,
  "support_height": 0.04,
  "min_support_force": 1.0,
  "wall_force_ratio": 5.0,
  "command_name": "twist",
  "command_threshold": 0.1,
  "sensor_name": "feet_ground_contact",
  "height_sensor_name": "feet_terrain_height",
}

DEFAULT_RESUME = (
  "/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/"
  "2026-09-30_10-05-21/model_10500.pt"
)


def _load_cfgs(task: str):
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  return load_env_cfg(task), load_rl_cfg(task)


def build_env_cfg(args: argparse.Namespace):
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  env_cfg, agent_cfg = _load_cfgs(args.task)

  term = env_cfg.rewards["foot_clearance"]
  if args.group == "window":
    site_names = tuple(term.params["asset_cfg"].site_names or args.foot_sites.split(","))
    params = dict(WINDOW_PARAMS)
    params.update(json.loads(args.window_params) if args.window_params else {})
    params["height_range"] = tuple(params["height_range"])
    params["asset_cfg"] = SceneEntityCfg("robot", site_names=site_names)
    term.func = FeetClearanceWindow
    term.params = params
    term.weight = args.clearance_weight if args.clearance_weight is not None else -0.25
  elif args.group == "control":
    if args.clearance_weight is not None:
      term.weight = args.clearance_weight
  else:
    raise ValueError(f"unknown group {args.group!r}")

  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  if args.terrain_init_level is not None:
    # Resumed runs otherwise restart the terrain curriculum at a uniform
    # random level in [0, num_rows-1], far harder than the level the
    # checkpoint was trained to.
    env_cfg.scene.terrain.max_init_terrain_level = int(args.terrain_init_level)
  if args.num_steps_per_env is not None:
    agent_cfg.num_steps_per_env = args.num_steps_per_env
  if args.iters is not None:
    agent_cfg.max_iterations = args.iters
  agent_cfg.save_interval = max(1, args.save_interval)

  if args.lr is not None:
    # Pin the LR instead of inheriting the adaptive schedule's restored 1e-5
    # and letting it ramp 1.5x per iteration.
    agent_cfg.algorithm.learning_rate = float(args.lr)
    agent_cfg.algorithm.schedule = args.schedule
  for override in args.agent_set or []:
    dotted, _, raw = override.partition("=")
    obj: Any = agent_cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
      obj = obj[int(part)] if part.isdigit() else (
        obj[part] if isinstance(obj, dict) else getattr(obj, part)
      )
    setattr(obj, parts[-1], json.loads(raw))

  for override in args.set or []:
    dotted, _, raw = override.partition("=")
    obj: Any = env_cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
      obj = obj[int(part)] if part.isdigit() else (
        obj[part] if isinstance(obj, dict) else getattr(obj, part)
      )
    value = json.loads(raw)
    if parts[-1].isdigit():
      obj[int(parts[-1])] = value
    elif isinstance(obj, dict):
      obj[parts[-1]] = value
    else:
      setattr(obj, parts[-1], value)

  return env_cfg, agent_cfg


def compatible_agent_yaml(cfg: dict) -> dict:
  """Match the historical ``params/agent.yaml`` schema.

  The archived runs (2026-09-29/30) were written by a checkout whose
  ``dump_yaml`` dropped ``None`` values, so their ``actor`` entry has no
  ``rnn_*``/``cnn_cfg`` keys.  The installed ``rsl_rl`` dataclasses default
  ``rnn_hidden_dim`` to 256, which makes the repository's checkpoint loader
  (``src/utils/training_report.py:load_checkpoint_actor``) refuse to build a
  simulation actor for the fixed-scenario evaluation.  Pruning those unused
  fields keeps the saved metadata compatible without changing the policy: the
  HIM actor has ``rnn_type=None``, so no RNN is ever constructed.
  """
  unused = {"rnn_type", "rnn_hidden_dim", "rnn_num_layers", "cnn_cfg"}

  def clean(obj):
    if isinstance(obj, dict):
      return {k: clean(v) for k, v in obj.items() if v is not None and k not in unused}
    return obj

  return clean(cfg)


def _gpu_stats() -> dict[str, float]:
  if not torch.cuda.is_available():
    return {}
  free, total = torch.cuda.mem_get_info()
  return {
    "gpu_allocated_MiB": round(torch.cuda.memory_allocated() / 2**20, 1),
    "gpu_reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 1),
    "gpu_peak_MiB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    "gpu_free_MiB": round(free / 2**20, 1),
    "gpu_total_MiB": round(total / 2**20, 1),
  }


def run(args: argparse.Namespace) -> dict[str, Any]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
  configure_torch_backends()

  env_cfg, agent_cfg = build_env_cfg(args)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  out_dir = Path(args.out) if args.out else Path("logs/window") / f"{args.group}_{int(time.time())}"
  out_dir.mkdir(parents=True, exist_ok=True)

  info: dict[str, Any] = {
    "group": args.group,
    "task": args.task,
    "resume": args.resume,
    "num_envs": int(env_cfg.scene.num_envs),
    "num_steps_per_env": int(agent_cfg.num_steps_per_env),
    "iters": int(agent_cfg.max_iterations),
    "clearance_weight": float(env_cfg.rewards["foot_clearance"].weight),
    "clearance_func": getattr(env_cfg.rewards["foot_clearance"].func, "__name__", str(env_cfg.rewards["foot_clearance"].func)),
    "window_params": {
      k: v for k, v in env_cfg.rewards["foot_clearance"].params.items() if k != "asset_cfg"
    },
  }

  if args.dry_run:
    info["dry_run"] = True
    print(json.dumps(info, indent=2, default=str))
    return info

  t0 = time.time()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  info["build_seconds"] = round(time.time() - t0, 2)
  info.update(_gpu_stats())

  if args.smoke:
    _smoke(env, args, info)
    env.close()
    return info

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), str(out_dir), device)
  if args.resume:
    runner.load(args.resume)
    info["loaded_iteration"] = int(runner.current_learning_iteration)
  info["memory_after_runner_MiB"] = round(torch.cuda.memory_allocated() / 2**20, 1)

  # Make the output directory a first-class training run for the repository's
  # analysis / fixed-condition evaluation tooling (scripts/best_model.py).
  from mjlab.utils.os import dump_yaml
  from src.utils.training_chain import write_lineage

  (out_dir / "params").mkdir(parents=True, exist_ok=True)
  dump_yaml(out_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(out_dir / "params" / "agent.yaml", compatible_agent_yaml(asdict(agent_cfg)))
  write_lineage(
    out_dir,
    args.task,
    Path(args.resume) if args.resume else None,
    int(runner.current_learning_iteration) if args.resume else None,
  )
  try:
    runner.add_git_repo_to_log(__file__)
  except Exception as exc:  # pragma: no cover - git metadata is best effort
    print(f"[window_lab] git metadata skipped: {exc}")

  t1 = time.time()
  runner.learn(
    num_learning_iterations=int(agent_cfg.max_iterations),
    init_at_random_ep_len=True,
  )
  info["train_seconds"] = round(time.time() - t1, 1)
  info["seconds_per_iter"] = round(
    info["train_seconds"] / max(1, int(agent_cfg.max_iterations)), 2
  )
  info.update(_gpu_stats())
  env.close()

  info["summary"] = summarize(out_dir)
  (out_dir / "experiment.json").write_text(json.dumps(info, indent=2, default=str))
  return info


def _smoke(env: Any, args: argparse.Namespace, info: dict[str, Any]) -> None:
  actions = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  diag: dict[str, list[float]] = {}
  t0 = time.time()
  steps = args.smoke_steps
  for _ in range(steps):
    env.step(actions)
    for key, value in (env.extras.get("log") or {}).items():
      if key.startswith("Diag/"):
        diag.setdefault(key, []).append(float(value))
  info["smoke_seconds"] = round(time.time() - t0, 2)
  info["smoke_seconds_per_step"] = round((time.time() - t0) / steps, 4)
  info["smoke_diagnostics"] = {k: round(sum(v) / len(v), 5) for k, v in diag.items()}
  info["reward_terms"] = sorted(env.reward_manager.active_terms)
  info.update(_gpu_stats())
  print(json.dumps(info, indent=2, default=str))


def summarize(log_dir: Path, tail: int = 50) -> dict[str, float]:
  """Average the last ``tail`` values of selected TensorBoard scalars."""
  import glob

  from tensorboard.backend.event_processing import event_accumulator as ea

  events = sorted(glob.glob(str(log_dir / "events.out.tfevents.*")))
  if not events:
    return {}
  acc = ea.EventAccumulator(events[-1], size_guidance={ea.SCALARS: 0})
  acc.Reload()
  keys = [
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Curriculum/terrain_levels",
    "Metrics/twist/error_vel_xy",
    "Metrics/twist/error_vel_yaw",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/track_angular_velocity",
    "Episode_Reward/foot_clearance",
    "Episode_Reward/foot_gait",
    "Episode_Reward/stumble",
    "Episode_Termination/fell_over",
    "Metrics/slip_velocity_mean",
    "Perf/total_fps",
    "Diag/Window/raw_cost",
    "Diag/Window/last_deficit",
    "Diag/Window/age",
    "Diag/Window/late_frac",
    "Diag/Window/full_cost_frac",
    "Diag/Window/support_frac",
    "Diag/Window/flight_frac",
    "Diag/Window/peak_mean",
    "Diag/Window/steps_settled",
    "Diag/Window/total_settled",
  ]
  out: dict[str, float] = {}
  tags = set(acc.Tags()["scalars"])
  for key in keys:
    if key not in tags:
      continue
    values = [s.value for s in acc.Scalars(key)][-tail:]
    if values:
      out[key] = round(sum(values) / len(values), 5)
  return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--task", default="Ri-4438-HIM-Rough")
  p.add_argument("--group", default="window", choices=("window", "control"))
  p.add_argument("--resume", default=DEFAULT_RESUME)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--terrain-init-level", type=int, default=None,
                 help="max_init_terrain_level for resumed runs (default: mjlab's None = full random)")
  p.add_argument("--lr", type=float, default=None, help="pin algorithm.learning_rate")
  p.add_argument("--schedule", default="fixed", choices=("fixed", "adaptive"))
  p.add_argument("--agent-set", action="append", default=[], help="dotted agent-cfg override key=json")
  p.add_argument("--num-steps-per-env", type=int, default=None)
  p.add_argument("--iters", type=int, default=None)
  p.add_argument("--save-interval", type=int, default=25)
  p.add_argument("--clearance-weight", type=float, default=None)
  p.add_argument("--window-params", default=None, help="JSON dict merged over the doc defaults")
  p.add_argument("--out", default=None)
  p.add_argument("--gpu", default="0")
  p.add_argument("--foot-sites", default="FL,FR,RL,RR")
  p.add_argument("--set", action="append", default=[], help="dotted env-cfg override key=json")
  p.add_argument("--smoke", action="store_true")
  p.add_argument("--smoke-steps", type=int, default=5)
  p.add_argument("--dry-run", action="store_true")
  return p.parse_args(argv)


def main() -> None:
  args = parse_args()
  info = run(args)
  print(json.dumps(info, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
  main()
