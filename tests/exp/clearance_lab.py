"""Short-horizon A/B harness for foot-clearance reward designs (sandbox).

Runs a real (short) training / fine-tuning job on a task config whose
``foot_clearance`` term has been swapped in memory for one of the experimental
variants in :mod:`tests.exp.clearance_variants`.  Repository code is never
modified; every override lives only in the process that starts the run.

Typical use
-----------
# 1) smoke test: build the env, step twice, print diagnostics, exit
python -m tests.exp.clearance_lab --variant band_terrain_swing --smoke

# 2) 120-iteration fine-tune from an existing checkpoint
python -m tests.exp.clearance_lab \
    --variant band_terrain_swing \
    --resume /path/to/model_4500.pt \
    --iters 120 --out tests/exp/runs/terrain_swing

# 3) compare without touching the clearance term (control)
python -m tests.exp.clearance_lab --variant baseline_interval --iters 120 \
    --out tests/exp/runs/control
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

from tests.exp import clearance_variants as cv  # noqa: E402


# --------------------------------------------------------------------------- #
# Variant registry
# --------------------------------------------------------------------------- #
# Each entry: (callable, params) -- params are merged over the defaults below.
_COMMON = {"command_name": "twist", "command_threshold": 0.1}

VARIANTS: dict[str, dict[str, Any]] = {
  # --- historical / control -------------------------------------------------
  "baseline_interval": {
    "func": cv.feet_clearance_interval,
    "params": {"height_range": (0.10, 0.16), **_COMMON},
    "needs_height_sensor": False,
    "default_weight": -0.5,
  },
  "target_worldz": {
    "func": cv.feet_clearance_target,
    "params": {"target_height": 0.08, **_COMMON},
    "needs_height_sensor": False,
    "default_weight": -1.0,
  },
  # --- proposed -------------------------------------------------------------
  # Terrain-relative band, swing-gated, asymmetric, no velocity weighting.
  "band_terrain_swing": {
    "func": cv.ClearanceBand,
    "params": {
      "height_range": (0.05, 0.13),
      "ref": "terrain",
      "gate": "swing",
      "gain_below": 1.0,
      "gain_above": 0.25,
      "mode": "l2",
      "vel_weight": 0.0,
      "combine": "sum",
      "period": 0.6,
      "offset": [0.0, 0.5, 0.5, 0.0],
      "threshold": 0.56,
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "feet_terrain_height",
      **_COMMON,
    },
    "needs_height_sensor": True,
    "default_weight": -0.4,
  },
  # Same, but gated by the trot clock instead of measured contact.
  "band_terrain_phase": {
    "func": cv.ClearanceBand,
    "params": {
      "height_range": (0.05, 0.13),
      "ref": "terrain",
      "gate": "phase",
      "gain_below": 1.0,
      "gain_above": 0.25,
      "mode": "l2",
      "vel_weight": 0.0,
      "combine": "sum",
      "period": 0.6,
      "offset": [0.0, 0.5, 0.5, 0.0],
      "threshold": 0.56,
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "feet_terrain_height",
      **_COMMON,
    },
    "needs_height_sensor": True,
    "default_weight": -0.4,
  },
  # Proprioceptive reference: height above the foot's own last contact height.
  # No extra sensor, no privileged information.
  "band_touchdown_swing": {
    "func": cv.ClearanceBand,
    "params": {
      "height_range": (0.04, 0.12),
      "ref": "touchdown",
      "gate": "swing",
      "gain_below": 1.0,
      "gain_above": 0.25,
      "mode": "l2",
      "vel_weight": 0.0,
      "combine": "sum",
      "touchdown_tau": 0.2,
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "feet_terrain_height",
      **_COMMON,
    },
    "needs_height_sensor": False,
    "default_weight": -0.4,
  },
  # World-z version of the same swing-gated band (isolates the reference frame).
  "band_world_swing": {
    "func": cv.ClearanceBand,
    "params": {
      "height_range": (0.06, 0.14),
      "ref": "world",
      "gate": "swing",
      "gain_below": 1.0,
      "gain_above": 0.25,
      "mode": "l2",
      "vel_weight": 0.0,
      "combine": "sum",
      "sensor_name": "feet_ground_contact",
      **_COMMON,
    },
    "needs_height_sensor": False,
    "default_weight": -0.4,
  },
  # Positive incentive instead of a penalty.
  "reward_terrain_swing": {
    "func": cv.ClearanceReward,
    "params": {
      "ref": "terrain",
      "gate": "swing",
      "target_height": 0.09,
      "sigma": 0.04,
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "feet_terrain_height",
      "period": 0.6,
      "offset": [0.0, 0.5, 0.5, 0.0],
      "threshold": 0.56,
      **_COMMON,
    },
    "needs_height_sensor": True,
    "default_weight": 0.3,
  },
}

# Tracking-precision presets applied to other reward terms.
TRACKING_PRESETS: dict[str, dict[str, Any]] = {
  "off": {},
  "sharp_ang": {
    # legged_gym uses std 0.5 (lin) / 0.25 (yaw); angular yaw here is 0.5.
    "rewards.track_angular_velocity.params.std": 0.25,
  },
  "sharp_both": {
    "rewards.track_linear_velocity.params.std": 0.35,
    "rewards.track_angular_velocity.params.std": 0.25,
  },
  "stronger_track": {
    "rewards.track_linear_velocity.weight": 4.0,
    "rewards.track_angular_velocity.weight": 3.0,
  },
  "soft_gait": {
    "rewards.foot_gait.weight": 0.2,
  },
}


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def _load_cfgs(task: str):
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  return load_env_cfg(task), load_rl_cfg(task)


def _get_child(obj: Any, part: str) -> Any:
  if part.isdigit():
    return obj[int(part)]
  if isinstance(obj, dict):
    return obj[part]
  return getattr(obj, part)


def _set_child(obj: Any, part: str, value: Any) -> None:
  if part.isdigit():
    obj[int(part)] = value
  elif isinstance(obj, dict):
    obj[part] = value
  else:
    setattr(obj, part, value)


def _set_by_path(cfg: Any, dotted: str, value: Any) -> None:
  obj = cfg
  parts = dotted.split(".")
  for part in parts[:-1]:
    obj = _get_child(obj, part)
  _set_child(obj, parts[-1], value)


def _add_feet_terrain_height_sensor(cfg: Any, site_names: tuple[str, ...]) -> None:
  """Add the 4-site terrain-height ray sensor used by terrain-relative variants."""
  from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg

  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    TerrainHeightSensorCfg(
      name="feet_terrain_height",
      frame=tuple(
        ObjRef(type="site", name=name, entity="robot") for name in site_names
      ),
      pattern=RingPatternCfg.single_ring(radius=0.02, num_samples=4, include_center=True),
      ray_alignment="world",
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      reduction="min",
    ),
  )


def _make_terrain_curriculum(mode: str, value: float):
  """Return a replacement for the ``terrain_levels`` curriculum term.

  ``mode`` is one of ``relaxed`` (gentler demotion rule) or ``fixed`` (pin every
  env to a constant difficulty row so the policy is forced onto tall stairs).
  """
  import torch as _torch

  from mjlab.managers.scene_entity_config import SceneEntityCfg as _SceneEntityCfg

  if mode == "relaxed":

    def _relaxed(env, env_ids, command_name, asset_cfg=_SceneEntityCfg("robot")):
      asset = env.scene[asset_cfg.name]
      terrain = env.scene["terrain"]
      generator = terrain.cfg.terrain_generator
      command = env.command_manager.get_command(command_name)
      distance = _torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
      )
      move_up = distance > generator.size[0] / 2
      move_down = distance < (
        _torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * value
      )
      move_down = move_down & ~move_up
      terrain.update_env_origins(env_ids, move_up, move_down)
      return _torch.mean(terrain.terrain_levels.float())

    return _relaxed

  if mode == "fixed":
    level = int(value)

    def _fixed(env, env_ids, command_name, asset_cfg=_SceneEntityCfg("robot")):
      del command_name, asset_cfg
      terrain = env.scene["terrain"]
      max_row = terrain.terrain_origins.shape[0] - 1
      terrain.terrain_levels[env_ids] = min(level, max_row)
      terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
      ]
      return _torch.mean(terrain.terrain_levels.float())

    return _fixed

  return None


def build_env_cfg(args: argparse.Namespace):
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  env_cfg, agent_cfg = _load_cfgs(args.task)

  variant = VARIANTS[args.variant]
  term = env_cfg.rewards["foot_clearance"]
  site_names = tuple(term.params["asset_cfg"].site_names or ())
  if not site_names:
    site_names = tuple(args.foot_sites.split(","))
  params = dict(variant["params"])
  params["asset_cfg"] = SceneEntityCfg("robot", site_names=site_names)
  if variant["needs_height_sensor"]:
    existing = {s.name for s in (env_cfg.scene.sensors or ())}
    if "feet_terrain_height" not in existing:
      _add_feet_terrain_height_sensor(env_cfg, site_names)

  term.func = variant["func"]
  term.params = params
  term.weight = (
    variant["default_weight"] if args.clearance_weight is None else args.clearance_weight
  )

  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  if args.num_steps_per_env is not None:
    agent_cfg.num_steps_per_env = args.num_steps_per_env
  if args.iters is not None:
    agent_cfg.max_iterations = args.iters
  agent_cfg.save_interval = max(1, min(50, args.iters or 100))

  for key, value in TRACKING_PRESETS[args.tracking].items():
    _set_by_path(env_cfg, key, value)

  for override in args.set or []:
    dotted, _, raw = override.partition("=")
    _set_by_path(env_cfg, dotted, json.loads(raw))

  if args.terrain_curriculum != "default":
    mode, _, raw = args.terrain_curriculum.partition(":")
    func = _make_terrain_curriculum(mode, float(raw or 0.0))
    if func is not None:
      term_cfg = env_cfg.curriculum["terrain_levels"]
      term_cfg.func = func
      term_cfg.params = {"command_name": "twist"}

  return env_cfg, agent_cfg


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
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

  out_dir = Path(args.out) if args.out else Path("tests/exp/runs") / args.variant
  out_dir.mkdir(parents=True, exist_ok=True)

  t0 = time.time()
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  build_s = time.time() - t0

  info = {
    "variant": args.variant,
    "task": args.task,
    "num_envs": int(env_cfg.scene.num_envs),
    "num_steps_per_env": int(agent_cfg.num_steps_per_env),
    "iters": int(agent_cfg.max_iterations),
    "clearance_weight": float(env_cfg.rewards["foot_clearance"].weight),
    "tracking": args.tracking,
    "resume": args.resume,
    "build_seconds": round(build_s, 2),
  }
  if torch.cuda.is_available():
    info["gpu_allocated_MiB_after_build"] = round(
      torch.cuda.memory_allocated() / 2**20, 1
    )

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

  t1 = time.time()
  runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  info["train_seconds"] = round(time.time() - t1, 1)
  if torch.cuda.is_available():
    info["gpu_peak_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
  env.close()

  info["summary"] = summarize(out_dir)
  (out_dir / "experiment.json").write_text(json.dumps(info, indent=2))
  return info


def _smoke(env: Any, args: argparse.Namespace, info: dict[str, Any]) -> None:
  """Step the env a few times with zero actions and print reward diagnostics."""
  import torch as _torch

  actions = _torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  diag: dict[str, list[float]] = {}
  for _ in range(3):
    env.step(actions)
    for key, value in (env.extras.get("log") or {}).items():
      if key.startswith("Diag/"):
        diag.setdefault(key, []).append(float(value))
  info["smoke_diagnostics"] = {k: round(sum(v) / len(v), 5) for k, v in diag.items()}
  info["reward_terms"] = sorted(env.reward_manager.active_terms)
  info["num_sensors"] = sorted(env.scene.sensors.keys())
  print(json.dumps(info, indent=2))


def summarize(log_dir: Path, tail: int = 30) -> dict[str, float]:
  """Read the run's TensorBoard scalars and average the last ``tail`` points."""
  import glob

  from tensorboard.backend.event_processing import event_accumulator as ea

  events = sorted(glob.glob(str(log_dir / "events.out.tfevents.*")))
  if not events:
    return {}
  acc = ea.EventAccumulator(events[0], size_guidance={ea.SCALARS: 0})
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
    "Diag/Clearance/raw_cost",
    "Diag/Clearance/h_active_mean",
    "Diag/Clearance/active_fraction",
    "Diag/Clearance/below_frac",
    "Diag/Clearance/above_frac",
    "Diag/ClearanceReward/value",
    "Perf/total_fps",
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
  p.add_argument("--variant", default="baseline_interval", choices=sorted(VARIANTS))
  p.add_argument("--tracking", default="off", choices=sorted(TRACKING_PRESETS))
  p.add_argument("--clearance-weight", type=float, default=None)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--num-steps-per-env", type=int, default=None)
  p.add_argument("--iters", type=int, default=None)
  p.add_argument("--resume", default=None, help="absolute path to a .pt checkpoint")
  p.add_argument("--out", default=None, help="log directory for this run")
  p.add_argument("--gpu", default="0")
  p.add_argument("--foot-sites", default="FL,FR,RL,RR")
  p.add_argument("--set", action="append", default=[], help="extra dotted override, e.g. key=value")
  p.add_argument("--smoke", action="store_true", help="build env, step 3x, print diagnostics")
  p.add_argument("--no-obstacles", action="store_true", help=argparse.SUPPRESS)
  p.add_argument(
    "--terrain-curriculum",
    default="default",
    help=(
      "default | relaxed:<demote_factor, default 0.5->use 0.2> | fixed:<row, 0-9>. "
      "Applies in memory only."
    ),
  )
  return p.parse_args(argv)


def main() -> None:
  args = parse_args()
  info = run(args)
  print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
