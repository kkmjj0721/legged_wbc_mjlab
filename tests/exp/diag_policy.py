"""Rollout diagnostics for foot-clearance rewards with a trained checkpoint.

Loads an existing policy (read-only), rolls it out on the training terrain
distribution, and reports how the *current* world-z clearance reward behaves
depending on the terrain level under the robot.  This is the piece of evidence
that shows whether a clearance term fights stair climbing.

Nothing is trained and nothing under ``src/`` is touched; the harness adds a
terrain-height ray sensor in memory to measure true foot clearance.

Example
-------
python -m tests.exp.diag_policy \
    --checkpoint /home/sunteng/Downloads/.../model_4500.pt \
    --num-envs 256 --steps 400 --out tests/exp/diag/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))


def _add_height_sensor(cfg, site_names) -> None:
  from mjlab.sensor import ObjRef, RingPatternCfg, TerrainHeightSensorCfg

  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    TerrainHeightSensorCfg(
      name="feet_terrain_height",
      frame=tuple(ObjRef(type="site", name=n, entity="robot") for n in site_names),
      pattern=RingPatternCfg.single_ring(radius=0.02, num_samples=4, include_center=True),
      ray_alignment="world",
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      reduction="min",
    ),
  )


def _stats(x: torch.Tensor) -> dict[str, float]:
  x = x.flatten().float()
  if x.numel() == 0:
    return {"n": 0}
  return {
    "n": int(x.numel()),
    "mean": round(x.mean().item(), 4),
    "p10": round(torch.quantile(x, 0.10).item(), 4),
    "p50": round(torch.quantile(x, 0.50).item(), 4),
    "p90": round(torch.quantile(x, 0.90).item(), 4),
    "max": round(x.max().item(), 4),
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Ri-4438-HIM-Rough")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--steps", type=int, default=400)
  ap.add_argument("--warmup", type=int, default=50)
  ap.add_argument("--command-threshold", type=float, default=0.1)
  ap.add_argument("--band", type=float, nargs=2, default=(0.10, 0.16))
  ap.add_argument("--site-small", type=float, default=0.05)
  ap.add_argument("--out", default=None)
  ap.add_argument("--dump-npz", default=None,
                  help="Save raw per-step arrays for offline curriculum analysis.")
  args = ap.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.episode_length_s = 20.0

  term = env_cfg.rewards["foot_clearance"]
  site_names = tuple(term.params["asset_cfg"].site_names or ())
  existing = {s.name for s in (env_cfg.scene.sensors or ())}
  if "feet_terrain_height" not in existing:
    _add_height_sensor(env_cfg, site_names)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task)
  runner = runner_cls(env, asdict(agent_cfg), "", device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  unwrapped = env.unwrapped
  asset = unwrapped.scene["robot"]
  # NOTE: the cfg-level SceneEntityCfg is unresolved (site_ids stays slice(None),
  # which would include the "imu" site).  The reward manager owns the resolved
  # copy, so read the ids from there.
  resolved = unwrapped.reward_manager.get_term_cfg("foot_clearance").params["asset_cfg"]
  site_ids = resolved.site_ids
  if isinstance(site_ids, slice):
    _, site_ids = asset.find_sites(tuple(resolved.site_names))
  print(f"[diag] resolved foot site ids: {site_ids}", flush=True)
  contact = unwrapped.scene.sensors["feet_ground_contact"]

  obs = env.get_observations()

  # Per-step collectors.
  rec: dict[str, list[torch.Tensor]] = {k: [] for k in (
    "foot_z", "clearance", "contact", "vel_xy", "command", "distance", "ep_len",
    "base_lin_vel_b",
  )}
  h_min, h_max = args.band

  with torch.no_grad():
    for step in range(args.steps):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      if step < args.warmup:
        continue
      foot_z = asset.data.site_pos_w[:, site_ids, 2].detach()
      clearance = torch.nan_to_num(
        unwrapped.scene.sensors["feet_terrain_height"].data.heights
      ).detach()
      in_contact = (contact.data.current_contact_time > 0).detach()
      if in_contact.shape[1] != foot_z.shape[1]:
        # Fall back to primary names if the sensor order differs.
        prim = list(contact.primary_names)
        perm = [next(i for i, n in enumerate(prim) if n.startswith(s)) for s in site_names]
        in_contact = in_contact[:, torch.tensor(perm, device=foot_z.device)]
      vel_xy = torch.norm(asset.data.site_lin_vel_w[:, site_ids, :2], dim=-1).detach()
      command = unwrapped.command_manager.get_command("twist").detach()
      distance = torch.norm(
        asset.data.root_link_pos_w[:, :2] - unwrapped.scene.env_origins[:, :2], dim=1
      ).detach()
      rec["distance"].append(distance.cpu())
      rec["ep_len"].append(unwrapped.episode_length_buf.detach().cpu())
      rec["base_lin_vel_b"].append(asset.data.root_link_lin_vel_b.detach().cpu())
      rec["foot_z"].append(foot_z.cpu())
      rec["clearance"].append(clearance.cpu())
      rec["contact"].append(in_contact.cpu())
      rec["vel_xy"].append(vel_xy.cpu())
      rec["command"].append(command.cpu())

  env.close()

  foot_z = torch.stack(rec["foot_z"])          # [T, B, F]
  clearance = torch.stack(rec["clearance"])    # [T, B, F]
  in_contact = torch.stack(rec["contact"])     # [T, B, F]
  vel_xy = torch.stack(rec["vel_xy"])          # [T, B, F]
  command = torch.stack(rec["command"])        # [T, B, 3]
  distance = torch.stack(rec["distance"])      # [T, B]
  ep_len = torch.stack(rec["ep_len"])          # [T, B]
  v_b = torch.stack(rec["base_lin_vel_b"])     # [T, B, 3]

  # Decompose the tracking reward:  r = exp(-(e_xy + 2*v_bz^2)/0.25)
  e_xy = torch.square(command[:, :, :2] - v_b[:, :, :2]).sum(-1)     # [T, B]
  e_z = torch.square(v_b[:, :, 2])                                   # [T, B]
  std = 0.5  # math.sqrt(0.25) -> variance 0.25, the published standard
  r_full = torch.exp(-(e_xy + 2.0 * e_z) / std**2)
  r_noz = torch.exp(-(e_xy) / std**2)

  # Emulate the terrain curriculum decision of `terrain_levels_vel` to see
  # whether progression is even possible under the current command sampling.
  ter_size_half = 4.0  # terrain_generator.size[0] / 2
  cmd_xy_norm = torch.linalg.vector_norm(command[:, :, :2], dim=-1)
  move_up = distance > ter_size_half
  move_down = distance < (cmd_xy_norm * 20.0 * 0.5)
  move_down = move_down & ~move_up

  moving = (torch.linalg.vector_norm(command[:, :, :2], dim=-1) > args.command_threshold) | (
    command[:, :, 2].abs() > args.command_threshold
  )
  moving = moving.unsqueeze(-1)                # [T, B, 1]

  swing = (~in_contact) & moving               # [T, B, F]
  stance = in_contact                          # [T, B, F]

  # Terrain height under a foot = world z - clearance.
  terrain_z = foot_z - clearance

  # Classify envs by the ground level they stand on: a "step" env has stance
  # feet clearly above the world origin plane.
  # Per-env ground level: median terrain height under the stance feet.  Envs
  # whose ground level is clearly above 0 are standing on a step / rough patch.
  per_env_step_level = torch.where(
    stance.any(dim=0), terrain_z.median(dim=0).values, torch.zeros_like(terrain_z[0])
  )  # [B, F]
  elevated = (per_env_step_level > args.site_small).any(dim=1)  # [B]

  def _sel(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    m = mask
    while m.dim() < values.dim():
      m = m.unsqueeze(-1)
    return values[m.expand_as(values)]

  swing_clear = _sel(swing, clearance)
  swing_worldz = _sel(swing, foot_z)
  swing_vel = _sel(swing, vel_xy)

  # Legacy reward decomposition per foot, split by terrain class.
  delta = (h_min - foot_z).clamp_min(0.0) + (foot_z - h_max).clamp_min(0.0)
  legacy_foot_cost = delta * vel_xy
  legacy_env_cost = legacy_foot_cost.sum(dim=-1)  # [T, B]
  flat_sel = ~elevated
  step_sel = elevated

  report = {
    "checkpoint": args.checkpoint,
    "tracking_decomposition": {
      "mean_e_xy_sq": round(float(e_xy.mean()), 4),
      "rms_e_xy": round(float(e_xy.mean().sqrt()), 4),
      "mean_v_bz_sq": round(float(e_z.mean()), 5),
      "rms_v_bz": round(float(e_z.mean().sqrt()), 4),
      "mean_2vz_sq": round(float(2.0 * e_z.mean()), 4),
      "reward_full": round(float(r_full.mean()), 4),
      "reward_without_vz_term": round(float(r_noz.mean()), 4),
      "reward_loss_from_vz_frac": round(
        float(1.0 - r_full.mean() / r_noz.mean()), 4
      ),
      "mean_v_bz": round(float(v_b[:, :, 2].mean()), 4),
      "frac_steps_vbz_gt_0.1": round(float((v_b[:, :, 2].abs() > 0.1).float().mean()), 4),
    },
    "curriculum_probe": {
      "move_up_frac": round(float(move_up.float().mean()), 4),
      "move_down_frac": round(float(move_down.float().mean()), 4),
      "distance_mean": round(float(distance.mean()), 3),
      "distance_p90": round(float(torch.quantile(distance.flatten(), 0.9)), 3),
      "episode_length_mean": round(float(ep_len.float().mean()), 1),
      "cmd_xy_norm_mean": round(float(cmd_xy_norm.mean()), 3),
      "cmd_active_frac": round(float((cmd_xy_norm > 0.1).float().mean()), 4),
    },
    "task": args.task,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "band": [h_min, h_max],
    "envs_on_elevated_terrain": int(elevated.sum()),
    "swing_clearance_terrain_relative": _stats(swing_clear),
    "swing_height_world_z": _stats(swing_worldz),
    "swing_foot_speed_xy": _stats(swing_vel),
    "swing_clearance_below_h_min_frac": float((swing_clear < h_min).float().mean()),
    "swing_clearance_above_h_max_frac": float((swing_clear > h_max).float().mean()),
    "stance_clearance": _stats(_sel(stance, clearance)),
    "terrain_height_under_stance": _stats(_sel(stance, terrain_z)),
    "legacy_cost_flat_envs": round(float(legacy_env_cost[:, flat_sel].mean()), 5),
    "legacy_cost_elevated_envs": round(float(legacy_env_cost[:, step_sel].mean()), 5),
    "legacy_foot_cost_mean_by_foot": [
      round(float(legacy_foot_cost[:, :, i].mean()), 5) for i in range(foot_z.shape[-1])
    ],
    "legacy_delta_by_foot": [
      round(float(delta[:, :, i].mean()), 5) for i in range(foot_z.shape[-1])
    ],
    "command_xy_norm_mean": round(
      float(torch.linalg.vector_norm(command[:, :, :2], dim=-1).mean()), 5
    ),
  }

  if args.dump_npz:
    import numpy as np

    npz = Path(args.dump_npz)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
      npz,
      distance=distance.numpy().astype("float32"),
      cmd_xy_norm=cmd_xy_norm.numpy().astype("float32"),
      clearance=clearance.numpy().astype("float32"),
      foot_z=foot_z.numpy().astype("float32"),
      in_contact=in_contact.numpy().astype("bool"),
      ep_len=ep_len.numpy().astype("int32"),
    )

  text = json.dumps(report, indent=2, ensure_ascii=False)
  print(text)
  if args.out:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


if __name__ == "__main__":
  main()
