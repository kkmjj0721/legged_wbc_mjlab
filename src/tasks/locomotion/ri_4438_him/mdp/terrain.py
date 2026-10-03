"""Shared, reward-only terrain difficulty for the blind RI-4438 policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class TerrainDifficultyCfg:
  sensor_name: str = "terrain_scan"
  radius: float = 0.35
  flat_span: float = 0.02
  rough_span: float = 0.08
  rise_time: float = 0.06
  fall_time: float = 0.20
  min_valid_points: int = 6

  def __post_init__(self):
    if not (self.radius > 0 and 0 <= self.flat_span < self.rough_span):
      raise ValueError("Invalid terrain difficulty radius or height thresholds")
    if min(self.rise_time, self.fall_time) <= 0 or self.min_valid_points < 2:
      raise ValueError("Terrain smoothing times must be positive and need >=2 points")


class _TerrainDifficulty:
  def __init__(self, env: ManagerBasedRlEnv, cfg: TerrainDifficultyCfg):
    self.cfg = cfg
    self.value = torch.zeros(env.num_envs, device=env.device)
    self.initialized = torch.zeros_like(self.value, dtype=torch.bool)
    self.last_step = torch.full_like(self.value, -1, dtype=torch.long)
    self.cached_step = -1

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.value[ids] = 0
    self.initialized[ids] = False
    self.last_step[ids] = -1
    self.cached_step = -1

  def compute(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    step = env.common_step_counter
    if self.cached_step == step:
      return self.value

    data = env.scene[self.cfg.sensor_name].data
    hits = data.hit_pos_w
    delta_xy = hits[..., :2] - data.frame_pos_w[:, :1, :2]
    valid = (
      (data.distances >= 0)
      & torch.isfinite(hits).all(dim=-1)
      & (data.normals_w[..., 2] > 0)
      & (delta_xy.square().sum(dim=-1) <= self.cfg.radius**2)
    )
    count = valid.sum(dim=-1)
    heights = hits[..., 2].masked_fill(~valid, float("inf")).sort(dim=-1).values
    # Interpolate quantiles over valid points only, without a per-env Python loop.
    heights = torch.where(torch.isfinite(heights), heights, 0.0)
    quantiles = torch.tensor([0.1, 0.9], device=env.device)
    ranks = (count - 1).clamp_min(0).unsqueeze(-1) * quantiles
    lower = ranks.floor().long()
    upper = ranks.ceil().long()
    values = heights.gather(1, lower).lerp(heights.gather(1, upper), ranks - lower)
    span = values[:, 1] - values[:, 0]
    x = ((span - self.cfg.flat_span) / (self.cfg.rough_span - self.cfg.flat_span)).clamp(0, 1)
    raw = x.square() * (3 - 2 * x)
    sufficient = count >= self.cfg.min_valid_points
    # A missing scan must not suddenly tighten constraints on rough terrain.
    raw = torch.where(sufficient, raw, self.value)
    tau = torch.where(raw > self.value, self.cfg.rise_time, self.cfg.fall_time)
    alpha = -torch.expm1(-env.step_dt / tau)
    smoothed = self.value.lerp(raw, alpha)
    updated = torch.where(self.initialized, smoothed, raw)
    needs_update = self.last_step != step
    self.value.copy_(torch.where(needs_update, updated, self.value))
    self.initialized |= sufficient & needs_update
    self.last_step.fill_(step)
    self.cached_step = step
    env.extras["log"]["Metrics/terrain_difficulty"] = self.value.mean()
    env.extras["log"]["Metrics/terrain_scan_valid_fraction"] = sufficient.float().mean()
    return self.value


def terrain_difficulty(
  env: ManagerBasedRlEnv, cfg: TerrainDifficultyCfg | None = None,
) -> torch.Tensor:
  """Return one shared q per env and control step; None preserves flat rewards."""
  if cfg is None:
    return torch.zeros(env.num_envs, device=env.device)
  state = getattr(env, "_ri4438_terrain_difficulty", None)
  if state is None:
    state = _TerrainDifficulty(env, cfg)
    env._ri4438_terrain_difficulty = state
  elif state.cfg != cfg:
    raise ValueError("All RI-4438 rewards must use the same TerrainDifficultyCfg")
  return state.compute(env)


def reset_terrain_difficulty(env: ManagerBasedRlEnv, env_ids=None) -> None:
  """Reset event: clear only the selected envs, including explicit mid-episode resets."""
  state = getattr(env, "_ri4438_terrain_difficulty", None)
  if state is not None:
    state.reset(env_ids)
