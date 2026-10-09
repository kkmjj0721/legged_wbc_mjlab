from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  promotion_fraction: float = 0.8,
  min_commanded_distance: float = 1.0,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, UniformVelocityCommand)

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2]
    - command_term.episode_start_pos_w[env_ids],
    dim=1,
  )

  # 提前结束仍计入剩余时间的要求，避免摔倒降低考核难度。
  time_left = (
    env.max_episode_length_s
    - env.episode_length_buf[env_ids] * env.step_dt
  )
  commanded_distance = (
    torch.norm(command_term.commanded_displacement_w[env_ids], dim=1)
    + torch.norm(command_term.command[env_ids, :2], dim=1)
    * time_left.clamp(min=0.0)
  )

  move_up = distance > torch.clamp(
    commanded_distance * promotion_fraction,
    max=terrain_generator.size[0] / 2,
  )
  move_up &= commanded_distance >= min_commanded_distance

  move_down = distance < commanded_distance * 0.5
  move_down &= ~move_up

  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  terrain.update_env_origins(env_ids, move_up, move_down)

  return terrain.terrain_levels.float().mean()


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class RewardBasedVelocityCurriculum(ManagerTermBase):
  """Expand vx/vy from planar tracking and wz from angular tracking.

  Aggregate completed episodes over each check interval. Scores use the full
  episode duration, so early termination lowers the score as in legged_gym.
  Global progress survives per-environment resets and is saved by the HIM runner.
  """

  _range_names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
  _reward_names = ("track_linear_velocity", "track_angular_velocity")

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    self.command_name = params.get("command_name", "twist")
    self.command_term = env.command_manager.get_term(self.command_name)
    self.thresholds = (
      float(params.get("linear_threshold", 0.8)),
      float(params.get("angular_threshold", 0.8)),
    )
    self.increments = (
      float(params.get("linear_increment", 0.1)),
      float(params.get("angular_increment", 0.1)),
    )
    self.limits = {
      name: tuple(params.get(f"{name}_limit", (-1.0, 1.0)))
      for name in self._range_names
    }
    interval_s = float(params.get("check_interval_s", env.max_episode_length_s))
    if not math.isfinite(interval_s) or interval_s <= 0:
      raise ValueError("Velocity curriculum check_interval_s must be positive.")
    if any(not 0.0 <= value <= 1.0 for value in self.thresholds):
      raise ValueError("Velocity curriculum thresholds must be between 0 and 1.")
    if any(not math.isfinite(value) or value <= 0 for value in self.increments):
      raise ValueError("Velocity curriculum increments must be positive.")
    for name, (low, high) in self.limits.items():
      if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"Invalid velocity curriculum limit for {name}.")
      initial_low, initial_high = getattr(self.command_term.cfg.ranges, name)
      if not low <= initial_low <= initial_high <= high:
        raise ValueError(f"Initial {name} range must be inside its curriculum limit.")
    self.check_interval_steps = max(1, math.ceil(interval_s / env.step_dt))
    self.last_check_step = env.common_step_counter
    self.episode_count = 0
    self.reward_sums = torch.zeros(2, device=self.device)
    self.scores = torch.zeros_like(self.reward_sums)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    **params,
  ) -> dict[str, torch.Tensor | float]:
    del params  # The manager passes the same parameters used by __init__.
    if env.common_step_counter == 0:
      return self._metrics()
    if isinstance(env_ids, slice):
      env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
    # Ignore fresh/manual resets that have not collected any episode data.
    env_ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(env_ids):
      self.reward_sums += torch.stack([
        env.reward_manager._episode_sums[name][env_ids].sum()
        for name in self._reward_names
      ])
      self.episode_count += len(env_ids)
    if (
      env.common_step_counter - self.last_check_step < self.check_interval_steps
      or self.episode_count == 0
    ):
      return self._metrics()

    weights = torch.tensor([
      env.reward_manager.get_term_cfg(name).weight for name in self._reward_names
    ], device=self.device)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
      raise ValueError("Velocity curriculum tracking rewards need positive weights.")
    duration = (
      env.max_episode_length_s
      if env.reward_manager._scale_by_dt else env.max_episode_length
    )
    self.scores.copy_(self.reward_sums / (weights * duration * self.episode_count))
    if self.scores[0] > self.thresholds[0]:
      self._expand("lin_vel_x", self.increments[0])
      self._expand("lin_vel_y", self.increments[0])
    self.reward_sums.zero_()
    self.episode_count = 0
    self.last_check_step = env.common_step_counter
    return self._metrics()

  def _expand(self, name: str, increment: float) -> None:
    low, high = getattr(self.command_term.cfg.ranges, name)
    limit_low, limit_high = self.limits[name]
    setattr(self.command_term.cfg.ranges, name, (
      round(max(low - increment, limit_low), 10),
      round(min(high + increment, limit_high), 10),
    ))

  def _metrics(self) -> dict[str, torch.Tensor | float]:
    result: dict[str, torch.Tensor | float] = {
      "linear_score": self.scores[0], "angular_score": self.scores[1]
    }
    for name in self._range_names:
      low, high = getattr(self.command_term.cfg.ranges, name)
      result[f"{name}_min"] = low
      result[f"{name}_max"] = high
    return result

  def state_dict(self) -> dict:
    return {
      "version": 1,
      "command_name": self.command_name,
      "ranges": {
        name: tuple(getattr(self.command_term.cfg.ranges, name))
        for name in self._range_names
      },
      "last_check_step": self.last_check_step,
      "episode_count": self.episode_count,
      "reward_sums": self.reward_sums.detach().cpu().clone(),
      "scores": self.scores.detach().cpu().clone(),
    }

  def load_state_dict(self, state: dict) -> None:
    if state.get("version") != 1 or state.get("command_name") != self.command_name:
      raise ValueError("Incompatible velocity curriculum checkpoint state.")
    ranges = self._checked_ranges(state["ranges"])
    reward_sums = torch.as_tensor(state["reward_sums"], device=self.device)
    scores = torch.as_tensor(state["scores"], device=self.device)
    if (
      reward_sums.shape != (2,) or scores.shape != (2,)
      or not torch.isfinite(reward_sums).all() or not torch.isfinite(scores).all()
      or (reward_sums < 0).any() or (scores < 0).any()
      or int(state["episode_count"]) < 0 or int(state["last_check_step"]) < 0
    ):
      raise ValueError("Invalid velocity curriculum checkpoint statistics.")
    for name, bounds in ranges.items():
      setattr(self.command_term.cfg.ranges, name, bounds)
    self.reward_sums.copy_(reward_sums)
    self.scores.copy_(scores)
    self.episode_count = int(state["episode_count"])
    self.last_check_step = min(int(state["last_check_step"]), self._env.common_step_counter)

  def restore_legacy_ranges(self, saved_env_cfg: dict) -> bool:
    """Recover an older checkpoint's ranges from its saved step-based schedule."""
    ranges = dict(
      saved_env_cfg.get("commands", {}).get(self.command_name, {}).get("ranges", {})
    )
    if any(name not in ranges for name in self._range_names):
      return False
    stages = (
      saved_env_cfg.get("curriculum", {}).get("command_vel", {})
      .get("params", {}).get("velocity_stages", [])
    )
    for stage in stages:
      if self._env.common_step_counter > stage["step"]:
        for name in self._range_names:
          if stage.get(name) is not None:
            ranges[name] = stage[name]
    ranges = self._checked_ranges(ranges)
    for name, bounds in ranges.items():
      setattr(self.command_term.cfg.ranges, name, bounds)
    self.last_check_step = self._env.common_step_counter
    self.reward_sums.zero_()
    self.scores.zero_()
    self.episode_count = 0
    return True

  def _checked_ranges(self, ranges: dict) -> dict[str, tuple[float, float]]:
    ranges = dict(ranges)
    ranges["ang_vel_z"] = self.command_term.cfg.ranges.ang_vel_z
    result = {}
    for name, (limit_low, limit_high) in self.limits.items():
      low, high = (float(value) for value in ranges[name])
      if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"Invalid checkpoint command range for {name}.")
      # Respect any tighter limits explicitly configured for the resumed run.
      result[name] = (
        min(max(low, limit_low), limit_high),
        min(max(high, limit_low), limit_high),
      )
    return result


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
