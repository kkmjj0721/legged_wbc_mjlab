from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

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
) -> torch.Tensor:
  """ 重写速度课程训练
      实现：
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  """Increase the velocity command range after both command groups are solved.

  The original legged-gym implementation of this curriculum was a method on the
  environment, so it accessed ``self.episode_sums``, ``self.reward_scales`` and
  ``self.command_ranges`` directly.  In mjlab those values live in managers and
  in the command-term config instead:

  * ``env.reward_manager._episode_sums`` contains the completed episode sums.
  * ``env.reward_manager.get_term_cfg(name).weight`` is the reward scale.
  * ``env.command_manager.get_term(name).cfg.ranges`` is the mutable command
    range used by the next command resample.

  Environments with ids below 20% of ``num_envs`` are treated as the high-speed
  group; the remaining environments are the low-speed group.  The range is
  widened only when both groups have completed episodes whose normalized linear
  velocity tracking reward is above 80% of its maximum.  ``velocity_stages`` is
  used for the initial range and the final (maximum) range, while the adaptive
  increment is 0.2 m/s per successful update, matching the old curriculum.

  The curriculum manager calls this function before resetting the selected
  environments.  Consequently, the reward sums still contain the episode that
  has just finished at the time this function runs.
  """
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  # ``env_ids`` normally contains the environments that are about to reset.  A
  # full reset passes all ids, while a partial reset can contain only one of the
  # two groups.  Do not update the curriculum unless both means are defined.
  if env_ids is None or isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=getattr(env, "device", None))
  else:
    env_ids = env_ids.reshape(-1)

  # Apply the first stage only once, during the initial reset.  Without this,
  # the command config's (usually final) defaults would make the first stage
  # ineffective.  The marker is stored on the command term so it survives the
  # per-environment reset calls made by CurriculumManager.
  if not getattr(command_term, "_velocity_curriculum_initialized", False):
    if getattr(env, "common_step_counter", 0) == 0 and velocity_stages:
      first_stage = min(velocity_stages, key=lambda stage: stage["step"])
      for field_name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        stage_range = first_stage.get(field_name)
        if stage_range is not None:
          setattr(cfg.ranges, field_name, stage_range)
    setattr(command_term, "_velocity_curriculum_initialized", True)

  # Locate the linear-velocity tracking sum.  ``tracking_lin_vel`` is the name
  # used by legged-gym, while this task calls the term ``track_linear_velocity``.
  reward_manager = getattr(env, "reward_manager", None)
  episode_sums = getattr(reward_manager, "_episode_sums", None)
  if episode_sums is None:
    # Keep the function usable with lightweight test doubles and older manager
    # versions that exposed the buffer without the leading underscore.
    episode_sums = getattr(reward_manager, "episode_sums", None)
  if episode_sums is None:
    episode_sums = getattr(env, "episode_sums", None)

  tracking_name: str | None = None
  if episode_sums is not None:
    for candidate in (
      "track_linear_velocity",
      "tracking_lin_vel",
      "track_lin_vel",
    ):
      if candidate in episode_sums:
        tracking_name = candidate
        break

  tracking_sum = episode_sums[tracking_name] if tracking_name is not None else None

  # The reward manager stores weighted rewards.  With its default ``scale_by_dt``
  # setting, episode sums are in reward-seconds and must be divided by episode
  # duration; otherwise they are per-step sums and use max_episode_length.
  reward_weight = 1.0
  if tracking_name is not None and reward_manager is not None:
    try:
      reward_weight = float(reward_manager.get_term_cfg(tracking_name).weight)
    except (AttributeError, KeyError, TypeError, ValueError):
      reward_scales = getattr(env, "reward_scales", {})
      reward_weight = float(reward_scales.get(tracking_name, 1.0))

  scale_by_dt = bool(getattr(reward_manager, "_scale_by_dt", False))
  if scale_by_dt:
    episode_length = float(getattr(env, "max_episode_length_s", 0.0))
    if episode_length <= 0.0:
      episode_length = float(getattr(env, "max_episode_length", 0.0))
  else:
    episode_length = float(getattr(env, "max_episode_length", 0.0))
  if episode_length <= 0.0:
    episode_length = 1.0

  low_vel_env_ids = env_ids[env_ids > env.num_envs * 0.2]
  high_vel_env_ids = env_ids[env_ids < env.num_envs * 0.2]

  def _tracking_reward_mean(group_ids: torch.Tensor) -> float | None:
    if tracking_sum is None or group_ids.numel() == 0:
      return None
    sum_ids = group_ids.to(device=tracking_sum.device, dtype=torch.long)
    return float(torch.mean(tracking_sum[sum_ids]) / episode_length)

  low_reward = _tracking_reward_mean(low_vel_env_ids)
  high_reward = _tracking_reward_mean(high_vel_env_ids)
  solved_threshold = 0.8 * reward_weight
  if (
    low_reward is not None
    and high_reward is not None
    and low_reward > solved_threshold
    and high_reward > solved_threshold
  ):
    # The final stage supplies a task-specific upper/lower bound.  If no stages
    # are supplied, the current range is intentionally left unbounded here,
    # matching the old max_curriculum-based implementation.
    max_range: tuple[float, float] | None = None
    for stage in sorted(velocity_stages, key=lambda item: item["step"]):
      stage_range = stage.get("lin_vel_x")
      if stage_range is not None:
        max_range = stage_range

    current_range = cfg.ranges.lin_vel_x
    lower = current_range[0] - 0.2
    upper = current_range[1] + 0.2
    if max_range is not None:
      lower = max(lower, max_range[0])
      upper = min(upper, max_range[1])
    cfg.ranges.lin_vel_x = (lower, upper)

  return {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0], device=env_ids.device),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1], device=env_ids.device),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0], device=env_ids.device),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1], device=env_ids.device),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0], device=env_ids.device),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1], device=env_ids.device),
  }


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
