"""Reward terms for the standalone RI-4438 HIM task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.rewards import *  # noqa: F401, F403
from mjlab.tasks.velocity.mdp.rewards import *  # noqa: F401, F403
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .terrain import TerrainDifficultyCfg, terrain_difficulty

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_cfg: TerrainDifficultyCfg | None = None,
  rough_z_scale: float = 0.1,
) -> torch.Tensor:
  """ 
  跟踪期望线速度

  数学公式:
    r = exp( -( ||v*_xy - v^b_xy||^2 + 2 * a(q) * (v^b_z)^2 ) / std^2 )
    a(q) = 1 - (1 - rough_z_scale) * q

  符号说明:
    v*_xy  : 期望水平线速度 [vx*, vy*] (command[:, :2])
    v^b_xy : 机器人机体局部系当前水平线速度 (root_link_lin_vel_b[:, :2])
    v^b_z  : 机器人垂直线速度 (权重为 2，用于抑制躯干跳跃和上下颠簸)
    std    : 高斯核宽度参数
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  q = terrain_difficulty(env, terrain_cfg)
  z_scale = 1.0 - (1.0 - rough_z_scale) * q
  lin_vel_error = xy_error + 2.0 * z_scale * z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """ 
  跟踪期望角速度

  数学公式:
    r = exp( -( (w*_z - w^b_z)^2 + 0.05 * ||w^b_xy||^2 ) / std^2 )

  符号说明:
    w*_z   : 期望偏航角速度 (command[:, 2])
    w^b_z  : 机体局部系实际偏航角速度 (root_link_ang_vel_b[:, 2])
    w^b_xy : 机体横滚与俯仰角速度 [wx, wy] (权重 0.05，抑制机身侧倾晃动)
    std    : 高斯核宽度参数
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_cfg: TerrainDifficultyCfg | None = None,
  rough_scale: float = 0.5,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity = quat_apply_inverse(quat, asset.data.gravity_vec_w)
  else:
    gravity = asset.data.projected_gravity_b
  q = terrain_difficulty(env, terrain_cfg)
  return torch.square(gravity[:, :2]).sum(dim=1) * (1.0 - (1.0 - rough_scale) * q)


def angular_momentum_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: BuiltinSensor = env.scene[sensor_name]
  magnitude_sq = torch.square(sensor.data).sum(dim=-1)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.sqrt(magnitude_sq).mean()
  return magnitude_sq


def body_angular_velocity_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  velocity = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.square(velocity[:, :2]).sum(dim=1)


def feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
  terrain_cfg: TerrainDifficultyCfg | None = None,
  rough_strength: float = 0.15,
) -> torch.Tensor:
  """Keep the flat trot score and reduce timing costs on difficult terrain."""
  sensor: ContactSensor = env.scene[sensor_name]
  contact = sensor.data.current_contact_time > 0

  # Use the same integer-step phase as the actor/critic observations.
  period_steps = int(round(period / env.step_dt))
  phase = ((env.episode_length_buf % period_steps) / period_steps).unsqueeze(1)
  offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
  desired = ((phase + offsets) % 1.0) < threshold
  mismatch = (desired != contact).float().mean(dim=1)
  q = terrain_difficulty(env, terrain_cfg)
  strength = 1.0 - (1.0 - rough_strength) * q
  reward = 1.0 - strength * mismatch

  command = env.command_manager.get_command(command_name)
  if command is not None:
    active = (
      torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      > command_threshold
    ).float()
    reward *= active
  return reward


def stand_still(env: ManagerBasedRlEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  assert default is not None
  cost = torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]).sum(dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    cost *= (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) <= command_threshold).float()
  return cost


def hip_joint_deviation_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  terrain_cfg: TerrainDifficultyCfg | None = None,
  rough_scale: float = 0.3,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice) and asset_cfg.joint_names is None:
    joint_ids, _ = asset.find_joints(r".*_hip_joint")
  default = asset.data.default_joint_pos
  assert default is not None
  cost = torch.square(asset.data.joint_pos[:, joint_ids] - default[:, joint_ids]).sum(dim=1)
  q = terrain_difficulty(env, terrain_cfg)
  cost *= 1.0 - (1.0 - rough_scale) * q
  return cost * ((torch.abs(command[:, 1]) <= command_threshold) & (torch.abs(command[:, 2]) <= command_threshold)).to(cost.dtype)


class variable_posture:
  """Speed-dependent pose tolerance, widened on difficult terrain."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default = asset.data.default_joint_pos
    assert default is not None
    self.default_joint_pos = default
    _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.std_standing = torch.tensor(resolve_matching_names_values(cfg.params["std_standing"], names)[2], device=env.device)
    self.std_walking = torch.tensor(resolve_matching_names_values(cfg.params["std_walking"], names)[2], device=env.device)
    self.std_running = torch.tensor(resolve_matching_names_values(cfg.params["std_running"], names)[2], device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
    terrain_cfg: TerrainDifficultyCfg | None = None,
    rough_std_scale: float = 1.75,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    speed = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (speed < walking_threshold).unsqueeze(1)
    walking = ((speed >= walking_threshold) & (speed < running_threshold)).unsqueeze(1)
    std = torch.where(standing, self.std_standing, torch.where(walking, self.std_walking, self.std_running))
    q = terrain_difficulty(env, terrain_cfg)
    std = std * (1.0 + (rough_std_scale - 1.0) * q.unsqueeze(1))
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(error) / torch.square(std), dim=1))


def _support_contact(
  sensor: ContactSensor, min_force: float, horizontal_ratio: float,
) -> torch.Tensor:
  """World-frame vertical support; a side-wall hit alone is not a landing."""
  data = sensor.data
  assert data.found is not None and data.force is not None
  force = data.force
  vertical = force[..., 2].abs()
  return (
    (data.found > 0)
    & torch.isfinite(force).all(dim=-1)
    & (vertical > min_force)
    & (torch.linalg.vector_norm(force[..., :2], dim=-1) <= horizontal_ratio * vertical)
  )


def _moving_command(env: ManagerBasedRlEnv, name: str, threshold: float) -> torch.Tensor:
  command = env.command_manager.get_command(name)
  assert command is not None, f"Command '{name}' not found."
  return torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs() > threshold


def feet_clearance_adaptive(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  scan_sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.1,
  foot_radius: float = 0.01573,
  clearance_margin: float = 0.025,
  path_length: float = 0.10,
  path_half_width: float = 0.04,
  speed_scale: float = 0.5,
  max_deficit: float = 2.0,
  support_force: float = 1.0,
  support_horizontal_ratio: float = 1.0,
) -> torch.Tensor:
  """One-sided sole clearance cost in a short capsule along actual foot motion.

  The scan originates at the base, so an obstacle above the foot remains visible.
  Only upward, valid terrain hits inside each foot's capsule participate. Missing
  local samples contribute no cost. Height above the margin is never penalized.
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet = asset.data.site_pos_w[:, asset_cfg.site_ids]
  velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  finite_feet = torch.isfinite(feet).all(dim=-1) & torch.isfinite(velocity).all(dim=-1)
  feet = torch.nan_to_num(feet)
  velocity = torch.nan_to_num(velocity)
  speed = torch.linalg.vector_norm(velocity, dim=-1)
  direction = velocity / speed.clamp_min(1e-6).unsqueeze(-1)

  scan = env.scene[scan_sensor_name].data
  hits = scan.hit_pos_w
  valid_ray = (
    (scan.distances >= 0)
    & torch.isfinite(hits).all(dim=-1)
    & (scan.normals_w[..., 2] > 0)
  )
  delta = hits[:, None, :, :2] - feet[:, :, None, :2]
  along = (delta * direction[:, :, None]).sum(dim=-1).clamp(0, path_length)
  across = delta - along.unsqueeze(-1) * direction[:, :, None]
  nearby = (across.square().sum(dim=-1) <= path_half_width**2) & valid_ray[:, None]
  valid_terrain = nearby.any(dim=-1) & finite_feet
  obstacle_z = (
    hits[:, None, :, 2].expand_as(nearby)
    .masked_fill(~nearby, -float("inf")).amax(dim=-1)
  )
  obstacle_z = torch.where(valid_terrain, obstacle_z, feet[..., 2] - foot_radius)
  clearance = feet[..., 2] - foot_radius - obstacle_z
  deficit = ((clearance_margin - clearance) / clearance_margin).clamp(0, max_deficit)
  support = _support_contact(env.scene[sensor_name], support_force, support_horizontal_ratio)
  moving = _moving_command(env, command_name, command_threshold)
  mask = ~support & valid_terrain & moving.unsqueeze(1)
  motion_weight = (speed / speed_scale).clamp(0, 1)
  cost = (mask * motion_weight * deficit.square()).mean(dim=1)
  env.extras["log"]["Metrics/clearance_valid_fraction"] = valid_terrain.float().mean()
  env.extras["log"]["Metrics/clearance_deficit"] = (deficit * mask).sum() / mask.sum().clamp_min(1)
  return cost


class feet_swing_peak:
  """Flat-ground lift band, charged once after a debounced support landing.

  Peaks use world-z displacement from the last support, not instantaneous ground
  clearance. The maximum terrain difficulty during a swing relaxes this band.
  Rewards are per event: return cost / step_dt to cancel RewardManager's dt.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    shape = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].shape
    self.support_z = torch.zeros(shape, device=env.device)
    self.takeoff_z = torch.zeros_like(self.support_z)
    self.peak_lift = torch.zeros_like(self.support_z)
    self.swing_q = torch.zeros_like(self.support_z)
    self.air_time = torch.zeros_like(self.support_z)
    self.support_steps = torch.zeros_like(self.support_z, dtype=torch.long)
    self.has_support = torch.zeros_like(self.support_z, dtype=torch.bool)
    self.in_swing = torch.zeros_like(self.has_support)
    self.command_active = torch.zeros_like(self.has_support)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    for buffer in (
      self.support_z, self.takeoff_z, self.peak_lift, self.swing_q,
      self.air_time, self.support_steps, self.has_support, self.in_swing,
      self.command_active,
    ):
      buffer[ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    terrain_cfg: TerrainDifficultyCfg | None = None,
    command_threshold: float = 0.1,
    min_height: float = 0.06,
    max_height: float = 0.10,
    error_scale: float = 0.04,
    max_cost: float = 4.0,
    min_air_time: float = 0.04,
    support_force: float = 1.0,
    support_horizontal_ratio: float = 1.0,
    contact_debounce_steps: int = 2,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    finite = torch.isfinite(foot_z)
    foot_z = torch.nan_to_num(foot_z)
    support = _support_contact(env.scene[sensor_name], support_force, support_horizontal_ratio) & finite
    self.has_support &= finite
    self.in_swing &= finite
    self.support_steps.copy_(torch.where(support, self.support_steps + 1, 0))
    settled = self.support_steps >= contact_debounce_steps
    q = terrain_difficulty(env, terrain_cfg).unsqueeze(1)
    moving = _moving_command(env, command_name, command_threshold).unsqueeze(1)

    takeoff = self.has_support & ~support & ~self.in_swing
    self.takeoff_z.copy_(torch.where(takeoff, self.support_z, self.takeoff_z))
    self.peak_lift.masked_fill_(takeoff, 0)
    self.air_time.masked_fill_(takeoff, 0)
    self.swing_q.copy_(torch.where(takeoff, q, self.swing_q))
    self.command_active.copy_(torch.where(takeoff, moving, self.command_active))
    self.in_swing |= takeoff
    self.command_active &= moving
    airborne = self.in_swing & ~support
    self.air_time.add_(airborne * env.step_dt)
    lift = (foot_z - self.takeoff_z).clamp_min(0)
    self.peak_lift.copy_(torch.where(
      airborne, torch.maximum(self.peak_lift, lift), self.peak_lift,
    ))
    self.swing_q.copy_(torch.where(
      self.in_swing, torch.maximum(self.swing_q, q), self.swing_q,
    ))

    landing = self.in_swing & settled
    valid_landing = landing & self.command_active & (self.air_time + 1e-6 >= min_air_time)
    low = ((min_height - self.peak_lift).clamp_min(0) / error_scale).square()
    high = ((self.peak_lift - max_height).clamp_min(0) / error_scale).square()
    event_cost = (low + high).clamp_max(max_cost) * (1.0 - self.swing_q) * valid_landing
    num_landings = valid_landing.sum().clamp_min(1)
    env.extras["log"]["Metrics/swing_peak_lift"] = (self.peak_lift * valid_landing).sum() / num_landings
    env.extras["log"]["Metrics/swing_landings"] = valid_landing.float().sum(dim=1).mean()

    self.support_z.copy_(torch.where(settled, foot_z, self.support_z))
    self.has_support |= settled
    self.in_swing &= ~landing
    self.command_active &= ~landing
    for buffer in (self.peak_lift, self.swing_q, self.air_time):
      buffer.masked_fill_(landing, 0)
    return event_cost.sum(dim=1) / env.step_dt


def stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  ratio: float = 5.0,
) -> torch.Tensor:
  force = env.scene[sensor_name].data.force
  assert force is not None

  horizontal_force = torch.linalg.norm(force[..., :2], dim=-1)
  vertical_force = torch.abs(force[..., 2])

  return (horizontal_force > ratio * vertical_force).any(dim=1).float()
