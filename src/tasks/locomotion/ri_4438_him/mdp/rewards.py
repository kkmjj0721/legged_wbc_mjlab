"""Reward terms for the standalone RI-4438 HIM task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math

from mjlab.envs.mdp.rewards import *  # noqa: F401, F403
from mjlab.tasks.velocity.mdp.rewards import *  # noqa: F401, F403
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """ 
  跟踪期望线速度

  数学公式:
    r = exp( -( ||v*_xy - v^b_xy||^2 + 2 * (v^b_z)^2 ) / std^2 )

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
  lin_vel_error = xy_error + (2 * z_error)
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


def body_orientation_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    gravity = quat_apply_inverse(quat, asset.data.gravity_vec_w)
    return torch.square(gravity[:, :2]).sum(dim=1)
  return torch.square(asset.data.projected_gravity_b[:, :2]).sum(dim=1)


def angular_momentum_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: BuiltinSensor = env.scene[sensor_name]
  magnitude_sq = torch.square(sensor.data).sum(dim=-1)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.sqrt(magnitude_sq).mean()
  return magnitude_sq


def body_angular_velocity_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  velocity = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
  return torch.square(velocity[:, :2]).sum(dim=1)


def feet_gait(env: ManagerBasedRlEnv, period: float, offset: list[float], threshold: float, command_threshold: float, command_name: str, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  contact = sensor.data.current_contact_time > 0
  # phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)

  period_steps = int(round(period / env.step_dt))
  phase = ((env.episode_length_buf % period_steps) / period_steps).unsqueeze(1)

  offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
  desired = ((phase + offsets) % 1.0) < threshold
  reward = (desired == contact).float().mean(dim=1)
  command = env.command_manager.get_command(command_name)
  if command is not None:
    active = (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()
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


def hip_joint_deviation_penalty(env: ManagerBasedRlEnv, command_name: str, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice) and asset_cfg.joint_names is None:
    joint_ids, _ = asset.find_joints(r".*_hip_joint")
  default = asset.data.default_joint_pos
  assert default is not None
  cost = torch.square(asset.data.joint_pos[:, joint_ids] - default[:, joint_ids]).sum(dim=1)
  return cost * ((torch.abs(command[:, 1]) <= command_threshold) & (torch.abs(command[:, 2]) <= command_threshold)).to(cost.dtype)


class variable_posture:
  """Speed-dependent tolerance around the robot's default pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default = asset.data.default_joint_pos
    assert default is not None
    self.default_joint_pos = default
    _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.std_standing = torch.tensor(resolve_matching_names_values(cfg.params["std_standing"], names)[2], device=env.device)
    self.std_walking = torch.tensor(resolve_matching_names_values(cfg.params["std_walking"], names)[2], device=env.device)
    self.std_running = torch.tensor(resolve_matching_names_values(cfg.params["std_running"], names)[2], device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, std_standing, std_walking, std_running, asset_cfg: SceneEntityCfg, command_name: str, walking_threshold: float = 0.5, running_threshold: float = 1.5) -> torch.Tensor:
    del std_standing, std_walking, std_running
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    speed = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    standing = (speed < walking_threshold).unsqueeze(1)
    walking = ((speed >= walking_threshold) & (speed < running_threshold)).unsqueeze(1)
    std = torch.where(standing, self.std_standing, torch.where(walking, self.std_walking, self.std_running))
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - self.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.mean(torch.square(error) / torch.square(std), dim=1))


def feet_clearance(
  env: ManagerBasedRlEnv,
  height_range: tuple[float, float],
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  period: float = 0.6,
  offset: tuple[float, ...] = (0.0, 0.5, 0.5, 0.0),
  threshold: float = 0.56,
  height_sensor_name: str = "feet_terrain_height",
) -> torch.Tensor:
  # 保留 asset_cfg 参数，兼容现有配置。
  min_height, max_height = height_range
  if (
    not all(math.isfinite(v) for v in height_range)
    or not 0.0 < min_height <= max_height
  ):
    raise ValueError("height_range must satisfy 0 < min <= max")

  # 与 foot_gait 使用相同的步态时钟。
  period_steps = int(round(period / env.step_dt))
  phase = (
    (env.episode_length_buf % period_steps) / period_steps
  ).unsqueeze(1)
  offsets = torch.as_tensor(
    offset, device=env.device, dtype=phase.dtype
  ).view(1, -1)
  leg_phase = (phase + offsets) % 1.0

  # 按预期摆动相位计算，不使用实际接触状态或足端速度筛选。
  swing = leg_phase >= threshold
  progress = (
    (leg_phase - threshold) / (1.0 - threshold)
  ).clamp(0.0, 1.0)

  # 足端 site 相对脚下地面的高度。
  height = env.scene[height_sensor_name].data.heights

  # 起落脚时最低要求为零，摆动中段达到 min_height。
  lower = min_height * torch.sin(torch.pi * progress)
  error = (
    (lower - height).clamp_min(0.0)
    + (height - max_height).clamp_min(0.0)
  )
  cost = (
    (error / min_height).clamp(max=1.0) * swing
  ).sum(dim=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      active = (
        torch.linalg.norm(command[:, :2], dim=1)
        + command[:, 2].abs()
      ) > command_threshold
      cost = cost * active

  return cost


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
