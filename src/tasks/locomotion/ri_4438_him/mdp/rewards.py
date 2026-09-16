"""Reward terms for the standalone RI-4438 HIM task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.rewards import *  # noqa: F401, F403
from mjlab.tasks.velocity.mdp.rewards import *  # noqa: F401, F403
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, TerrainHeightSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity_l1(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)

  actual = asset.data.root_link_lin_vel_b
  error = torch.linalg.norm(
    command[:, :2] - actual[:, :2],
    dim=1,
  )

  active = (
    torch.linalg.norm(command[:, :2], dim=1)
    + torch.abs(command[:, 2])
    > command_threshold
  ).to(error.dtype)

  return error * active


def track_angular_velocity(env: ManagerBasedRlEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_ang_vel_b
  error = torch.square(command[:, 2] - actual[:, 2]) + 0.05 * torch.square(actual[:, :2]).sum(dim=1)
  return torch.exp(-error / std**2)


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


def feet_clearance(
  env: ManagerBasedRlEnv, 
  target_height: float, 
  height_sensor_name: str,
  command_name: str | None = None, 
  command_threshold: float = 0.1, 
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
  ) -> torch.Tensor:
  # asset: Entity = env.scene[asset_cfg.name]
  # heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  # velocities = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  # cost = (torch.abs(heights - target_height) * torch.linalg.norm(velocities, dim=-1)).sum(dim=1)
  # if command_name is not None:
  #   command = env.command_manager.get_command(command_name)
  #   if command is not None:
  #     cost *= (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()
  # return cost
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]

  # [B, N]，每只脚的脚端中心到局部地形表面的垂直距离
  foot_clearance = height_sensor.data.heights

  # 与脚端高度保持相同顺序
  foot_vel_xy = asset.data.site_lin_vel_w[
    :, asset_cfg.site_ids, :2
  ]
  foot_speed = torch.linalg.norm(foot_vel_xy, dim=-1)

  cost = (
    torch.abs(foot_clearance - target_height) * foot_speed
  ).sum(dim=1)

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      active = (
        torch.linalg.norm(command[:, :2], dim=1)
        + torch.abs(command[:, 2])
        > command_threshold
      ).float()
      cost *= active

  return cost


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

def feet_clearance_phase(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str,
  period: float,
  offset: list[float],
  threshold: float = 0.56,
  foot_radius: float = 0.0155,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """在期望摆动阶段惩罚净空不足，不依赖实际脚速或接触状态。"""
  if not 0.0 < threshold < 1.0:
    raise ValueError("threshold 必须在 0 和 1 之间")
  if not 0.0 <= foot_radius < target_height:
    raise ValueError("需要 0 <= foot_radius < target_height")

  period_steps = int(round(period / env.step_dt))
  if period_steps < 2 or abs(period_steps * env.step_dt - period) > 1e-6:
    raise ValueError("period 必须是控制步长的整数倍，且至少包含两个控制步")

  height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
  heights = height_sensor.data.heights
  if heights.ndim != 2 or heights.shape[1] != len(offset):
    raise ValueError("高度传感器的脚数量必须与 offset 长度一致")

  # 与 feet_gait 使用相同的周期计数与相位偏移。
  phase = (
    (env.episode_length_buf % period_steps).to(heights.dtype) / period_steps
  ).unsqueeze(1)
  offsets = torch.as_tensor(
    offset, device=heights.device, dtype=heights.dtype
  ).view(1, -1)
  leg_phase = (phase + offsets) % 1.0
  swing = leg_phase >= threshold

  # 摆动进度 0 -> 1；净空要求由脚半径升至 target_height，再下降。
  swing_phase = ((leg_phase - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
  amplitude = target_height - foot_radius
  required_height = foot_radius + amplitude * torch.sin(
    torch.pi * swing_phase
  ).square()

  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = (
    torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    > command_threshold
  )
  mask = (swing & active.unsqueeze(1)).to(heights.dtype)

  # 抬得比要求高不扣分；使用无量纲误差，并对四只脚取平均。
  shortfall = (required_height - heights).clamp_min(0.0)
  cost = ((shortfall / amplitude).square() * mask).mean(dim=1)

  env.extras["log"]["Metrics/swing_clearance_shortfall_m"] = (
    (shortfall * mask).sum() / mask.sum().clamp_min(1.0)
  )
  return cost