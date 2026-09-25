from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_com(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Return the selected bodies' COM offsets in their local body frames.

  The ``base_com`` domain-randomization event modifies MuJoCo's
  ``model.body_ipos`` field.  Reading that field here keeps the critic's
  privileged observation consistent with the randomized model parameters,
  while avoiding the world-frame position of the robot, which changes as the
  robot moves.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  return env.sim.model.body_ipos[:, body_ids].flatten(start_dim=1)


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(
  env: ManagerBasedRlEnv,
  period: float,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  global_phase = (env.episode_length_buf * env.step_dt) % period / period

  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

  command = env.command_manager.get_command(command_name)
  assert command is not None

  moving = (
    torch.linalg.norm(command[:, :2], dim=1)
    + torch.abs(command[:, 2])
    > command_threshold
  )

  return torch.where(
    moving.unsqueeze(1),
    phase,
    torch.zeros_like(phase),
  )
