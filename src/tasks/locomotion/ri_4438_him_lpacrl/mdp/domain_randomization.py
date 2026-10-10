# Adapted from RI-4438 HIM (workspace snapshot 2026-10-10).
"""Domain randomization helpers for the RI-4438 HIM task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def randomize_imu_errors(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  gyro_bias_range: tuple[float, float],
  mount_angle_range: tuple[float, float],
) -> None:
  """Sample fixed sensor calibration errors without changing simulation truth.

  The mounting quaternion is the reported IMU frame's orientation relative to
  the nominal frame. Both actor IMU terms use its inverse rotation. Gyro bias
  is expressed in that reported frame. Startup sampling keeps these errors
  constant through history updates and episode resets.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if not hasattr(env, "_ri4438_imu_mount_quat"):
    env._ri4438_imu_mount_quat = torch.zeros((env.num_envs, 4), device=env.device)
    env._ri4438_imu_mount_quat[:, 0] = 1.0
    env._ri4438_imu_gyro_bias = torch.zeros((env.num_envs, 3), device=env.device)
  angles = torch.empty((len(env_ids), 3), device=env.device).uniform_(
    *mount_angle_range
  )
  env._ri4438_imu_mount_quat[env_ids] = quat_from_euler_xyz(
    angles[:, 0], angles[:, 1], angles[:, 2]
  )
  env._ri4438_imu_gyro_bias[env_ids] = torch.empty_like(angles).uniform_(
    *gyro_bias_range
  )


def imu_ang_vel_with_errors(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Actor gyro with fixed mounting error and bias, before noise and delay."""
  angular_velocity = mdp.builtin_sensor(env, sensor_name)
  # ObservationManager checks shapes before startup events; play has no event.
  if not hasattr(env, "_ri4438_imu_mount_quat"):
    return angular_velocity
  return quat_apply_inverse(
    env._ri4438_imu_mount_quat, angular_velocity
  ) + env._ri4438_imu_gyro_bias


def imu_projected_gravity_with_errors(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Actor gravity in the same misaligned IMU frame, preserving vector norm.

  RI-4438's nominal IMU site is aligned with the base. Do not rotate the model's
  site itself: critic and HIM velocity supervision must retain the true frame.
  """
  gravity = mdp.projected_gravity(env)
  if not hasattr(env, "_ri4438_imu_mount_quat"):
    return gravity
  return quat_apply_inverse(env._ri4438_imu_mount_quat, gravity)


@requires_model_fields("geom_solref")
def randomize_foot_contact_softness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  time_constant_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Randomize positive-format solref time constant, shared by the four feet.

  Preserve damping ratio and solimp. Larger time constants mean softer contact.
  Declaring geom_solref above ensures each world owns its contact parameters.
  """
  low, high = time_constant_range
  if not 2.0 * env.physics_dt <= low <= high:
    raise ValueError("Contact time constants must be ordered and >= 2 * physics_dt")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  defaults = env.sim.get_default_field("geom_solref")[geom_ids]
  if (defaults <= 0).any():
    raise ValueError("Foot softness randomization requires positive-format solref")
  values = torch.empty((len(env_ids), 1), device=env.device).uniform_(low, high)
  env.sim.model.geom_solref[env_ids[:, None], geom_ids, 0] = values


def synchronize_sensor_delays(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  observation_group: str = "actor",
  term_groups: tuple[tuple[str, ...], ...] = (
    ("base_ang_vel", "projected_gravity"),
    ("joint_pos", "joint_vel"),
  ),
) -> None:
  """Give observations in one sensor packet identical lag trajectories.

  Run once at startup, after ObservationManager creates its delay buffers.
  mjlab 1.6 has one buffer per observation term and no shared-lag setting.
  Separate generators with the same seed synchronize a packet's lag draws
  and reset phases, while keeping its terms' observation histories separate.
  ObservationManager computes and resets every term at the same cadence,
  including partial resets. Different packets use independent RNG streams.
  """
  del env_ids  # Startup always initializes all environments together.
  manager = env.observation_manager
  buffers = manager._group_obs_term_delay_buffer[observation_group]
  settings = (
    "min_lag", "max_lag", "batch_size", "device", "per_env", "hold_prob",
    "update_period", "per_env_phase",
  )
  for names in term_groups:
    if not any(name in buffers for name in names):
      continue
    if not all(name in buffers for name in names):
      raise ValueError(f"Sensor packet {names} has only some delays enabled")
    reference = buffers[names[0]]
    if any(
      getattr(buffers[name], setting) != getattr(reference, setting)
      for name in names[1:]
      for setting in settings
    ):
      raise ValueError(f"Sensor packet {names} must use identical delay settings")
    seed = int(torch.randint(0, 2**31, ()).item())
    for name in names:
      buffer = buffers[name]
      buffer.generator = torch.Generator(device=buffer.device).manual_seed(seed)
      buffer.reset()
