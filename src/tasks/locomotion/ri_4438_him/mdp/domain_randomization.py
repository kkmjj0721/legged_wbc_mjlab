"""Domain randomization helpers for the RI-4438 HIM task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
