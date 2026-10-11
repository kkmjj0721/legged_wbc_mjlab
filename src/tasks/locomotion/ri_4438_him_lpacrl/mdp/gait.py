# Adapted from RI-4438 HIM (workspace snapshot 2026-10-10).
"""Shared gait clock for observations and rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def gait_phase(env: ManagerBasedRlEnv, period: float) -> torch.Tensor:
    """Return cycle phase in [0, 1), taking the remainder before scaling time."""
    if period <= 0.0:
        raise ValueError("Gait period must be positive.")
    steps_per_period = period / env.step_dt
    return torch.remainder(env.episode_length_buf, steps_per_period) / steps_per_period
