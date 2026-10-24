"""Shared command-based standing/moving classification for this task."""

from __future__ import annotations

import math

import torch


def command_motion_mask(
    command: torch.Tensor,
    command_threshold: float,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    """Select moving commands with separate translation/yaw units when enabled.

    Omitting the yaw threshold retains the historical summed-command gate for
    saved configurations. With separate thresholds, equality belongs to standing
    and the motion terms all use this same strict comparison.
    """
    linear_speed = torch.linalg.norm(command[:, :2], dim=1)
    yaw_speed = command[:, 2].abs()
    if angular_command_threshold is None:
        return linear_speed + yaw_speed > command_threshold
    if not (
        math.isfinite(command_threshold)
        and command_threshold >= 0.0
        and math.isfinite(angular_command_threshold)
        and angular_command_threshold >= 0.0
    ):
        raise ValueError(
            "Linear and angular motion thresholds must be finite and nonnegative."
        )
    return (linear_speed > command_threshold) | (
        yaw_speed > angular_command_threshold
    )
