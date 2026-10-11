"""Task-local terminal predicates (illegal_contact copied from mjlab 1.6.0)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
    assert data.found is not None
    return torch.any(data.found, dim=-1)


def boundary_truncation(env):
    from .curriculums import get_controller

    controller = get_controller(env)
    if controller.terrain_map.plane:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    displacement = (
        env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    ).abs()
    limit = (
        torch.tensor(controller.cfg.geometry.patch_size_m, device=env.device) / 2
        - controller.cfg.geometry.boundary_margin_m
    )
    return (displacement >= limit).any(dim=1)
