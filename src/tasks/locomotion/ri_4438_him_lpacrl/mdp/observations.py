# Adapted from RI-4438 HIM (workspace snapshot 2026-10-10).
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from src.config.ri_4438.ri_4438_him_lpacrl_config import (
    ACTION_CLIP,
    OBSERVATION_CLIP,
    RAW_OBSERVATION_ABORT,
)

from .gait import gait_phase as _gait_phase

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
    command_name: str = "twist",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    global_phase = _gait_phase(env, period)

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    command = env.command_manager.get_command(command_name)
    assert command is not None

    moving = (
        torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        > command_threshold
    )

    return torch.where(
        moving.unsqueeze(1),
        phase,
        torch.zeros_like(phase),
    )


def checked_observation(env, *, source_func, numerics_tag, raw_limit, **params):
    value = source_func(env, **params)
    bad = (~torch.isfinite(value) | (value.abs() > raw_limit)).flatten(1).any(1)
    if not hasattr(env, "_him_numerics_bad"):
        env._him_numerics_bad = torch.zeros(
            env.num_envs, dtype=torch.bool, device=value.device
        )
        env._him_numerics_evidence = {}
    env._him_numerics_bad.logical_or_(bad)
    # No host synchronization and no collectives in reset hooks: ranks can reset
    # different numbers of episodes. Keep evidence until the runner consumes it.
    previous = env._him_numerics_evidence.get(numerics_tag)
    if previous is None:
        previous = torch.zeros_like(value)
    mask = bad.reshape((-1,) + (1,) * (value.ndim - 1))
    env._him_numerics_evidence[numerics_tag] = torch.where(mask, value, previous)
    return value


def configure_numerical_observations(cfg):
    for group_name in ("actor", "critic"):
        group = cfg.observations[group_name]
        # Base actor/critic dictionaries share term objects. Own each copy before
        # adding the wrapper, otherwise a shared term would wrap itself twice.
        group.terms = {name: deepcopy(term) for name, term in group.terms.items()}
        for name, term in group.terms.items():
            bound = ACTION_CLIP if name == "actions" else OBSERVATION_CLIP
            term.clip = (-bound, bound)
            term.params = {
                **term.params,
                "source_func": term.func,
                "numerics_tag": f"{group_name}/{name}",
                "raw_limit": RAW_OBSERVATION_ABORT,
            }
            term.func = checked_observation
