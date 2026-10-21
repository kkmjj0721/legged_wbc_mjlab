"""Independent environment factories based on the complete ordinary HIM factory."""

from copy import deepcopy
from src.tasks.locomotion.ri_4438_him.config.env_cfgs import (
    ri_4438_rough_env_cfg as base_rough, ri_4438_flat_env_cfg as base_flat,
)
from src.tasks.locomotion.ri_4438_him_symmetry.rl.layouts import PHASE47, validate_env_cfg


def _build(factory, play):
    cfg = deepcopy(factory(play=play))
    validate_env_cfg(cfg, PHASE47)
    return cfg


def ri_4438_rough_env_cfg(play=False):
    return _build(base_rough, play)


def ri_4438_flat_env_cfg(play=False):
    return _build(base_flat, play)
