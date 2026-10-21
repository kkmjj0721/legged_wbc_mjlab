"""ri_4438_him_symmetry training configuration."""

from src.config.ri_4438.him_layouts import PHASE47
from src.tasks.locomotion.ri_4438_him_symmetry.rl.rl_cfg import make_runner_cfg


def ri_4438_him_runner_cfg():
    return make_runner_cfg(PHASE47, "ri_4438_him_symmetry")
