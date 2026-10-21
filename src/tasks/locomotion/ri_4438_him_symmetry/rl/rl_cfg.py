"""Serializable configuration for the RI-4438 HIM symmetry task."""

from dataclasses import dataclass, field, fields

from src.config.ri_4438.him_layouts import HimLayout, PHASE47
from src.tasks.locomotion.ri_4438_him.config.rl_cfg import (
    RslRlHimAlgorithmCfg, RslRlHimRunnerCfg, ri_4438_him_runner_cfg,
)


@dataclass
class HimSymmetryCfg:
    implementation: str = "homie"
    data_augmentation_func: str = "src.tasks.locomotion.ri_4438_him_symmetry.rl.symmetry:augment_phase47"
    use_data_augmentation: bool = True
    use_mirror_loss: bool = True
    mirror_loss_coeff: float = 1.0
    critic_mirror_loss_coeff: float = 1.0


@dataclass
class HimSymmetryAlgorithmCfg(RslRlHimAlgorithmCfg):
    symmetry_cfg: HimSymmetryCfg | None = field(default_factory=HimSymmetryCfg)
    fused_optimizer: bool = True
    reference_update: bool = True


@dataclass
class HimSymmetryRunnerCfg(RslRlHimRunnerCfg):
    algorithm: HimSymmetryAlgorithmCfg = field(default_factory=HimSymmetryAlgorithmCfg)
    observation_layout: str = "phase47_v1"
    class_name: str = "src.tasks.locomotion.ri_4438_him_symmetry.rl.runner:HimSymmetryRunner"


def make_runner_cfg(layout: HimLayout, experiment_name: str) -> HimSymmetryRunnerCfg:
    if layout != PHASE47:
        raise ValueError("RI-4438 HIM symmetry requires phase47_v1")
    base = ri_4438_him_runner_cfg()
    cfg = HimSymmetryRunnerCfg(**{
        f.name: getattr(base, f.name) for f in fields(base)
        if f.name not in ("algorithm", "class_name")
    })
    cfg.experiment_name = experiment_name
    cfg.observation_layout = layout.name
    # Match the reference rollout budget; symmetry stores two transitions per step.
    cfg.num_steps_per_env = 50
    cfg.actor.num_one_step_obs = layout.frame_size
    cfg.actor.history_size = layout.history_size
    cfg.actor.history_term_dims = layout.term_dims
    cfg.actor.action_observation_slice = layout.action_slice
    cfg.algorithm.estimator_velocity_slice = layout.velocity_slice
    cfg.algorithm.estimator_target_slices = layout.target_slices
    return cfg
