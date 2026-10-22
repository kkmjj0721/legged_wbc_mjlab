"""Serializable configuration for the RI-4438 HIM symmetry task."""

from dataclasses import dataclass, field, fields

from src.config.ri_4438.him_layouts import HimLayout, PHASE47
from src.tasks.locomotion.ri_4438_him.config.rl_cfg import (
    RslRlHimAlgorithmCfg, RslRlHimRunnerCfg, ri_4438_him_runner_cfg,
)


@dataclass
class HimSymmetryCfg:
    implementation: str = "ppo"
    data_augmentation_func: str = "src.tasks.locomotion.ri_4438_him_symmetry.rl.symmetry:augment_phase47"
    use_data_augmentation: bool = True
    use_mirror_loss: bool = True
    mirror_loss_coeff: float = 0.1
    actor_mirror_reduction: str = "mean"
    critic_mirror_loss_coeff: float = 1.0


@dataclass
class HimSymmetryAlgorithmCfg(RslRlHimAlgorithmCfg):
    symmetry_cfg: HimSymmetryCfg | None = field(default_factory=HimSymmetryCfg)
    estimator_lr_mode: str = "follow"
    estimator_lr_cap: float | None = None
    estimator_use_mirror: bool = True
    fused_optimizer: bool = False
    reference_update: bool = False
    kl_mode: str = "standard"

    @classmethod
    def legacy_homie(cls):
        """Explicit historical reproduction; includes independently sampled mirrors."""
        return cls(symmetry_cfg=HimSymmetryCfg(implementation="homie",
            use_data_augmentation=True, use_mirror_loss=True, mirror_loss_coeff=1.,
            actor_mirror_reduction="sum", critic_mirror_loss_coeff=1.),
            estimator_use_mirror=True, reference_update=True, kl_mode="legacy_homie",
            fused_optimizer=True)


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
    # Keep the real sampling budget; paired augmentation adds mirrored samples.
    cfg.num_steps_per_env = 100
    cfg.actor.num_one_step_obs = layout.frame_size
    cfg.actor.history_size = layout.history_size
    cfg.actor.history_term_dims = layout.term_dims
    cfg.actor.action_observation_slice = layout.action_slice
    cfg.algorithm.estimator_velocity_slice = layout.velocity_slice
    cfg.algorithm.estimator_target_slices = layout.target_slices
    return cfg
