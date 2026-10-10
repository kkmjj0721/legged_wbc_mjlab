# Adapted from RI-4438 HIM (workspace snapshot 2026-10-10).
"""RL configuration for the RI-4438 HIM velocity task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.config.ri_4438.ri_4438_him_lpacrl_config import HimLearningCfg
from .numerics import (
  ACTION_CLIP, OBSERVATION_CLIP, RAW_ACTION_WARN, RAW_OBSERVATION_ABORT,
  ACTION_OBSERVATION_SLICE,
)


ri_4438_him_cfg = HimLearningCfg()


@dataclass
class RslRlHimActorCfg(RslRlModelCfg):
  """HIM actor and estimator configuration."""

  hidden_dims: tuple[int, ...] = tuple(ri_4438_him_cfg.policy.actor_hidden_dims)
  activation: str = ri_4438_him_cfg.policy.activation
  obs_normalization: bool = ri_4438_him_cfg.policy.obs_normalization
  observation_clip: float = OBSERVATION_CLIP
  action_clip: float = ACTION_CLIP
  action_observation_slice: tuple[int, int] = ACTION_OBSERVATION_SLICE
  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
      "init_std": float(ri_4438_him_cfg.policy.init_noise_std),
      "std_type": "scalar",
    },
  )
  num_one_step_obs: int = ri_4438_him_cfg.policy.num_one_step_obs
  history_size: int = ri_4438_him_cfg.policy.history_size
  history_term_dims: tuple[int, ...] = field(
    default_factory=lambda: tuple(ri_4438_him_cfg.policy.history_term_dims),
  )
  history_order: str = ri_4438_him_cfg.policy.history_order
  estimator_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "enc_hidden_dims": tuple(ri_4438_him_cfg.policy.encoder_hidden_dims),
      "tar_hidden_dims": tuple(ri_4438_him_cfg.policy.target_hidden_dims),
      "activation": ri_4438_him_cfg.policy.activation,
      "learning_rate": float(ri_4438_him_cfg.algorithm.estimator_learning_rate),
      "max_grad_norm": float(ri_4438_him_cfg.algorithm.estimator_max_grad_norm),
      "num_prototype": int(ri_4438_him_cfg.policy.num_prototype),
      "temperature": float(ri_4438_him_cfg.policy.temperature),
      "sinkhorn_eps": float(ri_4438_him_cfg.policy.sinkhorn_eps),
      "sinkhorn_iters": int(ri_4438_him_cfg.policy.sinkhorn_iters),
    },
  )
  class_name: str = "rsl_rl.models.him_actor_model:HIMActorModel"


@dataclass
class RslRlHimAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration extended with HIM estimator supervision."""

  entropy_coef: float = ri_4438_him_cfg.algorithm.entropy_coef
  num_learning_epochs: int = ri_4438_him_cfg.algorithm.num_learning_epochs
  num_mini_batches: int = ri_4438_him_cfg.algorithm.num_mini_batches
  learning_rate: float = ri_4438_him_cfg.algorithm.learning_rate
  schedule: str = ri_4438_him_cfg.algorithm.schedule
  gamma: float = ri_4438_him_cfg.algorithm.gamma
  lam: float = ri_4438_him_cfg.algorithm.lam
  desired_kl: float = ri_4438_him_cfg.algorithm.desired_kl
  max_grad_norm: float = ri_4438_him_cfg.algorithm.max_grad_norm
  value_loss_coef: float = ri_4438_him_cfg.algorithm.value_loss_coef
  use_clipped_value_loss: bool = ri_4438_him_cfg.algorithm.use_clipped_value_loss
  clip_param: float = ri_4438_him_cfg.algorithm.clip_param
  estimator_learning_rate: float = ri_4438_him_cfg.algorithm.estimator_learning_rate
  estimator_max_grad_norm: float = ri_4438_him_cfg.algorithm.estimator_max_grad_norm
  estimator_obs_groups: tuple[str, ...] = ("estimator",)
  estimator_velocity_slice: tuple[int, int] = ri_4438_him_cfg.algorithm.estimator_velocity_slice
  estimator_target_slices: tuple[tuple[int, int], ...] = ri_4438_him_cfg.algorithm.estimator_target_slices
  class_name: str = "rsl_rl.algorithms.him_ppo:HIMPPO"


@dataclass
class RslRlHimRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner configuration selecting the dedicated HIM training lifecycle."""

  actor: RslRlHimActorCfg = field(default_factory=RslRlHimActorCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=tuple(ri_4438_him_cfg.policy.critic_hidden_dims),
      activation=ri_4438_him_cfg.policy.activation,
      obs_normalization=ri_4438_him_cfg.policy.obs_normalization,
    ),
  )
  algorithm: RslRlHimAlgorithmCfg = field(default_factory=RslRlHimAlgorithmCfg)
  class_name: str = "src.tasks.locomotion.ri_4438_him_lpacrl.rl.runner:LPACRLRunner"
  clip_actions: float = ACTION_CLIP
  numerics: dict[str, Any] = field(default_factory=lambda: {
    "raw_action_warn": RAW_ACTION_WARN,
    "raw_observation_abort": RAW_OBSERVATION_ABORT,
    "history_steps": 8,
  })


def ri_4438_him_runner_cfg(experiment=None) -> RslRlHimRunnerCfg:
  """Create the RSL-RL HIM runner configuration."""
  cfg = RslRlHimRunnerCfg(
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    logger="tensorboard", experiment_name="ri_4438_him_lpacrl", upload_model=False,
  )
  if experiment is None:
    from src.config.ri_4438.ri_4438_him_lpacrl_config import ExperimentCfg
    experiment = ExperimentCfg()
  learn = experiment.learning
  cfg.seed = experiment.training.seed
  cfg.num_steps_per_env = experiment.training.rollout_steps
  cfg.max_iterations = experiment.training.max_iterations
  cfg.save_interval = experiment.training.save_interval
  cfg.actor.hidden_dims = tuple(learn.policy.actor_hidden_dims)
  cfg.critic.hidden_dims = tuple(learn.policy.critic_hidden_dims)
  cfg.actor.activation = cfg.critic.activation = learn.policy.activation
  cfg.actor.obs_normalization = cfg.critic.obs_normalization = learn.policy.obs_normalization
  cfg.actor.distribution_cfg["init_std"] = learn.policy.init_noise_std
  for key in ("num_one_step_obs", "history_size", "history_term_dims", "history_order"):
    setattr(cfg.actor, key, getattr(learn.policy, key))
  for target, source in (("enc_hidden_dims", "encoder_hidden_dims"), ("tar_hidden_dims", "target_hidden_dims"),
                          ("num_prototype", "num_prototype"), ("temperature", "temperature"),
                          ("sinkhorn_eps", "sinkhorn_eps"), ("sinkhorn_iters", "sinkhorn_iters"), ("activation", "activation")):
    cfg.actor.estimator_cfg[target] = getattr(learn.policy, source)
  cfg.actor.estimator_cfg["learning_rate"] = learn.algorithm.estimator_learning_rate
  cfg.actor.estimator_cfg["max_grad_norm"] = learn.algorithm.estimator_max_grad_norm
  for key, value in vars(learn.algorithm).items():
    if hasattr(cfg.algorithm, key):
      setattr(cfg.algorithm, key, value)
  return cfg
