"""RSL-RL HIM configuration for Unitree Go2."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.config.go2.go2_him_config import Go2HimPPO


go2_him_cfg = Go2HimPPO()


@dataclass
class RslRlHimActorCfg(RslRlModelCfg):
  """HIM actor and estimator configuration."""

  hidden_dims: tuple[int, ...] = tuple(go2_him_cfg.policy.actor_hidden_dims)
  activation: str = go2_him_cfg.policy.activation
  obs_normalization: bool = go2_him_cfg.policy.obs_normalization
  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
      "init_std": float(go2_him_cfg.policy.init_noise_std),
      "std_type": "scalar",
    }
  )
  num_one_step_obs: int = go2_him_cfg.policy.num_one_step_obs
  history_size: int = go2_him_cfg.policy.history_size
  history_term_dims: tuple[int, ...] = field(
    default_factory=lambda: tuple(go2_him_cfg.policy.history_term_dims)
  )
  history_order: str = go2_him_cfg.policy.history_order
  estimator_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "enc_hidden_dims": tuple(go2_him_cfg.policy.encoder_hidden_dims),
      "tar_hidden_dims": tuple(go2_him_cfg.policy.target_hidden_dims),
      "activation": go2_him_cfg.policy.activation,
      "learning_rate": float(go2_him_cfg.algorithm.estimator_learning_rate),
      "max_grad_norm": float(go2_him_cfg.algorithm.estimator_max_grad_norm),
      "num_prototype": int(go2_him_cfg.policy.num_prototype),
      "temperature": float(go2_him_cfg.policy.temperature),
      "sinkhorn_eps": float(go2_him_cfg.policy.sinkhorn_eps),
      "sinkhorn_iters": int(go2_him_cfg.policy.sinkhorn_iters),
    }
  )
  class_name: str = "rsl_rl.models.him_actor_model:HIMActorModel"


@dataclass
class RslRlHimAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration extended with HIM estimator supervision."""

  entropy_coef: float = go2_him_cfg.algorithm.entropy_coef
  num_learning_epochs: int = go2_him_cfg.algorithm.num_learning_epochs
  num_mini_batches: int = go2_him_cfg.algorithm.num_mini_batches
  learning_rate: float = go2_him_cfg.algorithm.learning_rate
  schedule: str = go2_him_cfg.algorithm.schedule
  gamma: float = go2_him_cfg.algorithm.gamma
  lam: float = go2_him_cfg.algorithm.lam
  desired_kl: float = go2_him_cfg.algorithm.desired_kl
  max_grad_norm: float = go2_him_cfg.algorithm.max_grad_norm
  value_loss_coef: float = go2_him_cfg.algorithm.value_loss_coef
  use_clipped_value_loss: bool = go2_him_cfg.algorithm.use_clipped_value_loss
  clip_param: float = go2_him_cfg.algorithm.clip_param
  estimator_learning_rate: float = go2_him_cfg.algorithm.estimator_learning_rate
  estimator_max_grad_norm: float = go2_him_cfg.algorithm.estimator_max_grad_norm
  estimator_obs_groups: tuple[str, ...] = ("critic",)
  estimator_velocity_slice: tuple[int, int] = go2_him_cfg.algorithm.estimator_velocity_slice
  estimator_target_slices: tuple[tuple[int, int], ...] = go2_him_cfg.algorithm.estimator_target_slices
  class_name: str = "rsl_rl.algorithms.him_ppo:HIMPPO"


@dataclass
class RslRlHimRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner configuration selecting the dedicated HIM lifecycle."""

  actor: RslRlHimActorCfg = field(default_factory=RslRlHimActorCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=tuple(go2_him_cfg.policy.critic_hidden_dims),
      activation=go2_him_cfg.policy.activation,
      obs_normalization=go2_him_cfg.policy.obs_normalization,
    )
  )
  algorithm: RslRlHimAlgorithmCfg = field(default_factory=RslRlHimAlgorithmCfg)
  class_name: str = "src.tasks.locomotion.go2_him.rl.runner:HIMOnPolicyRunner"


def unitree_go2_him_runner_cfg() -> RslRlHimRunnerCfg:
  """Create the RSL-RL HIM runner configuration for Go2."""

  return RslRlHimRunnerCfg(
    actor=RslRlHimActorCfg(),
    critic=RslRlModelCfg(
      hidden_dims=tuple(go2_him_cfg.policy.critic_hidden_dims),
      activation=go2_him_cfg.policy.activation,
      obs_normalization=go2_him_cfg.policy.obs_normalization,
    ),
    algorithm=RslRlHimAlgorithmCfg(
      value_loss_coef=float(go2_him_cfg.algorithm.value_loss_coef),
      use_clipped_value_loss=go2_him_cfg.algorithm.use_clipped_value_loss,
      clip_param=float(go2_him_cfg.algorithm.clip_param),
      entropy_coef=float(go2_him_cfg.algorithm.entropy_coef),
      num_learning_epochs=int(go2_him_cfg.algorithm.num_learning_epochs),
      num_mini_batches=int(go2_him_cfg.algorithm.num_mini_batches),
      learning_rate=float(go2_him_cfg.algorithm.learning_rate),
      schedule=go2_him_cfg.algorithm.schedule,
      gamma=float(go2_him_cfg.algorithm.gamma),
      lam=float(go2_him_cfg.algorithm.lam),
      desired_kl=float(go2_him_cfg.algorithm.desired_kl),
      max_grad_norm=float(go2_him_cfg.algorithm.max_grad_norm),
      estimator_learning_rate=float(go2_him_cfg.algorithm.estimator_learning_rate),
      estimator_max_grad_norm=float(go2_him_cfg.algorithm.estimator_max_grad_norm),
      estimator_velocity_slice=tuple(go2_him_cfg.algorithm.estimator_velocity_slice),
      estimator_target_slices=tuple(go2_him_cfg.algorithm.estimator_target_slices),
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    logger="tensorboard",
    experiment_name=go2_him_cfg.runner.experiment_name,
    save_interval=int(go2_him_cfg.runner.save_interval),
    num_steps_per_env=int(go2_him_cfg.runner.num_steps_per_env),
    max_iterations=int(go2_him_cfg.runner.max_iterations),
  )
