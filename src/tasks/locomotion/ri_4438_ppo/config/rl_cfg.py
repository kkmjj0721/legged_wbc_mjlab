"""RL configuration for ri 4438 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from config.ri_4438.ri_4438_config import Ri4438CfgPPO

ri_4438_ppo_cfg = Ri4438CfgPPO()

def ri_4438_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Ri 4438 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor = RslRlModelCfg(
      hidden_dims = tuple(ri_4438_ppo_cfg.policy.actor_hidden_dims),
      activation = ri_4438_ppo_cfg.policy.activation,
      obs_normalization = True,
      distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": float(ri_4438_ppo_cfg.policy.init_noise_std),
        "std_type": "scalar",
      },
    ),

    critic = RslRlModelCfg(
      hidden_dims = tuple(ri_4438_ppo_cfg.policy.critic_hidden_dims),
      activation = ri_4438_ppo_cfg.policy.activation,
      obs_normalization = True,
    ),

    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=ri_4438_ppo_cfg.algorithm.value_loss_coef,
      use_clipped_value_loss=ri_4438_ppo_cfg.algorithm.use_clipped_value_loss,
      clip_param=ri_4438_ppo_cfg.algorithm.clip_param,
      entropy_coef=ri_4438_ppo_cfg.algorithm.entropy_coef,
      num_learning_epochs=ri_4438_ppo_cfg.algorithm.num_learning_epochs,
      num_mini_batches=ri_4438_ppo_cfg.algorithm.num_mini_batches,
      learning_rate=ri_4438_ppo_cfg.algorithm.learning_rate,
      schedule=ri_4438_ppo_cfg.algorithm.schedule,
      gamma=ri_4438_ppo_cfg.algorithm.gamma,
      lam=ri_4438_ppo_cfg.algorithm.lam,
      desired_kl=ri_4438_ppo_cfg.algorithm.desired_kl,
      max_grad_norm=ri_4438_ppo_cfg.algorithm.max_grad_norm,
    ),

    logger = "tensorboard",         # tensorboard or wandb
    
    experiment_name = ri_4438_ppo_cfg.runner.experiment_name,
    save_interval = int(ri_4438_ppo_cfg.runner.save_interval),
    num_steps_per_env = int(ri_4438_ppo_cfg.runner.num_steps_per_env),
    max_iterations = int(ri_4438_ppo_cfg.runner.max_iterations),
  )