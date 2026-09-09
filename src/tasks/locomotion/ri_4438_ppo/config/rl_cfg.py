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
      hidden_dims = (512, 256, 128),
      activation = "elu",
      obs_normalization = True,
      distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),

    critic = RslRlModelCfg(
      hidden_dims = (512, 256, 128),
      activation = "elu",
      obs_normalization = True,
    ),

    algorithm = RslRlPpoAlgorithmCfg(
      value_loss_coef = 1.0,
      use_clipped_value_loss = True,
      clip_param = 0.2,
      entropy_coef = 0.01,
      num_learning_epochs = 5,
      num_mini_batches = 4,
      learning_rate = 1.0e-3,
      schedule = "adaptive",
      gamma = 0.99,
      lam = 0.95,
      desired_kl = 0.01,
      max_grad_norm = 1.0,
    ),

    logger = "tensorboard",         # tensorboard or wandb
    experiment_name = "ri_4438_ppo",
    save_interval = 100,
    num_steps_per_env = 24,
    max_iterations = 10001,
  )