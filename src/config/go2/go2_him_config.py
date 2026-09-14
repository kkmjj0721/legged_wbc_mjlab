"""Unitree Go2 configuration for HIM velocity training.

The HIM actor uses the same 47-dimensional frame as the RI-4438 setup:
angular velocity, projected gravity, command, gait phase, joint position,
joint velocity, and the previous action.  Six frames are retained for the
estimator while the critic receives privileged observations from the current
frame.
"""

from src.config.go2.go2_config import Go2Cfg, GO2CfgPPO


class Go2HimCfg(Go2Cfg):
  """Environment dimensions and domain-randomization defaults for Go2 HIM."""

  class env(Go2Cfg.env):
    num_envs = 4096
    num_actions = 12
    num_one_step_obs = 3 + 3 + 3 + 2 + num_actions * 3  # 47
    history_size = 6
    num_observations = num_one_step_obs * history_size  # 282

    # Compatibility aliases used by older HIM training scripts.
    num_action = num_actions
    num_obs_history = history_size
    num_obs_one_step = num_one_step_obs
    num_obs = num_observations

    # 47 actor features + 3 privileged velocity + 3 COM + 4 contacts +
    # 187 height-scan values.
    rough_num_privileged_obs = num_one_step_obs + 3 + 3 + 4 + 187  # 244
    flat_num_privileged_obs = rough_num_privileged_obs
    num_privileged_obs = rough_num_privileged_obs

  class comaman(Go2Cfg.comaman):
    rel_standing_envs = 0.05
    rel_forward_envs = 0.1

  class domain_rand(Go2Cfg.domain_rand):
    link_mass_range = [0.8, 1.2]
    KpKd_factor_range = [0.9, 1.1]
    motor_strength_range = [0.9, 1.1]


class Go2HimPPO(GO2CfgPPO):
  """HIM estimator, PPO, and runner hyperparameters for Go2."""

  class policy(GO2CfgPPO.policy):
    num_one_step_obs = Go2HimCfg.env.num_one_step_obs
    history_size = Go2HimCfg.env.history_size
    history_term_dims = (
      3,  # base_ang_vel
      3,  # projected_gravity
      3,  # command
      2,  # phase
      Go2HimCfg.env.num_actions,  # joint_pos
      Go2HimCfg.env.num_actions,  # joint_vel
      Go2HimCfg.env.num_actions,  # actions
    )
    history_order = "frame_major_oldest_first"
    encoder_hidden_dims = (128, 64, 16)
    target_hidden_dims = (128, 64)
    num_prototype = 32
    temperature = 3.0
    sinkhorn_eps = 0.05
    sinkhorn_iters = 3

  class algorithm(GO2CfgPPO.algorithm):
    estimator_learning_rate = 1.0e-3
    estimator_max_grad_norm = 10.0

    # Critic layout is actor frame (47), true base velocity (3), and the
    # remaining privileged terms.  Estimator targets replace command [6:9]
    # with privileged velocity [47:50].
    estimator_velocity_slice = (47, 50)
    estimator_target_slices = ((0, 6), (9, 47), (47, 50))

  class runner(GO2CfgPPO.runner):
    num_steps_per_env = 100
    max_iterations = 100000
    save_interval = 100
    experiment_name = "go2_him"
