"""RI-4438 HIM configuration shared by the task and rsl_rl adapters.

The old HIMLoco configuration described a 45-dimensional frame.  RI-4438's
actor also contains the two-dimensional sinusoidal gait phase, so the actual
frame is 47 dimensions and the six-frame history is 282 dimensions.
"""

from src.config.ri_4438.ri_4438_config import Ri4438CfgPPO, Ri4438Cfg


class Ri4438HimCfg(Ri4438Cfg):
    """Environment-side dimensions and robot defaults used by HIM."""

    class env(Ri4438Cfg.env):
        num_envs = 1024
        num_actions = 12
        num_one_step_obs = 3 + 3 + 3 + 2 + num_actions * 3  # 47
        history_size = 6
        num_observations = num_one_step_obs * history_size  # 282

        # Names used by the original HIMLoco configuration, retained as
        # read-only aliases for scripts that inspect environment dimensions.
        num_action = num_actions
        num_obs_history = history_size
        num_obs_one_step = num_one_step_obs
        num_obs = num_observations

        # The actor frame is followed by privileged terms in the critic.  The
        # exact rough/flat dimensions are documented here for validation; the
        # live MjLab observation manager remains the source of truth.
        # Rough and flat use the same critic layout so a HIM checkpoint can be
        # resumed after switching terrain.  On flat terrain ``height_scan`` is
        # retained and evaluates to a constant/near-constant scan.
        rough_num_privileged_obs = num_one_step_obs + 3 + 3 + 4 + 187  # 244
        flat_num_privileged_obs = rough_num_privileged_obs
        num_privileged_obs = rough_num_privileged_obs
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 20.0


class Ri4438CFGHimPPO(Ri4438CfgPPO):
    """HIM policy, estimator, PPO and runner hyperparameters."""

    class policy(Ri4438CfgPPO.policy):
        num_one_step_obs = Ri4438HimCfg.env.num_one_step_obs
        history_size = Ri4438HimCfg.env.history_size
        history_term_dims = (
            3,  # base_ang_vel
            3,  # projected_gravity
            3,  # command
            2,  # phase
            Ri4438HimCfg.env.num_actions,  # joint_pos
            Ri4438HimCfg.env.num_actions,  # joint_vel
            Ri4438HimCfg.env.num_actions,  # actions
        )
        history_order = "frame_major_oldest_first"
        encoder_hidden_dims = (128, 64, 16)
        target_hidden_dims = (128, 64)
        num_prototype = 32
        temperature = 3.0
        sinkhorn_eps = 0.05
        sinkhorn_iters = 3

    class algorithm(Ri4438CfgPPO.algorithm):
        estimator_learning_rate = 1.0e-3
        estimator_max_grad_norm = 10.0

        # critic = actor frame (47) + true base linear velocity (3) + other
        # privileged terms.  The target input removes command [6:9] and
        # appends the privileged velocity.
        estimator_velocity_slice = (47, 50)
        estimator_target_slices = ((0, 6), (9, 47), (47, 50))

    class runner(Ri4438CfgPPO.runner):
        num_steps_per_env = 100
        max_iterations = 100000
        save_interval = 100
        experiment_name = "ri_4438_him"
