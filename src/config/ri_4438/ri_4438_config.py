from src.config.base.base_config import BaseConfig


class Ri4438PiperCfg( BaseConfig ):
    class env:
        num_envs = 4096

    class comaman:
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
        resampling_time = [4.0, 8.0]  # [s] time before new command is given
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]
        
    class control:
        stiffness = {"hip": 30.0, "thigh": 30.0, "calf": 30.0,}  
        damping = {"hip": 0.6, "thigh": 0.6, "calf": 0.6,}  
        action_scale = 0.5
        decimation = 4  # control frequency = sim frequency / decimation
        hip_reduction = 0.5
        effort_limit = {"hip": 10.0, "thigh": 10.0, "calf": 10.0 }
        armature = 0.008234
        friction = 0.01
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class init_state:
        pos = [0.0, 0.0, 0.25]
        default_joint = {
            "FL_hip_joint": -0.0,
            "RL_hip_joint": -0.0,
            "FR_hip_joint": 0.0,
            "RR_hip_joint": 0.0,

            "FL_thigh_joint": 0.9,
            "RL_thigh_joint": 0.9,
            "FR_thigh_joint": 0.9,
            "RR_thigh_joint": 0.9,

            "FL_calf_joint": -1.8,
            "RL_calf_joint": -1.8,
            "FR_calf_joint": -1.8,
            "RR_calf_joint": -1.8,
        }

    class reset:
        base_offset = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {},
        }

        joint_offset = {
            "position_range": (-0.0, 0.0),
            "velocity_range": (-0.0, 0.0),
        }
   
    class domain_rand:
        pass

    class noise:
        pass

    class reward:
        pass
   
   
class Ri4438CfgPPO:
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        obs_normalization = True # Whether to normalize the observations

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3 #1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        num_steps_per_env = 24 # per iteration
        max_iterations = 20000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'test'
        


