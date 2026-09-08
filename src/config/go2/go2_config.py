from src.config.base.base_config import BaseConfig

class Go2Cfg( BaseConfig ):
    class env:
        num_envs = 4096
        num_action = 12
        num_obs_history = 1
        num_obs_one_step = 3 + 3 + 3 + num_action * 3
        num_obs = num_obs_one_step * num_obs_history
        num_privileged_obs = num_obs_one_step + 3 + 187
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class asset:
        file = "../../assets/robots/go2/xmls/go2.xml"
        name = "Go2"

    class control:
        stiffness = {"hip": 20.0, "thigh": 20.0, "calf": 40.0,}  
        damping = {"hip": 0.5, "thigh": 0.5, "calf": 1.0,}  
        action_scale = 0.25
        decimation = 4  # control frequency = sim frequency / decimation
        hip_reduction = 0.5
        effort_limit = {"hip": 23.7, "thigh": 23.7, "calf": 45.43,}
        armature = 0.01
        friction = 0.01
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10.0

    class init_state:
        pos = [0.0, 0.0, 0.32]
        default_joint = {
            "FL_hip_joint": -0.1,
            "RL_hip_joint": -0.1,
            "FR_hip_joint": 0.1,
            "RR_hip_joint": 0.1,

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
        pass

    class domain_rand:
        pass

    