from src.config.base.base_config import BaseConfig


class Riplus4438PiperCfg( BaseConfig ):
    class env:
        num_envs = 4096


    class asset:
        file = "../../assets/robots/go2_piper/xmls/ri_plus_4438.xml"
        name = "ri_plus_4438"
        
    class control:
        stiffness = {
            "hip": 30.0, "thigh": 30.0, "calf": 30.0,
            "shoulder_link_joint": 50.0, "upper_arm_link_joint": 50.0, "forearm_link_joint": 80.0, 
            "wrist1_link_joint": 30.0, "wrist2_link_joint": 30.0, "wrist3_link_joint": 20.0,
        }  
        damping = {
            "hip": 0.6, "thigh": 0.6, "calf": 0.6,
            "shoulder_link_joint": 3.0, "upper_arm_link_joint": 2.0, "forearm_link_joint": 3.0, 
            "wrist1_link_joint": 3.0, "wrist2_link_joint": 2.5, "wrist3_link_joint": 1.0,
        }  
        action_scale = 0.25
        decimation = 4  # control frequency = sim frequency / decimation
        hip_reduction = 0.5
        effort_limit = {
            "hip": 10.0, "thigh": 10.0, "calf": 10.0,
            "shoulder_link_joint": 10.0, "upper_arm_link_joint": 10.0, "forearm_link_joint": 10.0, 
            "wrist1_link_joint": 10.0, "wrist2_link_joint": 3.7, "wrist3_link_joint": 3.7,
        }
        armature = 0.01
        friction = 0.01
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class reward:
        pass