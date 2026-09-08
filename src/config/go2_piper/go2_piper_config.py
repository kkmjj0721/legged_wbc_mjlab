from src.config.base.base_config import BaseConfig


class Go2PiperCfg( BaseConfig ):
    class env:
        num_envs = 4096


    class asset:
        file = "../../assets/robots/go2_piper/xmls/go2piper.xml"
        name = "Go2piper"
        
    class control:
        stiffness = {
            "hip": 30.0, "thigh": 30.0, "calf": 30.0,
            "joint1": 50.0, "joint2": 50.0, "joint3": 80.0, "joint4": 30.0, "joint5": 30.0, "joint6": 20.0,
        }  
        damping = {
            "hip": 0.6, "thigh": 0.6, "calf": 0.6,
            "joint1": 3.0, "joint2": 2.0, "joint3": 3.0, "joint4": 3.0, "joint5": 2.5, "joint6": 1.0,
        }  
        action_scale = 0.25
        decimation = 4  # control frequency = sim frequency / decimation
        hip_reduction = 0.5
        effort_limit = {
            "hip": 23.7, "thigh": 23.7, "calf": 45.43,
            "joint1": 20.0, "joint2": 20.0, "joint3": 15.0, "joint4": 7.0, "joint5": 5.0, "joint6": 5.0,
        }
        armature = 0.01
        friction = 0.01
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class reward:
        pass
        


