from base.base_config import BaseConfig


class Go2PiperCfg( BaseConfig ):
    class env:
        pass

    class asset:
        file = "../../assets/robots/go2_piper/xmls/go2piper.xml"
        name = "Go2piper"
        
    class control:
        stiffness = {}  # [N*m/rad]
        damping = {}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4  # control frequency = sim frequency / decimation
        hip_reduction = 0.5
        effort_limit = None
        armature = None


