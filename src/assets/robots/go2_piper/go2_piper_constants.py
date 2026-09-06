from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


from config.go2_piper.go2_piper_config import Go2PiperCfg


CFG = Go2PiperCfg()

##
# MJCF
##

GO2PIPER_XML: Path = (
  SRC_PATH / "assets" / "robots" / "go2_piper" / "xmls" / "go2piper.xml"
)
assert GO2PIPER_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(GO2PIPER_XML))



##
# Actuator config.
##
GO2_PIPER_HIP = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*hip_.*",
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag =
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_CALF = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_ARM_JOINT1 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_ARM_JOINT2 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_ARM_JOINT3 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_ARM_JOINT4 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_ARM_JOINT5 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

GO2_PIPER_ARM_JOINT6 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      
  ),
  stiffness = 
  damping = 
  effort_limit = 
  armature = 
  delay_min_lag = 
  delay_max_lag = 
  delay_hold_prob =
  delay_update_period = 
)

##
# Keyframes.
##
INIT_STATE = EntityCfg.InitialStateCfg(
  pos = (0.0, 0.0, 0.32),
  joint_pos = {},
)




##
# Final config.
##