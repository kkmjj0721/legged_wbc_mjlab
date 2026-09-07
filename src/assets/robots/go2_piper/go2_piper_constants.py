from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg


from config.go2_piper.go2_piper_config import Go2PiperCfg


go2_piper_cfg = Go2PiperCfg()

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
  stiffness = go2_piper_cfg.control.stiffness["hip"],
  damping = go2_piper_cfg.control.damping["hip"],
  effort_limit = go2_piper_cfg.control.effort_limit["hip"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*thigh_.*",
  ),
  stiffness = go2_piper_cfg.control.stiffness["thigh"],
  damping = go2_piper_cfg.control.damping["thigh"],
  effort_limit = go2_piper_cfg.control.effort_limit["thigh"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_CALF = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*calf_.*",
  ),
  stiffness = go2_piper_cfg.control.stiffness["calf"],
  damping = go2_piper_cfg.control.damping["calf"],
  effort_limit = go2_piper_cfg.control.effort_limit["calf"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_ARM_JOINT1 = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^joint1$",
  ),
  stiffness = go2_piper_cfg.control.stiffness["joint1"],
  damping = go2_piper_cfg.control.damping["joint1"],
  effort_limit = go2_piper_cfg.control.effort_limit["joint1"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_ARM_JOINT2 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      "^joint2$",
    ),
  stiffness = go2_piper_cfg.control.stiffness["joint2"],
  damping = go2_piper_cfg.control.damping["joint2"],
  effort_limit = go2_piper_cfg.control.effort_limit["joint2"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_ARM_JOINT3 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      "^joint3$",
    ),
  stiffness = go2_piper_cfg.control.stiffness["joint3"],
  damping = go2_piper_cfg.control.damping["joint3"],
  effort_limit = go2_piper_cfg.control.effort_limit["joint3"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_ARM_JOINT4 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      "^joint4$",
    ),
  stiffness = go2_piper_cfg.control.stiffness["joint4"],
  damping = go2_piper_cfg.control.damping["joint4"],
  effort_limit = go2_piper_cfg.control.effort_limit["joint4"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_ARM_JOINT5 = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^joint5$",
  ),
  stiffness = go2_piper_cfg.control.stiffness["joint5"],
  damping = go2_piper_cfg.control.damping["joint5"],
  effort_limit = go2_piper_cfg.control.effort_limit["joint5"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)

GO2_PIPER_ARM_JOINT6 = BuiltinPositionActuatorCfg(
  target_names_expr = (
      "^joint6$",
    ),
  stiffness = go2_piper_cfg.control.stiffness["joint6"],
  damping = go2_piper_cfg.control.damping["joint6"],
  effort_limit = go2_piper_cfg.control.effort_limit["joint6"],
  armature = go2_piper_cfg.control.armature,
  delay_min_lag = go2_piper_cfg.control.delay_min_lag,
  delay_max_lag = go2_piper_cfg.control.delay_max_lag,
  delay_hold_prob = go2_piper_cfg.control.delay_hold_prob,
  delay_update_period = go2_piper_cfg.control.delay_update_period,
)


##
# Keyframes.
##
INIT_STATE = EntityCfg.InitialStateCfg(
  pos = (0.0, 0.0, 0.35),
  joint_pos = {
    # leg
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
    # arm
    "^joint[1-6]$": 0.0,
  },
)


##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr = (_foot_regex,),
  contype = 0,
  conaffinity = 1,
  condim = 3,
  priority = 1,
  friction = (0.6,),
  solimp = (0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr = (".*_collision",),
  contype = 1,
  conaffinity = 0,
  condim = {_foot_regex: 3, ".*_collision": 1},
  priority = {_foot_regex: 1},
  friction = {_foot_regex: (0.6,)},
  solimp = {_foot_regex: (0.9, 0.95, 0.023)},
)


##
# Final config.
##
GO2PIPER_ARTICULATION = EntityArticulationInfoCfg(
  actuators = (
    GO2_PIPER_HIP,
    GO2_PIPER_THIGH,
    GO2_PIPER_CALF,
    GO2_PIPER_ARM_JOINT1,
    GO2_PIPER_ARM_JOINT2,
    GO2_PIPER_ARM_JOINT3,
    GO2_PIPER_ARM_JOINT4,
    GO2_PIPER_ARM_JOINT5,
    GO2_PIPER_ARM_JOINT6,
  ),
  soft_joint_pos_limit_factor = 0.9,
)


def get_go2_piper_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state = INIT_STATE,
    collisions = (FULL_COLLISION,),
    spec_fn = get_spec,
    articulation = GO2PIPER_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_piper_robot_cfg())

  viewer.launch(robot.spec.compile())

