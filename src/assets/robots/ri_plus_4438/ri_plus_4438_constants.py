from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg


from config.ri_plus_4438.ri_plus_4438_config import Riplus4438PiperCfg


Riplus4438_piper_cfg = Riplus4438PiperCfg()

##
# MJCF
##
RIPLUS4438PIPER_XML: Path = (
  SRC_PATH / "assets" / "robots" / "ri_plus_4438" / "xmls" / "ri_plus_4438.xml"
)
assert RIPLUS4438PIPER_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(RIPLUS4438PIPER_XML))


##
# Actuator config.
##
RIPLUS4438_HIP = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*hip_.*",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["hip"],
  damping = Riplus4438_piper_cfg.control.damping["hip"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["hip"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*thigh_.*",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["thigh"],
  damping = Riplus4438_piper_cfg.control.damping["thigh"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["thigh"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_CALF = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*calf_.*",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["calf"],
  damping = Riplus4438_piper_cfg.control.damping["calf"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["calf"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^shoulder_link_joint$",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["shoulder_link_joint"],
  damping = Riplus4438_piper_cfg.control.damping["shoulder_link_joint"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["shoulder_link_joint"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_UPPER = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^upper_arm_link_joint$",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["upper_arm_link_joint"],
  damping = Riplus4438_piper_cfg.control.damping["upper_arm_link_joint"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["upper_arm_link_joint"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_FOREARM = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^forearm_link_joint$",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["forearm_link_joint"],
  damping = Riplus4438_piper_cfg.control.damping["forearm_link_joint"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["forearm_link_joint"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_WRIST1 = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^wrist1_link_joint$",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["wrist1_link_joint"],
  damping = Riplus4438_piper_cfg.control.damping["wrist1_link_joint"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["wrist1_link_joint"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_WRIST2 = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^wrist2_link_joint$",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["wrist2_link_joint"],
  damping = Riplus4438_piper_cfg.control.damping["wrist2_link_joint"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["wrist2_link_joint"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)

RIPLUS4438_WRIST3 = BuiltinPositionActuatorCfg(
  target_names_expr = (
    "^wrist3_link_joint$",
  ),
  stiffness = Riplus4438_piper_cfg.control.stiffness["wrist3_link_joint"],
  damping = Riplus4438_piper_cfg.control.damping["wrist3_link_joint"],
  effort_limit = Riplus4438_piper_cfg.control.effort_limit["wrist3_link_joint"],
  armature = Riplus4438_piper_cfg.control.armature,
  delay_min_lag = Riplus4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Riplus4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Riplus4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Riplus4438_piper_cfg.control.delay_update_period,
)


##
# Keyframes.
##
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.25),
    joint_pos={
        # 四足腿部
        ".*thigh_joint": 0.9,
        ".*calf_joint": -1.8,
        ".*R_hip_joint": 0.0,
        ".*L_hip_joint": -0.0,
        # 机械臂各关节归零
        ".*_link_joint": 0.0,
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
  priority = {_foot_regex: 1, ".*": 0,},
  friction = {_foot_regex: (0.6,)},
  solimp = {_foot_regex: (0.9, 0.95, 0.023)},
)


##
# Final config.
##
RIPLUS4438_ARTICULATION = EntityArticulationInfoCfg(
  actuators = (
    RIPLUS4438_HIP,
    RIPLUS4438_THIGH,
    RIPLUS4438_CALF,
    RIPLUS4438_SHOULDER,
    RIPLUS4438_UPPER,
    RIPLUS4438_FOREARM,
    RIPLUS4438_WRIST1,
    RIPLUS4438_WRIST2,
    RIPLUS4438_WRIST3
  ),
  soft_joint_pos_limit_factor = 0.9,
)


def get_ri_plus_4438_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state = INIT_STATE,
    collisions = (FULL_COLLISION,),
    spec_fn = get_spec,
    articulation = RIPLUS4438_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ri_plus_4438_robot_cfg())

  viewer.launch(robot.spec.compile())
