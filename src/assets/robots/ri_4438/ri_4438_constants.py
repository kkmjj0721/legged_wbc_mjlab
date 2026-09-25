from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg


from config.ri_4438.ri_4438_config import Ri4438Cfg


Ri4438_piper_cfg = Ri4438Cfg()

##
# MJCF
##
RI4438PIPER_XML: Path = (
  SRC_PATH / "assets" / "robots" / "ri_4438" / "xmls" / "ri_4438.xml"
)
assert RI4438PIPER_XML.exists()


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(RI4438PIPER_XML))


##
# Actuator config.
##
RI4438_HIP = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*hip_.*",
  ),
  stiffness = Ri4438_piper_cfg.control.stiffness["hip"],
  damping = Ri4438_piper_cfg.control.damping["hip"],
  effort_limit = Ri4438_piper_cfg.control.effort_limit["hip"],
  armature = Ri4438_piper_cfg.control.armature,
  delay_min_lag = Ri4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Ri4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Ri4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Ri4438_piper_cfg.control.delay_update_period,
)

RI4438_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*thigh_.*",
  ),
  stiffness = Ri4438_piper_cfg.control.stiffness["thigh"],
  damping = Ri4438_piper_cfg.control.damping["thigh"],
  effort_limit = Ri4438_piper_cfg.control.effort_limit["thigh"],
  armature = Ri4438_piper_cfg.control.armature,
  delay_min_lag = Ri4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Ri4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Ri4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Ri4438_piper_cfg.control.delay_update_period,
)

RI4438_CALF = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*calf_.*",
  ),
  stiffness = Ri4438_piper_cfg.control.stiffness["calf"],
  damping = Ri4438_piper_cfg.control.damping["calf"],
  effort_limit = Ri4438_piper_cfg.control.effort_limit["calf"],
  armature = Ri4438_piper_cfg.control.armature,
  delay_min_lag = Ri4438_piper_cfg.control.delay_min_lag,
  delay_max_lag = Ri4438_piper_cfg.control.delay_max_lag,
  delay_hold_prob = Ri4438_piper_cfg.control.delay_hold_prob,
  delay_update_period = Ri4438_piper_cfg.control.delay_update_period,
)

##
# Keyframes.
##
INIT_STATE = EntityCfg.InitialStateCfg(
    pos = tuple(Ri4438_piper_cfg.init_state.pos),
    joint_pos = dict(Ri4438_piper_cfg.init_state.default_joint),
    joint_vel = {".*": 0.0},
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
  conaffinity = 1,
  condim = {_foot_regex: 3, ".*_collision": 1},
  priority = {_foot_regex: 1, ".*": 0,},
  friction = {_foot_regex: (0.6,)},
  solimp = {_foot_regex: (0.9, 0.95, 0.023)},
)


##
# Final config.
##
RI4438_ARTICULATION = EntityArticulationInfoCfg(
  actuators = (
    RI4438_HIP,
    RI4438_THIGH,
    RI4438_CALF,
  ),
  soft_joint_pos_limit_factor = 0.9,
)


def get_ri_4438_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state = INIT_STATE,
    collisions = (FULL_COLLISION,),
    spec_fn = get_spec,
    articulation = RI4438_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_ri_4438_robot_cfg())

  viewer.launch(robot.spec.compile())
