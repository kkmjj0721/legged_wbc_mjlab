"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.spec_config import CollisionCfg

from config.go2.go2_config import Go2Cfg


go2_cfg = Go2Cfg

##
# MJCF and assets.
##

GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(GO2_XML))


##
# Actuator config.
##

GO2_HIP = BuiltinPositionActuatorCfg(
  target_names_expr = (
      ".*hip_.*",
    ),
  stiffness = go2_cfg.control.stiffness["hip"],
  damping = go2_cfg.control.damping["hip"],
  effort_limit = go2_cfg.control.effort_limit["hip"],
  armature = go2_cfg.control.armature,
  delay_min_lag = go2_cfg.control.delay_min_lag,
  delay_max_lag = go2_cfg.control.delay_max_lag,
  delay_hold_prob = go2_cfg.control.delay_hold_prob,
  delay_update_period = go2_cfg.control.delay_update_period,
)

GO2_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*thigh_.*",
  ),
  stiffness = go2_cfg.control.stiffness["thigh"],
  damping = go2_cfg.control.damping["thigh"],
  effort_limit = go2_cfg.control.effort_limit["thigh"],
  armature = go2_cfg.control.armature,
  delay_min_lag = go2_cfg.control.delay_min_lag,
  delay_max_lag = go2_cfg.control.delay_max_lag,
  delay_hold_prob = go2_cfg.control.delay_hold_prob,
  delay_update_period = go2_cfg.control.delay_update_period,
)

GO2_CALF = BuiltinPositionActuatorCfg(
  target_names_expr = (
    ".*calf_.*",
  ),
  stiffness = go2_cfg.control.stiffness["calf"],
  damping = go2_cfg.control.damping["calf"],
  effort_limit = go2_cfg.control.effort_limit["calf"],
  armature = go2_cfg.control.armature,
  delay_min_lag = go2_cfg.control.delay_min_lag,
  delay_max_lag = go2_cfg.control.delay_max_lag,
  delay_hold_prob = go2_cfg.control.delay_hold_prob,
  delay_update_period = go2_cfg.control.delay_update_period,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos = tuple(go2_cfg.init_state.pos),
  joint_pos = dict(go2_cfg.init_state.default_joint),
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
  geom_names_expr=(".*_collision",),
  condim = {_foot_regex: 3, ".*_collision": 1},
  priority = {_foot_regex: 1, ".*": 0},
  friction = {_foot_regex: (0.6,)},
  solimp = {_foot_regex: (0.9, 0.95, 0.023)},
  contype = 1,
  conaffinity = 0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators = (
    GO2_HIP,
    GO2_THIGH,
    GO2_CALF,
  ),
  soft_joint_pos_limit_factor = 0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state = INIT_STATE,
    collisions = (FULL_COLLISION,),
    spec_fn = get_spec,
    articulation = GO2_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())
