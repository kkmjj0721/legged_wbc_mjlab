"""RI-4438 HIM environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_ri_4438_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.locomotion.ri_4438_him.ri_4438_him_env_cfg import (
  make_velocity_env_cfg,
)

import src.tasks.locomotion.ri_4438_him.mdp as mdp

TerrainType = Literal["rough", "obstacles"]


def ri_4438_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the RI-4438 HIM rough-terrain environment."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = None
  cfg.scene.entities = {"robot": get_ri_4438_robot_cfg()}
  # Training keeps true terminal observations; play mode may auto-reset.
  cfg.auto_reset = bool(play)

  # Set raycast sensor frame to RI-4438 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")

  feet_height_cfg = TerrainHeightSensorCfg(
    name="feet_terrain_height",
    frame=tuple(
      ObjRef(
        type="site",
        name=name,
        entity="robot",
      )
      for name in site_names
    ),
    pattern=RingPatternCfg.single_ring(
      radius=0.02,
      num_samples=4,
      include_center=True,
    ),
    ray_alignment="world",
    max_distance=1.0,
    exclude_parent_body=True,

    # 地形几何体默认是 group=0，
    # 排除机器人自身的 visual/collision 几何体。
    include_geom_groups=(0,),

    # 取脚下局部地形中最高的点，作为保守 clearance。
    reduction="min",
  )

  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name = "feet_ground_contact",
    primary = ContactMatch(mode = "geom", pattern = geom_names, entity = "robot"),
    secondary = ContactMatch(mode = "body", pattern = "terrain"),
    fields = ("found", "force"),
    reduce = "netforce",
    num_slots = 1,
    track_air_time=True,
  )

  nonfoot_ground_cfg = ContactSensorCfg(
    name = "nonfoot_ground_touch",
    primary = ContactMatch(
      mode = "geom",
      entity = "robot",
      # Only base contact is considered an illegal ground contact.
      # Thigh/calf/hip contacts remain observable through the robot model
      # but no longer terminate the episode via this sensor.
      pattern = r"^base_collision$",
      exclude = (),
    ),
    secondary = ContactMatch(mode = "body", pattern = "terrain"),
    fields = ("found", "force"),
    reduce = "none",
    num_slots = 1,
    history_length = 4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    feet_height_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)
    cfg.auto_reset = True

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func = envs_mdp.randomize_terrain,
      mode = "reset",
      params = {},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def ri_4438_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the RI-4438 HIM flat-terrain environment."""
  cfg = ri_4438_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  # cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Keep the exact same critic observation schema as rough terrain.
  #
  # The critic is part of a HIM checkpoint (including its first-layer weight
  # and observation normalizer).  Removing ``terrain_scan`` and rebuilding the
  # flat critic with foot-only terms changes the input from 244 to 74 values,
  # which makes a rough <-> flat resume fail with a state-dict size mismatch.
  # A raycast against the plane is valid and produces a constant/near-constant
  # height scan, so retaining this term keeps both variants shape-compatible.
  # ``terrain_levels`` is still removed below because a plane has no terrain
  # curriculum.

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
