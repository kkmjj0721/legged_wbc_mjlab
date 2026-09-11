"""RI-4438 HIM velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.locomotion.ri_4438_him.mdp as mdp

from src.config.ri_4438.ri_4438_config import Ri4438PiperCfg

ri_4438_cfg = Ri4438PiperCfg()

def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base velocity tracking task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name = "terrain_scan",
    frame = ObjRef(type = "body", name = "", entity="robot"),  # Set per-robot.
    ray_alignment = "yaw",
    pattern = GridPatternCfg(size = (1.6, 1.0), resolution = 0.1),
    max_distance = 5.0,
    exclude_parent_body = True,
    debug_vis = True,
    viz = RayCastSensorCfg.VizCfg(show_normals = True),
  )

  ##
  # Observations
  ##

  # Keep this order in sync with ``Ri4438HimCfg`` and the estimator target
  # slices.  The actor history is represented as six frame-major observations,
  # each containing the following 47 features.
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func = mdp.builtin_sensor,
      params = {"sensor_name": "robot/imu_ang_vel"},
      noise = Unoise(n_min = -0.2, n_max = 0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func = mdp.projected_gravity,
      noise = Unoise(n_min = -0.05, n_max = 0.05),
    ),
    "command": ObservationTermCfg(
      func = mdp.generated_commands,
      params = {"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func = mdp.phase,
      params = {"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func = mdp.joint_pos_rel,
      noise = Unoise(n_min = -0.01, n_max = 0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func = mdp.joint_vel_rel,
      noise = Unoise(n_min = -1.5, n_max = 1.5),
    ),
    "actions": ObservationTermCfg(func = mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func = mdp.builtin_sensor,
      params = {"sensor_name": "robot/imu_lin_vel"},
      noise = Unoise(n_min = -0.5, n_max = 0.5),
    ),
    "base_com": ObservationTermCfg(
      func = mdp.base_com,
      params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "foot_contact": ObservationTermCfg(
      func = mdp.foot_contact,
      params = {"sensor_name": "feet_ground_contact"},
    ),
    "height_scan": ObservationTermCfg(
      func = envs_mdp.height_scan,
      params = {"sensor_name": "terrain_scan"},
      scale = 1 / terrain_scan.max_distance,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms = actor_terms,
      concatenate_terms = True,
      enable_corruption = True,
      history_length = 6,
      # HIMActorModel reverses this explicit time axis to current-first.
      flatten_history_dim = False,
    ),
    "critic": ObservationGroupCfg(
      terms = critic_terms,
      concatenate_terms = True,
      enable_corruption = False,
      history_length = 1,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func = mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
  ##

  def _build_action():
    base_action_scale = ri_4438_cfg.control.action_scale
    scale = {
      r".*_hip_joint": base_action_scale * ri_4438_cfg.control.hip_reduction,
      r".*_thigh_joint": base_action_scale,
      r".*_calf_joint": base_action_scale,
    }

    actions: dict[str, ActionTermCfg] = {
      "joint_pos": JointPositionActionCfg(
        entity_name = "robot",
        actuator_names = (".*",),
        scale = scale,
        use_default_offset = True,
      )
    }

    return actions

  ##
  # Commands
  ##
  def _build_commands():
    commands: dict[str, CommandTermCfg] = {
      "twist": UniformVelocityCommandCfg(
        entity_name = "robot",
        resampling_time_range = tuple(ri_4438_cfg.comaman.resampling_time),
        rel_standing_envs = float(ri_4438_cfg.comaman.rel_standing_envs),
        rel_forward_envs = float(ri_4438_cfg.comaman.rel_forward_envs),
        heading_command = ri_4438_cfg.comaman.heading_command,
        debug_vis = True,
        ranges = UniformVelocityCommandCfg.Ranges(
          lin_vel_x = tuple(ri_4438_cfg.comaman.ranges.lin_vel_x),
          lin_vel_y = tuple(ri_4438_cfg.comaman.ranges.lin_vel_y),
          ang_vel_z = tuple(ri_4438_cfg.comaman.ranges.ang_vel_yaw),
          heading=(
            tuple(ri_4438_cfg.comaman.ranges.heading)
            if ri_4438_cfg.comaman.heading_command
            else None
          ),
        ),
      )
    }

    return commands

  ##
  # Events
  ##
  # def _build_events():


  events = {
    # reset
    "reset_base": EventTermCfg(
      func = mdp.reset_root_state_uniform,
      mode = "reset",
      params = dict(ri_4438_cfg.reset.base_offset)
    ),
    "reset_robot_joints": EventTermCfg(
      func = mdp.reset_joints_by_offset,
      mode = "reset",
      params = {
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),

    # dr
    "push_robot": EventTermCfg(
      func = mdp.push_by_setting_velocity,
      mode = "interval",
      interval_range_s = (5.0, 6.0),
      params = {
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "pd_gains": EventTermCfg(
      func = dr.pd_gains,
      mode = "startup",
      params = {
        "kp_range": tuple(ri_4438_cfg.domain_rand.KpKd_factor_range),
        "kd_range": tuple(ri_4438_cfg.domain_rand.KpKd_factor_range),
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
      },
    ),

    "effort_limits": EventTermCfg(
      func = dr.effort_limits,
      mode = "startup",
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "effort_limit_range": tuple(ri_4438_cfg.domain_rand.motor_strength_range)
      }  
    ),

    # "body_mass": EventTermCfg(
    #   func = dr.body_mass,
    #   mode = "startup",
    #   params = {
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "ranges": tuple(ri_4438_cfg.domain_rand.link_mass_range),
    #     "operation": "scale",
    #   },
    # )

    "pseudo_inertia": EventTermCfg(
      func = dr.pseudo_inertia,
      mode = "startup",
      params = {
        # 默认选择 robot 的所有 body，包括 base_link 和各个腿部 link
        "asset_cfg": SceneEntityCfg("robot"),

        # pseudo_inertia 中质量缩放倍数为 exp(2 * alpha)
        # 这里把 [0.8, 1.2] 的质量缩放范围转换成 alpha 范围
        "alpha_range": (
          0.5 * math.log(ri_4438_cfg.domain_rand.link_mass_range[0]),
          0.5 * math.log(ri_4438_cfg.domain_rand.link_mass_range[1]),
        ),

        "distribution": "uniform",
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func = mdp.track_linear_velocity,
      weight = 1.0,
      params = {"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func = mdp.track_angular_velocity,
      weight = 0.5,
      params = {"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func = mdp.body_orientation_l2,
      weight = -1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func = mdp.body_angular_velocity_penalty,
      weight = -0.05,  # Override per-robot
      params = {"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    ),
    "angular_momentum": RewardTermCfg(
      func = mdp.angular_momentum_penalty,
      weight = -0.025,  # Override per-robot
      params = {"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight = -100.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight = -2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight = -10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight = -0.01),
    "smoothness": RewardTermCfg(func=mdp.action_acc_l2, weight = -0.01),
    "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight = -2.0e-5),
    "hip_pos": RewardTermCfg(
      func = mdp.hip_joint_deviation_penalty,
      weight = -0.5,
      params = {
        "command_name": "twist",
      }
    ),
    "foot_gait": RewardTermCfg(
      func = mdp.feet_gait,
      weight = 0.5,
      params = {
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func = mdp.feet_clearance,
      weight = -1.0,
      params = {
        "target_height": 0.08,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "foot_slip": RewardTermCfg(
      func = mdp.feet_slip,
      weight = -0.25,
      params = {
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func = mdp.soft_landing,
      weight = -1e-3,
      params = {
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func = mdp.stand_still,
      weight = -1.0,
      params = {
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }

  ##
  # Terminations
  ##
  def _build_terminations():
    terminations = {
      "time_out": TerminationTermCfg(func = mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(
        func = mdp.bad_orientation,
        params = {"limit_angle": math.radians(70.0)},
      ),
    }

    return terminations

  ##
  # Curriculum
  ##
  def _build_curriculum():
    curriculum = {
      "terrain_levels": CurriculumTermCfg(
        func = mdp.terrain_levels_vel,
        params = {"command_name": "twist"},
      ),
      "command_vel": CurriculumTermCfg(
        func = mdp.commands_vel,
        params = {
          "command_name": "twist",
          "velocity_stages": [
            {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
            {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0)},
          ],
        },
      ),
    }

    return curriculum

  
  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene = SceneCfg(
      terrain = TerrainEntityCfg(
        terrain_type = "generator",
        terrain_generator = replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level = 5,
      ),
      sensors = (terrain_scan,),
      num_envs = 1,
      extent = 2.0,
    ),
    observations = observations,
    actions = _build_action(),
    commands = _build_commands(),
    events = events,
    rewards = rewards,
    terminations = _build_terminations(),
    curriculum = _build_curriculum(),
    metrics = metrics,
    viewer = ViewerConfig(
      origin_type = ViewerConfig.OriginType.ASSET_BODY,
      entity_name = "robot",
      body_name = "",  # Set per-robot.
      distance = 3.0,
      elevation = -5.0,
      azimuth = 90.0,
    ),
    sim = SimulationCfg(
      nconmax = 256,
      njmax = 1500,
      mujoco = MujocoCfg(
        timestep = 0.005,
        iterations = 10,
        ls_iterations = 20,
      ),
    ),
    decimation = 4,
    episode_length_s = 20.0,
    # HIM training stores the pre-reset terminal observation before the
    # dedicated runner partially resets completed environments.  Play mode
    # overrides this to True in ``config/env_cfgs.py``.
    auto_reset = False,
  )
