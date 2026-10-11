# Adapted from RI-4438 HIM (workspace snapshot 2026-10-10).
"""RI-4438 HIM velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
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
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp
from . import mdp as him_mdp
from .mdp.velocity_command import TaskVelocityCommandCfg


def make_velocity_env_cfg(setup) -> ManagerBasedRlEnvCfg:
    """HIM behavior copied locally; task/geometry come from the compiler."""
    ri_4438_cfg = setup.cfg.robot

    ##
    # Sensors
    ##

    terrain_scan = RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
        ray_alignment="yaw",
        pattern=GridPatternCfg(
            size=setup.cfg.scan_size_m, resolution=setup.cfg.scan_resolution_m
        ),
        max_distance=5.0,
        exclude_parent_body=True,
        debug_vis=True,
        viz=RayCastSensorCfg.VizCfg(show_normals=True),
    )

    ##
    # Observations
    ##

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
        ),
        "phase": ObservationTermCfg(
            func=him_mdp.phase,
            params={"period": 0.6, "command_name": "twist"},
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"biased": True},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }

    critic_terms = {
        **actor_terms,
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_com": ObservationTermCfg(
            func=him_mdp.base_com,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
            },
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "height_scan": ObservationTermCfg(
            func=envs_mdp.height_scan,
            params={"sensor_name": "terrain_scan"},
            scale=1 / terrain_scan.max_distance,
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=ri_4438_cfg.env.history_size,
            # HIMActorModel reverses this explicit time axis to current-first.
            flatten_history_dim=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }

    # Replace actor terms only; critic and estimator retain unbiased true states.
    actor_terms["base_ang_vel"] = replace(
        actor_terms["base_ang_vel"],
        func=him_mdp.imu_ang_vel_with_errors,
    )
    actor_terms["projected_gravity"] = replace(
        actor_terms["projected_gravity"],
        func=him_mdp.imu_projected_gravity_with_errors,
    )

    for name in ("base_ang_vel", "projected_gravity"):
        actor_terms[name] = replace(
            actor_terms[name],
            delay_min_lag=0,
            delay_max_lag=1,
            delay_hold_prob=0.3,
            delay_update_period=10,
        )

    for name in ("joint_pos", "joint_vel"):
        actor_terms[name] = replace(
            actor_terms[name],
            delay_min_lag=0,
            delay_max_lag=1,
            delay_hold_prob=0.3,
            delay_update_period=10,
        )

    ##
    # Metrics
    ##

    metrics = {
        "mean_action_acc": MetricsTermCfg(
            func=mdp.mean_action_acc,
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
                entity_name="robot",
                actuator_names=(".*",),
                scale=scale,
                use_default_offset=True,
                clip={
                    r".*_hip_joint": (-1.50, 1.50),
                    r".*_thigh_joint": (-1.57, 3.49),
                    r".*_calf_joint": (-2.53, -0.84),
                },
            )
        }

        return actions

    ##
    # Commands
    ##
    def _build_commands():
        interval = setup.cfg.task_space.resampling_interval_s
        return {
            "twist": TaskVelocityCommandCfg(resampling_time_range=(interval, interval))
        }

    ##
    # Events
    ##
    # def _build_events():

    # Uniform alpha produces a log-uniform mass multiplier, exp(2 * alpha).
    base_mass_alpha_range = tuple(
        0.5 * math.log(scale) for scale in ri_4438_cfg.domain_rand.base_mass_range
    )
    link_mass_alpha_range = tuple(
        0.5 * math.log(scale) for scale in ri_4438_cfg.domain_rand.link_mass_range
    )

    events = {
        # reset
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params=dict(ri_4438_cfg.reset.base_offset),
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                **deepcopy(ri_4438_cfg.reset.joint_offset),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        # dr
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 6.0),
            params={
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-0.4, 0.4),
                    "roll": (-0.6, 0.6),
                    "pitch": (-0.6, 0.6),
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
                "ranges": (0.25, 1.6),
                "shared_random": True,  # All foot geoms share the same friction.
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": tuple(ri_4438_cfg.domain_rand.motor_offset_range),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.pseudo_inertia,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
                # Randomize base mass, inertia and COM together. The leg event below
                # excludes the base, so it cannot overwrite this COM perturbation.
                "alpha_range": base_mass_alpha_range,
                "t_range": tuple(ri_4438_cfg.domain_rand.com_displacement_range),
                "distribution": "uniform",
            },
        ),
        "pd_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "kp_range": tuple(ri_4438_cfg.domain_rand.KpKd_factor_range),
                "kd_range": tuple(ri_4438_cfg.domain_rand.KpKd_factor_range),
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
            },
        ),
        "effort_limits": EventTermCfg(
            func=dr.effort_limits,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "effort_limit_range": tuple(ri_4438_cfg.domain_rand.effort_limit_range),
            },
        ),
        "pseudo_inertia": EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=(r".*_(hip|thigh|calf)",),
                ),
                "alpha_range": link_mass_alpha_range,
                "distribution": "uniform",
            },
        ),
        "synchronize_sensor_delays": EventTermCfg(
            func=him_mdp.synchronize_sensor_delays,
            mode="startup",
        ),
        "joint_friction": EventTermCfg(
            func=dr.joint_friction,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_joint",)),
                "operation": "scale",
                "ranges": tuple(ri_4438_cfg.domain_rand.joint_friction_range),
            },
        ),
        "joint_armature": EventTermCfg(
            func=dr.joint_armature,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_joint",)),
                "operation": "scale",
                "ranges": tuple(ri_4438_cfg.domain_rand.joint_armature_range),
            },
        ),
        "joint_damping": EventTermCfg(
            func=dr.joint_damping,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_joint",)),
                "operation": "scale",
                "ranges": tuple(ri_4438_cfg.domain_rand.joint_damping_range),
            },
        ),
        "imu_errors": EventTermCfg(
            func=him_mdp.randomize_imu_errors,
            mode="startup",
            params={
                "gyro_bias_range": tuple(ri_4438_cfg.domain_rand.imu_gyro_bias_range),
                "mount_angle_range": tuple(
                    ri_4438_cfg.domain_rand.imu_mount_angle_range
                ),
            },
        ),
        "external_force": EventTermCfg(
            func=envs_mdp.apply_body_impulse,
            mode="step",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
                "force_range": tuple(ri_4438_cfg.domain_rand.external_force_range),
                "torque_range": (0.0, 0.0),
                "duration_s": tuple(ri_4438_cfg.domain_rand.external_force_duration_s),
                "cooldown_s": tuple(ri_4438_cfg.domain_rand.external_force_cooldown_s),
            },
        ),
        "foot_contact_softness": EventTermCfg(
            func=him_mdp.randomize_foot_contact_softness,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
                "time_constant_range": tuple(
                    ri_4438_cfg.domain_rand.foot_contact_time_constant_range
                ),
            },
        ),
    }

    ##
    # Basic PPO-style rewards, implemented and tuned locally for RI-4438 HIM.
    ##

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=him_mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "lin_vel_z": RewardTermCfg(
            func=him_mdp.lin_vel_z,
            weight=-2.0,
        ),
        "track_angular_velocity": RewardTermCfg(
            func=him_mdp.track_angular_velocity,
            weight=1.0,
            params={"command_name": "twist", "std": math.sqrt(0.25)},
        ),
        "body_orientation_l2": RewardTermCfg(
            func=him_mdp.body_orientation_l2,
            weight=-0.25,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=())
            },  # Set per-robot.
        ),
        "pose": RewardTermCfg(
            func=him_mdp.variable_posture,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "twist",
                "std_standing": {},  # 继续由机器人配置填入。
                "std_walking": {},
                "std_running": {},
                "walking_threshold": 0.1,
                "running_threshold": 1.0,
                "scan_sensor_name": "terrain_scan",
                "terrain_radius": 0.35,  # 局部扫描半径，米。
                "terrain_height_std": (0.01, 0.08),  # 地面高度起伏阈值，米。
                "terrain_tolerance": {  # 最大关节容差，弧度。
                    ".*_hip_joint": 0.15,
                    ".*_thigh_joint": 0.30,
                    ".*_calf_joint": 0.35,
                },
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=him_mdp.body_angular_velocity_penalty,
            weight=-0.08,  # Override per-robot
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=())
            },  # Set per-robot.
        ),
        "angular_momentum": RewardTermCfg(
            func=him_mdp.angular_momentum_penalty,
            weight=-0.025,  # Override per-robot
            params={"sensor_name": "robot/root_angmom"},
        ),
        "is_terminated": RewardTermCfg(func=him_mdp.is_terminated, weight=-100.0),
        "joint_acc_l2": RewardTermCfg(func=him_mdp.joint_acc_l2, weight=-2.5e-7),
        "joint_pos_limits": RewardTermCfg(func=him_mdp.joint_pos_limits, weight=-10.0),
        "action_rate_l2": RewardTermCfg(func=him_mdp.action_rate_l2, weight=-0.015),
        "smoothness": RewardTermCfg(func=him_mdp.action_acc_l2, weight=-0.015),
        "joint_torques_l2": RewardTermCfg(
            func=him_mdp.joint_torques_l2, weight=-2.0e-5
        ),
        "hip_pos": RewardTermCfg(
            func=him_mdp.hip_joint_deviation_penalty,
            weight=-0.5,
            params={
                "command_name": "twist",
            },
        ),
        "foot_gait": RewardTermCfg(
            func=him_mdp.feet_gait,
            weight=0.3,
            params={
                "period": 0.6,
                "offset": [0.0, 0.5, 0.5, 0.0],
                "threshold": 0.56,
                "command_threshold": 0.1,
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
            },
        ),
        "feet_air_time": RewardTermCfg(
            func=him_mdp.feet_air_time,
            weight=1.0,
            params={
                "sensor_name": "feet_ground_contact",
                "gait_name": "foot_gait",  # Target time and support pair follow this gait.
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        ),
        "foot_slip": RewardTermCfg(
            func=him_mdp.feet_slip,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
        "soft_landing": RewardTermCfg(
            func=him_mdp.soft_landing,
            weight=-2e-3,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
            },
        ),
        "stand_still": RewardTermCfg(
            func=him_mdp.stand_still,
            weight=-0.5,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "leg_collision": RewardTermCfg(
            func=mdp.illegal_contact,
            weight=-1.0,
            params={
                "sensor_name": "leg_ground_contact",
                "force_threshold": 1.0,
            },
        ),
        "self_collision": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-100.0,
            params={
                "sensor_name": "self_collision_contact",
                "force_threshold": 1.0,
            },
        ),
        "stumble": RewardTermCfg(
            func=him_mdp.stumble,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "ratio": 5.0,
            },
        ),
    }

    ##
    # Terminations
    ##
    def _build_terminations():
        terminations = {
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(
                func=mdp.bad_orientation,
                params={"limit_angle": math.radians(70.0)},
            ),
        }

        return terminations

    ##
    # Curriculum
    ##
    def _build_curriculum():
        return {
            "lpacrl": CurriculumTermCfg(
                func=him_mdp.CurriculumController,
                params={
                    "task_cfg": setup.cfg,
                    "mode": "train",
                    "expected_signature": "",
                    "fingerprints": {},
                },
            )
        }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=setup.terrain_map.entity_cfg(setup.cfg),
            sensors=(terrain_scan,),
            num_envs=setup.cfg.training.num_envs,
            extent=2.0,
        ),
        observations=observations,
        actions=_build_action(),
        commands=_build_commands(),
        events=events,
        rewards=rewards,
        terminations=_build_terminations(),
        curriculum=_build_curriculum(),
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=ri_4438_cfg.control.decimation,
        episode_length_s=setup.cfg.training.horizon_s,
        # The robot-specific config enables auto-reset with a recorder for fresh
        # terminal estimator targets. The base config retains manual-reset support.
        auto_reset=False,
    )
