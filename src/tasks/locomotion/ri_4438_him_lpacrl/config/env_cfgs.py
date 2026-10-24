# Adapted from RI-4438 HIM (workspace snapshot 2026-10-10).
"""Bind the RI-4438 robot and apply train/play/evaluation modes.

Task values come from the independent central config; reward definitions live
in ri_4438_him_lpacrl_env_cfg.py. This module performs environment assembly.
"""

from copy import deepcopy
from types import SimpleNamespace
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    RayCastSensorCfg,
)

from src.assets.robots import (
    get_ri_4438_robot_cfg,
)
from src.tasks.locomotion.ri_4438_him_lpacrl import mdp
from src.tasks.locomotion.ri_4438_him_lpacrl.mdp.observations import (
    configure_numerical_observations,
)
from src.tasks.locomotion.ri_4438_him_lpacrl.mdp.terminal_observations import (
    configure_estimator_observations,
)
from src.tasks.locomotion.ri_4438_him_lpacrl.ri_4438_him_lpacrl_env_cfg import (
    make_velocity_env_cfg,
)

from .curriculum_cfg import (
    LpacrlCfg,
    build_curriculum,
    environment_signature,
    finalize_fingerprints,
    make_lpacrl_cfg,
)

EnvironmentMode = Literal["train", "play", "evaluation"]


def build_env_cfg(
    task_cfg: LpacrlCfg, mode: EnvironmentMode = "train"
) -> ManagerBasedRlEnvCfg:
    """组装机器人和 LP 地形，并应用训练、播放或固定评估设置。"""
    if mode not in ("train", "play", "evaluation"):
        raise ValueError(f"Unknown environment mode: {mode!r}")
    setup = build_curriculum(task_cfg)
    cfg = make_velocity_env_cfg(setup)
    # Keep the collision and constraint buffers bounded for rough-terrain training.
    # njmax is per world; nconmax sizes the shared contact/CCD buffers.
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = 48
    cfg.sim.njmax = 300
    cfg.sim.broadphase = "nxn"
    cfg.sim.mujoco.solver = "cg"
    # The robot factory contains shared actuator configs. Own them before play
    # overrides so disabling play latency cannot affect a later training config.
    cfg.scene.entities = {"robot": deepcopy(get_ri_4438_robot_cfg())}
    # The recorder captures fresh estimator targets before the automatic reset.
    cfg.auto_reset = True

    # Set raycast sensor frame to RI-4438 base_link.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "base_link"
            sensor.include_geom_groups = (0,)

    foot_names = ("FL", "FR", "RL", "RR")
    site_names = ("FL", "FR", "RL", "RR")

    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            # Only base contact is considered an illegal ground contact.
            # Thigh/calf/hip contacts remain observable through the robot model
            # but no longer terminate the episode via this sensor.
            pattern=r"^base_collision$",
            exclude=(),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    leg_ground_cfg = ContactSensorCfg(
        name="leg_ground_contact",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            pattern=r"^(FR|FL|RR|RL)_(hip|thigh|calf).*_collision$",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=1,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r".*_collision$",
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern="base_link",
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=cfg.decimation,
        secondary_policy="error",
    )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfoot_ground_cfg,
        leg_ground_cfg,
        self_collision_cfg,
    )
    if setup.terrain_map.connected:
        cfg.scene.sensors += (ContactSensorCfg(
            name="feet_support", primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "pos", "normal"), reduce="maxforce", num_slots=1),)

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)

    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["foot_contact_softness"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    cfg.rewards["pose"].params["std_standing"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.25,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.35,  # Discourage sustained crouching.
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*(FR|FL|RR|RL)_hip_joint.*": 0.30,
        r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.4,
    }

    cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

    motion = getattr(setup.cfg, "motion_tracking", None)
    if motion is not None:
        if motion.low_speed_std > cfg.rewards["track_linear_velocity"].params["std"]:
            raise ValueError("Low-speed tracking std must not exceed the base tracking std")
        cfg.rewards["track_linear_velocity"].params.update({
            name: getattr(motion, name)
            for name in ("low_speed_std", "low_speed_threshold", "high_speed_threshold")
        })
        # Every terrain and command family shares these units and boundary
        # semantics. Keep lateral hip alignment's separate purpose unchanged.
        for name in ("foot_gait", "feet_air_time", "feet_clearance", "foot_slip",
                     "soft_landing", "stand_still"):
            if name in cfg.rewards:
                cfg.rewards[name].params.update(
                    command_threshold=motion.linear_command_threshold,
                    angular_command_threshold=motion.angular_command_threshold,
                )
        cfg.rewards["pose"].params.update(
            walking_threshold=motion.linear_command_threshold,
            angular_command_threshold=motion.angular_command_threshold,
        )

    gait_offset = [0.0, 0.5, 0.5, 0.0]

    cfg.rewards["foot_gait"].params["offset"] = list(gait_offset)

    foot_gait = cfg.rewards["foot_gait"]

    air_time_params = cfg.rewards["feet_air_time"].params
    for name in ("sensor_name", "command_name", "command_threshold", "angular_command_threshold"):
        if name not in foot_gait.params:
            continue
        air_time_params[name] = foot_gait.params[name]

    for group_name in ("actor", "critic"):
        phase_params = cfg.observations[group_name].terms["phase"].params
        phase_params["period"] = foot_gait.params["period"]
        phase_params["command_name"] = foot_gait.params["command_name"]
        phase_params["command_threshold"] = foot_gait.params["command_threshold"]
        if "angular_command_threshold" in foot_gait.params:
            phase_params["angular_command_threshold"] = foot_gait.params["angular_command_threshold"]

    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
    )

    from mjlab.managers.recorder_manager import RecorderTermCfg

    from ..mdp.curriculums import NaturalEpisodeRecorder
    from ..mdp.terminations import boundary_truncation

    cfg.seed = setup.cfg.training.seed
    cfg.scene.env_spacing = setup.cfg.robot.env.env_spacing
    robot_defaults = setup.cfg.robot
    cfg.scene.entities["robot"].init_state.pos = tuple(robot_defaults.init_state.pos)
    cfg.scene.entities["robot"].init_state.joint_pos = dict(
        robot_defaults.init_state.default_joint
    )
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        label = next(
            k
            for k in ("hip", "thigh", "calf")
            if any(k in expr for expr in actuator.target_names_expr)
        )
        for attr in ("stiffness", "damping", "effort_limit"):
            setattr(actuator, attr, getattr(robot_defaults.control, attr)[label])
        for attr in (
            "armature",
            "delay_min_lag",
            "delay_max_lag",
            "delay_hold_prob",
            "delay_update_period",
        ):
            setattr(actuator, attr, getattr(robot_defaults.control, attr))
    if not setup.terrain_map.plane:
        from ..mdp.terminations import map_boundary_truncation

        connected = setup.terrain_map.connected
        cfg.terminations["map_boundary_truncation" if connected else "boundary_truncation"] = TerminationTermCfg(
            func=map_boundary_truncation if connected else boundary_truncation, time_out=True)
    cfg.recorders["lpacrl_episode"] = RecorderTermCfg(func=NaturalEpisodeRecorder)
    if mode in ("evaluation", "play"):
        from ..mdp.domain_randomization import reset_nominal

        cfg.scene.num_envs = (
            setup.cfg.evaluation.num_envs
            if mode == "evaluation"
            else setup.cfg.training.num_envs
        )
        cfg.episode_length_s = (
            setup.cfg.evaluation.horizon_s
            if mode == "evaluation"
            else int(1e9)  # Interactive play; falls/boundaries still reset the scene.
        )
        cfg.auto_reset = mode == "play"
        cfg.recorders = {}
        cfg.events = {"reset_nominal": EventTermCfg(func=reset_nominal, mode="reset")}
        for group in cfg.observations.values():
            group.enable_corruption = False
            for term in group.terms.values():
                term.noise = None
                term.delay_min_lag = term.delay_max_lag = 0
                term.delay_hold_prob = 0.0
                term.delay_update_period = 0
        cfg.observations["actor"].terms["base_ang_vel"].func = envs_mdp.builtin_sensor
        cfg.observations["actor"].terms[
            "projected_gravity"
        ].func = envs_mdp.projected_gravity
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_min_lag = actuator.delay_max_lag = 0
            actuator.delay_hold_prob = 0.0
            actuator.delay_update_period = 0
    configure_numerical_observations(cfg)
    if mode == "train":
        from ..mdp.episode_diagnostics import observe_episode

        cfg.events["lpacrl_diagnostics"] = EventTermCfg(func=observe_episode, mode="step")
        behavior = getattr(setup.cfg, "behavior_monitor", None)
        if behavior is not None and behavior.enabled:
            from ..mdp.rewards import command_stall_cost
            from ..mdp.terminations import command_stall_failure

            cfg.rewards["command_stall"] = RewardTermCfg(
                func=command_stall_cost, weight=behavior.penalty_weight)
            if behavior.termination_enabled:
                cfg.terminations["command_stall_failure"] = TerminationTermCfg(
                    func=command_stall_failure, time_out=False)
    configure_estimator_observations(cfg, play=mode != "train")
    finalize_fingerprints(setup, cfg)
    params = cfg.curriculum["lpacrl"].params
    params["mode"] = mode
    params["fingerprints"] = dict(setup.fingerprints)
    params["expected_signature"] = environment_signature(cfg)
    return cfg


def ri_4438_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    task_cfg = make_lpacrl_cfg()
    task_cfg.task_space.terrains = {"flat": SimpleNamespace(levels={"base": {}})}
    return build_env_cfg(task_cfg, "play" if play else "train")


def ri_4438_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return build_env_cfg(make_lpacrl_cfg(), "play" if play else "train")
