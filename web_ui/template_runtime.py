"""Self-contained runtime copied into generated robot configuration packages.

This module deliberately imports no web package. Generated tasks own this source
and their model/configuration data and can run after the workbench is removed.
"""

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import re

import mujoco
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


class MigrationJointPositionAction(JointPositionAction):
    def _find_targets(self, cfg):
        # MjLab 1.6.0's joint-transmission branch ignores preserve_order.
        # All twelve targets were validated as the robot's actuated joints.
        return self._entity.find_joints(cfg.actuator_names, preserve_order=True)


@dataclass(kw_only=True)
class MigrationJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return MigrationJointPositionAction(self, env)


def exact_names(names):
    return tuple(re.escape(name) for name in names)


def joint_value(value, joint, default=None):
    if isinstance(value, dict):
        return value.get(joint["name"], value.get(joint["role"], default))
    return default if value is None else value


def bound_joints(asset, asset_cfg):
    """Use resolved IDs; literal model names must not be parsed as regex twice."""
    ids = asset_cfg.joint_ids
    names = asset.joint_names[ids] if isinstance(ids, slice) else [asset.joint_names[i] for i in ids]
    return ids, names


def model_spec(model_xml, bindings):
    """Keep asset names, replace imported motors, and provide the IMU contract."""
    spec = mujoco.MjSpec.from_file(str(model_xml))
    # The training actuator configuration is the single motor authority. Imported
    # keyframes can have actuator-state dimensions incompatible with those motors.
    for sensor in list(spec.sensors):
        if sensor.objtype == mujoco.mjtObj.mjOBJ_ACTUATOR:
            spec.delete(sensor)
    for item in list(spec.actuators) + list(spec.keys):
        spec.delete(item)
    for foot in bindings["feet"]:
        name = foot["site"]
        if spec.site(name) is None:
            spec.body(foot["body"]).add_site(
                name=name, pos=foot.get("position", [0, 0, 0]),
                size=[0.006, 0.006, 0.006], rgba=[0.75, 0.40, 0.15, 1],
            )
    imu_name = "migration_imu"
    if spec.site(imu_name) is None:
        spec.body(bindings["base_body"]).add_site(name=imu_name, pos=[0, 0, 0])
    definitions = (
        ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE, imu_name),
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtObj.mjOBJ_SITE, imu_name),
        ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtObj.mjOBJ_SITE, imu_name),
        ("root_angmom", mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
         mujoco.mjtObj.mjOBJ_BODY, bindings["base_body"]),
    )
    for name, kind, objtype, objname in definitions:
        existing = spec.sensor(name)
        if existing is not None:
            spec.delete(existing)
        spec.add_sensor(name=name, type=kind, objtype=objtype, objname=objname)
    return spec


def create_robot_cfg(model_xml, bindings, parameters, *, spec_fn=None):
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec_config import CollisionCfg

    control = parameters["control"]
    initial = parameters["init_state"]
    actuators = []
    for joint in bindings["joints"]:
        actuators.append(BuiltinPositionActuatorCfg(
            target_names_expr=exact_names([joint["name"]]),
            stiffness=float(joint_value(control["stiffness"], joint)),
            damping=float(joint_value(control["damping"], joint)),
            effort_limit=float(joint_value(control["effort_limit"], joint)),
            armature=joint_value(control.get("armature"), joint),
            frictionloss=joint_value(control.get("friction"), joint),
            viscous_damping=joint_value(control.get("passive_damping"), joint),
            **{key: control.get(key, 0) for key in (
                "delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period")},
        ))
    foot_geoms = exact_names([foot["geom"] for foot in bindings["feet"]])
    foot_pattern = "(?:" + "|".join(foot_geoms) + ")"
    collision = CollisionCfg(
        geom_names_expr=exact_names(bindings["all_geoms"]),
        contype=1, conaffinity=1,
        condim={foot_pattern: 3, ".*": 1},
        priority={foot_pattern: 1, ".*": 0},
        friction={foot_pattern: (0.6,)}, solref={foot_pattern: (0.02, 1.0)},
        solimp={foot_pattern: (0.9, 0.95, 0.023)},
        margin={re.escape(foot["geom"]): foot.get("margin", 0.001) for foot in bindings["feet"]},
    )
    return EntityCfg(
        spec_fn=spec_fn if spec_fn is not None else lambda: model_spec(model_xml, bindings),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(initial["pos"]), rot=tuple(initial.get("rot", [1, 0, 0, 0])),
            joint_pos={re.escape(j["name"]): float(initial["default_joint"][j["name"]])
                       for j in bindings["joints"]}, joint_vel={".*": 0.0},
        ),
        collisions=(collision,),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9),
    )


def _adapt_selectors(value, bindings, seen=None):
    """Resolve template selectors against explicit bindings before managers run."""
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    if seen is None:
        seen = set()
    if id(value) in seen or value is None or isinstance(value, (str, int, float, bool, type)):
        return
    seen.add(id(value))
    if isinstance(value, SceneEntityCfg):
        if value.name != "robot":
            return
        if value.joint_names is not None:
            patterns = (value.joint_names,) if isinstance(value.joint_names, str) else value.joint_names
            selected = [j for j in bindings["joints"] if any(re.fullmatch(p, j["name"]) for p in patterns)]
            if not selected:
                roles = {role for role in ("hip", "thigh", "calf") if any(role in p for p in patterns)}
                selected = [j for j in bindings["joints"] if not roles or j["role"] in roles]
            value.joint_names = exact_names([joint["name"] for joint in selected])
            value.preserve_order = True
        if value.body_names is not None:
            names = (value.body_names,) if isinstance(value.body_names, str) else value.body_names
            if any("base_link" in name for name in names):
                value.body_names = exact_names([bindings["base_body"]])
            elif any("hip|thigh|calf" in name for name in names):
                value.body_names = exact_names(bindings["leg_bodies"])
        if value.geom_names is not None:
            value.geom_names = exact_names([foot["geom"] for foot in bindings["feet"]])
            value.preserve_order = True
        if value.site_names is not None:
            value.site_names = exact_names([foot["site"] for foot in bindings["feet"]])
            value.preserve_order = True
        return
    children = (value.values() if isinstance(value, dict) else value
                if isinstance(value, (tuple, list)) else
                (getattr(value, f.name) for f in fields(value)) if is_dataclass(value) else ())
    for child in children:
        _adapt_selectors(child, bindings, seen)


def _assign(obj, values):
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown configuration field {type(obj).__name__}.{key}")
        old = getattr(obj, key)
        if is_dataclass(old) and isinstance(value, dict):
            _assign(old, value)
        else:
            setattr(obj, key, deepcopy(tuple(value) if isinstance(old, tuple) else value))


def apply_terrain_parameters(generator, values):
    """Preserve the concrete sub-terrain dataclasses while applying UI values."""
    shared = {key: value for key, value in values.items()
              if key != "sub_terrains" and hasattr(generator, key)}
    _assign(generator, shared)
    for name, overrides in values.get("sub_terrains", {}).items():
        if name not in generator.sub_terrains:
            raise ValueError(f"Unknown sub-terrain {name!r}")
        _assign(generator.sub_terrains[name], overrides)
    if sum(sub.proportion for sub in generator.sub_terrains.values()) <= 0:
        raise ValueError("子地形权重总和必须大于 0")


def adapt_env(cfg, model_xml, bindings, parameters, *, play=False, spec_fn=None):
    """Apply effective parameters to a fresh complete task configuration."""
    control = parameters["control"]
    cfg.scene.entities = {"robot": create_robot_cfg(model_xml, bindings, parameters, spec_fn=spec_fn)}
    _adapt_selectors(cfg, bindings)
    cfg.viewer.body_name = bindings["base_body"]
    env = parameters.get("environment", {})
    cfg.scene.num_envs = int(env.get("num_envs", cfg.scene.num_envs))
    cfg.scene.env_spacing = float(env.get("env_spacing", cfg.scene.env_spacing))
    cfg.episode_length_s = float(env.get("episode_length_s", cfg.episode_length_s))
    cfg.decimation = int(control.get("decimation", cfg.decimation))
    if "sim_dt" in env:
        cfg.sim.mujoco.timestep = float(env["sim_dt"])
    for name, value in env.get("mujoco", {}).items():
        setattr(cfg.sim.mujoco, name, tuple(value) if isinstance(value, list) else value)
    for field in ("nconmax", "njmax", "contact_sensor_maxmatch"):
        if field in env:
            setattr(cfg.sim, field, int(env[field]))
    original_action = cfg.actions["joint_pos"]
    action = MigrationJointPositionActionCfg(**{
        field.name: deepcopy(getattr(original_action, field.name))
        for field in fields(original_action) if field.init
    })
    cfg.actions["joint_pos"] = action
    action.actuator_names = exact_names([joint["name"] for joint in bindings["joints"]])
    action.preserve_order = True
    action.scale = {
        re.escape(joint["name"]): float(joint_value(control["action_scale"], joint)) *
        (float(control.get("hip_reduction", 1)) if joint["role"] == "hip" else 1)
        for joint in bindings["joints"]
    }
    action.clip = {re.escape(joint["name"]): tuple(control["action_clip"][joint["name"]])
                   for joint in bindings["joints"]}
    action.use_default_offset = True
    feet = exact_names([foot["geom"] for foot in bindings["feet"]])
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            sensor.frame.name = bindings["base_body"]
        elif sensor.name in ("feet_ground_contact", "feet_support"):
            sensor.primary.pattern = feet
        elif sensor.name == "nonfoot_ground_touch":
            sensor.primary.pattern = exact_names(bindings["base_geoms"])
        elif sensor.name == "leg_ground_contact":
            sensor.primary.pattern = exact_names(bindings["leg_geoms"])
        elif sensor.name == "self_collision_contact":
            sensor.primary.pattern = exact_names(bindings["all_geoms"])
            # Secondary subtree matching is a literal MuJoCo body reference.
            sensor.secondary.pattern = bindings["base_body"]
    commands = parameters.get("commands", {})
    command = cfg.commands.get("twist")
    if command is not None and hasattr(command, "ranges"):
        for key, val in commands.items():
            if key == "ranges":
                _assign(command.ranges, {("ang_vel_z" if k == "ang_vel_yaw" else k): v
                                         for k, v in val.items()})
            elif hasattr(command, "resampling_time_range" if key == "resampling_time" else key):
                _assign(command, {"resampling_time_range" if key == "resampling_time" else key: val})
    for name, values in parameters.get("rewards", {}).items():
        if name not in cfg.rewards:
            # LP-ACRL omits this penalty for play and for training presets that
            # explicitly disable the monitor (for example A0).
            if name == "command_stall" and "lpacrl" in cfg.curriculum:
                task_cfg = cfg.curriculum["lpacrl"].params.get("task_cfg")
                monitor = getattr(task_cfg, "behavior_monitor", None)
                if play or (monitor is not None and getattr(monitor, "enabled", None) is False):
                    continue
            raise ValueError(f"Unknown reward term {name!r}")
        if "weight" in values:
            cfg.rewards[name].weight = float(values["weight"])
        cfg.rewards[name].params.update(deepcopy(values.get("params", {})))
    pose = cfg.rewards.get("pose")
    if pose is not None:
        for key in ("std_standing", "std_walking", "std_running", "terrain_tolerance"):
            if key not in pose.params:
                continue
            old = pose.params[key]
            pose.params[key] = {
                re.escape(joint["name"]): old.get(re.escape(joint["name"]), next(
                    (v for pattern, v in old.items() if joint["role"] in pattern), 0.1))
                for joint in bindings["joints"]
            }
    gait = cfg.rewards.get("foot_gait")
    if gait is not None:
        for group in cfg.observations.values():
            if "phase" in group.terms:
                for key in ("period", "command_name", "command_threshold"):
                    if key in group.terms["phase"].params:
                        group.terms["phase"].params[key] = deepcopy(gait.params[key])
        if "feet_air_time" in cfg.rewards:
            for key in ("sensor_name", "command_name", "command_threshold"):
                cfg.rewards["feet_air_time"].params[key] = deepcopy(gait.params[key])
    observation = parameters.get("observations", {})
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            if name in group.terms:
                group.terms[name].params["asset_cfg"] = SceneEntityCfg(
                    "robot", joint_names=action.actuator_names, preserve_order=True,
                )
    actor = cfg.observations.get("actor")
    if actor is not None:
        actor.enable_corruption = observation.get("enable_corruption", actor.enable_corruption)
        for name, term in actor.terms.items():
            if name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
                for attr in ("delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period"):
                    if attr in observation:
                        setattr(term, attr, observation[attr])
            if name in observation.get("noise", {}):
                from mjlab.utils.noise import UniformNoiseCfg
                term.noise = UniformNoiseCfg(**observation["noise"][name])
    terrain = cfg.scene.terrain
    if terrain is not None and terrain.terrain_generator is not None:
        # LP uses its own map schema; that schema is applied in central config.
        apply_terrain_parameters(terrain.terrain_generator, parameters.get("terrain", {}))
    if not parameters.get("curriculum", {}).get("enabled", True):
        cfg.curriculum.pop("terrain_levels", None)
        cfg.curriculum.pop("command_vel", None)
        if terrain is not None and terrain.terrain_generator is not None:
            terrain.terrain_generator.curriculum = False
    random_events = ("push_robot", "foot_friction", "encoder_bias", "base_com", "pd_gains",
                     "effort_limits", "pseudo_inertia", "synchronize_sensor_delays", "joint_friction",
                     "joint_armature", "joint_damping", "imu_errors", "external_force", "foot_contact_softness")
    if not parameters.get("randomization", {}).get("enabled", True) or play:
        for name in random_events:
            cfg.events.pop(name, None)
    if play:
        cfg.episode_length_s = int(1e9)
        actor.enable_corruption = False
        for term in actor.terms.values():
            term.delay_min_lag = term.delay_max_lag = term.delay_update_period = 0
            term.delay_hold_prob = 0.0
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            actuator.delay_min_lag = actuator.delay_max_lag = actuator.delay_update_period = 0
            actuator.delay_hold_prob = 0.0
    for reward in cfg.rewards.values():
        if getattr(reward.func, "__name__", "").startswith("hip_joint_deviation"):
            reward.params["asset_cfg"] = SceneEntityCfg(
                "robot", joint_names=exact_names([j["name"] for j in bindings["joints"] if j["role"] == "hip"]),
                preserve_order=True,
            )
    # Reward params may contain explicit selectors supplied by configuration.
    _adapt_selectors(cfg, bindings)
    return cfg


def adapt_runner(cfg, parameters):
    learning = parameters.get("learning", {})
    runner = {key: val for key, val in learning.get("runner", {}).items() if hasattr(cfg, key)}
    _assign(cfg, runner)
    for key, val in learning.get("algorithm", {}).items():
        if hasattr(cfg.algorithm, key):
            _assign(cfg.algorithm, {key: val})
    policy = learning.get("policy", {})
    for model_name, hidden_name in (("actor", "actor_hidden_dims"), ("critic", "critic_hidden_dims")):
        model = getattr(cfg, model_name)
        if hidden_name in policy:
            model.hidden_dims = tuple(policy[hidden_name])
        for name in ("activation", "obs_normalization"):
            if name in policy:
                setattr(model, name, policy[name])
    if "init_noise_std" in policy:
        cfg.actor.distribution_cfg["init_std"] = float(policy["init_noise_std"])
    return cfg


def get_migration_metadata(env, run_path):
    """Describe the action/observation order, including reordered robot joints."""
    from mjlab.rl.exporter_utils import get_base_metadata

    metadata = get_base_metadata(env, run_path)
    action = env.action_manager.get_term("joint_pos")
    names = list(action.target_names)
    natural = list(env.scene["robot"].joint_names)
    order = [natural.index(name) for name in names]
    for field in ("joint_stiffness", "joint_damping", "default_joint_pos"):
        metadata[field] = [metadata[field][index] for index in order]
    metadata["joint_names"] = names
    metadata["action_joint_names"] = names
    metadata["joint_observation_names"] = names
    metadata["migration_contract"] = "explicit_joint_order_v1"
    return metadata
