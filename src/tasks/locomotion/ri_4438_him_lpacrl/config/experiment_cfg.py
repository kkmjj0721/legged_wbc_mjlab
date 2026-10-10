"""Resolve raw inputs once, then compile task, geometry and evaluation contracts."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, is_dataclass
import math
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from src.config.ri_4438.ri_4438_him_lpacrl_config import ExperimentCfg
from .presets import preset
from ..mdp.task_catalog import TaskCatalog, fingerprint, make_trial_table, plain
from ..mdp.terrains import TerrainMap


def merge(base: dict, override: dict, path=()):
    result = deepcopy(base)
    for key, value in override.items():
        here = path + (key,)
        replace = here == ("task_space", "terrains") or path == ("task_space", "command_axes")
        if key not in result:
            # Free-form dict keys are checked by their typed consumer/compiler.
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(result[key], dict) and not replace:
            result[key] = merge(result[key], value, here)
        else:
            result[key] = deepcopy(value)
    return result


def _convert(typ, value, path):
    if typ is Any:
        return value
    origin, args = get_origin(typ), get_args(typ)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _convert(candidate, value, path)
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
        raise ValueError(f"{path}: invalid value ({'; '.join(errors)})")
    if is_dataclass(typ):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected mapping")
        names = {f.name for f in fields(typ)}
        if set(value) - names:
            raise ValueError(f"{path}: unknown fields {set(value) - names}")
        hints = get_type_hints(typ)
        return typ(**{k: _convert(hints[k], v, f"{path}.{k}") for k, v in value.items()})
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected mapping")
        return {_convert(args[0], k, path): _convert(args[1], v, f"{path}.{k}") for k, v in value.items()}
    if origin in (list, tuple) or typ in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected sequence")
        if origin is tuple and args and args[-1] is not Ellipsis and len(value) != len(args):
            raise ValueError(f"{path}: expected {len(args)} elements")
        result = [_convert(args[i] if origin is tuple and args[-1] is not Ellipsis else args[0], v, path)
                  if args else v for i, v in enumerate(value)]
        return tuple(result) if origin is tuple or typ is tuple else result
    if typ is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if typ is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ValueError(f"{path}: expected integer")
    if typ in (str, bool, int, float, dict) and not isinstance(value, typ):
        raise ValueError(f"{path}: expected {typ.__name__}")
    return value


def from_dict(data) -> ExperimentCfg:
    return _convert(ExperimentCfg, data, "experiment")


def resolve_experiment(preset_name="ri_flat_smoke", config_path=None, overrides=()):
    import yaml
    data = asdict(preset(preset_name))
    if config_path:
        loaded = yaml.safe_load(Path(config_path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Experiment YAML must be a mapping")
        data = merge(data, loaded)
    for expression in overrides:
        if "=" not in expression:
            raise ValueError("--set requires dotted.path=value")
        key, text = expression.split("=", 1)
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown CLI path: {key}")
            target = target[part]
        # Complete axis replacement is allowed; mode changes alone are rejected.
        if parts[-1] == "mode" and parts[:2] == ["task_space", "command_axes"]:
            raise ValueError("Replace the complete command axis when changing mode")
        if parts[-1] not in target and parts[:1] != ["reward_weights"]:
            raise ValueError(f"Unknown CLI field: {key}")
        target[parts[-1]] = yaml.safe_load(text)
    return from_dict(data)


def validate(cfg):
    positive_counts = {
        "replicas": cfg.geometry.replicas, "trials_per_task": cfg.evaluation.trials_per_task,
        "interval_updates": cfg.evaluation.interval_updates, "evaluation.num_envs": cfg.evaluation.num_envs,
        "max_consecutive_failures": cfg.evaluation.max_consecutive_failures,
        "training.num_envs": cfg.training.num_envs, "rollout_steps": cfg.training.rollout_steps,
        "max_iterations": cfg.training.max_iterations, "save_interval": cfg.training.save_interval,
        "video_interval": cfg.training.video_interval, "video_length": cfg.training.video_length,
    }
    for name, val in positive_counts.items():
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, val in {"temperature": cfg.curriculum.temperature, "train_horizon": cfg.training.horizon_s,
                      "eval_horizon": cfg.evaluation.horizon_s, "scan_resolution": cfg.scan_resolution_m}.items():
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if cfg.curriculum.method not in {"lp", "alp", "uniform"}:
        raise ValueError("curriculum.method must be lp, alp or uniform")
    if not math.isfinite(cfg.curriculum.uniform_mix) or not 0 <= cfg.curriculum.uniform_mix < 1:
        raise ValueError("uniform_mix must be in [0, 1)")
    if cfg.curriculum.method == "uniform" and cfg.curriculum.uniform_mix:
        raise ValueError("Uniform requires uniform_mix=0")
    if cfg.training.resume and cfg.training.warm_start:
        raise ValueError("resume and warm_start are mutually exclusive")
    margin = cfg.geometry.boundary_margin_m
    if not math.isfinite(margin) or margin < 0 or any(not math.isfinite(s) or s <= 2 * margin for s in cfg.geometry.patch_size_m):
        raise ValueError("Patch must have a positive safe interior")
    if any(not math.isfinite(x) or x <= 0 for x in cfg.scan_size_m):
        raise ValueError("Scan sizes must be positive and finite")
    from mjlab.sensor import GridPatternCfg
    # Use the installed framework's rounding (CPU only), not a guessed formula.
    n = GridPatternCfg(size=cfg.scan_size_m, resolution=cfg.scan_resolution_m).generate_rays(None, "cpu")[0].shape[0]
    if n != 187:
        raise ValueError(f"HIM requires 187 scan rays, got {n}")
    if (cfg.robot.env.num_one_step_obs, cfg.robot.env.history_size, cfg.robot.env.num_actions) != (47, 6, 12):
        raise ValueError("HIM observation/action contract is fixed at 47 x 6 and 12")
    if cfg.robot.control.decimation <= 0:
        raise ValueError("decimation must be positive")
    if (cfg.learning.policy.num_one_step_obs, cfg.learning.policy.history_size,
        tuple(cfg.learning.policy.history_term_dims), cfg.learning.policy.history_order) != (
            47, 6, (3, 3, 3, 2, 12, 12, 12), "frame_major_oldest_first"):
        raise ValueError("HIM actor history contract cannot be changed in LPACRL v1")
    if tuple(cfg.learning.algorithm.estimator_velocity_slice) != (47, 50) or tuple(map(tuple, cfg.learning.algorithm.estimator_target_slices)) != ((0, 6), (9, 47), (47, 50)):
        raise ValueError("HIM estimator target contract cannot be changed in LPACRL v1")
    if cfg.training.rollout_steps * cfg.training.num_envs < cfg.learning.algorithm.num_mini_batches:
        raise ValueError("PPO batch is smaller than num_mini_batches")


def environment_signature(env_cfg):
    """Structural CLI changes must be recompiled before environment creation."""
    scene = env_cfg.scene
    terrain = deepcopy(scene.terrain)
    if terrain is not None:
        terrain.num_envs = 1
        terrain.env_spacing = 0.0
    return fingerprint({"terrain": terrain, "sensors": scene.sensors, "entities": scene.entities,
                        "observations": env_cfg.observations, "actions": env_cfg.actions,
                        "events": env_cfg.events, "rewards": env_cfg.rewards,
                        "terminations": env_cfg.terminations, "commands": env_cfg.commands,
                        "physics_dt": env_cfg.sim.mujoco.timestep, "scale_rewards_by_dt": env_cfg.scale_rewards_by_dt,
                        "decimation": env_cfg.decimation, "horizon": env_cfg.episode_length_s,
                        "is_finite_horizon": env_cfg.is_finite_horizon, "auto_reset": env_cfg.auto_reset})


@dataclass
class CompiledExperiment:
    cfg: ExperimentCfg
    catalog: TaskCatalog
    terrain_map: TerrainMap
    trials: list
    fingerprints: dict
    env_cfg: Any = None

    def describe(self):
        n = self.catalog.num_tasks
        m, e = self.cfg.evaluation.trials_per_task, self.cfg.evaluation.num_envs
        dt = 0.005 * self.cfg.robot.control.decimation
        steps = math.ceil(self.cfg.evaluation.horizon_s / dt)
        batches = math.ceil(n * m / e)
        train_steps = self.cfg.training.num_envs * self.cfg.training.rollout_steps * self.cfg.evaluation.interval_updates
        counts = {f: sum(t.terrain == f for t in self.catalog.records) for f in self.cfg.task_space.terrains}
        return dict(num_tasks=n, tasks=plain(self.catalog.records), tasks_per_terrain=counts,
                    geometry=self.terrain_map.summary(), fingerprints=self.fingerprints,
                    trials=self.trials, resolved_config=plain(self.cfg),
                    evaluation_budget=dict(trials=n*m, batches=batches, max_slot_steps=e*steps*batches,
                                           train_slot_steps_per_stage=train_steps,
                                           eval_to_train_slot_step_ratio=e*steps*batches/train_steps))


def compile_experiment(cfg: ExperimentCfg) -> CompiledExperiment:
    cfg = deepcopy(cfg)
    validate(cfg)
    catalog = TaskCatalog(cfg.task_space)
    terrains = TerrainMap(catalog, cfg)
    if not terrains.plane:
        offsets = cfg.robot.reset.base_offset.get("pose_range", {})
        for axis, size in zip(("x", "y"), cfg.geometry.patch_size_m):
            bounds = offsets.get(axis, (0.0, 0.0))
            if len(bounds) != 2 or max(abs(x) for x in bounds) >= size / 2 - cfg.geometry.boundary_margin_m:
                raise ValueError(f"Training spawn {axis} range is not inside the safe patch interior")
    trials = make_trial_table(catalog, cfg)
    protocol = asdict(cfg.evaluation)
    protocol.pop("num_envs")
    result = CompiledExperiment(cfg, catalog, terrains, trials, {
        "task": catalog.fingerprint,
        "geometry": fingerprint([cfg.geometry, terrains.columns, terrains.parameters]),
        "evaluation": fingerprint([protocol, trials]),
        "curriculum": fingerprint(cfg.curriculum),
        "him": fingerprint([cfg.learning, cfg.robot, cfg.scan_size_m, cfg.scan_resolution_m]),
    })
    from .env_cfgs import build_env_cfg
    result.env_cfg = build_env_cfg(result)
    result.fingerprints["reward"] = fingerprint([result.env_cfg.rewards, result.env_cfg.scale_rewards_by_dt,
                                                 result.env_cfg.sim.mujoco.timestep, result.env_cfg.decimation])
    result.fingerprints["training_protocol"] = fingerprint([cfg.training.horizon_s, cfg.training.rollout_steps])
    # Keep only raw config and hashes in the manager params (no cyclic graph).
    params = result.env_cfg.curriculum["lpacrl"].params
    params["expected_signature"] = environment_signature(result.env_cfg)
    params["fingerprints"] = dict(result.fingerprints)
    return result
