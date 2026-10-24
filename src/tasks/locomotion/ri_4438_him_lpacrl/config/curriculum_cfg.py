"""Build the LP task table and validate the contracts needed for resume.

把中央配置的嵌套 class 复制成独立的运行时参数，再建立任务表和续训校验信息。
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

from src.config.ri_4438.ri_4438_him_lpacrl_config import (
    Ri4438HimLpacrlCfg,
    Ri4438HimLpacrlCfgPPO,
)

from ..mdp.task_catalog import TaskCatalog, fingerprint, make_trial_table
from ..mdp.terrains import TerrainMap


def _copy_section(section: type) -> SimpleNamespace:
    """复制一个配置分组，包括可变字典/列表；实例之间不共享调参结果。"""
    return SimpleNamespace(
        **{
            name: deepcopy(value)
            for name, value in vars(section).items()
            if not name.startswith("_")
        }
    )


@dataclass
class LpacrlCfg:
    """任务内部的数据快照；默认值只在中央配置的两个 class 中维护。

    平时调参无需修改此类；环境工厂和评估器各自持有独立副本。
    """

    robot: SimpleNamespace
    learning: SimpleNamespace
    training: SimpleNamespace
    task_space: SimpleNamespace
    geometry: SimpleNamespace
    curriculum: SimpleNamespace
    evaluation: SimpleNamespace
    scan_size_m: tuple[float, float]
    scan_resolution_m: float
    behavior_monitor: SimpleNamespace | None = None


def make_lpacrl_cfg(*, heading_command: bool | None = None, experiment: str | None = None) -> LpacrlCfg:
    """Copy legacy defaults or select an explicit historical/Compact protocol."""
    env_cfg, ppo_cfg = Ri4438HimLpacrlCfg, Ri4438HimLpacrlCfgPPO
    compact = experiment in {"Compact", "Compact-Coverage", "Compact-Uniform", "Compact-Rough"}
    if compact:
        if heading_command is True:
            raise ValueError("Compact commands use body-frame vx/vy/wz, not heading targets")
        heading_command = False
    heading_command = env_cfg.commands.heading_command if heading_command is None else heading_command
    if type(heading_command) is not bool:
        raise ValueError("heading_command must be boolean")
    axes = ("vx", "vy", "heading" if heading_command else "wz")
    robot = SimpleNamespace(
        **{
            name: _copy_section(getattr(env_cfg, name))
            for name in ("control", "domain_rand", "init_state", "reset")
        }
    )
    robot.env = SimpleNamespace(
        **{
            name: getattr(env_cfg.env, name)
            for name in (
                "env_spacing",
                "history_size",
                "num_actions",
                "num_one_step_obs",
            )
        }
    )
    result = LpacrlCfg(
        robot=robot,
        learning=SimpleNamespace(
            policy=_copy_section(ppo_cfg.policy),
            algorithm=_copy_section(ppo_cfg.algorithm),
        ),
        training=SimpleNamespace(
            num_envs=env_cfg.env.num_envs,
            horizon_s=env_cfg.env.episode_length_s,
            rollout_steps=ppo_cfg.runner.num_steps_per_env,
            max_iterations=ppo_cfg.runner.max_iterations,
            save_interval=ppo_cfg.runner.save_interval,
            seed=ppo_cfg.runner.seed,
        ),
        task_space=SimpleNamespace(
            heading_command=heading_command,
            heading_control_stiffness=env_cfg.commands.heading_control_stiffness,
            ang_vel_z_range=tuple(env_cfg.commands.ang_vel_z_range),
            resampling_interval_s=env_cfg.commands.resampling_interval_s,
            sample_bin_centers=env_cfg.commands.sample_bin_centers,
            fixed_tasks=env_cfg.commands.fixed_tasks or not heading_command,
            random_process=env_cfg.commands.random_process and heading_command,
            process_amplitudes=tuple(env_cfg.commands.process_amplitudes),
            process_heading_range=tuple(env_cfg.commands.process_heading_range),
            process_hold_s=tuple(env_cfg.commands.process_hold_s),
            command_axes={
                axis: SimpleNamespace(
                    **deepcopy({**env_cfg.commands.axis_defaults, **env_cfg.commands.axes[axis]})
                )
                for axis in axes
            },
            terrains={
                family: SimpleNamespace(levels=deepcopy(levels))
                for family, levels in env_cfg.terrain.levels.items()
            },
            include=deepcopy(env_cfg.commands.include),
            exclude=deepcopy(env_cfg.commands.exclude),
        ),
        geometry=SimpleNamespace(
            **{
                name: deepcopy(getattr(env_cfg.terrain, name))
                for name in ("replicas", "patch_size_m", "boundary_margin_m", "seed",
                             "map_protocol", "core_rows", "surrounding_rings",
                             "outer_flat_width_m", "pass_margin_m", "pass_confirm_s")
            },
            map_levels=deepcopy(env_cfg.terrain.levels),
        ),
        curriculum=_copy_section(env_cfg.curriculum),
        evaluation=_copy_section(env_cfg.evaluation),
        scan_size_m=tuple(env_cfg.height_scan.size_m),
        scan_resolution_m=env_cfg.height_scan.resolution_m,
        behavior_monitor=_copy_section(env_cfg.behavior_monitor),
    )
    if experiment is None:
        return result
    if compact:
        _configure_compact(result, rough=experiment == "Compact-Rough")
        if experiment == "Compact-Coverage":
            result.behavior_monitor.penalty_weight = 0.0
        if experiment == "Compact-Uniform":
            result.curriculum.method, result.curriculum.uniform_mix = "uniform", 0.0
        return result
    if experiment == "Random-Uniform":
        result.curriculum.method, result.curriculum.uniform_mix = "uniform", 0.0
        return result
    if experiment not in {"A0", "A1", "A2", "A3"}:
        raise ValueError("Unknown LPACRL experiment; use A0–A3, Compact, Compact-Coverage, Compact-Uniform, Compact-Rough or Random-Uniform")
    # Preserve the named historical experiments and their resume fingerprints.
    result.task_space.fixed_tasks = True
    result.curriculum.forward_stairs_fraction = 0.25
    if experiment in {"A0", "A1"}:
        result.task_space.random_process = False
    if experiment == "A0":
        result.geometry.map_protocol = "legacy_v6"
    if experiment == "A2":
        result.curriculum.method, result.curriculum.uniform_mix = "uniform", 0.0
    return result


def _configure_compact(cfg: LpacrlCfg, *, rough: bool = False) -> None:
    definition = Ri4438HimLpacrlCfg.compact
    extension = Ri4438HimLpacrlCfg.compact_rough if rough else None
    random_ranges = extension.random_ranges if rough else definition.random_ranges
    space = cfg.task_space
    space.compact = True
    space.fixed_tasks, space.random_process = True, False
    space.resampling_interval_s = None
    space.sequence_hold_s = definition.sequence_hold_s
    space.process_hold_s = tuple(definition.process_hold_s)
    space.command_axes = {
        axis: SimpleNamespace(**deepcopy({**Ri4438HimLpacrlCfg.commands.axis_defaults,
                                         "mode": "continuous", "range": bounds}))
        for axis, bounds in zip(("vx", "vy", "wz"), random_ranges)
    }
    terrain_levels = cfg.geometry.map_levels if rough else definition.terrain_levels
    space.terrains = {
        name: SimpleNamespace(levels=deepcopy(levels))
        for name, levels in terrain_levels.items()
    }
    space.compact_tasks = []

    def add(name, family, terrain, level, mode, **parameters):
        space.compact_tasks.append(dict(name=name, family=family, terrain=terrain,
                                        level=level, mode=mode, **parameters))

    flat_terrain = extension.flat_terrain if rough else ("flat", "base")
    flat_commands = dict(definition.steady_flat)
    rough_commands = dict(definition.steady_rough)
    sequences = dict(definition.sequences)
    if rough:
        flat_commands.update(extension.steady_flat_extra)
        rough_commands.update(extension.steady_rough_extra)
        sequences.update(extension.sequences_extra)
    for name, command in flat_commands.items():
        add(name, "steady", *flat_terrain, "fixed", command=command)
    for level in space.terrains["random_rough"].levels:
        for name, command in rough_commands.items():
            add(name, "steady", "random_rough", level, "fixed", command=command)
    stairs_speeds = extension.stairs_speeds if rough else definition.stairs_speeds
    for terrain, spec in space.terrains.items():
        if terrain.startswith("stairs_"):
            for level in spec.levels:
                for speed in stairs_speeds:
                    add(f"vx_{speed:g}", "stairs", terrain, level, "fixed", command=(speed, 0.0, 0.0))
    for name, commands in sequences.items():
        add(name, "sequence", *flat_terrain, "sequence", commands=commands)
    random_terrains = (
        [(terrain, level) for terrain, spec in space.terrains.items() for level in spec.levels]
        if rough else [("flat", "base"), ("random_rough", "low")]
    )
    for terrain, level in random_terrains:
        add("body_ranges", "random", terrain, level, "random_body", ranges=random_ranges)
    if rough:
        cfg.geometry.map_protocol = "connected_v7"
    else:
        cfg.geometry.map_protocol = "legacy_v6"
        cfg.geometry.map_levels = deepcopy(definition.terrain_levels)
    cfg.curriculum.forward_stairs_fraction = 0.0
    cfg.curriculum.family_fractions = deepcopy(definition.family_fractions)
    cfg.curriculum.uniform_families = tuple(definition.uniform_families)
    cfg.evaluation.interaction_metrics_version = definition.interaction_metrics_version
    # Match the independent deployment v2 low-speed gates; retain yaw gates.
    threshold = definition.interaction_linear_threshold
    cfg.evaluation.deployment_change_thresholds = (
        threshold, threshold, cfg.evaluation.deployment_change_thresholds[2],
    )
    cfg.evaluation.deployment_stall_command = (
        threshold, cfg.evaluation.deployment_stall_command[1],
    )
    cfg.evaluation.deployment_tracking_abs = (
        definition.interaction_linear_tolerance, definition.interaction_linear_tolerance,
        cfg.evaluation.deployment_tracking_abs[2],
    )
    cfg.behavior_monitor.enabled = True


def make_compact_lpacrl_cfg() -> LpacrlCfg:
    """Build the small, explicit body-command curriculum without changing legacy defaults."""
    return make_lpacrl_cfg(experiment="Compact")


@dataclass
class CurriculumSetup:
    """确定的任务目录、几何映射和配对评估表；不持有仿真环境。"""

    cfg: LpacrlCfg
    catalog: TaskCatalog
    terrain_map: TerrainMap
    fingerprints: dict
    _trials: list | None = field(default=None, repr=False)

    @property
    def trials(self):
        """Only offline evaluation materializes the fixed trial table."""
        if self._trials is None:
            self._trials = make_trial_table(self.catalog, self.cfg)
        return self._trials


def validate(cfg):
    positive_counts = {
        "replicas": cfg.geometry.replicas,
        "trials_per_task": cfg.evaluation.trials_per_task,
        "stage_length_updates": cfg.curriculum.stage_length_updates,
        "min_episodes_per_group": cfg.curriculum.min_episodes_per_group,
        "evaluation.num_envs": cfg.evaluation.num_envs,
        "training.num_envs": cfg.training.num_envs,
        "rollout_steps": cfg.training.rollout_steps,
        "max_iterations": cfg.training.max_iterations,
        "save_interval": cfg.training.save_interval,
    }
    for name, val in positive_counts.items():
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, val in {
        "temperature": cfg.curriculum.temperature,
        "train_horizon": cfg.training.horizon_s,
        "eval_horizon": cfg.evaluation.horizon_s,
        "scan_resolution": cfg.scan_resolution_m,
    }.items():
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be finite and positive")
    interval = cfg.task_space.resampling_interval_s
    if interval is not None and (not math.isfinite(interval) or interval <= 0):
        raise ValueError("command_resampling_interval must be None or finite and positive")
    heading_command = getattr(cfg.task_space, "heading_command", False)
    if type(heading_command) is not bool:
        raise ValueError("heading_command must be boolean")
    if heading_command:
        stiffness = cfg.task_space.heading_control_stiffness
        bounds = cfg.task_space.ang_vel_z_range
        if not math.isfinite(stiffness) or stiffness <= 0:
            raise ValueError("heading_control_stiffness must be finite and positive")
        if (len(bounds) != 2 or not all(math.isfinite(x) for x in bounds)
                or not bounds[0] < 0 < bounds[1]):
            raise ValueError("ang_vel_z_range must contain finite negative/positive limits")
    if cfg.curriculum.method not in {"lp", "uniform"}:
        raise ValueError("curriculum.method must be lp or uniform")
    coverage = cfg.curriculum.forward_stairs_fraction
    if not math.isfinite(coverage) or not 0 <= coverage < 1:
        raise ValueError("forward_stairs_fraction must be in [0, 1)")
    if (not math.isfinite(cfg.curriculum.forward_stairs_velocity)
            or cfg.curriculum.forward_stairs_velocity <= 0):
        raise ValueError("forward_stairs_velocity must be finite and positive")
    heights = cfg.curriculum.forward_stairs_heights_m
    if not heights or any(not math.isfinite(h) or h <= 0 for h in heights):
        raise ValueError("forward_stairs_heights_m must contain positive finite heights")
    if type(cfg.curriculum.seed_bin) is not int or cfg.curriculum.seed_bin < 0:
        raise ValueError("curriculum.seed_bin must be a nonnegative integer")
    if type(cfg.task_space.sample_bin_centers) is not bool:
        raise ValueError("sample_bin_centers must be boolean")
    if cfg.evaluation.protocol_version != 7:
        raise ValueError("This evaluator requires protocol_version=7")
    if cfg.geometry.map_protocol not in {"connected_v7", "legacy_v6"}:
        raise ValueError("Unknown map protocol")
    if cfg.geometry.map_protocol == "connected_v7":
        if cfg.geometry.core_rows != 5 or cfg.geometry.surrounding_rings != 1:
            raise ValueError("V7 map requires five core rows and one surrounding ring")
        if cfg.geometry.replicas != 1:
            raise ValueError("Connected maps use explicit spawn lists, not replica rows")
    for name in ("outer_flat_width_m", "pass_margin_m", "pass_confirm_s"):
        if not math.isfinite(getattr(cfg.geometry, name)) or getattr(cfg.geometry, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if type(cfg.task_space.random_process) is not bool:
        raise ValueError("random_process must be boolean")
    fixed_tasks = getattr(cfg.task_space, "fixed_tasks", True)
    if type(fixed_tasks) is not bool:
        raise ValueError("fixed_tasks must be boolean")
    if not fixed_tasks and not cfg.task_space.random_process:
        raise ValueError("Enable fixed_tasks or random_process")
    if not fixed_tasks and coverage:
        raise ValueError("forward_stairs_fraction must be zero when fixed_tasks is disabled")
    if getattr(cfg.task_space, "compact", False):
        if heading_command or cfg.task_space.random_process or interval is not None:
            raise ValueError("Compact requires body commands and per-task command hold timers")
        if coverage:
            raise ValueError("Compact uses fixed family allocation without a separate coverage branch")
        fractions = cfg.curriculum.family_fractions
        if (set(fractions) != {"steady", "stairs", "sequence", "random"}
                or any(not math.isfinite(v) or v <= 0 for v in fractions.values())
                or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-12)):
            raise ValueError("Compact family fractions must be positive and sum to one")
        if not set(cfg.curriculum.uniform_families) <= set(fractions):
            raise ValueError("Unknown uniform Compact family")
        holds = cfg.task_space.process_hold_s
        if (len(holds) != 2 or not all(math.isfinite(v) for v in holds)
                or not 2.0 <= holds[0] < holds[1]):
            raise ValueError("Compact process holds require 2 <= low < high")
        if (not math.isfinite(cfg.task_space.sequence_hold_s)
                or cfg.task_space.sequence_hold_s <= 0):
            raise ValueError("Compact sequence_hold_s must be positive and finite")
    monitor = getattr(cfg, "behavior_monitor", None)
    if monitor is not None and monitor.enabled:
        if (not math.isfinite(monitor.penalty_weight) or monitor.penalty_weight > 0
                or type(monitor.termination_enabled) is not bool):
            raise ValueError("Behavior monitor requires a nonpositive penalty and boolean termination flag")
        for name in ("linear_command_threshold", "angular_command_threshold",
                     "linear_progress_floor", "angular_progress_floor", "progress_ratio",
                     "smoothing_s", "penalty_sustain_s", "termination_sustain_s",
                     "heading_change_threshold"):
            if not math.isfinite(getattr(monitor, name)) or getattr(monitor, name) <= 0:
                raise ValueError(f"behavior_monitor.{name} must be positive and finite")
        if not math.isfinite(monitor.grace_s) or monitor.grace_s < 0:
            raise ValueError("behavior_monitor.grace_s must be finite and nonnegative")
        if (len(monitor.command_change_thresholds) != 3
                or any(not math.isfinite(v) or v <= 0 for v in monitor.command_change_thresholds)):
            raise ValueError("Behavior monitor needs three positive command change thresholds")
    if cfg.task_space.random_process:
        if not heading_command or interval is not None:
            raise ValueError("Random processes require heading mode and their own hold timers")
        amplitudes = cfg.task_space.process_amplitudes
        if (len(amplitudes) != 4 or len(set(amplitudes)) != 4
                or any(not math.isfinite(a) or a <= 0 for a in amplitudes)):
            raise ValueError("process_amplitudes requires four distinct positive ranges")
        for name in ("process_hold_s", "process_heading_range"):
            bounds = getattr(cfg.task_space, name)
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid {name}")
        if cfg.task_space.process_hold_s[0] < 2.0:
            raise ValueError("Process holds must accommodate the complete 2 s switch window")
        if max(abs(x) for x in cfg.task_space.process_heading_range) > math.pi:
            raise ValueError("Process headings must be in [-pi, pi]")
    floors = cfg.evaluation.relative_error_floors
    if len(floors) != 3 or any(not math.isfinite(x) or x <= 0 for x in floors):
        raise ValueError("relative_error_floors must contain three positive finite values")
    for name in ("success_alive_fraction", "success_tracking_error"):
        value = getattr(cfg.evaluation, name)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
    for name in ("deployment_stall_s", "deployment_stall_window_s",
                 "deployment_transition_grace_s", "deployment_transition_window_s",
                 "deployment_segment_s"):
        if not math.isfinite(getattr(cfg.evaluation, name)) or getattr(cfg.evaluation, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, size in (("deployment_stall_command", 2), ("deployment_stall_speed", 2),
                       ("deployment_change_thresholds", 3), ("deployment_tracking_abs", 3),
                       ("deployment_tracking_rel", 3)):
        values = getattr(cfg.evaluation, name)
        if len(values) != size or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"{name} needs {size} positive finite values")
    if cfg.evaluation.deployment_segment_s <= (
        cfg.evaluation.deployment_transition_grace_s + cfg.evaluation.deployment_transition_window_s
    ):
        raise ValueError("Deployment segments must allow grace plus a complete tracking window")
    if (
        not math.isfinite(cfg.curriculum.uniform_mix)
        or not 0 <= cfg.curriculum.uniform_mix < 1
    ):
        raise ValueError("uniform_mix must be in [0, 1)")
    if cfg.curriculum.method == "uniform" and cfg.curriculum.uniform_mix:
        raise ValueError("Uniform requires uniform_mix=0")
    margin = cfg.geometry.boundary_margin_m
    if (
        not math.isfinite(margin)
        or margin < 0
        or any(
            not math.isfinite(s) or s <= 2 * margin for s in cfg.geometry.patch_size_m
        )
    ):
        raise ValueError("Patch must have a positive safe interior")
    if any(not math.isfinite(x) or x <= 0 for x in cfg.scan_size_m):
        raise ValueError("Scan sizes must be positive and finite")
    from mjlab.sensor import GridPatternCfg

    # Use the installed framework's rounding (CPU only), not a guessed formula.
    n = (
        GridPatternCfg(size=cfg.scan_size_m, resolution=cfg.scan_resolution_m)
        .generate_rays(None, "cpu")[0]
        .shape[0]
    )
    if n != 187:
        raise ValueError(f"HIM requires 187 scan rays, got {n}")
    if (
        cfg.robot.env.num_one_step_obs,
        cfg.robot.env.history_size,
        cfg.robot.env.num_actions,
    ) != (47, 6, 12):
        raise ValueError("HIM observation/action contract is fixed at 47 x 6 and 12")
    if cfg.robot.control.decimation <= 0:
        raise ValueError("decimation must be positive")
    if (
        cfg.learning.policy.num_one_step_obs,
        cfg.learning.policy.history_size,
        tuple(cfg.learning.policy.history_term_dims),
        cfg.learning.policy.history_order,
    ) != (47, 6, (3, 3, 3, 2, 12, 12, 12), "frame_major_oldest_first"):
        raise ValueError("HIM actor history contract cannot be changed in LPACRL v1")
    if tuple(cfg.learning.algorithm.estimator_velocity_slice) != (47, 50) or tuple(
        map(tuple, cfg.learning.algorithm.estimator_target_slices)
    ) != ((0, 6), (9, 47), (47, 50)):
        raise ValueError("HIM estimator target contract cannot be changed in LPACRL v1")
    if (
        cfg.training.rollout_steps * cfg.training.num_envs
        < cfg.learning.algorithm.num_mini_batches
    ):
        raise ValueError("PPO batch is smaller than num_mini_batches")


def environment_signature(env_cfg):
    """Structural CLI changes must be recompiled before environment creation."""
    scene = env_cfg.scene
    terrain = deepcopy(scene.terrain)
    if terrain is not None:
        terrain.num_envs = 1
        terrain.env_spacing = 0.0
    return fingerprint(
        {
            "terrain": terrain,
            "sensors": scene.sensors,
            "entities": scene.entities,
            "observations": env_cfg.observations,
            "actions": env_cfg.actions,
            "events": env_cfg.events,
            "rewards": env_cfg.rewards,
            "terminations": env_cfg.terminations,
            "commands": env_cfg.commands,
            "physics_dt": env_cfg.sim.mujoco.timestep,
            "scale_rewards_by_dt": env_cfg.scale_rewards_by_dt,
            "decimation": env_cfg.decimation,
            "horizon": env_cfg.episode_length_s,
            "is_finite_horizon": env_cfg.is_finite_horizon,
            "auto_reset": env_cfg.auto_reset,
        }
    )


def build_curriculum(cfg: LpacrlCfg) -> CurriculumSetup:
    """Compile immutable task definitions without constructing environment configs.

    Runtime managers use this lightweight validation path. They must never call
    build_env_cfg(), which itself installs the curriculum manager configuration.
    """
    cfg = deepcopy(cfg)
    validate(cfg)
    catalog = TaskCatalog(cfg.task_space)
    # Command mode and process semantics belong to the strict resume identity.
    command_identity = [catalog.fingerprint, cfg.task_space.resampling_interval_s]
    if getattr(cfg.task_space, "heading_command", False):
        command_identity.append({
            "heading_command": True,
            "heading_control_stiffness": cfg.task_space.heading_control_stiffness,
            "ang_vel_z_range": cfg.task_space.ang_vel_z_range,
            "heading_reference": "episode_spawn_yaw_plus_task_offset",
        })
    terrains = TerrainMap(catalog, cfg)
    command_identity.append({
        "protocol": cfg.geometry.map_protocol,
        "task_semantics": "birth_terrain_command_process_v7",
        "process": {k: v for k, v in vars(cfg.task_space).items() if k.startswith("process_") or k == "random_process"},
        "spawn": cfg.robot.reset.base_offset,
        "map": terrains.manifest,
        "score": "whole_episode_tracking_2to1_failure_budget_admin_mean_v7",
    })
    if catalog.compact:
        command_identity.append({
            "compact_protocol": "body_commands_family_allocation_v1",
            "sequence_hold_s": cfg.task_space.sequence_hold_s,
            "behavior_monitor": cfg.behavior_monitor,
            "interaction_metrics": {
                "version": cfg.evaluation.interaction_metrics_version,
                **{name: value for name, value in vars(cfg.evaluation).items()
                   if name.startswith("deployment_")},
            },
        })
    elif cfg.behavior_monitor is not None and cfg.behavior_monitor.enabled:
        # Explicit opt-in must invalidate resume when behavior changes, while
        # retaining the identity of all unmodified legacy presets.
        command_identity.append({"behavior_monitor": cfg.behavior_monitor})
    if not terrains.plane:
        offsets = cfg.robot.reset.base_offset.get("pose_range", {})
        for axis, size in zip(("x", "y"), cfg.geometry.patch_size_m):
            bounds = offsets.get(axis, (0.0, 0.0))
            if (
                len(bounds) != 2
                or max(abs(x) for x in bounds)
                >= size / 2 - cfg.geometry.boundary_margin_m
            ):
                raise ValueError(
                    f"Training spawn {axis} range is not inside the safe patch interior"
                )
    result = CurriculumSetup(
        cfg,
        catalog,
        terrains,
        {
            "task": fingerprint(command_identity),
            "geometry": fingerprint(
                [{k: v for k, v in vars(cfg.geometry).items() if not k.startswith("pass_")},
                 terrains.columns, terrains.parameters, terrains.manifest]
            ),
            "curriculum": fingerprint(cfg.curriculum),
            "him": fingerprint(
                [cfg.learning, cfg.robot, cfg.scan_size_m, cfg.scan_resolution_m]
            ),
        },
    )
    return result


def finalize_fingerprints(setup: CurriculumSetup, env_cfg) -> None:
    """Include the effective rewards and runner settings in checkpoint identity.

    Called after environment assembly, or after supported agent CLI overrides.
    Task/protocol changes intentionally invalidate the old LP history.
    """
    cfg = setup.cfg
    setup.fingerprints["environment"] = environment_signature(env_cfg)
    setup.fingerprints["him"] = fingerprint(
        [
            cfg.learning,
            cfg.robot,
            cfg.scan_size_m,
            cfg.scan_resolution_m,
        ]
    )
    setup.fingerprints["reward"] = fingerprint(
        [
            env_cfg.rewards,
            env_cfg.scale_rewards_by_dt,
            env_cfg.sim.mujoco.timestep,
            env_cfg.decimation,
        ]
    )
    training_protocol = [cfg.training.horizon_s, cfg.training.rollout_steps]
    if setup.catalog.compact:
        # Exact family slots depend on the effective local environment count.
        training_protocol.append({"num_envs": cfg.training.num_envs,
                                  "family_allocation": "largest_remainder_v1"})
    setup.fingerprints["training_protocol"] = fingerprint(training_protocol)
