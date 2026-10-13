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


def make_lpacrl_cfg(*, heading_command: bool | None = None) -> LpacrlCfg:
    """只在环境/agent 工厂调用一次，将中央默认值复制为可修改的任务参数。"""
    env_cfg, ppo_cfg = Ri4438HimLpacrlCfg, Ri4438HimLpacrlCfgPPO
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
    return LpacrlCfg(
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
                for name in ("replicas", "patch_size_m", "boundary_margin_m", "seed")
            }
        ),
        curriculum=_copy_section(env_cfg.curriculum),
        evaluation=_copy_section(env_cfg.evaluation),
        scan_size_m=tuple(env_cfg.height_scan.size_m),
        scan_resolution_m=env_cfg.height_scan.resolution_m,
    )


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
    if type(cfg.curriculum.seed_bin) is not int or cfg.curriculum.seed_bin < 0:
        raise ValueError("curriculum.seed_bin must be a nonnegative integer")
    if type(cfg.task_space.sample_bin_centers) is not bool:
        raise ValueError("sample_bin_centers must be boolean")
    if cfg.evaluation.protocol_version != 3:
        raise ValueError("This evaluator requires protocol_version=3")
    floors = cfg.evaluation.relative_error_floors
    if len(floors) != 3 or any(not math.isfinite(x) or x <= 0 for x in floors):
        raise ValueError("relative_error_floors must contain three positive finite values")
    for name in ("success_alive_fraction", "success_tracking_error"):
        value = getattr(cfg.evaluation, name)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be in (0, 1)")
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
    # Direct-velocity snapshots retain their historical task fingerprint.
    command_identity = [catalog.fingerprint, cfg.task_space.resampling_interval_s]
    if getattr(cfg.task_space, "heading_command", False):
        command_identity.append({
            "heading_command": True,
            "heading_control_stiffness": cfg.task_space.heading_control_stiffness,
            "ang_vel_z_range": cfg.task_space.ang_vel_z_range,
            "heading_reference": "episode_spawn_yaw_plus_task_offset",
        })
    terrains = TerrainMap(catalog, cfg)
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
                [cfg.geometry, terrains.columns, terrains.parameters]
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
    setup.fingerprints["training_protocol"] = fingerprint(
        [
            cfg.training.horizon_s,
            cfg.training.rollout_steps,
        ]
    )
