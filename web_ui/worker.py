"""Isolated CPU checks and bounded GPU simulation jobs for migrated tasks."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from copy import deepcopy


def _plain(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}"
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, default=str).encode()).hexdigest()


def register_tasks(stage_root=None):
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
    import src.tasks  # noqa: F401

    before = {task: _digest((load_env_cfg(task), load_rl_cfg(task))) for task in list_tasks()}
    if stage_root:
        # Keep the original modules available and prepend only generated package roots.
        # This runs in a fresh process, so no staged module survives validation.
        stage_root = Path(stage_root)
        for name in ("src", "src.config", "config", "src.assets", "src.assets.robots",
                     "src.tasks", "src.tasks.locomotion"):
            package = importlib.import_module(name)
            relative = Path("src") / Path(*name.split(".")[1:]) if name != "config" else Path("src/config")
            candidate = stage_root / relative
            if candidate.is_dir():
                package.__path__ = [str(candidate), *package.__path__]
        for directory in sorted((stage_root / "src/tasks/locomotion").iterdir()):
            if directory.is_dir():
                importlib.import_module(f"src.tasks.locomotion.{directory.name}")
                if (directory / "config/__init__.py").is_file():
                    importlib.import_module(f"src.tasks.locomotion.{directory.name}.config")
        for task, digest in before.items():
            if _digest((load_env_cfg(task), load_rl_cfg(task))) != digest:
                raise ValueError(f"生成任务修改了原有任务的配置: {task}")
    return before


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{name} 必须是有限数值")


def _equal(actual, expected, name):
    import numpy as np

    _require(np.allclose(actual, expected, rtol=1e-6, atol=1e-8), f"最终配置 {name} 不一致: {actual!r} != {expected!r}")


def _resolve(value, name, role, default=None):
    if not isinstance(value, dict):
        return default if value is None else value
    if name in value:
        return value[name]
    if role in value:
        return value[role]
    for pattern, result in value.items():
        if re.fullmatch(pattern, name):
            return result
    return default


def _parameters_for(spec, task_id):
    if not spec:
        return None, None, None
    slug = spec["robot"]["slug"]
    module = importlib.import_module(f"src.config.{slug}.migration")
    suffix = task_id.removeprefix(spec["robot"]["prefix"])
    algorithm = ("him_lpacrl" if suffix.startswith("-HIM-LPACRL-") else
                 "him_symmetry" if suffix.startswith("-HIM-Symmetry-") else
                 "him" if suffix.startswith("-HIM-") else "ppo")
    return module.PARAMETERS[algorithm], algorithm, module.BINDINGS


def _saved_spec(robot_cfg):
    module_name = getattr(robot_cfg.spec_fn, "__module__", "")
    if not module_name.startswith("src.config.") or not module_name.endswith((".migration", ".migration_runtime")):
        return None
    module = importlib.import_module(module_name)
    manifest = Path(module.__file__).with_name("migration_manifest.json")
    return json.loads(manifest.read_text(encoding="utf-8"))["spec"] if manifest.is_file() else None


def _check_effective(cfg, model, spec, parameters, algorithm):
    import numpy as np

    if not parameters:
        return
    control = parameters.get("control", {})
    initial = parameters.get("init_state", {})
    environment = parameters.get("environment", {})
    robot = cfg.scene.entities["robot"]
    for name in ("pos", "rot"):
        if name in initial:
            _equal(getattr(robot.init_state, name), initial[name], f"init_state.{name}")
    for name, position in initial.get("default_joint", {}).items():
        _equal(_resolve(robot.init_state.joint_pos, name, ""), position, f"init_state.default_joint.{name}")
    for key, actual in (("num_envs", cfg.scene.num_envs), ("env_spacing", cfg.scene.env_spacing),
                        ("episode_length_s", cfg.episode_length_s), ("decimation", cfg.decimation)):
        if key in environment:
            _equal(actual, environment[key], f"environment.{key}")
    if "sim_dt" in environment:
        _equal(cfg.sim.mujoco.timestep, environment["sim_dt"], "environment.sim_dt")
    for key, expected in environment.get("mujoco", {}).items():
        actual = getattr(cfg.sim.mujoco, key)
        if isinstance(expected, str) or key in ("enableflags", "disableflags"):
            _require(_plain(actual) == expected, f"environment.mujoco.{key} 不一致")
        else:
            _equal(actual, expected, f"environment.mujoco.{key}")
    if "decimation" in control:
        _equal(cfg.decimation, control["decimation"], "control.decimation")
    for key in ("nconmax", "njmax", "contact_sensor_maxmatch"):
        if key in environment:
            _equal(getattr(cfg.sim, key), environment[key], f"environment.{key}")
    bindings = spec["bindings"]
    names = {item["name"]: item for item in bindings["joints"]}
    for actuator in robot.articulation.actuators:
        matched = [name for name in names if any(re.fullmatch(expr, name) for expr in actuator.target_names_expr)]
        _require(matched, "存在未绑定到机器人关节的执行器")
        for name in matched:
            role = names[name]["role"]
            for key in ("stiffness", "damping", "effort_limit", "armature"):
                expected = _resolve(control.get(key), name, role)
                if expected is not None:
                    actual = _resolve(getattr(actuator, key), name, role)
                    _equal(actual, expected, f"control.{key}.{name}")
            for key in ("delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period"):
                if key in control:
                    _equal(getattr(actuator, key), control[key], f"control.{key}.{name}")
    for item in bindings["joints"]:
        joint = model.joint(item["name"])
        for name, array in (("armature", model.dof_armature), ("friction", model.dof_frictionloss),
                            ("passive_damping", model.dof_damping)):
            expected = _resolve(control.get(name), item["name"], item["role"])
            if expected is not None:
                _equal(array[int(joint.dofadr[0])], expected, f"model.{name}.{item['name']}")
    action = cfg.actions["joint_pos"]
    _require(list(action.actuator_names) == [re.escape(name) for name in names], "动作顺序与机器人绑定关节顺序不一致")
    for name, item in names.items():
        expected = _resolve(control.get("action_clip"), name, item["role"])
        if expected is not None:
            actual = _resolve(action.clip, name, item["role"])
            _equal(actual, expected, f"action_clip.{name}")
        expected_scale = _resolve(control.get("action_scale"), name, item["role"])
        if expected_scale is not None:
            if item["role"] == "hip":
                expected_scale *= control.get("hip_reduction", 1)
            _equal(_resolve(action.scale, name, item["role"]), expected_scale, f"action_scale.{name}")
    for item in bindings["feet"]:
        _require(model.geom(item["geom"]).id >= 0, f"足端碰撞对象不存在: {item['geom']}")
        _require(model.site(item["site"]).id >= 0, f"足端 site 不存在: {item['site']}")
    _require(model.body(bindings["base_body"]).id >= 0, "机体对象不存在")
    for name in bindings.get("base_geoms", []) + bindings.get("leg_geoms", []):
        _require(model.geom(name).id >= 0, f"碰撞对象不存在: {name}")
    if algorithm == "him_symmetry":
        joints = {(item["leg"], item["role"]): item for item in bindings["joints"]}
        for left, right in (("FL", "FR"), ("RL", "RR")):
            for role in ("hip", "thigh", "calf"):
                a, b = joints[left, role], joints[right, role]
                _equal(initial["default_joint"][a["name"]],
                       initial["default_joint"][b["name"]] * a["sign"], f"mirror.{left}.{role}.default")
    _require(np.isfinite(model.qpos0).all(), "模型初始状态包含非有限数值")


def _mirror_geometry(model, bindings, initial):
    import mujoco
    import numpy as np

    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    base = model.body(bindings["base_body"]).id
    orientation = data.xmat[base].reshape(3, 3)
    origin = data.xpos[base]
    reflect = np.array([1, -1, 1])
    axial = np.array([-1, 1, -1])

    def local_position(value):
        return (value - origin) @ orientation

    joints = {(item["leg"], item["role"]): item for item in bindings["joints"]}
    feet = {item["leg"]: item for item in bindings["feet"]}
    for left, right in (("FL", "FR"), ("RL", "RR")):
        for role in ("hip", "thigh", "calf"):
            a, b = joints[left, role], joints[right, role]
            _require(a["sign"] == b["sign"], f"{left}/{right} {role} 的镜像符号不一致")
            _equal(initial["default_joint"][a["name"]], initial["default_joint"][b["name"]] * a["sign"],
                   f"mirror.{left}.{role}.default")
            ja, jb = model.joint(a["name"]).id, model.joint(b["name"]).id
            delta = local_position(data.xanchor[ja]) * reflect - local_position(data.xanchor[jb])
            _require(np.max(np.abs(delta)) <= 0.005,
                     f"{left}/{right} {role} 关节位置不满足左右镜像，偏差 {np.max(np.abs(delta)):.5f} m")
            axis_a, axis_b = data.xaxis[ja] @ orientation, data.xaxis[jb] @ orientation
            _require(np.allclose(axis_a * axial * a["sign"], axis_b, rtol=0, atol=1e-4),
                     f"{left}/{right} {role} 关节轴与镜像符号不兼容")
        a, b = model.site(feet[left]["site"]).id, model.site(feet[right]["site"]).id
        delta = local_position(data.site_xpos[a]) * reflect - local_position(data.site_xpos[b])
        _require(np.max(np.abs(delta)) <= 0.005,
                 f"{left}/{right} 足端站姿不满足左右镜像，偏差 {np.max(np.abs(delta)):.5f} m")


def precheck_spec(spec):
    from .models import resolve_model, validate_spec
    from .generator import _model_details, _parameters, _registration
    from .template_runtime import create_robot_cfg
    from mjlab.tasks.registry import list_tasks
    import src.tasks  # noqa: F401

    issues = validate_spec(spec)
    errors = [item["message"] for item in issues if item.get("level") == "error"]
    _require(not errors, "；".join(errors))
    existing = set(list_tasks())
    for algorithm, selected in spec["tasks"].items():
        _, task_ids = _registration(spec["robot"]["slug"], spec["robot"]["prefix"], algorithm, selected["terrains"])
        conflicts = sorted(existing.intersection(task_ids))
        _require(not conflicts, "任务 ID 已存在，拒绝覆盖: " + ", ".join(conflicts))
    task = spec.get("tasks", {}).get("him_symmetry")
    if task:
        _, _, bindings, limits, _ = _model_details(spec)
        parameters = _parameters(spec, bindings, limits)["him_symmetry"]
        entity = create_robot_cfg(resolve_model(spec), bindings, parameters).build()
        model = entity.compile()
        _mirror_geometry(model, bindings, parameters["init_state"])
    return dict(status="passed", mode="precheck", issues=issues)


def _compile_sensor_scene(cfg):
    """Compile complete robot/sensor bindings without allocating Warp state.

    Rough terrain uses one plane with the same terrain object names. The real
    procedural map is covered by the explicitly requested simulation check.
    """
    from mjlab.scene import Scene

    scene_cfg = deepcopy(cfg.scene)
    scene_cfg.num_envs = 1
    proxy = scene_cfg.terrain is not None and scene_cfg.terrain.terrain_type == "generator"
    if proxy:
        scene_cfg.terrain.terrain_type = "plane"
        scene_cfg.terrain.terrain_generator = None
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    return dict(sensors=model.nsensor, sensor_channels=model.nsensordata,
                terrain="plane proxy" if proxy else "configured", device="cpu")


def cpu_check(task_ids, spec, stage_root=None):
    import mujoco
    import numpy as np
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    before = register_tasks(stage_root)
    result = []
    for task in task_ids:
        print(f"[CPU] 检查 {task}", flush=True)
        _require(task in list_tasks(), f"任务注册缺失: {task}")
        cfg, rl = load_env_cfg(task), load_rl_cfg(task)
        second = load_env_cfg(task)
        original_count = second.scene.num_envs
        cfg.scene.num_envs += 1
        _require(second.scene.num_envs == original_count and load_env_cfg(task).scene.num_envs == original_count,
                 f"{task} 的配置实例未隔离")
        cfg.scene.num_envs = original_count
        robot_cfg = cfg.scene.entities["robot"]
        task_spec = spec or _saved_spec(robot_cfg)
        if robot_cfg.articulation and robot_cfg.articulation.actuators:
            actuator = robot_cfg.articulation.actuators[0]
            old = second.scene.entities["robot"].articulation.actuators[0].delay_max_lag
            actuator.delay_max_lag += 1
            _require(second.scene.entities["robot"].articulation.actuators[0].delay_max_lag == old,
                     f"{task} 的执行器配置实例未隔离")
            actuator.delay_max_lag -= 1
        _require(cfg.decimation > 0 and cfg.sim.mujoco.timestep > 0, "控制时序必须为正数")
        _require(cfg.scene.num_envs > 0 and cfg.episode_length_s > 0, "环境数和回合长度必须为正数")
        for name, reward in cfg.rewards.items():
            _number(reward.weight, f"reward.{name}.weight")
        entity = robot_cfg.build()
        model = entity.compile()
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        _require(np.isfinite(data.qpos).all() and np.isfinite(data.xpos).all(), "前向运动学出现非有限数值")
        _require(len(entity.joint_names) == 12, f"当前工具仅支持 12 个关节，得到 {len(entity.joint_names)}")
        _require(model.nu == 12, f"模型应有 12 个位置执行器，得到 {model.nu}")
        parameters, algorithm, bindings = _parameters_for(task_spec, task)
        if task_spec:
            task_spec = dict(task_spec, bindings=bindings)
        _check_effective(cfg, model, task_spec, parameters, algorithm)
        sensor_scene = _compile_sensor_scene(cfg)
        if algorithm == "him_symmetry":
            _mirror_geometry(model, bindings, parameters["init_state"])
        is_him = "estimator" in cfg.observations
        if is_him:
            slug = task_spec["robot"]["slug"] if task_spec else None
            layout_module = importlib.import_module(f"src.config.{slug}.him_layouts" if slug else "src.config.ri_4438.him_layouts")
            layout = layout_module.PHASE47
            for group in ("actor", "critic", "estimator"):
                _require(tuple(cfg.observations[group].terms) == layout.terms(group), f"{group} 的 phase47_v1 观测顺序不匹配")
            actor = cfg.observations["actor"]
            _require(actor.history_length == layout.history_size and not actor.flatten_history_dim,
                     "HIM 历史必须为显式 [B,6,47]")
            _require(rl.actor.num_one_step_obs == layout.frame_size and rl.actor.history_size == layout.history_size,
                     "HIM actor 维度与历史契约不一致")
            _require(tuple(rl.actor.history_term_dims) == layout.term_dims and rl.actor.history_order == "frame_major_oldest_first",
                     "HIM actor 历史顺序或各项维度不一致")
            estimator = cfg.observations["estimator"]
            _require(not estimator.enable_corruption and not estimator.history_length, "estimator 监督不能包含历史或噪声")
            if algorithm == "him_symmetry":
                module = importlib.import_module(f"src.tasks.locomotion.{slug}_him_symmetry.rl.layouts")
                module.validate_env_cfg(cfg, layout)
        result.append(dict(task_id=task, joints=len(entity.joint_names), actuators=model.nu,
                           num_envs=cfg.scene.num_envs, decimation=cfg.decimation,
                           experiment_name=rl.experiment_name, layout="phase47_v1" if is_him else "ppo",
                           sensor_scene=sensor_scene))
        print(f"[CPU] 通过 {task}: 12 joints / 12 actuators", flush=True)
    return dict(status="passed", mode="cpu", tasks=result, original_task_count=len(before),
                checks=["任务注册", "原任务配置摘要", "配置实例隔离", "模型编译与前向运动学",
                        "最终控制与初始状态", "对象绑定与动作顺序", "HIM 布局与镜像站姿", "完整传感器场景编译"],
                note="CPU 场景检查将程序地形替换为同名平面；实际地形生成与动态运行由短仿真检查")


def simulation_check(task_ids):
    import torch
    from mjlab.tasks.registry import load_env_cfg
    from src.envs import ManagerBasedRlEnv

    register_tasks()
    if not torch.cuda.is_available():
        print("GPU 不可用，未运行短仿真", flush=True)
        return dict(status="skipped", mode="simulation", reason="GPU 不可用，未运行", seed=42, num_envs=4, steps=200)
    results = []
    for task in task_ids:
        cfg = load_env_cfg(task)
        cfg.scene.num_envs, cfg.seed = 4, 42
        started = time.monotonic()
        env = None
        try:
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
            observations, _ = env.reset(seed=42)
            action = torch.zeros((4, env.action_manager.total_action_dim), device=env.device)
            resets = 0
            with torch.inference_mode():
                for step in range(200):
                    observations, reward, terminated, truncated, _ = env.step(action)
                    _require(torch.isfinite(reward).all().item(), "仿真奖励出现非有限数值")
                    for name, value in observations.items():
                        _require(torch.isfinite(value).all().item(), f"观测 {name} 出现非有限数值")
                    resets += int((terminated | truncated).sum().item())
                    if (step + 1) % 25 == 0:
                        print(f"[GPU] {task}: {step + 1}/200 steps, resets={resets}", flush=True)
            results.append(dict(task_id=task, device=torch.cuda.get_device_name(0), seed=42,
                                num_envs=4, steps=200, resets=resets, elapsed_s=time.monotonic() - started))
        finally:
            if env is not None:
                env.close()
    return dict(status="passed", mode="simulation", tasks=results,
                note="固定零动作短仿真只验证运行链路与有限性，不评价策略性能")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--mode", choices=("cpu", "simulation", "precheck"), required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root / "src"))
    started = time.monotonic()
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else None
        if args.mode == "cpu":
            result = cpu_check(args.task_id, spec, args.stage_root)
        elif args.mode == "precheck":
            result = precheck_spec(spec)
        else:
            result = simulation_check(args.task_id)
        code = 0
    except Exception as exc:
        traceback.print_exc()
        result = dict(status="failed", error=str(exc), mode=args.mode)
        code = 1
    result["elapsed_s"] = time.monotonic() - started
    args.result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
