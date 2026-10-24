"""Render complete, reviewable robot tasks without modifying the repository."""

from copy import deepcopy
import ast
import hashlib
import keyword
import importlib
import inspect
from pathlib import Path
import pprint
import re

from .templates_catalog import CONTRACT_FIELDS, LP_PRESETS, TEMPLATES, catalog
from . import motor_inertia

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGS = ("FL", "FR", "RL", "RR")
ROLES = ("hip", "thigh", "calf")
OTHER = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
TASK_SUFFIX = {"ppo": "", "him": "-HIM", "him_symmetry": "-HIM-Symmetry", "him_lpacrl": "-HIM-LPACRL"}


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _literal(value):
    return pprint.pformat(value, width=100, sort_dicts=True)


def _sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def _validate(spec):
    if spec.get("version") != 1:
        raise ValueError("MigrationSpec.version 必须为 1")
    if spec.get("confirmed") is not True:
        raise ValueError("请先确认机器人绑定与配置")
    slug = spec["robot"]["slug"]
    prefix = spec["robot"]["prefix"]
    if not re.fullmatch(r"[a-z][a-z0-9_]*", slug) or keyword.iskeyword(slug):
        raise ValueError("robot.slug 必须为小写 Python 包名（字母开头，字母、数字或下划线）")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", prefix):
        raise ValueError("robot.prefix 必须为字母开头的英文任务前缀")
    if not spec.get("tasks") or set(spec["tasks"]) - set(TEMPLATES):
        raise ValueError("请选择 PPO、HIM、HIM Symmetry 或 HIM LP-ACRL 模板")
    for key, task in spec["tasks"].items():
        terrains = task.get("terrains", [])
        if not terrains or set(terrains) - {"Flat", "Rough"} or len(set(terrains)) != len(terrains):
            raise ValueError(f"{key}.terrains 必须从 Flat / Rough 中选择且不重复")
    binding = spec["bindings"]
    joints = binding["joints"]
    if len(joints) != 12 or len({joint["name"] for joint in joints}) != 12:
        raise ValueError("需要 12 个唯一驱动关节")
    if {(joint["leg"], joint["role"]) for joint in joints} != {(leg, role) for leg in LEGS for role in ROLES}:
        raise ValueError("四条腿均须绑定 hip / thigh / calf 三个关节，且不得重复")
    if any(joint.get("sign") not in (-1, 1) for joint in joints):
        raise ValueError("各关节的镜像 sign 必须为 +1 或 -1")
    motor_inertia.validate(spec, binding)
    feet = binding["feet"]
    if len(feet) != 4 or {foot["leg"] for foot in feet} != set(LEGS):
        raise ValueError("需要 FL / FR / RL / RR 四个足端绑定")
    if len({foot["geom"] for foot in feet}) != 4:
        raise ValueError("四个足端碰撞 geom 必须唯一")
    if not binding.get("base_geoms"):
        raise ValueError("请至少绑定一个机体碰撞 geom，用于非法接触检查")
    if any(key != "ppo" for key in spec["tasks"]) and not binding.get("leg_geoms"):
        raise ValueError("HIM 模板需要腿部碰撞 geom，用于腿部接触观测")


def _model_details(spec):
    import mujoco
    import numpy as np

    root = Path(spec["model"]["root"]).resolve()
    entry = (root / spec["model"]["xml"]).resolve()
    if not entry.is_relative_to(root) or not entry.is_file():
        raise ValueError("入口 XML 必须位于模型目录内")
    model = mujoco.MjModel.from_xml_path(str(entry))
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, model.nbody), (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom), (mujoco.mjtObj.mjOBJ_SITE, model.nsite)):
        if any("/" in (mujoco.mj_id2name(model, kind, i) or "") for i in range(count)):
            raise ValueError("MjLab 将 '/' 解释为实体命名空间，机器人 body/joint/geom/site 名称不能包含 '/'")
    if sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) != 1:
        raise ValueError("模型必须包含一个浮动根关节")
    movable = [model.joint(i).name for i in range(model.njnt)
               if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if set(movable) != {joint["name"] for joint in spec["bindings"]["joints"]} or len(movable) != 12:
        raise ValueError("模型必须恰好包含绑定的 12 个单自由度关节")
    if any(model.joint(name).type[0] != mujoco.mjtJoint.mjJNT_HINGE for name in movable):
        raise ValueError("第一版仅支持 12 个旋转关节")
    bindings = deepcopy(spec["bindings"])
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bindings["base_body"]) < 0:
        raise ValueError("找不到绑定的 base_body")
    bindings["feet"] = sorted(bindings["feet"], key=lambda foot: LEGS.index(foot["leg"]))
    bindings["all_geoms"] = list(dict.fromkeys(bindings["base_geoms"] + bindings.get("leg_geoms", []) +
                                             [foot["geom"] for foot in bindings["feet"]]))
    floating_id = next(int(model.jnt_bodyid[i]) for i in range(model.njnt)
                       if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    base_ids = {floating_id}
    for i in range(floating_id + 1, model.nbody):
        if int(model.body_parentid[i]) in base_ids and model.body_jntnum[i] == 0:
            base_ids.add(i)
    if model.body(bindings["base_body"]).id not in base_ids or model.body(bindings["base_body"]).mass[0] <= 0:
        raise ValueError("机体必须绑定浮动基座或其固定连接的正质量 body")
    # These selectors feed inertia randomization. URDF fixed frame-only links
    # have zero inertia and must remain massless rather than enter that event.
    bindings["leg_bodies"] = [model.body(i).name for i in range(1, model.nbody)
                              if model.body(i).name and i not in base_ids and model.body_mass[i] > 0]
    for name in bindings["base_geoms"] + bindings.get("leg_geoms", []) + [foot["geom"] for foot in bindings["feet"]]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            raise ValueError(f"找不到碰撞 geom: {name}")
    for foot in bindings["feet"]:
        foot_type = int(model.geom(foot["geom"]).type[0])
        # Warp MULTICCD rejects positive margins for mesh/box contact pairs.
        # The RI sphere-foot preset is only applicable to other foot shapes.
        foot["margin"] = 0.0 if foot_type in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX) else 0.001
        if not foot.get("site"):
            foot["site"] = f"migration_{foot['leg'].lower()}_foot"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, foot["site"]) < 0:
            if not foot.get("body") or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot["body"]) < 0:
                raise ValueError(f"{foot['leg']} 缺少足端 site，需要有效 body 和 position")
            position = np.asarray(foot.get("position", [0, 0, 0]), dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"{foot['leg']} 足端 position 必须为三个有限数值")
    if len({foot["site"] for foot in bindings["feet"]}) != 4:
        raise ValueError("足端 site 必须唯一")
    limits = {}
    for name in movable:
        joint = model.joint(name)
        if not joint.limited[0] or not np.isfinite(joint.range).all():
            raise ValueError(f"关节 {name} 需要有限的 XML 运动范围，以建立动作裁剪")
        limits[name] = joint.range.tolist()
    return root, entry, bindings, limits, model


def _parameters(spec, bindings, limits, *, allow_unconfigured_pd=False, motor_inertia_scope=None):
    import math

    templates = {item["key"]: item for item in catalog()}
    result = {}
    for algorithm, selected in spec["tasks"].items():
        defaults = templates[algorithm]["defaults"]
        allowed = {
            "control": set(defaults["control"]) | {"action_clip"},
            "environment": set(defaults["environment"]),
            "init_state": {"pos", "rot", "default_joint"},
            "observations": set(defaults["observations"]),
        }
        for supplied in (spec.get("parameters", {}), selected.get("parameters", {})):
            unknown_groups = set(supplied) - set(defaults)
            if unknown_groups:
                raise ValueError(f"{algorithm}: 未知参数分组 {sorted(unknown_groups)}")
            for group, names in allowed.items():
                if group in supplied and set(supplied[group]) - names:
                    raise ValueError(f"{algorithm}: 未知 {group} 字段 {sorted(set(supplied[group]) - names)}")
            for group, values in supplied.get("learning", {}).items():
                if group not in defaults["learning"] or set(values) - set(defaults["learning"][group]):
                    raise ValueError(f"{algorithm}: learning.{group} 包含未支持的训练配置字段")
        p = merge(defaults, spec.get("parameters", {}))
        p = merge(p, selected.get("parameters", {}))
        from .terrain_parameters import validate as validate_terrain
        validate_terrain(p.get("terrain", {}), f"{algorithm}: terrain")
        p["_motor_inertia"] = motor_inertia.apply(
            spec, algorithm if motor_inertia_scope is None else motor_inertia_scope, bindings, p["control"])
        # Gains belong to the new robot, never to the source template.  Missing
        # fields in older drafts are treated exactly like explicit nulls.
        from .template_runtime import joint_value
        supplied_control = merge(spec.get("parameters", {}).get("control", {}),
                                 selected.get("parameters", {}).get("control", {}))
        missing_pd = []
        for field, test_reference in (("stiffness", 20.0), ("damping", 0.5)):
            supplied = supplied_control.get(field)
            values = {}
            for joint in bindings["joints"]:
                value = joint_value(supplied, joint)
                if value is None:
                    missing_pd.append({"joint": joint["name"], "field": field})
                    if allow_unconfigured_pd:
                        value = test_reference
                values[joint["name"]] = value
            p["control"][field] = values
        if missing_pd and not allow_unconfigured_pd:
            item = missing_pd[0]
            raise ValueError(f"{algorithm}: {item['joint']} 的 PD 尚未配置（{item['field']}）；"
                             "请先运行自动 PD 搜索并应用候选，或手动填写完整 PD。测试基准 20.0/0.5 不会自动写入任务。")
        if allow_unconfigured_pd:
            p["_pd_reference"] = {"baseline_used": bool(missing_pd),
                                  "reference_only": bool(missing_pd),
                                  "unconfigured": missing_pd,
                                  "label": (("测试基准 20.0 / 0.5（未应用）" if len(missing_pd) == 2 * len(bindings["joints"])
                                             else "测试对照（缺失项按 20.0 / 0.5 补全，未应用）")
                                            if missing_pd else "已配置 PD"),
                                  "kp": 20.0, "kd": 0.5}
        if algorithm == "him_lpacrl" and p.get("lpacrl", {}).get("height_scan") != defaults["lpacrl"]["height_scan"]:
            raise ValueError("LP-ACRL height_scan 为 phase47_v1 固定观测契约，不允许修改")
        p["control"]["action_clip"] = merge(limits, p["control"].get("action_clip", {}))
        source_positions = defaults["init_state"]["default_joint"]
        provided_positions = merge(spec.get("parameters", {}).get("init_state", {}).get("default_joint", {}),
                                   selected.get("parameters", {}).get("init_state", {}).get("default_joint", {}))
        p["init_state"]["default_joint"] = {
            joint["name"]: provided_positions.get(joint["name"], source_positions[f"{joint['leg']}_{joint['role']}_joint"])
            for joint in bindings["joints"]
        }
        p["learning"]["runner"]["experiment_name"] = f"{spec['robot']['slug']}_{algorithm}"
        for section in ("policy", "algorithm"):
            for name in CONTRACT_FIELDS & set(defaults["learning"][section]):
                if p["learning"][section].get(name) != defaults["learning"][section][name]:
                    raise ValueError(f"{algorithm}: {name} 为固定训练契约，不允许修改")
        control = p["control"]
        reduction = control.get("hip_reduction", 1)
        if (isinstance(reduction, bool) or not isinstance(reduction, (int, float))
                or not math.isfinite(reduction) or not 0 < reduction <= 1):
            raise ValueError(f"{algorithm}: hip_reduction 必须为 (0, 1] 内的有限数值")
        for joint in bindings["joints"]:
            name = joint["name"]
            for field in ("stiffness", "damping", "effort_limit", "action_scale", "armature", "friction", "passive_damping"):
                from .template_runtime import joint_value
                val = joint_value(control.get(field), joint)
                if val is None and field in ("armature", "friction", "passive_damping"):
                    continue
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0 or (field in ("effort_limit", "action_scale") and val == 0):
                    raise ValueError(f"{algorithm}: {name} 的 {field} 必须为有效非负数（力矩和动作缩放须大于零）")
            lo, hi = control["action_clip"][name]
            angle = p["init_state"]["default_joint"][name]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi, angle)):
                raise ValueError(f"{name} 的限位和初始角度必须为有限数值")
            if not limits[name][0] <= lo < hi <= limits[name][1] or not lo <= angle <= hi:
                raise ValueError(f"{name} 的动作裁剪或初始角度超出模型限位")
        for group in (control, p.get("observations", {})):
            low, high = group.get("delay_min_lag", 0), group.get("delay_max_lag", 0)
            period = group.get("delay_update_period", 0)
            if any(type(v) is not int or v < 0 for v in (low, high, period)) or high < low:
                raise ValueError("延迟必须为非负整数，且最大延迟不能小于最小延迟")
            if not 0 <= group.get("delay_hold_prob", 0) <= 1:
                raise ValueError("delay_hold_prob 必须位于 [0,1]")
        if type(control.get("decimation")) is not int or control["decimation"] <= 0:
            raise ValueError("decimation 必须为正整数")
        environment = p["environment"]
        from .simulation_config import validate_options
        validate_options(environment.get("mujoco", {}))
        if type(environment["num_envs"]) is not int or environment["num_envs"] <= 0:
            raise ValueError("num_envs 必须为正整数")
        if any(not isinstance(environment[key], (int, float)) or not math.isfinite(environment[key]) or environment[key] <= 0
               for key in ("env_spacing", "episode_length_s", "sim_dt")):
            raise ValueError("环境间距、回合长度和仿真步长必须为有限正数")
        for field in ("nconmax", "njmax", "contact_sensor_maxmatch"):
            if type(environment[field]) is not int or environment[field] <= 0:
                raise ValueError(f"environment.{field} 必须为正整数")
        runner = p["learning"]["runner"]
        for field in ("num_steps_per_env", "max_iterations", "save_interval"):
            if type(runner[field]) is not int or runner[field] <= 0:
                raise ValueError(f"learning.runner.{field} 必须为正整数")
        for field in ("actor_hidden_dims", "critic_hidden_dims"):
            layers = p["learning"]["policy"][field]
            if not isinstance(layers, (tuple, list)) or not layers or any(type(size) is not int or size <= 0 for size in layers):
                raise ValueError(f"learning.policy.{field} 必须为非空正整数列表")
        if len(p["init_state"]["pos"]) != 3 or not all(math.isfinite(x) for x in p["init_state"]["pos"]):
            raise ValueError("初始位置必须为三个有限数值")
        rot = p["init_state"].get("rot", [1, 0, 0, 0])
        if len(rot) != 4 or not all(math.isfinite(x) for x in rot) or abs(sum(x*x for x in rot) - 1) > 1e-5:
            raise ValueError("初始旋转必须为归一化四元数 [w,x,y,z]")
        result[algorithm] = p
    return result


def _validate_symmetry(model, bindings, p):
    import mujoco
    import numpy as np
    from .template_runtime import joint_value

    joints = {(j["leg"], j["role"]): j for j in bindings["joints"]}
    data = mujoco.MjData(model)
    for joint in bindings["joints"]:
        data.qpos[model.joint(joint["name"]).qposadr[0]] = p["init_state"]["default_joint"][joint["name"]]
    mujoco.mj_forward(model, data)
    base_id = model.body(bindings["base_body"]).id
    frame = data.xmat[base_id].reshape(3, 3)
    origin = data.xpos[base_id]
    for joint in bindings["joints"]:
        peer = joints[OTHER[joint["leg"]], joint["role"]]
        sign = joint["sign"]
        label = f"{joint['name']} ↔ {peer['name']}"
        if sign != peer["sign"]:
            raise ValueError(f"Symmetry: {label} 两端 sign 必须一致")
        if not np.isclose(p["init_state"]["default_joint"][joint["name"]],
                          sign * p["init_state"]["default_joint"][peer["name"]], atol=1e-6):
            raise ValueError(f"Symmetry: {label} 初始关节角不满足镜像 sign")
        a, b = model.joint(joint["name"]).id, model.joint(peer["name"]).id
        anchor_a = (data.xanchor[a] - origin) @ frame
        anchor_b = (data.xanchor[b] - origin) @ frame
        axis_a, axis_b = data.xaxis[a] @ frame, data.xaxis[b] @ frame
        if not np.allclose(anchor_a * [1, -1, 1], anchor_b, atol=0.003, rtol=0):
            raise ValueError(f"Symmetry: {label} 关节位置不满足机体 Y 轴镜像（容差 3 mm）")
        if not np.allclose(axis_a * [-1, 1, -1], sign * axis_b, atol=1e-4, rtol=0):
            raise ValueError(f"Symmetry: {label} 关节轴与 sign 不兼容")
        for field in ("stiffness", "damping", "effort_limit", "action_scale", "armature", "friction", "passive_damping"):
            va, vb = joint_value(p["control"].get(field), joint), joint_value(p["control"].get(field), peer)
            if va != vb:
                raise ValueError(f"Symmetry: {label} 的 {field} 不对称")
        clip_a = p["control"]["action_clip"][joint["name"]]
        clip_b = sorted(sign * x for x in p["control"]["action_clip"][peer["name"]])
        if not np.allclose(clip_a, clip_b, atol=1e-5):
            raise ValueError(f"Symmetry: {label} 的动作限位不满足镜像")
    feet = {foot["leg"]: foot for foot in bindings["feet"]}
    for leg, foot in feet.items():
        def position(item):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, item["site"])
            if site_id >= 0:
                point = data.site_xpos[site_id]
            else:
                body = model.body(item["body"]).id
                point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ np.asarray(item.get("position", [0, 0, 0]))
            return (point - origin) @ frame
        if not np.allclose(position(foot) * [1, -1, 1], position(feet[OTHER[leg]]), atol=0.003, rtol=0):
            raise ValueError(f"Symmetry: {leg} 与 {OTHER[leg]} 足端位置不满足镜像（容差 3 mm）")


def _replace(text, slug, prefix):
    replacements = {"ri_4438": slug, "Ri4438": "Robot", "RI4438": slug.upper(),
                    "ri4438": slug, "RI-4438": prefix, "Ri-4438": prefix}
    # Replace only original template tokens: a slug/prefix may contain one too.
    text = re.sub("|".join(map(re.escape, replacements)), lambda match: replacements[match[0]], text)
    return re.sub(r"from config\.", "from src.config.", text)


def _central_config(algorithm, p):
    """Materialize one selected algorithm, including its inherited defaults."""
    env_class, learning_class = {
        "ppo": ("RobotCfg", "RobotCfgPPO"), "him": ("RobotHimCfg", "RobotCFGHimPPO"),
        "him_symmetry": ("RobotHimCfg", "RobotCFGHimPPO"),
        "him_lpacrl": ("RobotHimLpacrlCfg", "RobotHimLpacrlCfgPPO"),
    }[algorithm]
    groups = {"control": "control", "init_state": "init_state", "reset": "reset",
              "randomization": "domain_rand", "environment": "env",
              "commands": "commands" if algorithm == "him_lpacrl" else "comaman",
              "observations": "observations", "rewards": "rewards",
              "terrain": "terrain", "curriculum": "curriculum"}
    original_module = importlib.import_module("src.config.ri_4438." + (
        "ri_4438_config" if algorithm == "ppo" else "ri_4438_him_lpacrl_config"
        if algorithm == "him_lpacrl" else "ri_4438_him_config"))
    original_env = getattr(original_module, TEMPLATES[algorithm][2])
    original_learning = getattr(original_module, TEMPLATES[algorithm][3])

    def restore(original, value):
        if isinstance(original, tuple) and isinstance(value, (list, tuple)):
            return tuple(restore(original[i] if i < len(original) else None, v) for i, v in enumerate(value))
        if isinstance(original, dict) and isinstance(value, dict):
            return {k: restore(original.get(k), v) for k, v in value.items()}
        return value

    overrides = {target: deepcopy(p[group]) for group, target in groups.items() if group in p}
    for group, values in p.get("lpacrl", {}).items():
        overrides[group] = merge(overrides.get(group, {}), values)
    if algorithm == "him_lpacrl" and not p["curriculum"].get("enabled", True):
        overrides["curriculum"].update(method="uniform", uniform_mix=0.0)

    def declaration(original, name, supplied, level=0, *, base=""):
        indent = "    " * level
        lines = [f"{indent}class {name}{base}:"]
        names = {key for key in dir(original) if not key.startswith("_")
                 and (inspect.isclass(getattr(original, key)) or not callable(getattr(original, key)))}
        names.update(supplied)
        for key in sorted(names):
            old = getattr(original, key, None)
            if inspect.isclass(old) or (level == 0 and key in supplied):
                lines.extend(declaration(old, key, supplied.get(key, {}), level + 1))
            else:
                value = restore(old, supplied.get(key, old))
                literal = _literal(value).splitlines()
                assignment = f"{indent}    {key} = "
                lines.append(assignment + literal[0])
                lines.extend(" " * len(assignment) + part for part in literal[1:])
        if len(lines) == 1:
            lines.append(f"{indent}    pass")
        return lines

    from src.config.base.base_config import BaseConfig
    base = "(BaseConfig)" if issubclass(original_env, BaseConfig) else ""
    lines = ['"""Editable defaults for this algorithm; tasks read these classes directly."""',
             "from src.utils.robot_config import parameters_from_config"]
    if base:
        lines.append("from src.config.base.base_config import BaseConfig")
    # LP constants are used directly by its task configuration.
    for name in sorted(vars(original_module)):
        if name.isupper():
            lines.append(f"{name} = {_literal(getattr(original_module, name))}")
    if algorithm in ("him", "him_symmetry"):
        numerics = (REPO_ROOT / "src/config/ri_4438/numerics.py").read_text()
        lines.extend(["", *numerics.splitlines()[2:]])
    lines.extend(["", *declaration(original_env, env_class, overrides, base=base), "",
                  *declaration(original_learning, learning_class, p["learning"]), "",
                  "def get_parameters():",
                  '    """Read current class values into a fresh, independent runtime snapshot."""',
                  f"    return parameters_from_config({env_class}, {learning_class})"])
    return "\n".join(lines) + "\n"


def _robot_constants(slug, prefix, bindings, model_xml, config_modules):
    """Keep robot structure in its asset package and tunables in one config."""
    def assignment(name, value):
        parts = _literal(value).splitlines()
        prefix = f"{name} = "
        return prefix + parts[0] + "\n" + "".join(
            " " * len(prefix) + part + "\n" for part in parts[1:])

    binding_source = assignment("BASE_BODY", bindings["base_body"]) + "\nJOINTS = [\n"
    for joint in bindings["joints"]:
        binding_source += "    " + pprint.pformat(joint, width=100_000, sort_dicts=False) + ",\n"
    binding_source += "]\n\nFEET = [\n"
    for foot in bindings["feet"]:
        binding_source += "\n".join("    " + line for line in _literal(foot).splitlines()) + ",\n"
    binding_source += "]\n\n"
    for key, name in (("base_geoms", "BASE_GEOMS"), ("leg_geoms", "LEG_GEOMS"),
                      ("leg_bodies", "LEG_BODIES")):
        binding_source += assignment(name, bindings.get(key, [])) + "\n"
    names = {"base_body": "BASE_BODY", "joints": "JOINTS", "feet": "FEET",
             "base_geoms": "BASE_GEOMS", "leg_geoms": "LEG_GEOMS", "leg_bodies": "LEG_BODIES"}
    binding_source += "BINDINGS = {\n"
    for key, value in bindings.items():
        if key == "all_geoms":
            expression = ('list(dict.fromkeys(BASE_GEOMS + LEG_GEOMS + '
                          '[foot["geom"] for foot in FEET]))')
        else:
            expression = names.get(key, _literal(value))
        binding_source += f"    {key!r}: {expression},\n"
    binding_source += "}\n\n"
    header = (
        '"""Robot model, explicit bindings and position-actuator factory.\n\n'
        'Tune control and initial pose in the selected algorithm configuration.\n'
        'The task passes those values here when constructing its robot.\n"""\n\n'
        'from copy import deepcopy\nfrom importlib import import_module\n'
        'from pathlib import Path\nimport re\n\n'
        'from mjlab.actuator import BuiltinPositionActuatorCfg\n'
        'from mjlab.entity import EntityArticulationInfoCfg, EntityCfg\n'
        'from mjlab.utils.spec_config import CollisionCfg\n'
        'from src.utils.robot_config import exact_names, joint_value, model_spec\n\n'
        f'ROBOT_SLUG = {slug!r}\nROBOT_PREFIX = {prefix!r}\n'
        f'CONFIG_MODULES = {_literal(config_modules)}\n\n'
        f'MODEL_XML = Path(__file__).resolve().parent / "model" / {model_xml!r}\n\n'
        f'{binding_source}'
        'def get_spec():\n'
        '    """Load the model, foot sites and IMU sensors for MjLab."""\n'
        '    return model_spec(MODEL_XML, BINDINGS)\n\n'
        f'def get_{slug}_robot_cfg(control=None, init_state=None):\n'
        '    """Construct independent actuators from this task\'s effective values."""\n'
        '    if control is None or init_state is None:\n'
        '        defaults = import_module(next(iter(CONFIG_MODULES.values()))).get_parameters()\n'
        '        control = defaults["control"] if control is None else control\n'
        '        init_state = defaults["init_state"] if init_state is None else init_state\n'
        '    control, initial = deepcopy(control), deepcopy(init_state)\n'
    )
    body = '''    actuators = []
    for joint in BINDINGS["joints"]:
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
    foot_geoms = exact_names([foot["geom"] for foot in BINDINGS["feet"]])
    foot_pattern = "(?:" + "|".join(foot_geoms) + ")"
    collision = CollisionCfg(
        geom_names_expr=exact_names(BINDINGS["all_geoms"]),
        contype=1, conaffinity=1,
        condim={foot_pattern: 3, ".*": 1},
        priority={foot_pattern: 1, ".*": 0},
        friction={foot_pattern: (0.6,)}, solref={foot_pattern: (0.02, 1.0)},
        solimp={foot_pattern: (0.9, 0.95, 0.023)},
        margin={re.escape(foot["geom"]): foot.get("margin", 0.001) for foot in BINDINGS["feet"]},
    )
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(initial["pos"]), rot=tuple(initial.get("rot", [1, 0, 0, 0])),
            joint_pos={re.escape(j["name"]): float(initial["default_joint"][j["name"]])
                       for j in BINDINGS["joints"]}, joint_vel={".*": 0.0},
        ),
        collisions=(collision,),
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9),
    )
'''
    return header + body


def _registration(slug, prefix, algorithm, terrains):
    runner_name = {"ppo": "VelocityOnPolicyRunner", "him": "HIMOnPolicyRunner",
                   "him_symmetry": "HimSymmetryRunner", "him_lpacrl": "LPACRLRunner"}[algorithm]
    runner_module = "rl.runner" if algorithm in ("him_symmetry", "him_lpacrl") else "rl"
    config_function = f"{slug}_ppo_runner_cfg" if algorithm == "ppo" else f"{slug}_him_runner_cfg"
    lines = ['"""Generated task registration; imported automatically by src.tasks."""',
             "from mjlab.tasks.registry import register_mjlab_task",
             f"from ..{runner_module} import {runner_name}",
             f"from .env_cfgs import {slug}_flat_env_cfg, {slug}_rough_env_cfg",
             f"from .rl_cfg import {config_function}"]
    task_ids = []
    if algorithm == "him_lpacrl":
        lines.extend(["from .env_cfgs import build_env_cfg", "from .curriculum_cfg import make_lpacrl_cfg"])
    for terrain in terrains:
        suffixes = [""]
        if algorithm == "him_lpacrl" and terrain == "Rough":
            suffixes += [item["suffix"] for item in LP_PRESETS if item["suffix"]]
        for suffix in suffixes:
            task_id = prefix + TASK_SUFFIX[algorithm] + "-" + terrain + ("-" + suffix if suffix else "")
            task_ids.append(task_id)
            factory = f"{slug}_{terrain.lower()}_env_cfg"
            train, play = f"{factory}()", f"{factory}(play=True)"
            if suffix:
                train = f"build_env_cfg(make_lpacrl_cfg(experiment={suffix!r}), 'train')"
                play = f"build_env_cfg(make_lpacrl_cfg(experiment={suffix!r}), 'play')"
            lines.append(f"\nregister_mjlab_task(task_id={task_id!r},\n"
                         f"    env_cfg={train}, play_env_cfg={play},\n"
                         f"    rl_cfg={config_function}(), runner_cls={runner_name})")
    return "\n".join(lines) + "\n", task_ids


def _symmetry_source(text, bindings, slug):
    joints = {(j["leg"], j["role"]): j for j in bindings["joints"]}
    joint_map = {j["name"]: (joints[OTHER[j["leg"]], j["role"]]["name"], float(j["sign"]))
                 for j in bindings["joints"]}
    feet = {f["leg"]: f["geom"] for f in bindings["feet"]}
    foot_map = {name: feet[OTHER[leg]] for leg, name in feet.items()}
    start, end = text.index("def joint_rule("), text.index("def ray_rule(")
    mapping = (f"JOINT_MIRROR = {_literal(joint_map)}\nFOOT_MIRROR = {_literal(foot_map)}\n\n"
               "def joint_rule(names):\n"
               "    names = [name.split('/')[-1] for name in names]\n"
               "    if len(names) != 12 or set(names) != set(JOINT_MIRROR):\n"
               "        raise ValueError(f'Expected the twelve explicitly bound joints, got {names}')\n"
               "    return ([names.index(JOINT_MIRROR[name][0]) for name in names],\n"
               "            [JOINT_MIRROR[name][1] for name in names])\n\n"
               "def foot_rule(names):\n"
               "    names = [name.split('/')[-1] for name in names]\n"
               "    if len(names) != 4 or set(names) != set(FOOT_MIRROR):\n"
               "        raise ValueError(f'Expected the four explicitly bound foot geoms, got {names}')\n"
               "    return [names.index(FOOT_MIRROR[name]) for name in names]\n\n\n")
    text = text[:start] + mapping + text[end:]
    start = text.index("        # Validate the actual model axes")
    end = text.index("        foot_sensor =", start)
    text = text[:start] + ("        # Joint anchors, axes, foot positions, initial pose and action limits\n"
                           "        # were checked against this explicit mapping before generation.\n") + text[end:]
    return text


def _manifest_spec(spec):
    """Keep reproducible inputs and PD summaries without duplicate trajectories."""
    saved = deepcopy(spec)
    saved.pop("workflow", None)
    for scope in saved.get("pd_analysis", {}).get("scopes", {}).values():
        report = scope.get("response_report")
        if not isinstance(report, dict):
            continue
        report.pop("videos", None)
        report.pop("cleanup", None)
        for candidate in report.get("candidates", []):
            candidate.pop("joints", None)
            candidate.pop("videos", None)
            for joint in candidate.get("tracking", {}).get("joints", []):
                joint.pop("trace", None)
            for trial in candidate.get("ground", {}).get("trials", []):
                trial.pop("trajectory", None)
        report["trajectories_omitted"] = True
    return saved


def render(spec):
    """Return stable repo-relative source/resources; never write source paths."""
    spec = deepcopy(spec)
    _validate(spec)
    root, entry, bindings, limits, model = _model_details(spec)
    from .motor_curves import compile_envelopes, image_path
    compile_envelopes(spec, [joint["name"] for joint in bindings["joints"]])
    for curve in spec.get("motor_curves", []):
        if curve.get("image"):
            image_path(curve["image"])
    from .models import check_mesh_budget, _dependencies
    check_mesh_budget(spec["model"].get("mesh_face_limit"), [
        {"name": model.mesh(i).name, "faces": int(model.mesh_facenum[i])}
        for i in range(model.nmesh)
    ])
    effective = _parameters(spec, bindings, limits)
    # A complete explicit mapping shadows template role defaults. Keep only
    # the twelve values that the robot actually consumes in its editable file.
    joint_names = [joint["name"] for joint in bindings["joints"]]
    for parameters in effective.values():
        for field in ("stiffness", "damping", "effort_limit", "armature", "friction",
                      "passive_damping", "action_scale", "action_clip"):
            values = parameters["control"].get(field)
            if isinstance(values, dict) and all(name in values for name in joint_names):
                parameters["control"][field] = {name: values[name] for name in joint_names}
    if "him_symmetry" in effective:
        _validate_symmetry(model, bindings, effective["him_symmetry"])
    slug, prefix = spec["robot"]["slug"], spec["robot"]["prefix"]
    config_root = f"src/config/{slug}"
    asset_root = f"src/assets/robots/{slug}"
    files = {}
    model_hashes, template_hashes = {}, {}
    for path in _dependencies(root, entry):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        files[f"{asset_root}/model/{relative}"] = data
        model_hashes[relative] = _sha(data)
    files[f"{asset_root}/__init__.py"] = f"from .{slug}_constants import get_{slug}_robot_cfg\n"
    config_modules = {
        algorithm: f"src.config.{slug}.{slug}_config" if algorithm == "ppo" else
                   f"src.config.{slug}.{slug}_{algorithm}_config"
        for algorithm in effective
    }
    files[f"{asset_root}/{slug}_constants.py"] = _robot_constants(
        slug, prefix, bindings, entry.relative_to(root).as_posix(), config_modules)
    runtime_path = REPO_ROOT / "src/utils/robot_config.py"
    if runtime_path.is_file():
        template_hashes["src/utils/robot_config.py"] = _sha(runtime_path.read_text())
    central = REPO_ROOT / "src/config/ri_4438"
    task_ids, output_roots = [], [asset_root, config_root]
    for algorithm, p in effective.items():
        source_name = "ri_4438_config.py" if algorithm == "ppo" else (
            "ri_4438_him_lpacrl_config.py" if algorithm == "him_lpacrl" else "ri_4438_him_config.py")
        target_name = f"{slug}_config.py" if algorithm == "ppo" else f"{slug}_{algorithm}_config.py"
        text = (central / source_name).read_text()
        template_hashes[f"src/config/ri_4438/{source_name}"] = _sha(text)
        if algorithm in ("him", "him_symmetry"):
            base_text = (central / "ri_4438_config.py").read_text()
            template_hashes["src/config/ri_4438/ri_4438_config.py"] = _sha(base_text)
            template_hashes["src/config/ri_4438/numerics.py"] = _sha((central / "numerics.py").read_text())
        files[f"{config_root}/{target_name}"] = _central_config(algorithm, p)
        source_package = f"ri_4438_{algorithm}"
        task_root = f"src/tasks/locomotion/{slug}_{algorithm}"
        output_roots.append(task_root)
        source = REPO_ROOT / "src/tasks/locomotion" / source_package
        for path in sorted(source.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            original = path.read_text()
            template_hashes[path.relative_to(REPO_ROOT).as_posix()] = _sha(original)
            relative = path.relative_to(source).as_posix().replace("ri_4438", slug)
            text = _replace(original, slug, prefix)
            text = text.replace(f"src.config.{slug}.him_layouts", "src.config.base.him_layouts")
            text = text.replace(f"src.config.{slug}.numerics", config_modules[algorithm])
            if "base_action_scale = " in text:
                text, replaced = re.subn(
                    r"^(?P<indent>[ \t]+)base_action_scale = [^\n]+\n(?P=indent)scale = \{.*?\n(?P=indent)\}",
                    lambda match: f"{match['indent']}scale = _action_scales(_get_parameters()['control'], BINDINGS)",
                    text, flags=re.S | re.M)
                if replaced != 1:
                    raise ValueError(f"{path}: 无法定位模板动作缩放构造")
                text += ("\nfrom src.utils.robot_config import action_scales as _action_scales\n"
                         f"from src.assets.robots.{slug}.{slug}_constants import BINDINGS\n"
                         f"from {config_modules[algorithm]} import get_parameters as _get_parameters\n")
            if algorithm == "him_symmetry":
                text = text.replace(f"src.config.{slug}.{slug}_him_config", f"src.config.{slug}.{slug}_him_symmetry_config")
            if relative == "config/env_cfgs.py":
                text = re.sub(r"from src.assets.robots import \(\s*get_" + re.escape(slug) + r"_robot_cfg,?\s*\)",
                              f"from src.assets.robots.{slug} import get_{slug}_robot_cfg", text)
                text += ("\nfrom src.utils.robot_config import adapt_env\n"
                         f"from src.assets.robots.{slug}.{slug}_constants import MODEL_XML, BINDINGS, get_spec\n"
                         f"from {config_modules[algorithm]} import get_parameters as _get_parameters\n\n"
                         "def _configure_env(cfg, *, play=False):\n"
                         "    return adapt_env(cfg, MODEL_XML, BINDINGS, _get_parameters(),\n"
                         f"                     play=play, spec_fn=get_spec, robot_cfg_fn=get_{slug}_robot_cfg)\n")
                if algorithm == "him_lpacrl":
                    # Original LP assumes three role-named actuators. The generated
                    # factory uses twelve explicit selectors and owns these values.
                    a = text.index("    for actuator in cfg.scene.entities[\"robot\"].articulation.actuators:")
                    b = text.index("    if not setup.terrain_map.plane:", a)
                    text = text[:a] + text[b:]
                    text = text.replace("    finalize_fingerprints(setup, cfg)",
                                        "    cfg = _configure_env(cfg, play=mode != 'train')\n"
                                        "    finalize_fingerprints(setup, cfg)")
                else:
                    for terrain in ("rough", "flat"):
                        function = f"{slug}_{terrain}_env_cfg"
                        text += (f"\n_template_{function} = {function}\n"
                                 f"def {function}(play=False):\n"
                                 f"    return _configure_env(_template_{function}(play=play), play=play)\n")
            if relative == "config/rl_cfg.py":
                function = f"{slug}_ppo_runner_cfg" if algorithm == "ppo" else f"{slug}_him_runner_cfg"
                text += ("\nfrom src.utils.robot_config import adapt_runner\n"
                         f"from {config_modules[algorithm]} import get_parameters as _get_parameters\n"
                         f"_template_{function} = {function}\n"
                         f"def {function}(*args, **kwargs):\n"
                         f"    return adapt_runner(_template_{function}(*args, **kwargs), _get_parameters())\n")
            if relative == "rl/symmetry.py" and algorithm == "him_symmetry":
                text = _symmetry_source(text, bindings, slug)
            if relative == "mdp/rewards.py" and 'asset.find_joints(cfg.params["asset_cfg"].joint_names)' in text:
                text = text.replace('asset.find_joints(cfg.params["asset_cfg"].joint_names)',
                                    'bound_joints(asset, cfg.params["asset_cfg"])')
                text += "\nfrom src.utils.robot_config import bound_joints\n"
            if "get_base_metadata" in original and relative.startswith("rl/"):
                text += ("\nfrom src.utils.robot_config import "
                         "get_migration_metadata as get_base_metadata\n")
            if relative == "config/__init__.py":
                text, ids = _registration(slug, prefix, algorithm, spec["tasks"][algorithm]["terrains"])
                task_ids.extend(ids)
            files[f"{task_root}/{relative}"] = text
    if any(key != "ppo" for key in effective):
        layout_path = REPO_ROOT / "src/config/base/him_layouts.py"
        if layout_path.is_file():
            template_hashes["src/config/base/him_layouts.py"] = _sha(layout_path.read_text())
    for path, source in files.items():
        if path.endswith(".py"):
            ast.parse(source, filename=path)
    manifest = {
        "version": 1, "robot": deepcopy(spec["robot"]), "spec": _manifest_spec(spec),
        "bindings": bindings, "effective_parameters": effective,
        "motor_inertia": {algorithm: deepcopy(p["_motor_inertia"]) for algorithm, p in effective.items()},
        "model_hashes": model_hashes, "template_hashes": template_hashes,
        "task_ids": task_ids, "output_roots": output_roots,
        "files": {path: _sha(value) for path, value in sorted(files.items())},
        "symmetry_checked": "him_symmetry" in effective,
        "output_layout": [
            {"path": asset_root, "purpose": "机器人运行模型：入口 XML、被引用的网格和资源、机器人工厂"},
            {"path": config_root, "purpose": "仅所选算法配置；各算法配置是运行参数的唯一来源",
             "config_files": [f"{config_root}/{slug}_config.py" if key == "ppo" else
                              f"{config_root}/{slug}_{key}_config.py" for key in effective]},
            *({"path": f"src/tasks/locomotion/{slug}_{key}", "algorithm": key,
               "config_file": f"{config_root}/{slug}_config.py" if key == "ppo" else f"{config_root}/{slug}_{key}_config.py",
               "purpose": f"{TEMPLATES[key][0]} 任务注册、环境、奖励与训练 runner"} for key in effective),
        ],
        "training_outputs": [{"algorithm": key, "experiment_name": p["learning"]["runner"]["experiment_name"],
                              "path_template": f"logs/rsl_rl/{p['learning']['runner']['experiment_name']}/<timestamp>[_<run_name>]/",
                              "task_ids": [task for task in task_ids if task in [
                                  prefix + TASK_SUFFIX[key] + "-" + terrain for terrain in spec["tasks"][key]["terrains"]]
                                  or (key == "him_lpacrl" and task.startswith(prefix + TASK_SUFFIX[key] + "-Rough-"))]}
                             for key, p in effective.items()],
    }
    return {"files": files, "task_ids": task_ids, "manifest": manifest}
