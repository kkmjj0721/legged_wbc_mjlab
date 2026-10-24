"""CPU-only, explainable PD initial estimates for the migration workbench.

These are local locked-coordinate inertia estimates, not contact dynamics or a
closed-loop stability calculation. No model files or draft parameters are changed.
"""

from copy import deepcopy
import json
import math

from . import generator, models
from .template_runtime import joint_value


DEFAULTS = {
    "frequency_hz": 5.0,
    "damping_ratio": 1.0,
    "pose_delta_rad": 0.15,
    "action_amplitude": 1.0,
    "hardware_pd_hz": None,
    "hardware_delay_ms": None,
    "gain_grouping": "auto",
    "joints": {},
}
JOINT_FIELDS = {"rotor_inertia", "gear_ratio", "continuous_torque", "velocity_limit"}


def _number(value, label, low, high, *, optional=False, strict_low=False):
    if optional and value is None:
        return None
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value > high or value < low or (strict_low and value == low)):
        bracket = "(" if strict_low else "["
        raise ValueError(f"{label} 必须是 {bracket}{low}, {high}] 内的有限数值")
    return float(value)


def _settings(options, names):
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("PD 分析 options 必须是对象")
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"未知 PD 分析选项：{sorted(unknown)}")
    settings = {**deepcopy(DEFAULTS), **deepcopy(options)}
    if settings["gain_grouping"] not in ("auto", "individual", "mirror_pairs"):
        raise ValueError("gain_grouping 必须为 auto、individual 或 mirror_pairs")
    # These bounds only guard input/numerical scale; they are not hardware limits.
    bounds = {
        "frequency_hz": (0.1, 100.0),
        "damping_ratio": (0.05, 10.0),
        "pose_delta_rad": (0.0, math.pi),
        "action_amplitude": (0.0, 1000.0),
        "hardware_pd_hz": (1.0, 100000.0),
        "hardware_delay_ms": (0.0, 1000.0),
    }
    for name, (low, high) in bounds.items():
        settings[name] = _number(settings[name], name, low, high,
                                 optional=name.startswith("hardware_"))
    supplied = settings["joints"]
    if not isinstance(supplied, dict) or set(supplied) - set(names):
        raise ValueError("PD joints 必须是以已绑定关节名称为键的对象")
    settings["joints"] = {}
    for name, raw in supplied.items():
        if not isinstance(raw, dict) or set(raw) - JOINT_FIELDS:
            raise ValueError(f"{name} 的硬件参数包含未知字段或不是对象")
        values = {
            "rotor_inertia": _number(raw.get("rotor_inertia"), f"{name}.rotor_inertia", 0, 100, optional=True),
            "gear_ratio": _number(raw.get("gear_ratio"), f"{name}.gear_ratio", 0, 10000,
                                  optional=True, strict_low=True),
            "continuous_torque": _number(raw.get("continuous_torque"), f"{name}.continuous_torque", 0, 1e6,
                                         optional=True, strict_low=True),
            "velocity_limit": _number(raw.get("velocity_limit"), f"{name}.velocity_limit", 0, 1e5,
                                      optional=True, strict_low=True),
        }
        if (values["rotor_inertia"] is None) != (values["gear_ratio"] is None):
            raise ValueError(f"{name} 的转子惯量和减速比必须同时填写或同时留空")
        settings["joints"][name] = values
    return settings


def _parameters(spec, scope):
    """Use the generator's exact resolution, isolating the requested scope."""
    if not isinstance(spec, dict):
        raise ValueError("MigrationSpec 必须是对象")
    try:
        json.dumps(spec, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("配置必须是有限 JSON 数据，不能包含 NaN 或 Infinity") from exc
    tasks = spec.get("tasks")
    if not isinstance(tasks, dict) or not tasks or set(tasks) - set(generator.TEMPLATES):
        raise ValueError("请先选择至少一个受支持的任务模板")
    if not isinstance(scope, str) or (scope != "common" and scope not in tasks):
        raise ValueError("PD 分析范围必须为 common 或已选择的任务模板")
    models.resolve_model(spec)
    joints = spec.get("bindings", {}).get("joints", [])
    expected = {(leg, role) for leg in generator.LEGS for role in generator.ROLES}
    if (len(joints) != 12 or len({j["name"] for j in joints}) != 12
            or {(j["leg"], j["role"]) for j in joints} != expected
            or any(type(j.get("sign")) is not int or j["sign"] not in (-1, 1) for j in joints)):
        raise ValueError("PD 分析需要四条腿各 hip/thigh/calf 的 12 个唯一关节及 ±1 镜像符号")
    _, _, bindings, limits, model = generator._model_details(spec)
    algorithm = next(iter(tasks)) if scope == "common" else scope
    selected = deepcopy(spec)
    selected["tasks"] = {algorithm: deepcopy(tasks[algorithm])}
    if scope == "common":
        selected["tasks"][algorithm]["parameters"] = {}
    parameters = generator._parameters(selected, bindings, limits)[algorithm]
    return algorithm, bindings, model, parameters


def _budget(kp, kd, joint, settings, *, design_inertia=None):
    inertia = joint["inertia"]
    scale, initial = joint["action_scale"], joint["initial"]
    low, high = joint["action_clip"]
    amplitude = settings["action_amplitude"]
    minus = max(low, min(high, initial - amplitude * scale))
    plus = max(low, min(high, initial + amplitude * scale))
    torque_minus, torque_plus = kp * (minus - initial), kp * (plus - initial)
    peak = max(abs(torque_minus), abs(torque_plus))
    frequency = [math.sqrt(kp / inertia[key]) / (2 * math.pi) for key in ("max", "min")]
    damping = ([((kd + joint["passive_damping"]) / (2 * math.sqrt(kp * inertia[key])))
                for key in ("max", "min")] if kp > 0 else [None, None])
    result = {
        "available": True, "kp": kp, "kd": kd,
        "unit_action_torque": kp * scale,
        "unit_action_effort_ratio": kp * scale / joint["effort_limit"],
        "target_minus": minus, "target_plus": plus,
        "proportional_torque_minus": torque_minus, "proportional_torque_plus": torque_plus,
        "peak_proportional_torque": peak,
        "effort_ratio": peak / joint["effort_limit"],
        "continuous_ratio": (peak / joint["continuous_torque"] if joint["continuous_torque"] else None),
        "frequency_hz_range": frequency, "damping_ratio_range": damping,
    }
    if design_inertia is not None:
        result["design_inertia"] = design_inertia
    return result


def analyze(spec: dict, scope="common", options=None) -> dict:
    """Return JSON data and explicit parameter proposals, without applying them."""
    import mujoco
    import numpy as np

    algorithm, bindings, model, parameters = _parameters(spec, scope)
    settings = _settings(options, [joint["name"] for joint in bindings["joints"]])
    needs_symmetry = scope == "him_symmetry" or (scope == "common" and "him_symmetry" in spec["tasks"])
    grouping = settings["gain_grouping"]
    if grouping == "auto":
        grouping = "mirror_pairs" if needs_symmetry else "individual"
    control, initial = parameters["control"], parameters["init_state"]
    hip_reduction = _number(control.get("hip_reduction", 1), "control.hip_reduction", 0, 1000)
    sim_dt = _number(parameters["environment"]["sim_dt"], "environment.sim_dt", 0, 10, strict_low=True)
    for field in ("pos", "rot"):
        for index, value in enumerate(initial[field]):
            _number(value, f"init_state.{field}[{index}]", -1e6, 1e6)
    warnings = []

    def warn(code, message, joint=None, method=None):
        item = {"code": code, "message": message}
        if joint is not None:
            item["joint"] = joint
        if method is not None:
            item["method"] = method
        warnings.append(item)

    rows = []
    for binding in bindings["joints"]:
        name = binding["name"]
        dof = int(model.joint(name).dofadr[0])
        hardware = settings["joints"].get(name, {})
        armature = joint_value(control.get("armature"), binding)
        armature_source = "XML" if armature is None else "配置"
        armature = float(model.dof_armature[dof]) if armature is None else armature
        if hardware.get("rotor_inertia") is not None:
            armature = hardware["rotor_inertia"] * hardware["gear_ratio"] ** 2
            armature_source = "转子惯量 × 减速比²（替换配置/XML）"
        armature = _number(armature, f"{name}.armature", 0, 1e10)
        passive = joint_value(control.get("passive_damping"), binding)
        passive_source = "XML" if passive is None else "配置"
        passive = float(model.dof_damping[dof]) if passive is None else passive
        passive = _number(passive, f"{name}.passive_damping", 0, 1e10)
        model.dof_armature[dof] = armature
        model.dof_damping[dof] = passive
        values = {}
        for field in ("stiffness", "damping", "effort_limit", "action_scale"):
            values[field] = _number(joint_value(control[field], binding), f"{name}.{field}", 0, 1e10,
                                    strict_low=field in ("effort_limit", "action_scale"))
        angle = _number(initial["default_joint"][name], f"{name}.initial", -1e6, 1e6)
        clip = [_number(value, f"{name}.action_clip", -1e6, 1e6) for value in control["action_clip"][name]]
        row = {
            "name": name, "leg": binding["leg"], "role": binding["role"],
            "armature": armature, "armature_source": armature_source,
            "passive_damping": passive, "passive_damping_source": passive_source,
            "action_scale": values["action_scale"] * (hip_reduction if binding["role"] == "hip" else 1),
            "effort_limit": values["effort_limit"], "initial": angle, "action_clip": clip,
            "continuous_torque": hardware.get("continuous_torque"),
            "velocity_limit": hardware.get("velocity_limit"),
            "current": {"kp": values["stiffness"], "kd": values["damping"]},
        }
        rows.append(row)
        if row["continuous_torque"] and row["continuous_torque"] > row["effort_limit"]:
            warn("continuous_above_configured_limit", "填写的连续力矩大于当前仿真限矩，请核对单位与输出端定义。", name)

    nominal = {row["name"]: row["initial"] for row in rows}
    samples = [{"name": "nominal", "positions": nominal}]
    for role in generator.ROLES:
        for direction in (-1, 1):
            pose = deepcopy(nominal)
            for binding, row in zip(bindings["joints"], rows):
                if binding["role"] != role:
                    continue
                mirror = binding["sign"] if binding["leg"] in ("FR", "RR") else 1
                proposed = row["initial"] + direction * mirror * settings["pose_delta_rad"]
                pose[row["name"]] = max(row["action_clip"][0], min(row["action_clip"][1], proposed))
            samples.append({"name": f"{role}_{'minus' if direction < 0 else 'plus'}", "positions": pose})

    data = mujoco.MjData(model)
    free_id = next(i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    free_qpos = int(model.jnt_qposadr[free_id])
    data.qpos[free_qpos:free_qpos + 3] = initial["pos"]
    data.qpos[free_qpos + 3:free_qpos + 7] = initial["rot"]
    matrix = np.zeros((model.nv, model.nv), dtype=np.float64)
    sampled = {row["name"]: [] for row in rows}
    for sample in samples:
        for name, angle in sample["positions"].items():
            data.qpos[int(model.joint(name).qposadr[0])] = angle
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("模型质量矩阵出现非有限值，请检查质量、惯量与关节定义")
        for row in rows:
            dof = int(model.joint(row["name"]).dofadr[0])
            value = float(matrix[dof, dof])
            if value <= 0:
                raise ValueError(f"{row['name']} 的锁定惯量不为正，无法计算 PD 初值")
            sampled[row["name"]].append(value)

    omega = 2 * math.pi * settings["frequency_hz"]
    zeta = settings["damping_ratio"]
    for row in rows:
        values = sampled[row["name"]]
        row["inertia"] = {"min": min(values), "nominal": values[0], "max": max(values)}
        row["current"] = _budget(row["current"]["kp"], row["current"]["kd"], row, settings)
    by_leg_role = {(row["leg"], row["role"]): row for row in rows}
    if grouping == "mirror_pairs":
        warn("mirror_grouped_gains", "为共享左右增益，每对关节按较大采样惯量设计 KP，KD 取两侧所需非负值的较大者；这是工作台的工程合并规则，各侧响应仍单独估算。")
        for row in rows:
            if row["leg"] not in ("FL", "RL"):
                continue
            peer = by_leg_role[(generator.OTHER[row["leg"]], row["role"])]
            differing = [field for field in ("armature", "passive_damping", "effort_limit", "action_scale")
                         if row[field] != peer[field]]
            binding = next(j for j in bindings["joints"] if j["name"] == row["name"])
            peer_binding = next(j for j in bindings["joints"] if j["name"] == peer["name"])
            if joint_value(control.get("friction"), binding) != joint_value(control.get("friction"), peer_binding):
                differing.append("friction")
            if any(abs(a - b) > 1e-6 for a, b in zip(row["action_clip"], sorted(binding["sign"] * v for v in peer["action_clip"]))):
                differing.append("action_clip")
            if abs(row["initial"] - binding["sign"] * peer["initial"]) > 1e-6:
                differing.append("initial")
            if differing:
                warn("symmetry_control_mismatch", f"{row['name']} 与 {peer['name']} 的 {', '.join(differing)} 不满足镜像相等；这些真实参数未被修改，Symmetry 生成仍可能拒绝。", row["name"])
    elif needs_symmetry:
        warn("individual_symmetry_gains", "当前包含 Symmetry 模板但使用逐关节候选；左右增益若不完全相等，生成器会拒绝。可选择镜像成组。")
    proposals = {method: {"control": {"stiffness": {}, "damping": {}, "armature": {}}}
                 for method in ("model", "armature")}
    for row in rows:
        name = row["name"]
        group = [row]
        if grouping == "mirror_pairs":
            group.append(by_leg_role[(generator.OTHER[row["leg"]], row["role"])])
        group_names = sorted(item["name"] for item in group)
        inertia = max(item["inertia"]["max"] for item in group)
        motor_inertia = max(item["armature"] for item in group)
        motor_available = all(item["armature"] > 0 for item in group)
        kd_raw = 2 * zeta * inertia * omega - row["passive_damping"]
        kd = max(max(0.0, 2 * zeta * inertia * omega - item["passive_damping"]) for item in group)
        if kd_raw < 0:
            message = ("被动阻尼已超过设计总阻尼，候选 KD 截为 0；实际阻尼比高于设计值。" if kd == 0
                       else "被动阻尼已超过本关节的设计总阻尼；镜像组沿用另一侧所需 KD，实际阻尼比高于设计值。")
            warn("passive_damping_clamp", message, name, "model")
        row["candidates"] = {
            "model": _budget(inertia * omega**2, kd, row, settings, design_inertia=inertia),
            "armature": (_budget(motor_inertia * omega**2, 2 * zeta * motor_inertia * omega,
                                  row, settings, design_inertia=motor_inertia)
                         if motor_available else {"available": False, "reason": "本关节或镜像组的 armature 为 0，电机惯量法不可用"}),
        }
        for candidate in row["candidates"].values():
            candidate["gain_group"] = group_names
        if not row["candidates"]["armature"]["available"]:
            warn("zero_armature", "本关节或镜像组的折算电机惯量为 0；仍可使用模型法，电机惯量法须补充可靠惯量。", name, "armature")
            proposals["armature"] = None
        for method, candidate in (("current", row["current"]), *row["candidates"].items()):
            if not candidate["available"]:
                continue
            if candidate["effort_ratio"] > 1:
                warn("proportional_effort_exceeded", "默认角、零速度下指定动作幅值的比例项超过仿真限矩；这只是静态预算。", name, method)
            if candidate["continuous_ratio"] is not None and candidate["continuous_ratio"] > 1:
                warn("proportional_continuous_exceeded", "该静态比例项超过填写的连续力矩；尚未计入速度、负载与温升。", name, method)
            if method != "current" and proposals[method] is not None:
                patch = proposals[method]["control"]
                patch["stiffness"][name] = candidate["kp"]
                patch["damping"][name] = candidate["kd"]
                patch["armature"][name] = row["armature"]

    decimation = control["decimation"]
    timing = {
        "sim_dt": sim_dt, "decimation": decimation,
        "physics_hz": 1 / sim_dt, "policy_hz": 1 / (sim_dt * decimation),
        "hardware_pd_hz": settings["hardware_pd_hz"], "hardware_delay_ms": settings["hardware_delay_ms"],
        "simulation_delay_ms": [control.get(f"delay_{which}_lag", 0) * sim_dt * 1000 for which in ("min", "max")],
        "physics_samples_per_design_period": 1 / (sim_dt * settings["frequency_hz"]),
        "hardware_samples_per_design_period": (settings["hardware_pd_hz"] / settings["frequency_hz"]
                                                if settings["hardware_pd_hz"] is not None else None),
        "hardware_delay_phase_deg": (360 * settings["frequency_hz"] * settings["hardware_delay_ms"] / 1000
                                      if settings["hardware_delay_ms"] is not None else None),
    }
    if timing["physics_samples_per_design_period"] < 20:
        warn("coarse_physics_sampling", "设计频率每周期少于 20 个物理步，应做离散响应检查；20 点仅为筛查提示，不是稳定界。")
    if timing["hardware_pd_hz"] is None:
        warn("unknown_hardware_pd_rate", "尚未填写电机内部 PD 更新频率；策略或命令发送频率不能替代该值。")
    elif timing["hardware_samples_per_design_period"] < 20:
        warn("coarse_hardware_sampling", "设计频率每周期少于 20 个硬件 PD 更新点，应核查离散控制；此阈值不是稳定性证明。")
    if timing["hardware_delay_ms"] is None:
        warn("unknown_hardware_delay", "实机闭环延迟尚未提供；仿真配置延迟不是实测硬件延迟。")
    elif timing["hardware_delay_ms"] > 0:
        warn("hardware_delay", "报告给出设计频率处的纯延迟相位滞后；未计算闭环相位裕度，也不能据此保证稳定。")
    if control.get("delay_hold_prob", 0) > 0:
        warn("simulation_correlated_lag", "仿真保持概率用于保持当前 lag 数，数据仍每步更新；lag 在配置范围内随时间相关变化，不等同实机延迟。")
    warn("unvalidated_initial_gains", "结果是待验证的增益初值；未进行动态仿真、策略性能评估或实机标定。")
    warn("unknown_torque_speed_envelope", "速度上限仅作为输入记录；缺少电压、温度与力矩—速度曲线，尚未检查完整执行器包络。")

    result = {
        "version": 1, "scope": scope, "template": algorithm, "gain_grouping": grouping,
        "model": {key: spec["model"].get(key) for key in ("id", "name", "sha256")},
        "settings": settings, "timing": timing, "samples": samples,
        "joints": rows, "proposals": proposals, "warnings": warnings,
        "method_notes": [
            "当前范围：" + (f"公共参数，以 {algorithm} 模板为默认，不包含任务覆盖。" if scope == "common"
                         else f"{scope} 的真实有效参数：模板 → 公共参数 → 任务覆盖。"),
            "Mii 是锁定基座与其他自由度时的关节惯量，已含有效 armature；不是足端支撑/接触闭环的有效惯量。",
            "采样为默认站姿与 hip/thigh/calf 各自的对称 ±扰动，并按目标角限位裁剪；max 仅为这些姿态的最大值，不是全姿态上界。",
            "模型法：KP=Jmax·(2πf)²，KD=max(0, 2ζ·Jmax·2πf−被动阻尼)。电机法：KP=Iarm·(2πf)²，KD=2ζ·Iarm·2πf，不扣被动阻尼。",
            ("镜像成组：模型法取左右组内较大 Jmax，KD 取两侧需求最大值；电机法取组内较大 armature 设计共享增益。保持真实惯量/限矩不变，各侧频率与阻尼范围分别报告。这是满足共享增益契约的工程规则，不是论文中的独立整定算法。"
             if grouping == "mirror_pairs" else "逐关节设计：每个关节独立使用自身采样最大惯量和 armature。"),
            "转子惯量与减速比同时提供时，以 N²·Jrotor 替换 armature，不叠加；候选应用同时携带该 armature。",
            "频率与阻尼比范围均由完整模型的采样锁定惯量估算；电机法的负载响应通常不同于输入设计频率。",
            "控制律按 τ=clip[KP(qd−q)−KD·qdot]，目标速度与前馈力矩均为 0；qd=clip[q0+scale·action]，hip 缩放包含 hip_reduction。",
            "单位动作力矩 KP·scale 是目标角裁剪前的比例项增量；±幅值预算则先裁剪目标角，再计算两端比例力矩。",
            "力矩预算仅在 q=q0、qdot=0 比较 ±动作幅值的比例项，未计入重力负载、运动误差、阻尼力矩、摩擦或动力学随机化。动作幅值是本次检查输入，不表示策略输出恒限于 ±1。",
            "连续力矩与速度均应填写关节输出端数据；KP/KD 按 Nm/rad、Nm·s/rad。实机接口若是电流/PWM 等量纲，必须先换算。",
            "频率/阻尼目标属于设计假设。采样、相位滞后和静态力矩检查不能证明实机稳定性；应用只更新当前草稿。",
        ],
        "references": [
            {"title": "BeyondMimic：附录 S1 的 armature / 频率 / 阻尼与动作缩放启发式", "url": "https://arxiv.org/html/2508.08241v4"},
            {"title": "BeyondMimic：G1 参数参考实现", "url": "https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py"},
            {"title": "Tan 等：Sim-to-Real，电机模型、饱和和延迟", "url": "https://arxiv.org/html/1804.10332"},
            {"title": "Hwangbo 等：真实执行器数据与 actuator network", "url": "https://arxiv.org/html/1901.08652"},
            {"title": "legged_gym：动作缩放、PD 与力矩限幅实现", "url": "https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py"},
        ],
    }
    # Reject non-finite derived quantities too, rather than sending invalid JSON.
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("参数计算超出有限数值范围，请检查输入尺度") from exc
    return result
