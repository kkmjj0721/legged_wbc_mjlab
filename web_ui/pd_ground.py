"""Whole-robot, free-base PD standing and disturbance screening on a flat floor.

The editable model and draft are never changed.  Every candidate owns an MjModel
and every trial starts from the same grounded initial pose.  This is a bounded
simulation screen, not hardware calibration or proof of policy performance.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy, deepcopy
import math
import os
import time

SEED = 42
MAX_WORKERS = 4
MAX_TRIAL_STEPS = 100_000
MAX_TOTAL_STEPS = 2_000_000
TRACE_HZ = 30
TRIALS = (
    {"name": "standing", "duration_s": 10.0, "axis": None},
    {"name": "push_x", "duration_s": 5.2, "axis": 0},
    {"name": "push_y", "duration_s": 5.2, "axis": 1},
)
# These are explicit screening thresholds, not identified robot properties.
PASS_CRITERIA = {
    "max_tilt_deg": 30.0,
    "min_base_height_fraction": 0.5,
    "max_base_drop_fraction": 0.45,
    "max_drift_m": 0.30,
    "max_nonfoot_contact_time_s": 0.10,
    "min_foot_contact_fraction": 0.90,
    "max_foot_slip_p95_m_s": 0.25,
    "max_steady_joint_error_rad": 0.35,
    "max_steady_joint_speed_rms_rad_s": 0.50,
    "max_final_tilt_deg": 15.0,
    "max_steady_base_speed_m_s": 0.10,
    "max_steady_base_angular_speed_rad_s": 0.25,
    "max_saturation_fraction": 0.20,
    "steady_window_s": 1.0,
    "contact_settle_grace_s": 0.5,
    "contact_force_threshold_n": 0.1,
    "illegal_contact_force_threshold_n": 10.0,
}


def _trials(settings=None):
    frequency = float((settings or {}).get("tracking_frequency_hz", 0.5))
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("落地动作跟踪频率必须有限且大于零")
    return (*deepcopy(TRIALS), {
        "name": "target_motion", "duration_s": 3.0 + 2.0 / frequency,
        "axis": None, "settle_s": 2.0, "cycles": 2,
        "tracking_frequency_hz": frequency, "recovery_s": 1.0,
        "normalized_action_amplitude": float((settings or {}).get("action_amplitude", 1.0)) * 0.25,
    })


def _contact_contract(analysis, bindings):
    policy = analysis.get("settings", {}).get("ground_contact_policy", "task")
    if policy not in ("task", "feet_only"):
        raise ValueError("落地接触标准必须为 task 或 feet_only")
    template = analysis.get("template", "him")
    if template not in ("ppo", "him", "him_symmetry", "him_lpacrl"):
        raise ValueError(f"无法确定模板 {template!r} 的落地接触合同")
    return {
        "policy": policy, "template": template,
        "illegal_geom_names": list(bindings["base_geoms"]),
        "leg_geom_names": list(bindings.get("leg_geoms", [])),
        "illegal_force_threshold_n": (PASS_CRITERIA["contact_force_threshold_n"] if policy == "feet_only"
                                       else PASS_CRITERIA["illegal_contact_force_threshold_n"]),
        "max_illegal_contact_time_s": PASS_CRITERIA["max_nonfoot_contact_time_s"],
        "leg_warning_force_threshold_n": PASS_CRITERIA["contact_force_threshold_n"],
        "leg_warning_contact_time_s": PASS_CRITERIA["max_nonfoot_contact_time_s"],
        "source": "src/utils/robot_config.py:adapt_env nonfoot_ground_touch binds base_geoms; selected RI-4438 template illegal_contact force_threshold=10N",
        "scope_note": "Web-generated PPO/HIM/Symmetry/LP-ACRL use the bound base geoms for termination. Native Go2 PPO uses all nonfoot geoms and is a different contact contract.",
        "duration_note": "finite screening rejects cumulative threshold exceedance over 0.1s; training termination has its own sensor history and any-step semantics",
        "force_note": "sum positive contact normal forces per geometry; this screen does not reproduce the training world-frame net-force history sensor",
    }


def resolve_timing(parameters, settings):
    """Resolve the user's timing without altering dt or silently rounding rates."""
    dt = float(parameters["environment"]["sim_dt"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("落地 PD 测试需要有限且大于零的 environment.sim_dt")
    if math.ceil(10.0 / dt) > MAX_TRIAL_STEPS:
        raise ValueError(f"落地 PD 测试每试验最多 {MAX_TRIAL_STEPS} 个物理步；请增大 sim_dt 或另做专门验证，不会自动更改步长")
    decimation = parameters["control"]["decimation"]
    if type(decimation) is not int or decimation < 1:
        raise ValueError("control.decimation 必须为正整数")
    lag_max = parameters["control"].get("delay_max_lag", 0)
    if type(lag_max) is not int or lag_max < 0 or lag_max > MAX_TRIAL_STEPS:
        raise ValueError(f"执行器目标延迟超出本次 {MAX_TRIAL_STEPS} 步历史预算")
    hz = settings.get("hardware_pd_hz")
    delay = settings.get("hardware_delay_ms")
    if hz is not None and (not math.isfinite(float(hz)) or float(hz) <= 0):
        raise ValueError("实机内部 PD 频率必须大于零且有限")
    if delay is not None and (not math.isfinite(float(delay)) or float(delay) < 0):
        raise ValueError("实机闭环延迟必须非负且有限")
    if hz is not None and float(hz) > 1 / dt * (1 + 1e-12):
        raise ValueError(f"实机 PD 频率 {hz:g} Hz 超过物理频率 {1 / dt:g} Hz；请将 sim_dt 设为不大于 {1 / float(hz):.9g} s 后重新测试，不能按较低频率代测")
    discrete = hz is not None or delay is not None
    effective = float(hz) if hz is not None else 1 / dt
    return {
        "sim_dt": dt, "physics_hz": 1 / dt, "decimation": decimation,
        "policy_hz": 1 / (dt * decimation), "target_period_s": dt * decimation,
        "hardware_pd_hz": hz, "hardware_delay_ms": delay,
        "native_pd_hz": 1 / dt, "effective_pd_hz": effective,
        "pd_hz": effective,
        "mode": "hardware_discrete_pd" if discrete else "native_position_pd",
        "feedback_delay_s": float(delay or 0) / 1000,
        "scheduling": "first physics boundary at/after each requested tick; measured jitter reported",
        "target_schedule": "initial pose or synthetic target_motion sampled at policy ticks; training target-delay history retained",
        "observation_delay_included": False,
        "hardware_matches_training_native_pd": not discrete,
    }


def validate_budget(timing, candidate_count, settings=None):
    """Preflight all ground trials before spending time on joint screening."""
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("落地 PD 测试至少需要一个候选")
    dt = timing["sim_dt"]
    per_trial = [math.ceil(trial["duration_s"] / dt) for trial in _trials(settings)]
    if max(per_trial) > MAX_TRIAL_STEPS:
        raise ValueError(f"落地 PD 测试每试验最多 {MAX_TRIAL_STEPS} 个物理步；请调整 sim_dt 后重新测试，不会自动更改步长")
    planned = sum(per_trial) * candidate_count
    if planned > MAX_TOTAL_STEPS:
        raise ValueError(f"落地 PD 测试需 {planned} 个物理步，超过本次 {MAX_TOTAL_STEPS} 步预算；请调整配置后再测，不会缩短试验或改变步长")
    return planned


def _geom_bottom(model, data, geom_id):
    """Exact vertical support of primitives or compiled mesh vertices."""
    import mujoco
    import numpy as np

    size = model.geom_size[geom_id]
    zaxis = data.geom_xmat[geom_id].reshape(3, 3)[2]
    kind = int(model.geom_type[geom_id])
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = size[0] + abs(zaxis[2]) * size[1]
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius = np.linalg.norm(zaxis * size)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = size[0] * np.linalg.norm(zaxis[:2]) + size[1] * abs(zaxis[2])
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        radius = np.dot(np.abs(zaxis), size)
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        return float(data.geom_xpos[geom_id, 2] + np.min(model.mesh_vert[start:start + count] @ zaxis))
    else:
        raise ValueError(f"足端碰撞几何 {model.geom(geom_id).name} 类型不能精确贴地，需使用有限实体碰撞几何")
    return float(data.geom_xpos[geom_id, 2] - radius)


def build_model(spec, bindings, parameters, analysis):
    """Return the task's native robot entity plus a floor, grounded initial data."""
    import mujoco
    import numpy as np
    from mjlab.sim import MujocoCfg
    from . import models
    from .template_runtime import create_robot_cfg

    effective = deepcopy(parameters)
    # Legacy PD-only J/N inputs may override training armature for this report.
    effective["control"]["armature"] = {row["name"]: row["armature"] for row in analysis["joints"]}
    entity = create_robot_cfg(str(models.resolve_model(spec)), bindings, effective).build()
    ground_name = "pd_ground"
    if entity.spec.geom(ground_name) is not None:
        raise ValueError("机器人模型已有 pd_ground 几何，无法建立无歧义测试平地")
    entity.spec.worldbody.add_geom(
        name=ground_name, type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10, 10, 0.1], contype=1, conaffinity=1, condim=3,
        friction=[1.0, 0.005, 0.0001], rgba=[0.32, 0.36, 0.40, 1],
    )
    model = entity.compile()
    env = effective["environment"]
    MujocoCfg(timestep=float(env["sim_dt"]), **env.get("mujoco", {})).apply(model)
    if (model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
            or model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
            or model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
            or model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)):
        raise ValueError("当前仿真配置关闭接触、约束、重力或执行器，不能执行真实承重 PD 测试；请先启用")
    gravity = model.opt.gravity.copy()
    if not np.all(np.isfinite(gravity)) or gravity[2] >= 0 or np.linalg.norm(gravity[:2]) > 1e-8:
        raise ValueError("落地 PD 测试需要竖直向下的非零重力，请核对 environment.mujoco.gravity")
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("落地 PD 测试需要且仅支持一个未锁定的自由基座")
    if model.nu != len(bindings["joints"]):
        raise ValueError("完整机器人执行器数与绑定关节数不一致")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    foot_ids = [model.geom(foot["geom"]).id for foot in bindings["feet"]]
    initial_bottom = [_geom_bottom(model, data, geom_id) for geom_id in foot_ids]
    offset = -min(initial_bottom)
    root_qadr = int(model.jnt_qposadr[free[0]])
    data.qpos[root_qadr + 2] += offset
    mujoco.mj_forward(model, data)
    base_id = model.body(bindings["base_body"]).id
    base_height = float(data.xpos[base_id, 2])
    if base_height <= 0:
        raise ValueError("贴地后机身中心不在地面上方，请检查初始姿态和足端碰撞绑定")
    nonfoot = [model.geom(name).id for name in bindings["all_geoms"] if name not in {f["geom"] for f in bindings["feet"]}]
    # Reject intersecting confirmed poses; never silently rotate/change angles.
    intersections = [model.geom(i).name for i in nonfoot if _geom_bottom(model, data, i) < -0.002]
    if intersections:
        raise ValueError("当前初姿按足端贴地后存在非足端穿地，请调整初始姿态：" + ", ".join(intersections))
    timing = resolve_timing(parameters, analysis.get("settings", {}))
    return model, data, {
        "settings": deepcopy(analysis.get("settings", {})),
        "contact_contract": _contact_contract(analysis, bindings),
        "timing": timing, "floor_geom": ground_name, "base_body": bindings["base_body"],
        "free_qpos_address": root_qadr,
        "initial_qpos": data.qpos.tolist(), "initial_base_height_m": base_height,
        "initial_base_rotation": data.xmat[base_id].reshape(3, 3).tolist(),
        "vertical_translation_m": float(offset), "original_initial_foot_bottom_z_m": initial_bottom,
        "initial_foot_lowest_z_m": [_geom_bottom(model, data, i) for i in foot_ids],
        "total_mass_kg": float(np.sum(model.body_mass)),
        "gravity_m_s2": gravity.tolist(), "floor_friction": model.geom_friction[model.geom(ground_name).id].tolist(),
        "fixed_base": False, "contacts": True, "gravity": True,
        "collision_source": "unmodified model plus shared task robot factory create_robot_cfg; no collision or self-collision overrides for screening",
        "collision_geoms": {model.geom(i).name: {
            "contype": int(model.geom_contype[i]), "conaffinity": int(model.geom_conaffinity[i]),
            "margin_m": float(model.geom_margin[i]), "friction": model.geom_friction[i].tolist(),
        } for i in range(model.ngeom) if i != model.geom(ground_name).id
            and (model.geom_contype[i] or model.geom_conaffinity[i])},
    }


def _conditions(metadata, parameters):
    timing = metadata["timing"]
    return {
        **deepcopy(metadata), "seed": SEED, "device": "cpu", "trace_hz": TRACE_HZ,
        "trials": _trials(metadata.get("settings")), "pass_criteria": deepcopy(PASS_CRITERIA),
        "push": {"start_s": 2.0, "duration_s": 0.2, "weight_fraction": 0.05,
                 "force_n": metadata["total_mass_kg"] * abs(metadata["gravity_m_s2"][2]) * 0.05,
                 "application": "horizontal world x/y force at base body center of mass; no applied torque"},
        "target": "initial pose for standing/push; target_motion is a synthetic symmetric thigh +u/calf -u action with hips held; no learned policy or feed-forward gravity compensation",
        "randomization": "disabled; common deterministic initial state and floor for all candidates",
        "foot_friction": 0.6,
        "foot_contact_fraction_definition": "fraction of physical steps after the settling grace with at least three load-bearing feet",
        "foot_slip_definition": "tangential world velocity of load-bearing foot contact points; 95th percentile after settling grace",
        "inference_schedule": {"policy_hz": timing["policy_hz"], "command": "initial pose or synthetic action target sampled and held at each policy tick"},
        "control_delay": {key: parameters["control"].get(key, 0) for key in (
            "delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period")},
        "target_delay_note": "seeded training target-delay lag/hold/update-period semantics retained; target_motion makes the delay observable; constant standing targets remain identical for every lag",
        "continuous_torque_note": "per-joint RMS over the entire trial is compared with supplied continuous torque; unknown stays unknown. This is a simulation RMS budget, not thermal validation or a motor temperature model.",
        "hardware_note": ("native training position actuator updated each physics step" if timing["mode"] == "native_position_pd" else
                          "explicit sampled position/velocity PD with zero-order-held torque and sample-to-execution delay; this differs from training native position PD"),
        "threshold_note": "explicit simulation screening thresholds, not hardware requirements; loading deflection is allowed and no unloaded 2% settling band is used",
    }


def _simulate_trial(model, initial_qpos, bindings, analysis, timing, candidate, trial, metadata, envelopes):
    import mujoco
    import numpy as np
    from .motor_curves import torque_limit

    dt, duration = timing["sim_dt"], trial["duration_s"]
    count = math.ceil(duration / dt)
    base_id = model.body(bindings["base_body"]).id
    floor_id = model.geom(metadata["floor_geom"]).id
    foot_ids = {model.geom(foot["geom"]).id for foot in bindings["feet"]}
    contract = metadata.get("contact_contract") or _contact_contract(analysis, bindings)
    base_ids = {model.geom(name).id for name in bindings["base_geoms"]}
    leg_ids = {model.geom(name).id for name in bindings.get("leg_geoms", [])} - base_ids - foot_ids
    rows = analysis["joints"]
    qadr = np.array([model.joint(row["name"]).qposadr[0] for row in rows], dtype=int)
    dadr = np.array([model.joint(row["name"]).dofadr[0] for row in rows], dtype=int)
    joint_ids = np.array([model.joint(row["name"]).id for row in rows], dtype=int)
    actuator_ids = np.array([next(a for a in range(model.nu) if int(model.actuator_trnid[a, 0]) == j) for j in joint_ids], dtype=int)
    if len(set(actuator_ids)) != len(rows) or not np.allclose(model.actuator_gear[actuator_ids, 0], 1):
        raise ValueError("落地 PD 需要每绑定关节独立的单位传动比位置执行器")
    patch = candidate["proposal"]["control"]
    kp = np.array([patch["stiffness"][row["name"]] for row in rows], dtype=float)
    kd = np.array([patch["damping"][row["name"]] for row in rows], dtype=float)
    if not np.isfinite(kp).all() or not np.isfinite(kd).all() or (kp < 0).any() or (kd < 0).any():
        raise ValueError("候选 PD 必须有限且非负")
    initial_target = np.array([row["initial"] for row in rows])
    motion = trial["name"] == "target_motion"
    scales = np.array([row["action_scale"] for row in rows])
    clip_low = np.array([row["action_clip"][0] for row in rows])
    clip_high = np.array([row["action_clip"][1] for row in rows])
    motion_direction = np.array([1.0 if row["role"] == "thigh" else -1.0 if row["role"] == "calf" else 0.0 for row in rows])
    target = initial_target.copy()
    limits = np.array([row["effort_limit"] for row in rows], dtype=float)
    velocity_limits = np.array([row.get("velocity_limit") or np.inf for row in rows])
    model.actuator_gainprm[actuator_ids, 0] = kp
    model.actuator_biasprm[actuator_ids, 1] = -kp
    model.actuator_biasprm[actuator_ids, 2] = -kd
    # Each trial may follow a discrete trial; overwrite every relevant field.
    discrete = timing["mode"] == "hardware_discrete_pd"
    if discrete:
        model.actuator_gainprm[actuator_ids, 0] = 1
        model.actuator_biasprm[actuator_ids, :] = 0
    model.actuator_forcerange[actuator_ids, 0] = -limits
    model.actuator_forcerange[actuator_ids, 1] = limits
    data = mujoco.MjData(model)
    data.qpos[:] = initial_qpos
    data.ctrl[actuator_ids] = 0 if discrete else target
    mujoco.mj_forward(model, data)
    initial_pos = data.xpos[base_id].copy()
    initial_up = data.xmat[base_id].reshape(3, 3)[:, 2].copy()
    start_warning = np.array([w.number for w in data.warning])
    force = metadata["total_mass_kg"] * abs(metadata["gravity_m_s2"][2]) * 0.05
    queue, held = deque(), np.zeros(len(rows))
    period = 1 / timing["effective_pd_hz"]
    next_tick, max_jitter, max_delay_error, ticks, applied = 0.0, 0.0, 0.0, 0, 0
    pd_tick_times = []
    next_trace = 0.0
    traces = {"time": [0.0], "qpos": [data.qpos.tolist()]}
    arrays = {name: [] for name in ("height", "tilt", "drift", "contact_feet", "nonfoot", "base_contact", "leg_contact", "illegal_contact", "saturation", "error", "joint_speed", "torque", "foot_slip", "speed_exceeded", "base_speed", "base_angular_speed")}
    steady_error, steady_speed, contact_slip = [], [], []
    reasons = []
    warnings = []
    finite = True
    fall = False
    # These draws use the same lag/hold/update-period semantics as DelayBuffer.
    control_delay = metadata.get("control_delay", {})
    rng = np.random.default_rng(SEED)
    low, high = (int(control_delay.get(f"delay_{which}_lag", 0)) for which in ("min", "max"))
    lag_period = int(control_delay.get("delay_update_period", 0))
    phase = int(rng.integers(lag_period)) if lag_period else 0
    lag = int(rng.integers(low, high + 1))
    sampled_low, sampled_high = lag, lag
    lag_ticks = 0
    target_ticks = 0
    held_target = initial_target.copy()
    previous_intended, previous_applied = initial_target.copy(), initial_target.copy()
    intended_changes = applied_changes = clipping_ticks = 0
    intended_peak = np.zeros(len(rows))
    applied_peak = np.zeros(len(rows))
    target_gap_peak = np.zeros(len(rows))
    raw_peak = np.zeros(len(rows))
    intended_samples, applied_samples, motion_errors = [], [], []
    geom_contacts = {}
    base_force_peak = leg_force_peak = nonfoot_force_peak = illegal_force_peak = 0.0
    target_history = deque((initial_target.copy() for _ in range(high + 1)), maxlen=high + 1)
    contact_force = np.empty(6)
    geom_velocity = np.empty(6)
    # Record all metrics at every physical step, independent of video sampling.
    for step in range(count):
        t = step * dt
        if step % timing["decimation"] == 0:
            raw_action = np.zeros(len(rows))
            if motion and trial["settle_s"] <= t < trial["settle_s"] + trial["cycles"] / trial["tracking_frequency_hz"] - 1e-10:
                phase_angle = 2 * math.pi * trial["tracking_frequency_hz"] * (t - trial["settle_s"])
                raw_action = motion_direction * trial["normalized_action_amplitude"] * math.sin(phase_angle)
            raw_target = initial_target + scales * raw_action
            held_target = np.clip(raw_target, clip_low, clip_high)
            raw_peak = np.maximum(raw_peak, np.abs(raw_action))
            if np.any(np.abs(raw_target - held_target) > 1e-12):
                clipping_ticks += 1
            if np.any(np.abs(held_target - previous_intended) > 1e-12):
                intended_changes += 1
                if len(intended_samples) < 16:
                    intended_samples.append({"time_s": t, "target_rad": held_target.tolist()})
            previous_intended = held_target.copy()
            intended_peak = np.maximum(intended_peak, np.abs(held_target - initial_target))
            target_ticks += 1
        target_history.append(held_target.copy())
        proposed_lag = int(rng.integers(low, high + 1))
        take = bool(rng.random() >= float(control_delay.get("delay_hold_prob", 0)))
        if (not lag_period or (step + phase) % lag_period == 0) and take:
            lag = proposed_lag
            lag_ticks += 1
        sampled_low, sampled_high = min(sampled_low, lag), max(sampled_high, lag)
        target = target_history[-1 - lag]
        if np.any(np.abs(target - previous_applied) > 1e-12):
            applied_changes += 1
            if len(applied_samples) < 16:
                applied_samples.append({"time_s": t, "target_rad": target.tolist()})
        previous_applied = target.copy()
        applied_peak = np.maximum(applied_peak, np.abs(target - initial_target))
        target_gap_peak = np.maximum(target_gap_peak, np.abs(target - held_target))
        if discrete and t + 1e-10 >= next_tick:
            jitter = max(0.0, t - next_tick)
            max_jitter = max(max_jitter, jitter)
            torque = kp * (target - data.qpos[qadr]) - kd * data.qvel[dadr]
            queue.append((t + timing["feedback_delay_s"], torque.copy()))
            ticks += 1
            pd_tick_times.append(t)
            next_tick = ticks * period
        if discrete:
            while queue and queue[0][0] <= t + 1e-10:
                due, held = queue.popleft()
                max_delay_error = max(max_delay_error, max(0.0, t - due))
                applied += 1
            data.ctrl[actuator_ids] = held
        else:
            data.ctrl[actuator_ids] = target
        active_limits = limits.copy()
        for j, row in enumerate(rows):
            if row["name"] in envelopes:
                active_limits[j] = min(active_limits[j], torque_limit(envelopes[row["name"]], data.qvel[dadr[j]]))
        model.actuator_forcerange[actuator_ids, 0] = -active_limits
        model.actuator_forcerange[actuator_ids, 1] = active_limits
        data.xfrc_applied[base_id, :] = 0
        if trial["axis"] is not None and 2.0 <= t < 2.2 - 1e-10:
            data.xfrc_applied[base_id, trial["axis"]] = force
        mujoco.mj_step(model, data)
        # Capture the forces actually used by this physical step. mj_forward
        # below recomputes a different force at the resulting state.
        signed_torque = data.qfrc_actuator[dadr].copy()
        # mj_step may reset unstable states internally; warnings/time detect it.
        bad_warning = any(data.warning[i].number > start_warning[i] for i in (
            int(mujoco.mjtWarning.mjWARN_BADQPOS), int(mujoco.mjtWarning.mjWARN_BADQVEL), int(mujoco.mjtWarning.mjWARN_BADQACC), int(mujoco.mjtWarning.mjWARN_BADCTRL)))
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or not np.isfinite(data.qacc).all() or not np.isfinite(data.qfrc_actuator).all() or bad_warning
                or data.time < (step + 1) * dt - dt * 0.5):
            finite = False
            reasons.append("完整模型出现非有限状态、MuJoCo 数值警告或内部重置")
            break
        # Recompute contacts/velocities at the state used for metrics and video.
        mujoco.mj_forward(model, data)
        height = float(data.xpos[base_id, 2])
        up = data.xmat[base_id].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(up @ initial_up, -1, 1))))
        drift = float(np.linalg.norm(data.xpos[base_id, :2] - initial_pos[:2]))
        touching, contact_points, force_by_geom = {}, {}, {}
        for con_id in range(data.ncon):
            contact = data.contact[con_id]
            if floor_id not in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(model, data, con_id, contact_force)
            other = int(contact.geom2 if contact.geom1 == floor_id else contact.geom1)
            force_by_geom[other] = force_by_geom.get(other, 0.0) + max(0.0, float(contact_force[0]))
            contact_points.setdefault(other, []).append(contact.pos.copy())
        threshold = PASS_CRITERIA["contact_force_threshold_n"]
        nonfoot = base_contact = leg_contact = illegal_contact = False
        for geom_id, normal_force in force_by_geom.items():
            contact_diag = geom_contacts.setdefault(model.geom(geom_id).name, {
                "time_over_contact_threshold_s": 0.0, "time_over_task_threshold_s": 0.0,
                "peak_normal_force_n": 0.0,
                "category": "foot" if geom_id in foot_ids else "base" if geom_id in base_ids else "leg" if geom_id in leg_ids else "other_nonfoot",
            })
            contact_diag["peak_normal_force_n"] = max(contact_diag["peak_normal_force_n"], normal_force)
            contact_diag["time_over_contact_threshold_s"] += dt if normal_force > threshold else 0.0
            contact_diag["time_over_task_threshold_s"] += dt if normal_force > PASS_CRITERIA["illegal_contact_force_threshold_n"] else 0.0
            if geom_id in foot_ids:
                if normal_force > threshold:
                    touching[geom_id] = contact_points[geom_id]
                continue
            nonfoot_force_peak = max(nonfoot_force_peak, normal_force)
            nonfoot |= normal_force > threshold
            if geom_id in base_ids:
                base_force_peak = max(base_force_peak, normal_force)
                base_contact |= normal_force > threshold
            elif geom_id in leg_ids:
                leg_force_peak = max(leg_force_peak, normal_force)
                leg_contact |= normal_force > threshold
            illegal_geom = contract["policy"] == "feet_only" or geom_id in base_ids or geom_id not in leg_ids
            if illegal_geom:
                illegal_force_peak = max(illegal_force_peak, normal_force)
                illegal_contact |= normal_force > contract["illegal_force_threshold_n"]
        slip = []
        for geom_id, contact_positions in touching.items():
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, geom_velocity, 0)
            for position in contact_positions:
                velocity = geom_velocity[3:] + np.cross(geom_velocity[:3], position - data.geom_xpos[geom_id])
                slip.append(float(np.linalg.norm(velocity[:2])))
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, geom_velocity, 0)
        base_speed = float(np.linalg.norm(geom_velocity[3:]))
        base_angular_speed = float(np.linalg.norm(geom_velocity[:3]))
        joint_error = np.abs(data.qpos[qadr] - target)
        if motion and trial["settle_s"] <= t < trial["settle_s"] + trial["cycles"] / trial["tracking_frequency_hz"]:
            motion_errors.append(joint_error.copy())
        joint_speed = np.abs(data.qvel[dadr])
        actual_torque = np.abs(signed_torque)
        force_low, force_high = -active_limits.copy(), active_limits.copy()
        for j, joint_id in enumerate(joint_ids):
            if model.jnt_actfrclimited[joint_id]:
                force_low[j] = max(force_low[j], model.jnt_actfrcrange[joint_id, 0])
                force_high[j] = min(force_high[j], model.jnt_actfrcrange[joint_id, 1])
        saturated = ((signed_torque <= force_low + np.abs(force_low) * 1e-6)
                     | (signed_torque >= force_high - np.abs(force_high) * 1e-6))
        for name, value in (
            ("height", height), ("tilt", tilt), ("drift", drift),
            ("contact_feet", len(touching)), ("nonfoot", nonfoot),
            ("base_contact", base_contact), ("leg_contact", leg_contact), ("illegal_contact", illegal_contact),
            ("saturation", saturated), ("error", joint_error), ("joint_speed", joint_speed),
            ("torque", actual_torque), ("foot_slip", max(slip, default=0.0)),
            ("base_speed", base_speed), ("base_angular_speed", base_angular_speed),
            ("speed_exceeded", bool(np.any(joint_speed > velocity_limits))),
        ):
            arrays[name].append(value)
        if data.time >= PASS_CRITERIA["contact_settle_grace_s"]:
            contact_slip.extend(slip)
        if data.time >= duration - PASS_CRITERIA["steady_window_s"]:
            steady_error.append(joint_error)
            steady_speed.append(joint_speed)
        if data.time + 1e-10 >= next_trace + 1 / TRACE_HZ or step == count - 1:
            traces["time"].append(float(data.time))
            traces["qpos"].append(data.qpos.tolist())
            next_trace = len(traces["time"]) / TRACE_HZ - 1 / TRACE_HZ
        if height < initial_pos[2] * 0.25 or tilt > 75:
            fall = True
            reasons.append("机器人倒地，提前终止该试验")
            break
    # Preserve the last finite real state in failed-trial video too.
    if finite and data.time > traces["time"][-1] + 1e-10:
        traces["time"].append(float(data.time))
        traces["qpos"].append(data.qpos.tolist())
    if not arrays["height"]:
        return {"name": trial["name"], "status": "failed", "passed": False,
                "metrics": {"finite": finite, "fallen": fall, "completed_s": 0.0},
                "reasons": reasons or ["未获得有效物理步数据"], "warnings": warnings, "trajectory": traces}
    values = {name: np.asarray(value) for name, value in arrays.items()}
    n = len(values["height"])
    after_grace = np.arange(1, n + 1) * dt >= PASS_CRITERIA["contact_settle_grace_s"]
    contact_fraction = float(np.mean(values["contact_feet"][after_grace] >= 3)) if after_grace.any() else 0.0
    # Report per-joint worst saturation; a single overloaded actuator matters.
    saturation_fraction = np.mean(values["saturation"], axis=0)
    errors = np.array(steady_error) if steady_error else values["error"][-max(1, int(1 / dt)):]
    speeds = np.array(steady_speed) if steady_speed else values["joint_speed"][-max(1, int(1 / dt)):]
    steady_slice = slice(-max(1, int(1 / dt)), None)
    torque_rms = np.sqrt(np.mean(values["torque"] ** 2, axis=0))
    continuous_ratios = {row["name"]: (float(torque_rms[i] / row["continuous_torque"])
                         if row.get("continuous_torque") else None) for i, row in enumerate(rows)}
    known_ratios = [value for value in continuous_ratios.values() if value is not None]
    metrics = {
        "finite": bool(finite), "fallen": bool(fall), "completed_s": n * dt,
        "physics_steps": n, "initial_base_height_m": float(initial_pos[2]),
        "min_base_height_m": float(np.min(values["height"])),
        "final_base_height_m": float(values["height"][-1]),
        "max_base_drop_m": max(0.0, float(initial_pos[2] - np.min(values["height"]))),
        "max_tilt_deg": float(np.max(values["tilt"])), "max_drift_m": float(np.max(values["drift"])),
        "final_tilt_deg": float(values["tilt"][-1]),
        "final_drift_m": float(values["drift"][-1]),
        "steady_base_speed_rms_m_s": float(np.sqrt(np.mean(values["base_speed"][steady_slice] ** 2))),
        "steady_base_angular_speed_rms_rad_s": float(np.sqrt(np.mean(values["base_angular_speed"][steady_slice] ** 2))),
        "min_contact_feet": int(np.min(values["contact_feet"][after_grace])) if after_grace.any() else 0,
        "foot_contact_fraction": contact_fraction,
        "nonfoot_contact_time_s": float(np.sum(values["nonfoot"]) * dt),
        "base_contact_time_s": float(np.sum(values["base_contact"]) * dt),
        "leg_contact_time_s": float(np.sum(values["leg_contact"]) * dt),
        "leg_contact_fraction": float(np.mean(values["leg_contact"])),
        "illegal_contact_time_s": float(np.sum(values["illegal_contact"]) * dt),
        "peak_base_normal_force_n": base_force_peak,
        "peak_leg_normal_force_n": leg_force_peak,
        "peak_nonfoot_normal_force_n": nonfoot_force_peak,
        "peak_illegal_normal_force_n": illegal_force_peak,
        "contact_by_geom": geom_contacts,
        "contact_policy": contract["policy"],
        "foot_slip_p95_m_s": float(np.percentile(contact_slip, 95)) if contact_slip else 0.0,
        "peak_foot_slip_m_s": float(np.max(values["foot_slip"])),
        "steady_joint_error_max_rad": float(np.max(np.mean(errors, axis=0))),
        "steady_joint_speed_rms_rad_s": float(np.max(np.sqrt(np.mean(speeds**2, axis=0)))),
        "max_saturation_fraction": float(np.max(saturation_fraction)),
        "saturation_fraction_by_joint": {row["name"]: float(saturation_fraction[i]) for i, row in enumerate(rows)},
        "peak_joint_velocity_rad_s": float(np.max(values["joint_speed"])),
        "peak_torque_nm": float(np.max(values["torque"])),
        "torque_rms_nm_by_joint": {row["name"]: float(torque_rms[i]) for i, row in enumerate(rows)},
        "continuous_torque_ratio_by_joint": continuous_ratios,
        "continuous_torque_passed": (max(known_ratios) <= 1.0 + 1e-9) if known_ratios else None,
        "continuous_torque_known_joints": len(known_ratios),
        "continuous_torque_total_joints": len(rows),
        "velocity_limit_passed": not bool(np.any(values["speed_exceeded"])),
        "timing_actual": {"pd_ticks": ticks if discrete else n,
                          "torque_commands_applied": applied if discrete else n,
                          "target_ticks": target_ticks,
                          "max_pd_tick_jitter_s": max_jitter,
                          "pd_tick_interval_min_s": float(np.min(np.diff(pd_tick_times))) if len(pd_tick_times) > 1 else dt,
                          "pd_tick_interval_max_s": float(np.max(np.diff(pd_tick_times))) if len(pd_tick_times) > 1 else dt,
                          "sampled_pd_tick_times_first_16_s": pd_tick_times[:16] if discrete else [i * dt for i in range(min(n, 16))],
                          "pd_tick_count_window": "samples at step starts from t=0 through completed_s-sim_dt; no fictitious sample beyond final physics step",
                          "max_execution_delay_quantization_s": max_delay_error,
                          "sampled_target_lag_steps": [sampled_low, sampled_high],
                          "target_lag_updates": lag_ticks},
        "target_motion": {
            "active": motion, "synthetic_not_policy_gait": True,
            "target_definition": "intended is policy-held and joint-clipped; applied is the PD input after training target delay. Discrete hardware torque feedback/execution delay is counted separately in timing_actual.",
            "intended_target_changes": intended_changes, "applied_target_changes": applied_changes,
            "target_clip_policy_ticks": clipping_ticks,
            "raw_normalized_action_peak_by_joint": {row["name"]: float(raw_peak[i]) for i, row in enumerate(rows)},
            "intended_target_offset_peak_rad_by_joint": {row["name"]: float(intended_peak[i]) for i, row in enumerate(rows)},
            "applied_target_offset_peak_rad_by_joint": {row["name"]: float(applied_peak[i]) for i, row in enumerate(rows)},
            "target_delay_gap_peak_rad": float(np.max(target_gap_peak)),
            "target_delay_gap_peak_rad_by_joint": {row["name"]: float(target_gap_peak[i]) for i, row in enumerate(rows)},
            "intended_target_changes_first_16": intended_samples,
            "applied_target_changes_first_16": applied_samples,
            "tracking_error_max_rad": float(np.max(motion_errors)) if motion_errors else None,
            "tracking_error_rms_rad_by_joint": {row["name"]: float(value) for row, value in zip(rows, np.sqrt(np.mean(np.array(motion_errors) ** 2, axis=0)))} if motion_errors else {},
        },
    }
    pc = PASS_CRITERIA
    checks = (
        (metrics["max_tilt_deg"] <= pc["max_tilt_deg"], "机身相对初姿倾角超过阈值"),
        (metrics["min_base_height_m"] >= initial_pos[2] * pc["min_base_height_fraction"], "机身高度过低"),
        (metrics["max_base_drop_m"] <= initial_pos[2] * pc["max_base_drop_fraction"], "机身下沉超过初始高度阈值"),
        (metrics["max_drift_m"] <= pc["max_drift_m"], "机身水平漂移过大"),
        (metrics["illegal_contact_time_s"] <= contract["max_illegal_contact_time_s"] + 1e-12,
         "非足端持续触地（feet_only严格标准）" if contract["policy"] == "feet_only" else "机体或未分类非足端接触超过任务法向力与累计时间门槛"),
        (metrics["foot_contact_fraction"] >= pc["min_foot_contact_fraction"], "至少三足承重接触的时间占比不足"),
        (metrics["foot_slip_p95_m_s"] <= pc["max_foot_slip_p95_m_s"], "承重足端滑移过大"),
        (metrics["steady_joint_error_max_rad"] <= pc["max_steady_joint_error_rad"], "承重关节稳态误差过大"),
        (metrics["steady_joint_speed_rms_rad_s"] <= pc["max_steady_joint_speed_rms_rad_s"], "稳态关节持续抖动"),
        (metrics["final_tilt_deg"] <= pc["max_final_tilt_deg"], "试验结束时机身倾斜未恢复"),
        (metrics["steady_base_speed_rms_m_s"] <= pc["max_steady_base_speed_m_s"], "最后一秒机身仍明显平移"),
        (metrics["steady_base_angular_speed_rms_rad_s"] <= pc["max_steady_base_angular_speed_rad_s"], "最后一秒机身仍明显旋转"),
        (metrics["max_saturation_fraction"] <= pc["max_saturation_fraction"], "关节力矩长期饱和"),
        (metrics["velocity_limit_passed"], "关节速度超过用户填写的工作上限"),
        (n == count, "试验未完成完整持续时间"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    if metrics["leg_contact_time_s"] > contract["leg_warning_contact_time_s"] + 1e-12:
        warnings.append("腿部持续触地；task标准单独报告并计入排名代价，feet_only标准同时按非法接触判定")
    if metrics["continuous_torque_passed"] is False:
        reasons.append("整个试验的关节力矩RMS超过填写的连续力矩；仅为仿真RMS预算，不是热验证")
    if len(known_ratios) < len(rows):
        warnings.append("部分或全部关节未填写连续力矩，相关RMS预算未知，不能作热能力结论")
    if motion and metrics["target_motion"]["tracking_error_max_rad"] is not None and metrics["target_motion"]["tracking_error_max_rad"] > pc["max_steady_joint_error_rad"]:
        reasons.append("负载动作跟踪角误差超过落地关节误差门槛")
    if clipping_ticks:
        warnings.append("小幅合成动作触及所填关节目标裁剪；实际裁剪次数与幅值已记录，另需通过全幅局部跟踪工况")
    passed = finite and not fall and not reasons
    return {"name": trial["name"], "status": "passed" if passed else "failed", "passed": passed,
            "metrics": metrics, "reasons": reasons, "warnings": warnings, "trajectory": traces}


def _candidate(model, initial_qpos, bindings, analysis, timing, candidate, metadata, envelopes):
    # The shared model is read-only. Copy after entering the bounded executor,
    # so at most MAX_WORKERS candidate-owned models are resident at once.
    model = copy(model)
    requests = _trials(analysis.get("settings", metadata.get("settings", {})))
    trials = [_simulate_trial(model, initial_qpos, bindings, analysis, timing, candidate, trial, metadata, envelopes)
              for trial in requests]
    passed = all(trial["passed"] for trial in trials)
    available = [trial["metrics"] for trial in trials if "max_tilt_deg" in trial["metrics"]]
    metrics = {}
    for key in ("max_tilt_deg", "max_base_drop_m", "max_drift_m", "nonfoot_contact_time_s",
                "base_contact_time_s", "leg_contact_time_s", "illegal_contact_time_s",
                "peak_base_normal_force_n", "peak_leg_normal_force_n", "peak_nonfoot_normal_force_n", "peak_illegal_normal_force_n",
                "foot_slip_p95_m_s", "steady_joint_error_max_rad", "steady_joint_speed_rms_rad_s",
                "max_saturation_fraction", "peak_joint_velocity_rad_s", "peak_torque_nm", "final_tilt_deg",
                "steady_base_speed_rms_m_s", "steady_base_angular_speed_rms_rad_s"):
        metrics[key] = max((m[key] for m in available), default=0.0)
    for key in ("min_base_height_m", "foot_contact_fraction"):
        metrics[key] = min((m[key] for m in available), default=0.0)
    metrics["finite"] = all(t["metrics"].get("finite", False) for t in trials)
    metrics["fallen"] = any(t["metrics"].get("fallen", False) for t in trials)
    metrics["completed_trials"] = sum(t["metrics"].get("completed_s", 0) >= request["duration_s"] - timing["sim_dt"] / 2 for t, request in zip(trials, requests))
    metrics["total_trials"] = len(requests)
    completed_duration = sum(t["metrics"].get("completed_s", 0.0) for t in trials)
    metrics["leg_contact_fraction"] = sum(m.get("leg_contact_time_s", 0.0) for m in available) / completed_duration if completed_duration else 0.0
    metrics["contact_time_aggregation"] = "reported contact times are worst trial; leg_contact_fraction is weighted across all completed physics time"
    metrics["contact_policy"] = (metadata.get("contact_contract") or _contact_contract(analysis, bindings))["policy"]
    metrics["strict_feet_only_passed"] = passed and all(m.get("nonfoot_contact_time_s", 0.0) <= PASS_CRITERIA["max_nonfoot_contact_time_s"] + 1e-12 for m in available)
    metrics["torque_rms_nm_by_joint"] = {row["name"]: max((m.get("torque_rms_nm_by_joint", {}).get(row["name"], 0.0) for m in available), default=0.0) for row in analysis["joints"]}
    metrics["continuous_torque_ratio_by_joint"] = {
        row["name"]: max((m["continuous_torque_ratio_by_joint"][row["name"]] for m in available), default=0.0)
        if row.get("continuous_torque") else None for row in analysis["joints"]}
    continuous = [m.get("continuous_torque_passed") for m in available if m.get("continuous_torque_passed") is not None]
    metrics["continuous_torque_passed"] = all(continuous) if continuous else None
    # Ranking is only a screen heuristic; all eligibility gates remain explicit.
    metrics["score"] = (metrics["max_tilt_deg"] / 30 + metrics["max_drift_m"] / .3
                        + metrics["max_saturation_fraction"] * 2 + metrics["steady_joint_error_max_rad"] / .35
                        + metrics["leg_contact_fraction"] * .15)
    reasons = [f"{trial['name']}: {reason}" for trial in trials for reason in trial["reasons"]]
    warnings = [f"{trial['name']}: {warning}" for trial in trials for warning in trial.get("warnings", [])]
    return {"status": "passed" if passed else "failed", "passed": passed,
            "metrics": metrics, "reasons": reasons, "warnings": warnings, "trials": trials,
            "strict_feet_only_passed": metrics["strict_feet_only_passed"]}


def evaluate_candidates(spec, bindings, parameters, analysis, candidates, progress=print):
    """Evaluate all candidates with bounded independent-model CPU parallelism."""
    from .motor_curves import compile_envelopes
    from .pd_response import _effective_options

    started = time.monotonic()
    timing = resolve_timing(parameters, analysis.get("settings", {}))
    planned = validate_budget(timing, len(candidates), analysis.get("settings", {}))
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("PD 候选 ID 必须唯一")
    model, data, metadata = build_model(spec, bindings, parameters, analysis)
    conditions = _conditions(metadata, parameters)
    conditions["mujoco_effective"] = _effective_options(model)
    metadata["control_delay"] = conditions["control_delay"]
    envelopes = compile_envelopes(spec, [row["name"] for row in analysis["joints"]])
    conditions["torque_speed_envelopes"] = {name: [list(point) for point in points] for name, points in envelopes.items()}
    workers = min(MAX_WORKERS, max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1), len(candidates))
    progress(f"落地承重测试：{len(candidates)} 组候选，最多 {workers} 组并行；自由基座、重力和接触开启。")
    results = {}
    # Each running task copies once; queued candidates allocate no model.
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pd-ground") as executor:
        pending = {executor.submit(_candidate, model, data.qpos.copy(), bindings, analysis,
                                   timing, candidate, metadata, envelopes): candidate for candidate in candidates}
        for future in as_completed(pending):
            candidate = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "passed": False, "metrics": {}, "trials": [],
                          "reasons": [f"完整机器人落地试验异常：{type(exc).__name__}: {exc}"], "warnings": []}
            results[candidate["id"]] = result
            progress(f"落地 [{len(results)}/{len(candidates)}] {candidate.get('label', candidate['id'])}："
                     + ("静站、双向扰动及合成目标变化全部通过" if result["passed"] else "未通过；" + "; ".join(result["reasons"][:2])))
    # Stable input ordering makes reports reproducible despite completion order.
    return {"conditions": conditions, "results": {c["id"]: results[c["id"]] for c in candidates},
            "parallel": {"backend": "thread", "max_workers": workers,
                         "independent_model_per_candidate": True, "candidate_count": len(candidates),
                         "planned_physics_steps": planned, "elapsed_s": time.monotonic() - started}}
