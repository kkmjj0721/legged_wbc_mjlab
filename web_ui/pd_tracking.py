"""Full action-amplitude tracking under the configured target and PD timing.

These independent, locked-coordinate experiments retain each reduced model's
inertia and passive joint properties. They expose target-update, actuator-delay
and force-limit effects, but do not establish whole-body or hardware stability.
No input model, gain proposal, analysis result or editable parameter is changed.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import math
import os
import time

from .pd_ground import resolve_timing


SEED = 42
HOLD_S = 1.0
CYCLES = 2
MAX_TRACE_POINTS = 257
MAX_JOINT_STEPS = 100_000
MAX_TOTAL_STEPS = 2_000_000
MAX_SATURATION_FRACTION = 0.10


def _workers(count):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, min(4, available, count))


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"动作跟踪测试的 {name} 必须为有限正数")
    return float(value)


def validate_budget(parameters, settings, candidate_count, joint_count=12):
    """Reject a full tracking sweep before starting any other response trials."""
    if type(candidate_count) is not int or candidate_count < 1 or type(joint_count) is not int or joint_count < 1:
        raise ValueError("动作跟踪预算需要正整数候选和关节数")
    frequency = _positive(settings.get("tracking_frequency_hz", 0.5), "tracking_frequency_hz")
    _positive(settings.get("max_tracking_error_ratio", 0.2), "max_tracking_error_ratio")
    _positive(settings.get("action_amplitude", 1.0), "action_amplitude")
    timing = resolve_timing(parameters, settings)
    duration = HOLD_S * 2 + CYCLES / frequency
    per_joint = math.ceil(duration / timing["sim_dt"])
    planned = per_joint * joint_count * candidate_count
    if per_joint > MAX_JOINT_STEPS or planned > MAX_TOTAL_STEPS:
        raise ValueError(f"完整动作跟踪需要每关节 {per_joint}、共 {planned} 个物理步，超过预算；"
                         "请调整配置或另做专门验证，不会缩幅、缩时或改变步长")
    return planned


def _schedule(row, joint_index, control, settings, timing):
    """Build real target history, rather than delaying a single fixed command."""
    import numpy as np

    dt = timing["sim_dt"]
    frequency = settings["tracking_frequency_hz"]
    active_end = HOLD_S + CYCLES / frequency
    count = math.ceil((active_end + HOLD_S) / dt)
    times = np.arange(count, dtype=float) * dt
    amplitude = abs(float(row["action_scale"])) * settings["action_amplitude"]
    initial = float(row["initial"])
    low, high = map(float, row["action_clip"])
    available_minus = max(0.0, initial - low)
    available_plus = max(0.0, high - initial)
    enough_range = min(available_minus, available_plus) >= amplitude - 1e-10
    raw = np.full(count, initial, dtype=float)
    moving = (times >= HOLD_S - 1e-10) & (times < active_end - 1e-10)
    raw[moving] += amplitude * np.sin(2 * math.pi * frequency * (times[moving] - HOLD_S))
    # Keep requested amplitude in the report. Limited commands are diagnostic
    # only and cannot qualify, even if the resulting clipped motion tracks well.
    bounded = np.clip(raw, low, high)
    intended = np.empty(count)
    held = initial
    for step in range(count):
        if step % timing["decimation"] == 0:
            held = float(bounded[step])
        intended[step] = held

    lag_low = control.get("delay_min_lag", 0)
    lag_high = control.get("delay_max_lag", 0)
    period = control.get("delay_update_period", 0)
    hold = control.get("delay_hold_prob", 0.0)
    if (type(lag_low) is not int or type(lag_high) is not int
            or not 0 <= lag_low <= lag_high <= MAX_JOINT_STEPS
            or type(period) is not int or period < 0
            or type(hold) not in (int, float) or not math.isfinite(hold) or not 0 <= hold <= 1):
        raise ValueError("动作跟踪测试的执行器目标延迟/保持参数无效")
    rng = np.random.default_rng(SEED + joint_index)
    phase = int(rng.integers(period)) if period else 0
    lag = int(rng.integers(lag_low, lag_high + 1))
    history = deque((initial for _ in range(lag_high + 1)), maxlen=lag_high + 1)
    delayed = np.empty(count)
    lags = np.empty(count, dtype=int)
    updates = np.zeros(count, dtype=bool)
    for step in range(count):
        history.append(float(intended[step]))
        proposed_lag = int(rng.integers(lag_low, lag_high + 1))
        take = bool(rng.random() >= hold)
        if (not period or (step + phase) % period == 0) and take:
            lag = proposed_lag
            updates[step] = True
        delayed[step] = history[-1 - lag]
        lags[step] = lag
    return {
        "time": times, "raw": raw, "intended": intended, "applied": delayed,
        "active": moving, "amplitude": amplitude, "enough_range": bool(enough_range),
        "available_minus": available_minus, "available_plus": available_plus,
        "lags": lags, "lag_updates": updates,
        "target_updates": math.ceil(count / timing["decimation"]),
    }


def _trace(arrays):
    import numpy as np

    count = len(arrays["time"])
    selected = set(np.linspace(0, count - 1, min(count, MAX_TRACE_POINTS - 12), dtype=int).tolist())
    for key in ("position", "torque", "raw_target", "applied_target", "unclamped_torque"):
        selected.update((int(np.argmin(arrays[key])), int(np.argmax(arrays[key]))))
    indices = sorted(selected)
    return {key: np.asarray(value)[indices].tolist() for key, value in arrays.items()}


def _joint(model, row, binding, joint_index, candidate, control, settings, timing, envelope):
    import mujoco
    import numpy as np
    from .motor_curves import torque_limit

    if model.nq != 1 or model.nv != 1 or model.nu != 1:
        raise ValueError("动作跟踪需要一个关节、一个执行器的独立约化模型")
    if not math.isclose(float(model.opt.timestep), timing["sim_dt"], rel_tol=1e-12):
        raise ValueError("动作跟踪模型与配置的物理步长不同")
    name = row["name"]
    kp = _positive(candidate["proposal"]["control"]["stiffness"][name], f"{name}.Kp")
    kd = candidate["proposal"]["control"]["damping"][name]
    if type(kd) not in (int, float) or not math.isfinite(kd) or kd < 0:
        raise ValueError(f"{name}.Kd 必须为有限非负数")
    schedule = _schedule(row, joint_index, control, settings, timing)
    amplitude = schedule["amplitude"]
    if amplitude <= 0:
        raise ValueError(f"{name} 的有效动作幅值为零，不能验证周期跟踪")
    initial = float(row["initial"])
    fixed_limit = _positive(row["effort_limit"], f"{name}.effort_limit")
    discrete = timing["mode"] == "hardware_discrete_pd"
    model.actuator_gainprm[0, :] = 0
    model.actuator_biasprm[0, :] = 0
    if discrete:
        model.actuator_gainprm[0, 0] = 1
    else:
        model.actuator_gainprm[0, 0] = kp
        model.actuator_biasprm[0, 1:3] = [-kp, -kd]
    data = mujoco.MjData(model)
    data.qpos[0] = initial
    data.ctrl[0] = 0 if discrete else initial
    mujoco.mj_forward(model, data)
    dt = timing["sim_dt"]
    period = 1 / timing["effective_pd_hz"]
    next_tick, tick_count = 0.0, 0
    tick_times = []
    pending = deque()
    held_torque = 0.0
    max_tick_error, max_delay_error, torque_updates = 0.0, 0.0, 0
    reasons = []
    if not schedule["enough_range"]:
        reasons.append(
            f"所需双向幅值 {amplitude:.6g} rad 超过动作限位余量"
            f"（负向 {schedule['available_minus']:.6g}、正向 {schedule['available_plus']:.6g} rad）；"
            "本条仅显示限位后的诊断，不能推荐")
    arrays = {key: [] for key in ("time", "raw_target", "intended_target", "applied_target",
                                  "position", "velocity", "torque", "unclamped_torque")}
    errors, torques, raw_torques, saturation = [], [], [], []
    finite = True
    peak_velocity = 0.0
    bad_warnings = tuple(int(value) for value in (
        mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL))
    for step, t in enumerate(schedule["time"]):
        position, velocity = float(data.qpos[0]), float(data.qvel[0])
        target = float(schedule["applied"][step])
        if discrete and t + 1e-10 >= next_tick:
            max_tick_error = max(max_tick_error, max(0.0, float(t) - next_tick))
            requested = kp * (target - position) - kd * velocity
            pending.append((float(t) + timing["feedback_delay_s"], requested))
            tick_times.append(float(t))
            tick_count += 1
            next_tick = tick_count * period
        if discrete:
            while pending and pending[0][0] <= t + 1e-10:
                due, held_torque = pending.popleft()
                max_delay_error = max(max_delay_error, max(0.0, float(t) - due))
                torque_updates += 1
            raw_torque = held_torque
            data.ctrl[0] = held_torque
        else:
            raw_torque = kp * (target - position) - kd * velocity
            data.ctrl[0] = target
        limit = min(fixed_limit, torque_limit(envelope, velocity)) if envelope else fixed_limit
        model.actuator_forcerange[0] = [-limit, limit]
        low, high = -limit, limit
        if model.jnt_actfrclimited[0]:
            low = max(low, float(model.jnt_actfrcrange[0, 0]))
            high = min(high, float(model.jnt_actfrcrange[0, 1]))
        if low > high:
            raise ValueError(f"{name} 的关节及执行器力矩限幅没有交集")
        mujoco.mj_step(model, data)
        actual_torque = float(data.qfrc_actuator[0])
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or not np.isfinite(data.qacc).all() or not math.isfinite(actual_torque)
                or any(data.warning[w].number for w in bad_warnings)
                or data.time < (step + 1) * dt - dt * 0.5):
            finite = False
            reasons.append("动作跟踪出现非有限状态、MuJoCo 数值警告或内部重置")
            break
        peak_velocity = max(peak_velocity, abs(velocity), abs(float(data.qvel[0])))
        for key, value in (
            ("time", float(t)), ("raw_target", float(schedule["raw"][step])),
            ("intended_target", float(schedule["intended"][step])), ("applied_target", target),
            ("position", position), ("velocity", velocity),
            ("torque", actual_torque), ("unclamped_torque", raw_torque),
        ):
            arrays[key].append(value)
        if schedule["active"][step]:
            errors.append(float(schedule["intended"][step]) - position)
            torques.append(actual_torque)
            raw_torques.append(raw_torque)
            if high - low < 1e-12:
                saturated = abs(raw_torque - high) > 1e-9
            else:
                saturated = (actual_torque <= low + abs(low) * 1e-6
                             or actual_torque >= high - abs(high) * 1e-6)
            saturation.append(saturated)
        if abs(data.qvel[0]) > 1e4 or not row["action_clip"][0] - 0.25 <= data.qpos[0] <= row["action_clip"][1] + 0.25:
            reasons.append("动作跟踪超出诊断角度或速度数值边界")
            break
    if not errors:
        raise ValueError("动作跟踪未获得周期运动窗口数据")
    error = np.asarray(errors)
    rmse = float(np.sqrt(np.mean(error**2)))
    ratio = rmse / amplitude
    saturation_fraction = float(np.mean(saturation))
    torque_rms = float(np.sqrt(np.mean(np.square(torques))))
    continuous = row.get("continuous_torque")
    continuous_ratio = torque_rms / float(continuous) if continuous is not None else None
    continuous_ok = continuous_ratio <= 1.0 if continuous_ratio is not None else None
    speed_limit = row.get("velocity_limit")
    speed_ok = peak_velocity <= float(speed_limit) if speed_limit is not None else None
    completed_steps = len(arrays["time"])
    complete = completed_steps == len(schedule["time"])
    if ratio > settings["max_tracking_error_ratio"]:
        reasons.append(f"周期跟踪 RMS 误差/请求幅值 {ratio:.4g} 超过 {settings['max_tracking_error_ratio']:.4g}")
    if saturation_fraction > MAX_SATURATION_FRACTION:
        reasons.append(f"周期力矩饱和比例 {saturation_fraction:.1%} 超过 {MAX_SATURATION_FRACTION:.0%}")
    if continuous_ok is False:
        reasons.append(f"周期力矩 RMS {torque_rms:.6g} Nm 超过已知连续力矩 {continuous:.6g} Nm")
    if speed_ok is False:
        reasons.append(f"峰值速度 {peak_velocity:.6g} rad/s 超过填写的 {speed_limit:.6g} rad/s")
    if not complete and not reasons:
        reasons.append("未完成全部动作跟踪物理步")
    metrics = {
        "finite": bool(finite), "completed": bool(complete),
        "requested_amplitude_rad": amplitude, "range_passed": schedule["enough_range"],
        "available_minus_rad": schedule["available_minus"], "available_plus_rad": schedule["available_plus"],
        "tracking_rmse_rad": rmse, "tracking_error_ratio": ratio,
        "tracking_error_p95_rad": float(np.percentile(np.abs(error), 95)),
        "max_tracking_error_rad": float(np.max(np.abs(error))),
        "saturation_fraction": saturation_fraction,
        "peak_torque_nm": float(np.max(np.abs(torques))),
        "peak_unclamped_torque_nm": float(np.max(np.abs(raw_torques))),
        "torque_rms_nm": torque_rms, "peak_velocity_rad_s": peak_velocity,
        "continuous_torque_nm": continuous, "continuous_torque_ratio": continuous_ratio,
        "continuous_torque_passed": continuous_ok, "velocity_limit_rad_s": speed_limit,
        "velocity_limit_passed": speed_ok,
        "evaluated_motion_samples": len(errors), "completed_steps": completed_steps,
        "target_update_count": math.ceil(completed_steps / timing["decimation"]),
        "sampled_training_lag_steps": [int(schedule["lags"][:completed_steps].min()),
                                       int(schedule["lags"][:completed_steps].max())],
        "training_lag_update_count": int(schedule["lag_updates"][:completed_steps].sum()),
        "max_target_delay_difference_rad": float(np.max(np.abs(schedule["applied"][:completed_steps]
                                                                 - schedule["intended"][:completed_steps]))),
        "hardware_pd_tick_count": tick_count, "hardware_torque_update_count": torque_updates,
        "hardware_pd_interval_s_range": ([float(np.min(np.diff(tick_times))), float(np.max(np.diff(tick_times)))]
                                         if len(tick_times) > 1 else None),
        "max_hardware_tick_quantization_s": max_tick_error,
        "max_hardware_delay_quantization_s": max_delay_error,
        "torque_speed_envelope_applied": bool(envelope),
    }
    passed = bool(finite and complete and schedule["enough_range"] and not reasons)
    return {
        "name": name, "leg": binding["leg"], "role": binding["role"],
        "kp": kp, "kd": float(kd), "status": "passed" if passed else "failed",
        "passed": passed, "metrics": metrics, "reasons": reasons, "trace": _trace(arrays),
    }


def _candidate(candidate, bindings, rows, reduced_models, control, settings, timing, envelopes):
    # A candidate owns all mutable model gains/limits; every joint has its own
    # independent model as well. Worker scheduling cannot alter another run.
    models = deepcopy(reduced_models)
    joints = []
    for index, (row, binding, model) in enumerate(zip(rows, bindings["joints"], models)):
        try:
            result = _joint(model, row, binding, index, candidate, control, settings,
                            timing, envelopes.get(row["name"]))
        except Exception as exc:
            result = {"name": row["name"], "leg": binding["leg"], "role": binding["role"],
                      "status": "failed", "passed": False, "metrics": {"finite": False},
                      "reasons": [f"该关节跟踪测试失败：{type(exc).__name__}: {exc}"]}
        joints.append(result)
    valid = [item["metrics"] for item in joints if "tracking_error_ratio" in item["metrics"]]
    reasons = [f"{item['name']}：{reason}" for item in joints for reason in item["reasons"]]
    passed = bool(joints and all(item["passed"] for item in joints))
    metrics = {
        "passed_joints": sum(item["passed"] for item in joints), "total_joints": len(joints),
        "finite": all(item["metrics"].get("finite") is True for item in joints),
        "range_passed": all(item["metrics"].get("range_passed") is True for item in joints),
        "max_tracking_error_ratio": max((m["tracking_error_ratio"] for m in valid), default=None),
        "max_tracking_rmse_rad": max((m["tracking_rmse_rad"] for m in valid), default=None),
        "max_tracking_error_rad": max((m["max_tracking_error_rad"] for m in valid), default=None),
        "max_saturation_fraction": max((m["saturation_fraction"] for m in valid), default=None),
        "max_peak_torque_nm": max((m["peak_torque_nm"] for m in valid), default=None),
        "max_peak_unclamped_torque_nm": max((m["peak_unclamped_torque_nm"] for m in valid), default=None),
        "max_torque_rms_nm": max((m["torque_rms_nm"] for m in valid), default=None),
        "max_peak_velocity_rad_s": max((m["peak_velocity_rad_s"] for m in valid), default=None),
        "continuous_torque_passed": (all(m["continuous_torque_passed"] is not False for m in valid)
                                     if any(m["continuous_torque_passed"] is not None for m in valid) else None),
        "known_continuous_torque_joints": sum(m["continuous_torque_nm"] is not None for m in valid),
        "velocity_limit_passed": (all(m["velocity_limit_passed"] is not False for m in valid)
                                 if any(m["velocity_limit_passed"] is not None for m in valid) else None),
    }
    return {"status": "passed" if passed else "failed", "passed": passed,
            "metrics": metrics, "joints": joints, "reasons": reasons}


def evaluate_candidates(analysis, bindings, parameters, candidates, reduced_models,
                        progress=print, *, envelopes=None):
    """Evaluate complete scaled action motion, preserving the caller's inputs."""
    started = time.monotonic()
    settings = deepcopy(analysis["settings"])
    settings["tracking_frequency_hz"] = _positive(settings.get("tracking_frequency_hz", 0.5), "tracking_frequency_hz")
    settings["max_tracking_error_ratio"] = _positive(settings.get("max_tracking_error_ratio", 0.2), "max_tracking_error_ratio")
    settings["action_amplitude"] = _positive(settings.get("action_amplitude", 1.0), "action_amplitude")
    timing = resolve_timing(parameters, settings)
    rows = analysis["joints"]
    if not candidates or len({item["id"] for item in candidates}) != len(candidates):
        raise ValueError("动作跟踪需要至少一个 ID 唯一的候选")
    if not len(rows) == len(bindings["joints"]) == len(reduced_models) or any(
        row["name"] != binding["name"] for row, binding in zip(rows, bindings["joints"])):
        raise ValueError("动作跟踪的模型、分析与绑定关节顺序不一致")
    duration = HOLD_S * 2 + CYCLES / settings["tracking_frequency_hz"]
    planned = validate_budget(parameters, settings, len(candidates), len(rows))
    workers = _workers(len(candidates))
    progress(f"动作跟踪：{len(candidates)} 组，最多 {workers} 组并行；"
             f"完整动作幅值 × {settings['action_amplitude']:g}，{settings['tracking_frequency_hz']:g} Hz 正弦两周期。")
    completed = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pd-tracking") as pool:
        pending = {pool.submit(_candidate, candidate, bindings, rows, reduced_models,
                               parameters["control"], settings, timing, envelopes or {}): candidate["id"]
                   for candidate in candidates}
        for future in as_completed(pending):
            identifier = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "passed": False, "metrics": {"finite": False},
                          "joints": [], "reasons": [f"该候选跟踪测试失败：{type(exc).__name__}: {exc}"]}
            completed[identifier] = result
            progress(f"    跟踪 [{len(completed)}/{len(candidates)}] {identifier}："
                     f"{result['metrics'].get('passed_joints', 0)}/{len(rows)} 关节通过。")
    conditions = {
        "seed": SEED, "mode": "independent_locked_coordinate_action_tracking",
        "timing": timing, "frequency_hz": settings["tracking_frequency_hz"],
        "normalized_action_amplitude": settings["action_amplitude"], "cycles": CYCLES,
        "initial_hold_s": HOLD_S, "final_hold_s": HOLD_S, "duration_s": duration,
        "ramp": "none; sine starts and ends at the initial position",
        "waveform": "initial + effective_action_scale * action_amplitude * sin(2*pi*f*t)",
        "scale_source": "analysis.joints.action_scale already includes hip_reduction exactly once",
        "target_schedule": "policy tick sample and zero-order hold, then physics-step actuator lag/hold/update-period history",
        "error_reference": "current intended policy-held target before actuator delay; full physics resolution during active cycles only",
        "force_limit": "configured actuator limit intersected with imported joint force limit and confirmed symmetric torque-speed envelope",
        "unclamped_torque": "native pre-integration proportional/velocity demand, or hardware sampled/delayed held torque before force clipping",
        "pass_criteria": {
            "finite_and_complete": True, "both_directions_fit_requested_amplitude": True,
            "max_tracking_error_ratio": settings["max_tracking_error_ratio"],
            "max_saturation_fraction": MAX_SATURATION_FRACTION,
            "known_velocity_limit": "peak over all physical steps <= supplied limit",
            "known_continuous_torque": "active-cycle torque RMS <= supplied continuous torque",
        },
        "planned_physics_steps": planned,
        "limitations": ["每个关节独立、基座及其他关节固定，关闭重力和接触；完整机器人另测。",
                        "有限频率及幅值的仿真筛查，不是策略性能、反馈稳定裕度或实机热模型验证。",
                        "未提供连续力矩或速度限值时对应结论为未知，不标作已验证。",
                        "填写硬件频率/延迟时使用理想离散采样与延迟，不含未建模的滤波或通信抖动。"],
    }
    return {
        "conditions": conditions,
        "results": {candidate["id"]: completed[candidate["id"]] for candidate in candidates},
        "parallel": {"workers": workers, "max_workers": 4, "candidate_count": len(candidates),
                     "elapsed_s": time.monotonic() - started},
    }
