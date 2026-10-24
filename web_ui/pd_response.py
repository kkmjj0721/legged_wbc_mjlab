"""Budgeted CPU joint-response experiments for the local migration workbench.

Every experiment retains the imported robot's rigid-body inertias and topology,
but freezes the base and all joints except one at the configured initial pose.
This is a controlled simulation experiment, not hardware identification or a
standing/contact stability test. Nothing here changes the model or draft.
"""

from copy import deepcopy
import argparse
import json
import math
from pathlib import Path
import time

from . import pd_analysis
from .template_runtime import joint_value


DURATION_S = 2.0
STEP_RAD = 0.05
MIN_STEP_RAD = 0.005
MAX_TRACE_POINTS = 161
MAX_RESPONSE_STEPS = 20000
SEED = 42
FREQUENCIES_HZ = (3.0, 5.0, 8.0)
DAMPING_RATIOS = (0.7, 1.0, 1.3)


def _reduced_model(full_model, full_data, binding, row, control, sim_dt, mujoco_options=None):
    """Bake nominal transforms, retaining exactly the tested hinge DOF.

    Native MuJoCo affine position actuators preserve the selected integration
    method and force clamp of the generated BuiltinPositionActuatorCfg.
    Resetting a full model's other qpos after stepping would not reproduce the
    locked-coordinate mass matrix; the actual reduced topology is used instead.
    """
    import mujoco
    import numpy as np
    from mjlab.sim import MujocoCfg

    coupled = [label for label, count in (
        ("tendon", full_model.ntendon), ("equality", full_model.neq),
        ("flex", full_model.nflex), ("plugin", full_model.nplugin),
    ) if count]
    if coupled:
        raise ValueError("单关节响应测试暂不支持模型中的耦合定义：" + ", ".join(coupled))
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.timestep = sim_dt
    active = full_model.joint(binding["name"])
    joint_id, body_id = active.id, int(active.bodyid[0])
    dof, qadr = int(active.dofadr[0]), int(active.qposadr[0])
    rotations = full_data.xmat.reshape((-1, 3, 3))
    bodies = {0: spec.worldbody}
    for index in range(1, full_model.nbody):
        parent = int(full_model.body_parentid[index])
        rotation = rotations[parent].T @ rotations[index]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, rotation.ravel())
        position = rotations[parent].T @ (full_data.xpos[index] - full_data.xpos[parent])
        body = bodies[parent].add_body(
            name=full_model.body(index).name or f"pd_body_{index}",
            pos=position, quat=quat,
            mass=float(full_model.body_mass[index]),
            ipos=full_model.body_ipos[index], iquat=full_model.body_iquat[index],
            inertia=full_model.body_inertia[index], explicitinertial=True,
        )
        bodies[index] = body
        if index != body_id:
            continue
        axis = rotations[index].T @ full_data.xaxis[joint_id]
        anchor = rotations[index].T @ (full_data.xanchor[joint_id] - full_data.xpos[index])
        friction = joint_value(control.get("friction"), binding)
        friction = float(full_model.dof_frictionloss[dof]) if friction is None else float(friction)
        body.add_joint(
            name=binding["name"], type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=anchor, axis=axis, ref=row["initial"],
            limited=True, range=full_model.jnt_range[joint_id],
            actfrclimited=bool(full_model.jnt_actfrclimited[joint_id]),
            actfrcrange=full_model.jnt_actfrcrange[joint_id],
            margin=float(full_model.jnt_margin[joint_id]),
            solref_limit=full_model.jnt_solref[joint_id],
            solimp_limit=full_model.jnt_solimp[joint_id],
            stiffness=float(full_model.jnt_stiffness[joint_id]),
            springref=float(full_model.qpos_spring[qadr]),
            armature=row["armature"], damping=row["passive_damping"],
            frictionloss=friction,
            solref_friction=full_model.dof_solref[dof],
            solimp_friction=full_model.dof_solimp[dof],
        )
    spec.add_actuator(
        name="pd_position", target=binding["name"],
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        dyntype=mujoco.mjtDyn.mjDYN_NONE,
        gaintype=mujoco.mjtGain.mjGAIN_FIXED, gainprm=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biastype=mujoco.mjtBias.mjBIAS_AFFINE, biasprm=[0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        ctrllimited=False, inheritrange=0,
        forcelimited=True, forcerange=[-row["effort_limit"], row["effort_limit"]],
    )
    model = spec.compile()
    # Match task runtime option application, including flags present in source
    # XML.  Only gravity/contact differ deliberately in this isolated experiment.
    model.opt.disableflags = full_model.opt.disableflags
    model.opt.enableflags = full_model.opt.enableflags
    MujocoCfg(timestep=sim_dt, **(mujoco_options or {})).apply(model)
    model.opt.gravity[:] = 0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if model.nq != 1 or model.nv != 1 or model.nu != 1:
        raise ValueError(f"{binding['name']} 的单关节模型自由度不符合预期")
    data = mujoco.MjData(model)
    data.qpos[0] = row["initial"]
    mujoco.mj_forward(model, data)
    inertia = np.empty((1, 1))
    mujoco.mj_fullM(model, data, inertia)
    if not math.isclose(float(inertia[0, 0]), row["inertia"]["nominal"], rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError(f"{binding['name']} 的约化惯量与完整模型锁定惯量不一致，已停止测试")
    return model


def _effective_options(model):
    """JSON description of the actual CPU model, after experiment overrides."""
    import mujoco

    result = {}
    for field, enum, prefix in (("integrator", mujoco.mjtIntegrator, "mjINT_"),
                                ("cone", mujoco.mjtCone, "mjCONE_"),
                                ("jacobian", mujoco.mjtJacobian, "mjJAC_"),
                                ("solver", mujoco.mjtSolver, "mjSOL_")):
        result[field] = enum(int(getattr(model.opt, field))).name.removeprefix(prefix).lower()
    for field in ("timestep", "impratio", "iterations", "tolerance", "ls_iterations",
                  "ls_tolerance", "ccd_iterations"):
        result[field] = getattr(model.opt, field)
    result["gravity"] = model.opt.gravity.tolist()
    for field, enum, prefix in (("disableflags", mujoco.mjtDisableBit, "mjDSBL_"),
                                ("enableflags", mujoco.mjtEnableBit, "mjENBL_")):
        mask = int(getattr(model.opt, field))
        result[field] = [name.removeprefix(prefix).lower() for name in dir(enum)
                         if name.startswith(prefix) and mask & int(getattr(enum, name))]
    return result


def _delay_targets(control, steps, sim_dt, joint_index, initial, target):
    """Use a fixed seeded target-delay sequence shared by every candidate.

    This follows DelayBuffer's lag/hold/update-period semantics. NumPy supplies
    reproducible draws here; the draws are not bitwise Torch training RNG draws.
    An initial hold creates real pre-step history, rather than backfilling the
    first step command and unintentionally erasing its delay.
    """
    import numpy as np

    low, high = (int(control.get(f"delay_{side}_lag", 0)) for side in ("min", "max"))
    period = int(control.get("delay_update_period", 0))
    hold = float(control.get("delay_hold_prob", 0))
    rng = np.random.default_rng(SEED + joint_index)
    phase = int(rng.integers(period)) if period else 0
    lag = 0
    prehold = max(high + 1, int(math.ceil(0.1 / sim_dt)))
    for count in range(prehold):
        candidate = int(rng.integers(low, high + 1))
        take = bool(rng.random() >= hold) if hold > 0 else True
        if (not period or (count + phase) % period == 0) and take:
            lag = candidate
    targets = np.empty(steps)
    for index in range(steps):
        count = prehold + index
        candidate = int(rng.integers(low, high + 1))
        take = bool(rng.random() >= hold) if hold > 0 else True
        if (not period or (count + phase) % period == 0) and take:
            lag = candidate
        targets[index] = target if index - lag >= 0 else initial
    return targets


def _metrics(times, positions, velocities, torques, initial, target, effort_limit,
             velocity_limit=None, *, stable=True, force_range=None, force_limits=None):
    """Compute metrics at full physics resolution; 2% settling needs 0.1 s hold."""
    import numpy as np

    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    torques = np.asarray(torques, dtype=float)
    if not all(np.isfinite(array).all() for array in (times, positions, velocities, torques)):
        raise ValueError("响应曲线出现非有限值")
    step = target - initial
    if abs(step) < MIN_STEP_RAD - 1e-12:
        raise ValueError("可用关节阶跃幅值过小，无法进行响应测试")
    tolerance = max(abs(step) * 0.02, 0.0001)
    outside = np.flatnonzero(np.abs(positions - target) > tolerance)
    settle_index = int(outside[-1] + 1) if len(outside) else 0
    settling = (float(times[settle_index]) if settle_index < len(times)
                and times[-1] - times[settle_index] >= 0.1 - 1e-10 else None)
    progress = (positions - initial) / step
    overshoot = max(0.0, float(np.max(progress)) - 1.0) * 100
    force_low, force_high = force_range or (-effort_limit, effort_limit)
    if force_limits is not None:
        bounds = np.asarray(force_limits, dtype=float)
        low, high = bounds[:, 0], bounds[:, 1]
    else:
        low, high = force_low, force_high
    saturation = float(np.mean((torques[:-1] <= low + np.abs(low) * 1e-6)
                               | (torques[:-1] >= high - np.abs(high) * 1e-6)))
    peak_velocity = float(np.max(np.abs(velocities)))
    velocity_ok = velocity_limit is None or peak_velocity <= velocity_limit
    result = {
        "settling_time_s": settling, "overshoot_pct": overshoot,
        "final_error_rad": abs(float(positions[-1]) - target),
        "peak_torque_nm": float(np.max(np.abs(torques))),
        "peak_velocity_rad_s": peak_velocity,
        "saturation_fraction": saturation, "settling_tolerance_rad": tolerance,
        "stable": bool(stable), "velocity_limit_passed": bool(velocity_ok),
        "passed": bool(stable and settling is not None and overshoot <= 20
                       and saturation <= 0.1 and velocity_ok),
    }
    return result


def _trace(times, positions, targets, torques):
    import numpy as np

    # Include extrema in addition to regular samples. Metrics use every step.
    selected = set(np.linspace(0, len(times) - 1, MAX_TRACE_POINTS - 6, dtype=int).tolist())
    selected.update((0, len(times) - 1, int(np.argmax(positions)), int(np.argmin(positions)),
                     int(np.argmax(torques)), int(np.argmin(torques))))
    indices = sorted(selected)
    return {name: np.asarray(values)[indices].tolist() for name, values in (
        ("time", times), ("position", positions), ("target", targets), ("torque", torques),
    )}


def _simulate(model, row, control, joint_index, direction, kp, kd, duration_s=DURATION_S,
              envelope=None):
    import mujoco
    import numpy as np
    if envelope is not None:
        from .motor_curves import torque_limit

    initial, low, high = row["initial"], *row["action_clip"]
    target = max(low, min(high, initial + direction * STEP_RAD))
    step = target - initial
    if abs(step) < MIN_STEP_RAD - 1e-12:
        return None
    sim_dt = float(model.opt.timestep)
    count = int(math.ceil(duration_s / sim_dt))
    if count > MAX_RESPONSE_STEPS:
        raise ValueError(f"sim_dt 过小：每条 2 秒响应最多允许 {MAX_RESPONSE_STEPS} 个物理步")
    delayed = _delay_targets(control, count, sim_dt, joint_index, initial, target)
    model.actuator_gainprm[0, 0] = kp
    model.actuator_biasprm[0, 1:3] = [-kp, -kd]
    data = mujoco.MjData(model)
    data.qpos[0] = initial
    times, positions, velocities, torques = [0.0], [initial], [0.0], []
    stable = True
    reasons = []
    force_limits = []
    model.actuator_forcerange[0] = [-row["effort_limit"], row["effort_limit"]]
    bad_warnings = tuple(int(value) for value in (
        mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
        mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
    ))
    for index in range(count):
        data.ctrl[0] = delayed[index]
        limit = min(row["effort_limit"], torque_limit(envelope, float(data.qvel[0]))) if envelope else row["effort_limit"]
        model.actuator_forcerange[0] = [-limit, limit]
        mujoco.mj_step(model, data)
        # qfrc_actuator belongs to the pre-integration state at t[index], and
        # includes any imported joint-level force clamp after actuator clamping.
        torque = float(data.qfrc_actuator[0])
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
                or not math.isfinite(torque)
                or any(data.warning[warning].number for warning in bad_warnings)):
            stable = False
            reasons.append("MuJoCo 报告了数值异常；曲线止于最后一个有限状态")
            break
        torques.append(torque)
        effective_range = [-limit, limit]
        if model.jnt_actfrclimited[0]:
            joint_low, joint_high = map(float, model.jnt_actfrcrange[0])
            effective_range = [max(joint_low, min(joint_high, value)) for value in effective_range]
        force_limits.append(effective_range)
        times.append((index + 1) * sim_dt)
        positions.append(float(data.qpos[0]))
        velocities.append(float(data.qvel[0]))
        if abs(data.qvel[0]) > 1e4 or not low - 0.25 <= data.qpos[0] <= high + 0.25:
            stable = False
            reasons.append("关节速度或角度超出本次诊断的数值边界")
            break
    # MuJoCo may reset bad states internally. Keep a failed curve's final force
    # consistent with its last recorded finite position and velocity.
    data.qpos[0], data.qvel[0] = positions[-1], velocities[-1]
    data.ctrl[0] = delayed[min(len(positions) - 1, count - 1)]
    mujoco.mj_forward(model, data)
    final_torque = float(data.qfrc_actuator[0])
    torques.append(final_torque if math.isfinite(final_torque) else 0.0)
    # A failed first step must still return a finite, clearly failed report.
    if len(times) == 1:
        times.append(sim_dt)
        positions.append(initial)
        velocities.append(0.0)
        torques.append(0.0)
        force_limits.append([-row["effort_limit"], row["effort_limit"]])
    force_range = [-row["effort_limit"], row["effort_limit"]]
    if model.jnt_actfrclimited[0]:
        joint_low, joint_high = map(float, model.jnt_actfrcrange[0])
        force_range = [max(joint_low, min(joint_high, value)) for value in force_range]
    metrics = _metrics(times, positions, velocities, torques, initial, target,
                       row["effort_limit"], row.get("velocity_limit"), stable=stable,
                       force_range=force_range, force_limits=force_limits if envelope else None)
    return {
        "direction": direction, "step_rad": step, "initial_rad": initial,
        "target_rad": target, "metrics": metrics, "failure_reasons": reasons,
        "effective_force_range_nm": force_range,
        "torque_speed_envelope_applied": bool(envelope),
        "trace": _trace(times, positions, [target] * len(times), torques),
    }


def _joint_metrics(directions):
    metrics = [direction["metrics"] for direction in directions]
    settling = [item["settling_time_s"] for item in metrics]
    return {
        "settling_time_s": max(settling) if all(value is not None for value in settling) else None,
        **{field: max(item[field] for item in metrics) for field in (
            "overshoot_pct", "final_error_rad", "peak_torque_nm", "peak_velocity_rad_s",
            "saturation_fraction", "settling_tolerance_rad",
        )},
        "stable": all(item["stable"] for item in metrics),
        "velocity_limit_passed": all(item["velocity_limit_passed"] for item in metrics),
        "passed": all(item["passed"] for item in metrics),
    }


def _candidate_proposals(analysis):
    rows = analysis["joints"]
    by_name = {row["name"]: row for row in rows}
    current = {"control": {field: {} for field in ("stiffness", "damping", "armature")}}
    for row in rows:
        for field, value in (("stiffness", row["current"]["kp"]),
                             ("damping", row["current"]["kd"]), ("armature", row["armature"])):
            current["control"][field][row["name"]] = value
    reference = analysis.get("pd_reference", {})
    candidates = [{"id": "current", "label": reference.get("label", "已配置 PD"), "frequency_hz": None,
                   "damping_ratio": None, "proposal": current}]
    for frequency in FREQUENCIES_HZ:
        for zeta in DAMPING_RATIOS:
            omega = frequency * 2 * math.pi
            proposal = {"control": {field: {} for field in ("stiffness", "damping", "armature")}}
            for row in rows:
                group = [by_name[name] for name in row["candidates"]["model"]["gain_group"]]
                inertia = max(item["inertia"]["max"] for item in group)
                proposal["control"]["stiffness"][row["name"]] = inertia * omega**2
                proposal["control"]["damping"][row["name"]] = max(
                    max(0.0, 2 * zeta * inertia * omega - item["passive_damping"]) for item in group)
                proposal["control"]["armature"][row["name"]] = row["armature"]
            candidates.append({
                "id": f"f{frequency:g}_z{zeta:g}".replace(".", "p"),
                "label": f"{frequency:g} Hz / ζ={zeta:g}",
                "frequency_hz": frequency, "damping_ratio": zeta, "proposal": proposal,
            })
    return candidates


def test_response(spec: dict, scope="common", options=None, progress=print, video_dir=None) -> dict:
    """Find gains from a candidate bank, without requiring or applying prior PD."""
    import mujoco

    started = time.monotonic()
    progress("读取模型、有效参数与采样惯量；准备 CPU 单关节响应测试……")
    analysis = pd_analysis.analyze(spec, scope=scope, options=options)
    from .motor_curves import compile_envelopes
    envelopes = compile_envelopes(spec, [row["name"] for row in analysis["joints"]])
    _, bindings, full_model, parameters = pd_analysis._parameters(spec, scope)
    control, initial = parameters["control"], parameters["init_state"]
    sim_dt = float(parameters["environment"]["sim_dt"])
    if math.ceil(DURATION_S / sim_dt) > MAX_RESPONSE_STEPS:
        raise ValueError(f"sim_dt 过小：每条 2 秒响应最多允许 {MAX_RESPONSE_STEPS} 个物理步")
    if control.get("delay_max_lag", 0) > MAX_RESPONSE_STEPS:
        raise ValueError("执行器延迟超过本次有限响应测试的历史预算")
    data = mujoco.MjData(full_model)
    free_id = next(i for i in range(full_model.njnt)
                   if full_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    free_qadr = int(full_model.jnt_qposadr[free_id])
    data.qpos[free_qadr:free_qadr + 3] = initial["pos"]
    data.qpos[free_qadr + 3:free_qadr + 7] = initial["rot"]
    for binding, row in zip(bindings["joints"], analysis["joints"]):
        source_dof = int(full_model.joint(binding["name"]).dofadr[0])
        full_model.dof_armature[source_dof] = row["armature"]
        data.qpos[int(full_model.joint(binding["name"]).qposadr[0])] = row["initial"]
    mujoco.mj_forward(full_model, data)
    requested_options = parameters["environment"].get("mujoco", {})
    reduced = [_reduced_model(full_model, data, binding, row, control, sim_dt, requested_options)
               for binding, row in zip(bindings["joints"], analysis["joints"])]
    effective_options = _effective_options(reduced[0])
    candidates = _candidate_proposals(analysis)
    warnings = []
    for index, row in enumerate(analysis["joints"]):
        available = [min(STEP_RAD, max(0.0, side * (row["action_clip"][0 if side < 0 else 1]
                                                  - row["initial"]))) for side in (-1, 1)]
        if max(available) < MIN_STEP_RAD - 1e-12:
            raise ValueError(f"{row['name']} 两侧可用阶跃均小于 {MIN_STEP_RAD:g} rad，请调整站姿或动作限位")
        if min(available) < MIN_STEP_RAD - 1e-12:
            warnings.append({"code": "one_sided_response", "joint": row["name"],
                             "message": f"{row['name']} 靠近限位，仅测试有至少 {MIN_STEP_RAD:g} rad 余量的一侧。"})
    for candidate_index, candidate in enumerate(candidates):
        progress(f"[{candidate_index + 1}/{len(candidates)}] {candidate['label']}：测试 12 个关节的双向阶跃……")
        joints = []
        for joint_index, (binding, row, model) in enumerate(zip(bindings["joints"], analysis["joints"], reduced)):
            name = row["name"]
            kp = candidate["proposal"]["control"]["stiffness"][name]
            kd = candidate["proposal"]["control"]["damping"][name]
            directions = [value for direction in (-1, 1)
                          if (value := _simulate(model, row, control, joint_index, direction, kp, kd,
                                                 envelope=envelopes.get(name))) is not None]
            metrics = _joint_metrics(directions)
            worst = max(directions, key=lambda item: (
                not item["metrics"]["passed"], item["metrics"]["settling_time_s"] or DURATION_S * 3,
                item["metrics"]["overshoot_pct"], item["metrics"]["final_error_rad"]))
            joints.append({
                "name": name, "role": binding["role"], "leg": binding["leg"],
                "kp": kp, "kd": kd, "initial_rad": row["initial"],
                "step_rad": max(abs(item["step_rad"]) for item in directions),
                "metrics": metrics, "directions": directions, "trace": worst["trace"],
            })
        settling = [joint["metrics"]["settling_time_s"] for joint in joints]
        summary = {
            "passed_joints": sum(joint["metrics"]["passed"] for joint in joints),
            "total_joints": len(joints),
            "worst_settling_time_s": max(settling) if all(value is not None for value in settling) else None,
            "max_overshoot_pct": max(joint["metrics"]["overshoot_pct"] for joint in joints),
            "max_saturation_fraction": max(joint["metrics"]["saturation_fraction"] for joint in joints),
            "max_final_error_rad": max(joint["metrics"]["final_error_rad"] for joint in joints),
            "eligible": all(joint["metrics"]["passed"] for joint in joints),
        }
        summary["score"] = ((summary["worst_settling_time_s"] if summary["worst_settling_time_s"] is not None
                              else DURATION_S * 3) + summary["max_overshoot_pct"] / 100
                             + summary["max_saturation_fraction"] * 2
                             + summary["max_final_error_rad"] / STEP_RAD)
        candidate.update(summary=summary, joints=joints)
        progress(f"    通过 {summary['passed_joints']}/12；最大过冲 {summary['max_overshoot_pct']:.2f}%，"
                 f"最大饱和占比 {summary['max_saturation_fraction'] * 100:.2f}%")
    # The unconfigured reference is a comparison, never the answer to the gain
    # search.  An explicitly configured PD remains eligible for comparison.
    reference_only = analysis.get("pd_reference", {}).get("reference_only", False)
    eligible = [candidate for candidate in candidates if candidate["summary"]["eligible"]
                and not (reference_only and candidate["id"] == "current")]
    best = min(eligible, key=lambda candidate: candidate["summary"]["score"]) if eligible else None
    if not best:
        warnings.append({"code": "no_eligible_candidate", "message": "本次没有候选通过全部关节响应筛查，未给出推荐。请检查摩擦、限矩、站姿与时序设置，并审阅曲线。"})
    if analysis["settings"]["hardware_pd_hz"] is not None or analysis["settings"]["hardware_delay_ms"] is not None:
        warnings.append({"code": "hardware_not_simulated", "message": "硬件 PD 频率和硬件延迟仅作记录；本次使用当前训练配置的物理步长和执行器目标延迟，未模拟硬件内环。"})
    if any(raw.get("rotor_inertia") is not None for raw in analysis["settings"]["joints"].values()):
        warnings.append({"code": "hardware_armature_assumption", "message": "所有候选（含当前增益）均使用补充的转子惯量×减速比²作为有效 armature；该值是输入假设，尚未实机辨识。"})
    warnings.extend(item for item in analysis["warnings"] if item["code"] in (
        "symmetry_control_mismatch", "individual_symmetry_gains", "unknown_torque_speed_envelope"))
    result = {
        "version": 1, "status": "passed", "scope": scope, "template": analysis["template"],
        "model": deepcopy(analysis["model"]), "settings": deepcopy(analysis["settings"]),
        "gain_grouping": analysis["gain_grouping"],
        "pd_reference": deepcopy(analysis.get("pd_reference", {})),
        "conditions": {
            "seed": SEED, "device": "cpu", "sim_dt": sim_dt, "physics_hz": 1 / sim_dt,
            "duration_s": DURATION_S, "step_rad": STEP_RAD, "minimum_step_rad": MIN_STEP_RAD,
            "gravity": False, "contacts": False, "fixed_base": True,
            "locked_other_joints": True, "integrator": effective_options["integrator"],
            "mujoco_requested": deepcopy(requested_options),
            "mujoco_effective": effective_options,
            "actuator": "MuJoCo native position PD; target-only delay",
            "trace_max_points": MAX_TRACE_POINTS,
            **{field: control.get(field, 0) for field in (
                "delay_min_lag", "delay_max_lag", "delay_hold_prob", "delay_update_period")},
            "decimation": control["decimation"], "observation_delay_included": False,
            "pass_criteria": {"settling_band_fraction": 0.02, "minimum_settling_hold_s": 0.1,
                              "max_overshoot_pct": 20, "max_saturation_fraction": 0.1},
            "tested_poses": "configured initial pose only",
            "torque_speed_envelopes": {name: [list(point) for point in points]
                                       for name, points in envelopes.items()},
            "torque_speed_model": "symmetric magnitude envelope, per-physics-step velocity; no voltage/thermal dynamics",
        },
        "candidates": candidates, "recommended_id": best["id"] if best else None,
        "warnings": warnings,
        "method_notes": [
            "这是基于输入模型的受控单关节仿真，未连接或测试实机。基座和其他关节在初始站姿处固定，关闭重力、接触和外部负载。",
            "约化模型保留完整模型的刚体惯量与拓扑，烘焙固定关节变换后只保留被测铰链；逐关节校验其惯量等于完整模型的锁定 Mii。",
            "候选频率 3/5/8 Hz 与阻尼比 0.7/1/1.3 是扫描的设计目标，不是机器人实测固有参数。增益按姿态采样最大惯量计算；响应只在当前初始站姿测量。",
            "每关节测试 ±0.05 rad 阶跃，按动作限位裁剪；不足 0.005 rad 的方向跳过并给出提示。每条曲线固定 2 秒，指标使用完整物理步数据。",
            f"保留有效 armature、被动阻尼、干摩擦、关节弹簧、关节限位与仿真限矩。使用原生 MuJoCo PD 及配置选定的 {effective_options['integrator']} 积分；输入转子惯量时，当前增益和候选共用同一惯量假设。",
            "MuJoCo 积分器、求解器、摩擦锥、迭代数、容差及控制相关标志采用任务的有效配置，源 XML 标志与配置标志按任务运行时相同规则合并。仅重力固定为零、接触强制关闭，并固定基座和其他关节；requested/effective 同时记录这些实验覆盖。",
            "仅目标角按物理步延迟；内环位置/速度反馈保持当前值，符合 BuiltinPositionActuator。固定种子 42 的 NumPy 延迟抽样供候选共用，分布语义与框架一致，但不与训练 Torch 随机数逐位对应。",
            "到位时间按目标阶跃 2% 误差带计算，且剩余至少 0.1 秒持续在带内；未到位记为空。通过要求数值有限、到位、过冲≤20%、限矩饱和时间占比≤10%，以及不超过已填速度上限。",
            "推荐仅在所有关节通过的候选中按到位时间、过冲、饱和和残差的启发式分数选择；没有通过者时不推荐。该排名不证明步态、接触稳定性或策略性能。",
            "观测延迟、硬件 PD 更新率与硬件闭环延迟不作为电机内环反馈延迟混入此实验。",
            "仅已确认数值点的 T-N 曲线用于本次测试：每个物理步按关节速度插值可用力矩，并与固定限矩取较小值；正反向采用对称幅值近似。上传图片本身不生成数值约束。未模拟再生制动、电压瞬态和热限制；生成的训练任务仍使用固定限矩。",
            "新机器人无需预先填写 PD；缺失 PD 只在本次比较的深拷贝中使用 20.0/0.5 测试基准，不会自动应用到草稿。需审阅并应用候选或手动填写后才能生成任务。",
        ],
        "elapsed_s": time.monotonic() - started,
    }
    if envelopes:
        result["warnings"] = [item for item in result["warnings"] if item["code"] != "unknown_torque_speed_envelope"]
        if len(envelopes) < len(analysis["joints"]):
            result["warnings"].append({"code": "partial_torque_speed_envelope", "message": "仅部分关节有已确认数值 T-N 曲线；其余关节仍使用固定限矩。"})
    result["videos"] = []
    if video_dir is not None:
        try:
            from .pd_videos import create_comparison_videos
            result["videos"] = create_comparison_videos(full_model, data, bindings, candidates,
                                                         best["id"] if best else None, video_dir, progress)
        except Exception as exc:
            result["warnings"].append({"code": "response_video_unavailable", "message": f"响应数值测试已完成，但视频未生成：{exc}"})
    result["elapsed_s"] = time.monotonic() - started
    json.dumps(result, allow_nan=False)
    progress("响应测试完成。" + (f"本工况候选：{best['label']}；应用前请审阅曲线。" if best else "没有全部通过的候选。"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--video-dir", type=Path)
    args = parser.parse_args()
    artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
    if not args.request.resolve().is_relative_to(artifacts) or not args.result.resolve().is_relative_to(artifacts):
        parser.error("响应测试请求与结果必须位于本项目 .artifacts 内")
    if args.video_dir is not None and not args.video_dir.resolve().is_relative_to(artifacts):
        parser.error("响应视频必须位于本项目 .artifacts 内")
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = test_response(request["spec"], scope=request.get("scope", "common"),
                               options=request.get("options"), progress=lambda value: print(value, flush=True),
                               video_dir=args.video_dir)
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "type": type(exc).__name__}
        print(f"响应测试失败：{exc}", flush=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
