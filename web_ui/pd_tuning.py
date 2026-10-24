"""Explicit motion/torque constraints and ranking for finite PD candidates."""

import math

PROTOCOL = "joint_tracking_ground_v2"
WEIGHTS = {"maximum_effort_ratio": 1.0, "mean_effort_ratio": 0.25,
           "leg_contact_fraction": 0.15, "tracking_error_fraction_of_limit": 0.1}


def effective_effort_limit(row, model=None, envelope=None):
    """Account for both actuator and joint force caps when they are compiled."""
    limit = float(row["effort_limit"])
    if model is not None and model.actuator_forcelimited[0]:
        limit = min(limit, *map(abs, model.actuator_forcerange[0]))
    if model is not None and model.jnt_actfrclimited[0]:
        low, high = map(float, model.jnt_actfrcrange[0])
        limit = min(limit, -low, high)
    if envelope:
        from .motor_curves import torque_limit
        limit = min(limit, torque_limit(envelope, 0.0))
    return max(0.0, limit)


def torque_budget(candidate, analysis, reduced_models, envelopes=None):
    """Use requested action amplitude, actual hip scale and effective force caps.

    This is a zero-speed proportional budget, not a claim about measured policy
    duty cycle. The reserve leaves room for damping and external load.
    """
    settings = analysis["settings"]
    amplitude = settings["action_amplitude"]
    if amplitude <= 0:
        raise ValueError("自动 PD 测试的动作幅值必须大于零；零幅值不能验证动作力矩与跟踪")
    maximum = settings["max_unit_action_effort_ratio"]
    joints, reasons = [], []
    for row, model in zip(analysis["joints"], reduced_models):
        name = row["name"]
        kp = float(candidate["proposal"]["control"]["stiffness"][name])
        kd = float(candidate["proposal"]["control"]["damping"][name])
        limit = effective_effort_limit(row, model, (envelopes or {}).get(name))
        scale = float(row["action_scale"])
        requested = scale * amplitude
        low, high = row["action_clip"]
        minus, plus = row["initial"] - low, high - row["initial"]
        fits = min(minus, plus) >= requested - 1e-10
        demand = kp * requested
        ratio = demand / limit if limit > 0 else None
        joint_reasons = []
        if not fits:
            joint_reasons.append("所请求动作幅值超出确认角限位，不缩幅代测")
        if ratio is None or ratio > maximum + 1e-10:
            joint_reasons.append(f"动作比例力矩预算超过允许限矩占比 {maximum:.0%}")
        continuous = row.get("continuous_torque")
        joints.append({
            "name": name, "stiffness": kp, "damping": kd,
            "action_scale": scale, "action_amplitude": amplitude,
            "requested_amplitude_rad": requested,
            "available_minus_rad": float(minus), "available_plus_rad": float(plus),
            "effort_limit_nm": max(0.0, limit), "unit_action_torque_nm": kp * scale,
            "requested_action_torque_nm": demand, "effort_ratio": ratio,
            "saturation_action": limit / (kp * scale) if kp * scale > 0 and limit > 0 else None,
            "continuous_torque_nm": continuous,
            "continuous_effort_ratio": demand / continuous if continuous else None,
            "motion_range_passed": fits, "passed": not joint_reasons,
            "reasons": joint_reasons,
        })
        reasons.extend(f"{name}: {reason}" for reason in joint_reasons)
    ratios = [joint["effort_ratio"] for joint in joints if joint["effort_ratio"] is not None]
    saturation = [joint["saturation_action"] for joint in joints if joint["saturation_action"] is not None]
    return {"passed": not reasons and len(joints) == 12,
            "max_effort_ratio": max(ratios) if ratios else None,
            "mean_effort_ratio": sum(ratios) / len(ratios) if ratios else None,
            "minimum_saturation_action": min(saturation) if saturation else None,
            "reserve_fraction": 1 - maximum, "action_amplitude": amplitude,
            "limit_ratio": maximum, "joints": joints, "reasons": reasons,
            "condition": "zero velocity and nominal position; proportional demand before clipping; not a thermal or full-policy guarantee"}


def rank_candidate(candidate, settings):
    """Rank only admissible candidates; faster settling earns no extra reward."""
    budget = candidate["torque_budget"]
    ground = candidate["ground"]
    tracking = candidate["tracking"]
    metrics = tracking.get("metrics", {})
    error = metrics.get("max_tracking_error_ratio", metrics.get("tracking_error_ratio"))
    if (error is None or budget["max_effort_ratio"] is None
            or budget["mean_effort_ratio"] is None):
        return {"cost": None, "components": {},
                "reason": "缺少有效跟踪或力矩预算指标，不能参与排序"}
    trials = ground.get("trials", [])
    duration = sum(trial.get("metrics", {}).get("completed_s", 0) for trial in trials)
    leg_time = sum(trial.get("metrics", {}).get("leg_contact_time_s", 0) for trial in trials)
    components = {"maximum_effort_ratio": float(budget["max_effort_ratio"]),
                  "mean_effort_ratio": float(budget["mean_effort_ratio"]),
                  "leg_contact_fraction": leg_time / duration if duration else 0,
                  "tracking_error_fraction_of_limit": float(error) / settings["max_tracking_error_ratio"]}
    if not all(math.isfinite(value) for value in components.values()):
        raise ValueError("候选排序出现非有限指标")
    return {"cost": sum(WEIGHTS[key] * value for key, value in components.items()),
            "components": components}
