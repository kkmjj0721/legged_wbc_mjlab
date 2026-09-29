"""Stage-aware checkpoint ranking and conservative convergence diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re

import numpy as np

from .training_logs import LENGTH, REWARD, Run, Segment


POLICY_VERSION = "stage_reward_median_mad_v1"
LINEAR = "Episode_Reward/track_linear_velocity"
ANGULAR = "Episode_Reward/track_angular_velocity"
TERRAIN = "Curriculum/terrain_levels"
CLIP = "Loss/numerics/action_clip_fraction"


@dataclass(frozen=True)
class Goal:
    metric: str
    operator: str
    value: float

    @classmethod
    def parse(cls, expression: str):
        match = re.fullmatch(r"(.+?)(>=|<=)(.+)", expression)
        if not match:
            raise ValueError("goal must look like 'survival_ratio>=0.95' or 'Metrics/tag<=0.2'")
        value = float(match[3])
        if not math.isfinite(value):
            raise ValueError("goal threshold must be finite")
        return cls(match[1].strip(), match[2], value)


@dataclass
class Options:
    window: int = 100
    min_samples: int = 50
    warmup: int = 20
    penalty: float = 1.0
    top_k: int = 3
    convergence_windows: int = 3
    improvement: float = 0.02
    max_relative_mad: float = 0.05
    terrain_drift: float = 0.1
    max_action_clip: float = 0.1
    segment: int | None = None
    env_step_offset: int | None = None
    goals: list[Goal] = field(default_factory=list)

    def validate(self):
        if not 1 <= self.min_samples <= self.window:
            raise ValueError("require 1 <= min_samples <= window")
        if self.warmup < 0 or self.top_k < 1 or self.convergence_windows < 3:
            raise ValueError("warmup >= 0, top_k >= 1 and convergence_windows >= 3 are required")
        for name in ("penalty", "improvement", "max_relative_mad", "terrain_drift", "max_action_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


def window_stats(segment: Segment, start: int, end: int) -> dict:
    result = {}
    for tag, points in segment.series.items():
        values = [p.value for step, p in points.items() if start <= step <= end]
        if not values:
            continue
        finite = np.asarray([v for v in values if math.isfinite(v)])
        median = float(np.median(finite)) if len(finite) else None
        result[tag] = {
            "median": median,
            "mad": float(np.median(np.abs(finite - median))) if len(finite) else None,
            "count": len(finite), "invalid_count": len(values) - len(finite),
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    return result


def episode_limit(run: Run) -> int | None:
    try:
        dt = float(run.env["sim"]["mujoco"]["timestep"]) * int(run.env["decimation"])
        return math.ceil(float(run.env["episode_length_s"]) / dt)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def curriculum(run: Run, segment: Segment, options: Options) -> dict:
    """Use the actual saved environment counter, including resume offsets."""
    result = {"known": False, "reason": None, "boundaries": [], "offset": None,
              "counter_source": None, "terrain_active": "terrain_levels" in run.env.get("curriculum", {})}
    if not run.env:
        result["reason"] = "missing environment configuration"
        return result
    terms = run.env.get("curriculum") or {}
    stages = {}
    for name, config in terms.items():
        if name == "terrain_levels":
            continue
        params = config.get("params") or {}
        schedule = params.get("velocity_stages") or params.get("weight_stages")
        if not schedule:
            result["reason"] = f"unsupported curriculum term: {name}"
            return result
        for stage in schedule:
            step = int(stage["step"])
            stages.setdefault(step, {})[name] = {k: v for k, v in stage.items() if k != "step"}
    if not stages:
        result.update(known=True, counter_source="no_scheduled_curriculum",
                      boundaries=[{"iteration": segment.first_step, "env_step": None, "changes": {}}])
        return result
    steps = int(run.agent.get("num_steps_per_env", 0))
    if steps <= 0:
        result["reason"] = "missing num_steps_per_env"
        return result
    anchors = [c["env_steps"] - (c["iteration"] + 1) * steps for c in run.checkpoints
               if c["segment"] == segment.index and c["valid"] and c["env_steps"] is not None]
    if options.env_step_offset is not None:
        offset, source = options.env_step_offset, "explicit_override"
    elif anchors and len(set(anchors)) == 1:
        offset, source = anchors[0], "checkpoint.infos.env_state.common_step_counter"
    elif anchors:
        result["reason"] = "checkpoint environment counters disagree; specify --env-step-offset for this segment"
        return result
    elif segment.index == 0 and segment.first_step == 0 and run.agent.get("resume") is False:
        offset, source = 0, "fresh_run_inferred"
    else:
        result["reason"] = "cannot reconstruct resumed curriculum; supply --env-step-offset"
        return result
    # The curriculum checks counter > threshold during an iteration, hence floor.
    boundaries = [{"iteration": (step - offset) // steps, "env_step": step, "changes": changes}
                  for step, changes in sorted(stages.items())]
    result.update(known=True, offset=offset, counter_source=source, boundaries=boundaries)
    return result


def stage_at(schedule: dict, iteration: int) -> dict | None:
    if not schedule["known"]:
        return None
    active = [(i, b) for i, b in enumerate(schedule["boundaries"]) if b["iteration"] <= iteration]
    if not active:
        return None
    index, boundary = active[-1]
    return {"index": index, "start": boundary["iteration"],
            "final": index == len(schedule["boundaries"]) - 1, "changes": boundary["changes"]}


def derived_metrics(run: Run, metrics: dict) -> dict:
    derived = {}
    limit = episode_limit(run)
    length = metrics.get(LENGTH, {}).get("median")
    if limit and length is not None:
        derived["survival_ratio"] = length / limit
    # These are fixed-horizon reward indices, NOT physical tracking accuracy.
    if run.env.get("scale_rewards_by_dt", True):
        for name in ("track_linear_velocity", "track_angular_velocity"):
            if any((term.get("params") or {}).get("reward_name") == name
                   and (term.get("params") or {}).get("weight_stages")
                   for term in (run.env.get("curriculum") or {}).values()):
                continue
            weight = run.env.get("rewards", {}).get(name, {}).get("weight")
            value = metrics.get("Episode_Reward/" + name, {}).get("median")
            if weight and float(weight) > 0 and value is not None:
                derived[name + "_index"] = value / float(weight)
    return derived


def attainment(metrics: dict, derived: dict, goals: list[Goal]) -> dict:
    details = []
    for goal in goals:
        value = derived.get(goal.metric, metrics.get(goal.metric, {}).get("median"))
        passed = None if value is None else (value >= goal.value if goal.operator == ">=" else value <= goal.value)
        details.append({**asdict(goal), "observed": value, "passed": passed})
    state = ("unspecified" if not goals else "unknown" if any(g["passed"] is None for g in details)
             else "passed" if all(g["passed"] for g in details) else "failed")
    return {"status": state, "goals": details}


def health(metrics: dict, options: Options) -> list[str]:
    warnings = [f"non-finite samples in {tag}" for tag, stats in metrics.items() if stats["invalid_count"]]
    clip = metrics.get(CLIP, {}).get("median")
    if clip is not None and clip > options.max_action_clip:
        warnings.append(f"action clipping median {clip:.4g} exceeds {options.max_action_clip:.4g}")
    return warnings


def convergence(run: Run, segment: Segment, schedule: dict, options: Options) -> dict:
    end = segment.last_step
    stage = stage_at(schedule, end)
    result = {"status": "insufficient_data", "reason": "", "as_of_iteration": end,
              "windows": [], "trends": {}, "attainment": {"status": "unknown", "goals": []}}
    if stage is None:
        result.update(status="unknown_curriculum", reason=schedule["reason"] or "curriculum stage unavailable")
        return result
    start = max(stage["start"] + options.warmup, segment.first_step)
    latest = window_stats(segment, max(start, end - options.window + 1), end)
    result["attainment"] = attainment(latest, derived_metrics(run, latest), options.goals)
    result["warnings"] = health(latest, options)
    if not stage["final"]:
        result.update(status="curriculum_in_progress", reason="scheduled curriculum has not reached its final stage")
        return result
    required = options.window * options.convergence_windows
    if end - start + 1 < required:
        result["reason"] = f"need {required} iterations after stage warmup; have {max(end-start+1, 0)}"
        return result
    for index in range(options.convergence_windows):
        left = end - required + 1 + index * options.window
        metrics = window_stats(segment, left, left + options.window - 1)
        result["windows"].append({"start": left, "end": left + options.window - 1, "metrics": metrics})
    tracked = [REWARD, LENGTH] + [tag for tag in (LINEAR, ANGULAR) if tag in segment.series]
    tracked += [g.metric for g in options.goals if g.metric in segment.series and g.metric not in tracked]
    tracked += [TERRAIN] if schedule["terrain_active"] else []
    minimum = max(options.min_samples, math.ceil(options.window * 0.8))
    for tag in tracked:
        stats = [w["metrics"].get(tag, {}) for w in result["windows"]]
        if any(s.get("invalid_count", 0) for s in stats):
            result.update(status="unstable", reason=f"non-finite samples in {tag}")
            return result
        if any(s.get("count", 0) < minimum for s in stats):
            result["reason"] = f"insufficient window coverage for {tag}"
            return result
        medians = np.array([s["median"] for s in stats])
        scale = max(float(np.max(np.abs(medians))), 1.0 if tag == REWARD else 1e-6)
        slope = float(np.polyfit(np.arange(len(medians)), medians, 1)[0] / scale)
        result["trends"][tag] = {
            "medians": medians.tolist(), "relative_slope_per_window": slope,
            "relative_span": float(np.ptp(medians) / scale),
            "relative_mad": max(s["mad"] for s in stats) / scale,
        }
        if tag == TERRAIN and float(np.ptp(medians)) > options.terrain_drift:
            result.update(status="curriculum_in_progress", reason="terrain difficulty is still changing")
            return result
    reward = result["trends"][REWARD]
    if reward["relative_slope_per_window"] > options.improvement:
        result.update(status="improving", reason="reward still improves across recent windows")
    elif reward["relative_slope_per_window"] < -options.improvement:
        result.update(status="degrading", reason="reward decreases across recent windows")
    elif any(t["relative_span"] > options.improvement or t["relative_mad"] > options.max_relative_mad
             for tag, t in result["trends"].items() if tag != TERRAIN):
        result.update(status="unstable", reason="task metrics are still changing or fluctuating")
    elif result["warnings"]:
        result.update(status="unstable", reason="health diagnostics contain warnings")
    elif result["attainment"]["status"] == "passed" and all(tag in tracked for tag in (LINEAR, ANGULAR)):
        result.update(status="converged_candidate", reason="stable task metrics and explicit goals met; heuristic, not a proof")
    else:
        result.update(status="plateau", reason="stable metrics; task goals are missing, unmet, or incompletely observed")
    return result


def analyze(run: Run, options: Options) -> dict:
    options.validate()
    segment = next((s for s in run.segments if s.index == options.segment), None) if options.segment is not None else run.segments[-1]
    if segment is None:
        raise ValueError(f"segment {options.segment} does not exist")
    schedule = curriculum(run, segment, options)
    current_stage = stage_at(schedule, segment.last_step)
    candidates = []
    for checkpoint in run.checkpoints:
        row = {**checkpoint, "stage": None, "eligible": False, "score": None,
               "reason": None, "metrics": {}, "derived": {}, "warnings": []}
        iteration = checkpoint["iteration"]
        if checkpoint["segment"] != segment.index:
            row["reason"] = "checkpoint outside selected log segment or association is ambiguous"
        elif not checkpoint["valid"]:
            row["reason"] = checkpoint["error"]
        else:
            stage = stage_at(schedule, iteration)
            if stage is None:
                row["reason"] = schedule["reason"] or "unknown curriculum stage"
            else:
                row["stage"] = stage["index"]
                start = max(segment.first_step, stage["start"] + options.warmup, iteration-options.window+1)
                metrics = window_stats(segment, start, iteration)
                reward = metrics.get(REWARD, {})
                row.update(window_start=start, window_end=iteration, samples=reward.get("count", 0),
                           metrics=metrics, derived=derived_metrics(run, metrics), warnings=health(metrics, options))
                row["attainment"] = attainment(metrics, row["derived"], options.goals)
                if reward.get("count", 0) < options.min_samples:
                    row["reason"] = "insufficient post-warmup samples in this stage"
                elif reward.get("invalid_count", 0):
                    row["reason"] = "non-finite reward samples"
                else:
                    row.update(eligible=True, score=reward["median"] - options.penalty * reward["mad"])
        candidates.append(row)
    ranked = sorted((c for c in candidates if c["eligible"]), key=lambda c: (-c["score"], -c["iteration"]))
    by_stage = {}
    for candidate in ranked:
        by_stage.setdefault(str(candidate["stage"]), [])
        if len(by_stage[str(candidate["stage"])]) < options.top_k:
            by_stage[str(candidate["stage"])].append(candidate["iteration"])
    current = by_stage.get(str(current_stage["index"]), []) if current_stage else []
    return {
        "schema_version": 1, "selection_policy": POLICY_VERSION, "run": str(run.path),
        "options": asdict(options), "config_sha256": run.fingerprints, "event_snapshot": run.event_files,
        "segment": segment.index, "segments": [{"index": s.index, "first_step": s.first_step, "last_step": s.last_step} for s in run.segments],
        "as_of_iteration": segment.last_step, "max_episode_steps": episode_limit(run),
        "curriculum": schedule, "current_stage": current_stage,
        "best_current_stage": current[0] if current else None,
        "top_current_stage": current, "top_by_stage": by_stage,
        "candidates": candidates, "convergence": convergence(run, segment, schedule, options),
        "warnings": run.warnings + ["Training-window scores approximate checkpoint performance; no fixed-condition evaluation was run.",
                                   "Termination scalars are counts, not failure rates; tracking indices are not physical accuracy."],
    }
