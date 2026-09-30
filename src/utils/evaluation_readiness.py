"""Relative hardware-trial priorities; no claim of measured hardware performance."""

from collections import Counter, defaultdict
import math
from statistics import mean

from .evaluation_metrics import COMMANDS
from .evaluation_scenarios import TERRAIN_LABELS, expand_terrains
from .training_logs import model_key, model_label

REQUIRED_TERRAINS = tuple(expand_terrains(("flat", "rough", "stairs_up", "stairs_down")))
POLICY_VERSION = "hardware_trial_relative_v1"
WEIGHTS = {"tracking": 0.45, "stability": 0.25, "smoothness": 0.10, "stairs": 0.20}
REFERENCES = {"linear_floor_m_s": 0.3, "yaw_rad_s": 0.5, "tilt_deg": 20.0,
              "action_rate_s": 10.0, "action_accel_s2": 400.0}
SENSITIVITY = {"基础行走优先": {**WEIGHTS, "tracking": 0.55, "stairs": 0.10},
               "默认均衡": WEIGHTS, "楼梯能力优先": {**WEIGHTS, "tracking": 0.35, "stairs": 0.30}}
RANKING_POLICY = (POLICY_VERSION + ": 45% ground tracking, 25% survival/posture, "
                  "10% action smoothness, 20% stair completion; no all-pass gate; "
                  "fixed normalization references; exact tie uses source order then iteration")
EVIDENCE_GAPS = (
    "推荐依据为已有仿真表现，hardware_validated=false；实机表现和唯一最优尚未验证。",
    "当前协议关闭参数随机化、推扰、观测噪声和观测/执行器延迟。",
    "力矩 RMS 和动作裁剪比例不能代替逐关节峰值力矩、限幅持续时间及电机温升。",
    "实际部署 ONNX 的数值一致性、启停、指令切换及长时间运行仍需核对。",
)


def _components(rows):
    ground = [r for r in rows if r["terrain"] in ("flat", "rough")]
    stairs = [r for r in rows if r["terrain"].startswith("stairs_")]
    if not ground or not stairs:
        return None
    metrics = ("linear_rmse", "yaw_rmse", "tilt_rms_deg", "action_rate_rms", "action_accel_rms")
    if any(any(k not in r or not math.isfinite(r[k]) or r[k] < 0 for k in metrics) for r in ground):
        return None
    commands = {c[0]: c[2] for c in COMMANDS}
    if any(r["command"] not in commands for r in ground):
        return None
    tracking = mean(float(r["success"])
                    * max(0.0, 1 - r["linear_rmse"] / max(REFERENCES["linear_floor_m_s"], math.hypot(*commands[r["command"]][:2])))
                    * max(0.0, 1 - r["yaw_rmse"] / REFERENCES["yaw_rad_s"]) for r in ground)
    stability = 0.5 * mean(float(r["success"]) for r in rows) + 0.5 * mean(
        1 / (1 + (r["tilt_rms_deg"] / REFERENCES["tilt_deg"]) ** 2) for r in ground)
    smoothness = mean(0.5 / (1 + (r["action_rate_rms"] / REFERENCES["action_rate_s"]) ** 2)
                      + 0.5 / (1 + (r["action_accel_rms"] / REFERENCES["action_accel_s2"]) ** 2) for r in ground)
    return {"tracking": tracking, "stability": stability, "smoothness": smoothness,
            "stairs": mean(float(r["task_success"]) for r in stairs)}


def _score(components, weights=WEIGHTS):
    return 100 * sum(components[k] * w for k, w in weights.items())


def assess_hardware_readiness(data):
    """Task failures reduce relative scores; only incomparable data prevents ranking.

    Fixed scales and weights express an engineering preference, not a hardware
    success probability. No task needs a 100% pass rate.
    """
    grouped = defaultdict(list)
    for row in data.get("rows", []):
        grouped[model_key(row)].append(row)
    models, sample_sets = [], []
    for summary in data.get("summary", []):
        rows = grouped[model_key(summary)]
        samples = [(r["terrain"], r["command"], r.get("seed"), r["env_id"]) for r in rows]
        sample_sets.append(set(samples))
        terrains = []
        for terrain in REQUIRED_TERRAINS:
            sample = [r for r in rows if r["terrain"] == terrain]
            terrains.append({"terrain": terrain, "label": TERRAIN_LABELS[terrain], "episodes": len(sample),
                             "passed": sum(bool(r["task_success"]) for r in sample),
                             "terminated": sum(bool(r["failed"]) for r in sample),
                             "survived_but_incomplete": sum(bool(r["success"] and not r["task_success"]) for r in sample)})
        components = _components(rows)
        per_seed = []
        for seed in sorted({r.get("seed", 0) for r in rows}):
            values = _components([r for r in rows if r.get("seed", 0) == seed])
            if values is not None:
                per_seed.append({"seed": seed, "score": _score(values)})
        reasons = Counter(reason for r in rows for reason in r.get("failure_reason", "").split(";") if reason)
        models.append({"model_id": model_key(summary), "iteration": summary["iteration"], "label": model_label(summary),
                       "source_index": summary.get("source_index", 0), "episodes": len(rows),
                       "passed": sum(bool(r["task_success"]) for r in rows),
                       "terminated": sum(bool(r["failed"]) for r in rows), "failure_reasons": dict(reasons),
                       "duplicate_samples": len(samples) != len(set(samples)),
                       "score": _score(components) if components is not None else None,
                       "components": components, "seed_scores": per_seed, "terrains": terrains})
    comparable = (bool(models) and all(m["score"] is not None and not m["duplicate_samples"] for m in models)
                  and all(s == sample_sets[0] for s in sample_sets))
    ordered = sorted(models, key=lambda m: (-(m["score"] or 0), m["source_index"], m["iteration"], str(m["model_id"]))) if comparable else models
    choice = ({k: ordered[0][k] for k in ("model_id", "iteration", "label", "score")} if comparable else None)
    sensitivity = []
    if comparable:
        for label, weights in SENSITIVITY.items():
            ranked = sorted(models, key=lambda m: (-_score(m["components"], weights), m["source_index"], m["iteration"], str(m["model_id"])))
            sensitivity.append({"profile": label, "weights": weights, "winner": ranked[0]["label"], "model_id": ranked[0]["model_id"]})
    limits = []
    for index, terrain in enumerate(REQUIRED_TERRAINS):
        observations = [m["terrains"][index] for m in models if m["terrains"][index]["episodes"]]
        limits.append({"terrain": terrain, "label": TERRAIN_LABELS[terrain], "tested_models": len(observations),
                       "models_with_any_pass": sum(t["passed"] > 0 for t in observations),
                       "best_observed_pass_rate": max((t["passed"] / t["episodes"] for t in observations), default=None)})
    return {"status": "relative_recommendation" if choice else "insufficient_comparable_metrics",
            "hardware_recommendation": choice, "hardware_validated": False,
            "policy_version": POLICY_VERSION, "ranking_policy": RANKING_POLICY,
            "weights": WEIGHTS, "references": REFERENCES, "sensitivity": sensitivity,
            "scope": "relative hardware-trial priority from nominal simulation; task shortfalls allowed",
            "model_count": len(models), "control_dt_s": data.get("dt"), "duration_s": data.get("options", {}).get("duration"),
            "evidence_gaps": list(EVIDENCE_GAPS), "terrain_limits": limits, "models": ordered}


def apply_recommendation_ranking(data):
    """Use one recommendation order for artifacts, plots, terminal and HTML."""
    assessment = assess_hardware_readiness(data)
    data["hardware_assessment"] = assessment
    choice = assessment["hardware_recommendation"]
    if choice is None:
        return assessment
    lookup = {model_key(s): s for s in data["summary"]}
    for rank, model in enumerate(assessment["models"], 1):
        lookup[model["model_id"]].update(deployment_score=model["score"], score_components=model["components"],
                                         seed_deployment_scores=model["seed_scores"], recommendation_rank=rank)
    data["summary"] = [lookup[m["model_id"]] for m in assessment["models"]]
    data.update(best_iteration=choice["iteration"], best_model_id=choice["model_id"], best_label=choice["label"],
                ranking_policy=RANKING_POLICY)
    return assessment


def readiness_notes(assessment):
    choice = assessment["hardware_recommendation"]
    if choice:
        notes = [f"优先实机验证推荐：{choice['label']}。依据现有模型间的相对表现，允许楼梯等能力存在短板；实机表现尚未验证。",
                 "相对评分：基础跟踪 45%、稳定性 25%、动作平滑 10%、楼梯能力 20%；分数不是实机成功概率。"]
        winner = assessment["models"][0]
        notes.append(f"推荐模型任务通过 {winner['passed']}/{winner['episodes']}，提前终止 {winner['terminated']}；具体原因和未完成场景分别列出。")
        notes.append("偏重基础行走、默认均衡、偏重楼梯的权重对照第一名依次为：" + "；".join(s["winner"] for s in assessment["sensitivity"]) + "。这不是重新仿真或实机验证。")
    else:
        notes = ["实机优先推荐尚无足够的可比指标；保留本次仿真候选。缺少基础行走或楼梯指标、指标无效、样本不一致会阻止相对排序，任务未全部通过不会。"]
    for terrain in assessment["terrain_limits"]:
        if terrain["tested_models"] and not terrain["models_with_any_pass"]:
            notes.append(f"能力边界：{terrain['label']}在本轮 {terrain['tested_models']} 个模型中均无通过回合；保留此短板，不因此取消推荐。")
    return notes
