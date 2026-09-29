"""Compare completed evaluations without hiding GPU simulation variability."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from .training_logs import model_key, model_origin


def physics_fingerprint(model):
    """Hash compiled numeric physics, excluding randomized generated asset names."""
    digest = hashlib.sha256()
    for prefix, obj in (("model", model), ("opt", model.opt)):
        for name in sorted(dir(obj)):
            if name.startswith(("_", "name_")) or name in ("names", "names_map", "nnames", "paths", "npaths"):
                continue
            value = getattr(obj, name)
            if isinstance(value, np.ndarray):
                digest.update(f"{prefix}.{name}:{value.dtype}:{value.shape}".encode())
                digest.update(value.tobytes())
            elif isinstance(value, (int, float)):
                digest.update(f"{prefix}.{name}:{value}".encode())
    return digest.hexdigest()


def runtime_description(device):
    import torch
    versions = {}
    for package in ("torch", "mujoco", "mujoco-warp", "warp-lang", "mjlab"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {"packages": versions, "device": device, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "physics_determinism": "MuJoCo Warp GPU contact solving is not bitwise deterministic; fixed seeds alone do not ensure identical rollouts"}


def compare_evaluations(current, previous):
    """Pair the same episodes; input changes and numerical variation are distinct."""
    base = {"unique_best_established": False,
            "current_best": current["best_iteration"], "previous_best": None,
            "current_best_id": current.get("best_model_id", current["best_iteration"]),
            "current_best_label": current.get("best_label", f"model_{current['best_iteration']}"),
            "status": "not_checked", "comparable": False, "changes": []}
    ranked = current["summary"]
    if len(ranked) > 1:
        base["runner_up"] = ranked[1]["iteration"]
        base["lead_success_percentage_points"] = 100 * (ranked[0]["task_success"] - ranked[1]["task_success"])
        base["lead_success_episodes"] = round((ranked[0]["task_success"] - ranked[1]["task_success"]) * ranked[0]["episodes"])
    if previous is None:
        return base
    base["previous_best"] = previous.get("best_iteration")
    base["previous_best_id"] = previous.get("best_model_id", previous.get("best_iteration"))
    base["previous_best_label"] = previous.get("best_label", f"model_{previous.get('best_iteration')}")
    required = ("models", "saved_config_sha256", "scenarios", "options", "parallel_models", "environments", "rows", "summary", "best_iteration")
    missing = [key for key in required if key not in previous]
    for collection, fields in (("models", ("iteration", "sha256")),
                               ("rows", ("iteration", "terrain", "seed", "env_id", "initial_state_sha256", "task_success", "success", "survival_s", "linear_rmse", "yaw_rmse", "tilt_rms_deg")),
                               ("summary", ("iteration", "task_success"))):
        if any(any(field not in row for field in fields) for row in previous.get(collection, [])):
            missing.append(collection + " 缺少比较字段")
    if missing or not previous.get("rows") or not previous.get("summary"):
        base.update(status="insufficient_metadata", missing_fields=missing)
        return base
    changes = base["changes"]
    models = lambda d: {model_key(m): m["sha256"] for m in d["models"]}
    old, new = models(previous), models(current)
    if set(old) != set(new):
        changes.append("模型列表变化")
        base["added_models"] = sorted(set(new) - set(old), key=str)
        base["removed_models"] = sorted(set(old) - set(new), key=str)
    if any(old[i] != new[i] for i in old.keys() & new.keys()):
        changes.append("checkpoint 内容变化")
    if previous["saved_config_sha256"] != current["saved_config_sha256"]:
        changes.append("保存的训练配置变化")
    # A live protocol uses tuples; its saved JSON uses lists. Compare the
    # serialized values so reloading does not invent a changed input.
    canonical = lambda value: json.dumps(value, sort_keys=True)
    if canonical(previous["scenarios"]) != canonical(current["scenarios"]):
        changes.append("地形或指令变化")
    for key in ("num_envs", "duration", "seeds", "tilt_limit", "device"):
        if canonical(previous["options"].get(key)) != canonical(current["options"].get(key)):
            changes.append(f"参数 {key} 变化")
    if previous["parallel_models"] != current["parallel_models"]:
        changes.append("实际并行模型数量变化")
    for key in ("dt", "steps", "ranking_policy"):
        if previous.get(key) != current.get(key):
            changes.append(f"{key} 变化")
    for terrain in set(previous["environments"]) & set(current["environments"]):
        before, after = previous["environments"][terrain], current["environments"][terrain]
        if before["sha256"] != after["sha256"]:
            changes.append(f"{terrain} 环境配置变化")
        if before.get("physics_sha256") and after.get("physics_sha256") and before["physics_sha256"] != after["physics_sha256"]:
            changes.append(f"{terrain} 编译物理模型变化")
    if previous.get("runtime") and current.get("runtime") and previous["runtime"] != current["runtime"]:
        changes.append("运行库、设备或计算设置变化")
    base["runtime_recorded_both"] = bool(previous.get("runtime") and current.get("runtime"))
    key = lambda r: (model_key(r), r["terrain"], r["seed"], r["env_id"])
    prior = {key(r): r for r in previous["rows"]}
    present = {key(r): r for r in current["rows"]}
    if set(prior) != set(present) or len(prior) != len(previous["rows"]) or len(present) != len(current["rows"]):
        changes.append("逐回合样本编号不一致或重复")
    if any(prior[k]["initial_state_sha256"] != present[k]["initial_state_sha256"] for k in prior.keys() & present.keys()):
        changes.append("初始状态变化")
    if changes:
        base["status"] = "inputs_changed"
        return base
    base["comparable"] = True
    fields = ("task_success", "success", "survival_s", "linear_rmse", "yaw_rmse", "tilt_rms_deg")
    differences = {field: [] for field in fields}
    by_terrain = Counter()
    changed_rows = 0
    for k, row in present.items():
        old_row = prior[k]
        changed_rows += any(row[field] != old_row[field] for field in fields)
        if row["task_success"] != old_row["task_success"]:
            by_terrain[row["terrain"]] += 1
        for field in fields:
            differences[field].append(abs(float(row[field]) - float(old_row[field])))
    base.update(episodes=len(present), changed_metric_episodes=changed_rows,
                changed_task_success_episodes=sum(by_terrain.values()),
                changed_task_success_by_terrain=dict(by_terrain),
                max_absolute_episode_differences={field: max(values, default=0) for field, values in differences.items()})
    old_summary = {model_key(r): r for r in previous["summary"]}
    old_ranks = {model_key(r): rank for rank, r in enumerate(previous["summary"], 1)}
    base["models"] = [{"iteration": r["iteration"], **model_origin(r),
                       "previous_rank": old_ranks[model_key(r)], "current_rank": rank,
                       "previous_success": old_summary[model_key(r)]["task_success"],
                       "current_success": r["task_success"],
                       "delta_percentage_points": 100 * (r["task_success"] - old_summary[model_key(r)]["task_success"])} for rank, r in enumerate(ranked, 1)]
    base["max_model_success_delta_percentage_points"] = max(abs(r["delta_percentage_points"]) for r in base["models"])
    base["status"] = ("winner_changed" if base["previous_best_id"] != base["current_best_id"] else
                      "same_winner_scores_changed" if changed_rows else "identical")
    return base


def repeatability_notes(data):
    audit = data.get("repeatability", {})
    status = audit.get("status", "not_checked")
    notes = ["本轮第一名是单次评测的候选，不表示已经确认唯一最优。固定 3 组种子控制初始抽样，但不能保证 GPU 接触求解的浮点运算逐位一致。"]
    if "lead_success_episodes" in audit:
        notes.append(f"本轮第一名比第二名多通过 {audit['lead_success_episodes']} 个回合（{audit['lead_success_percentage_points']:.2f} 个百分点）；微小领先需要复测。")
    if status == "not_checked":
        notes.append("尚无上次完成的评测可比较；本次未验证排名重复性。相同命令再次运行后会自动核对。")
    elif status == "insufficient_metadata":
        notes.append("上次结果缺少复跑核对需要的逐回合数据或元信息，无法验证重复性；本次将保存完整记录供下次比较。")
    elif status == "inputs_changed":
        notes.append("与上次评测的条件不同，不能将排名变化归因于仿真波动：" + "；".join(audit["changes"]) + "。")
    else:
        if status == "winner_changed":
            notes.append(f"复跑第一名发生变化：{audit.get('previous_best_label', 'model_'+str(audit['previous_best']))} → {audit.get('current_best_label', 'model_'+str(audit['current_best']))}。在已记录的相同输入下排名不稳定，不能据此断言新模型更好。")
        elif status == "same_winner_scores_changed":
            notes.append("两次第一名相同，但逐回合成绩有变化；这次复测不能证明唯一最优。")
        else:
            notes.append("这两次的记录指标一致；这仅验证了两次结果，不能外推到其他设备或运行条件。")
        notes.append(f"共 {audit['episodes']} 个配对回合，其中 {audit['changed_task_success_episodes']} 个的通过/未通过判定改变；单个模型总通过率最大变化 {audit['max_model_success_delta_percentage_points']:.2f} 个百分点。")
        if not audit.get("runtime_recorded_both"):
            notes.append("旧报告没有完整运行库版本记录，比较范围限于已保存的模型、配置和初始状态。")
    return notes


def load_previous(output: Path):
    path = output / "evaluation.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("complete") else None
