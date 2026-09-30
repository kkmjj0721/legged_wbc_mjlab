"""Reports and optional checkpoint publication; simulation is never started."""

from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile

from .best_model import ANGULAR, LENGTH, LINEAR, REWARD, TERRAIN
from .training_logs import Run, sha256, model_key, model_label


def log_candidate_label(result):
    iteration = result.get("best_current_stage")
    if iteration is None:
        return "暂无合格日志候选"
    prefix = f"R{result['chain'][-1]['index']} / " if len(result.get("chain", [])) > 1 else ""
    return f"{prefix}model_{iteration}"


def recommendation(result, evaluation=None):
    """One explicit recommendation source shared by the CLI, JSON and report."""
    if evaluation is not None:
        winner = evaluation["summary"][0]
        identity = model_key(winner)
        if identity != evaluation.get("best_model_id", evaluation.get("best_iteration", identity)):
            raise ValueError("evaluation winner differs from the first ranked model")
        source = next(m for m in evaluation["models"] if model_key(m) == identity)
        trial = evaluation.get("hardware_assessment", {}).get("hardware_recommendation")
        if trial and trial["model_id"] != identity:
            raise ValueError("hardware-trial recommendation differs from the first ranked model")
        return {"basis": "simulation", "label": model_label(source), "model_id": identity,
                "iteration": source["iteration"], "source_checkpoint": source.get("path"),
                "artifact": "model_best_eval.pt", "unique_best_established": False,
                "intended_use": "hardware_trial_priority" if trial else "simulation_screening", "hardware_validated": False,
                "hardware_recommendation": trial}
    candidate = next((c for c in result.get("candidates", []) if c["iteration"] == result.get("best_current_stage")), None)
    return {"basis": "training_log", "label": log_candidate_label(result),
            "iteration": result.get("best_current_stage"), "source_checkpoint": candidate["path"] if candidate else None,
            "artifact": "model_best.pt" if result.get("published") else None,
            "unique_best_established": False, "intended_use": "training_log_analysis",
            "hardware_validated": False, "hardware_recommendation": None}


def _atomic_text(path: Path, writer):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint_actor(run: Run, checkpoint: Path, observations=None, device="cpu"):
    """Build a frozen actor from the saved architecture, optionally on live observations."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import HIMActorModel, MLPModel

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["actor_state_dict"]
    cfg = copy.deepcopy(run.agent.get("actor") or {})
    name = cfg.pop("class_name", "MLPModel")
    classes = {"MLPModel": MLPModel, "HIMActorModel": HIMActorModel,
               "rsl_rl.models.him_actor_model:HIMActorModel": HIMActorModel,
               "rsl_rl.models.mlp_model:MLPModel": MLPModel}
    if name not in classes:
        raise ValueError(f"unsupported actor class {name}")
    for key in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
        if cfg.get(key) is not None:
            raise ValueError(f"ONNX export does not support {key}")
        cfg.pop(key, None)
    distribution = cfg.get("distribution_cfg")
    if distribution:
        if distribution.get("class_name") not in ("GaussianDistribution", "rsl_rl.modules.distribution:GaussianDistribution"):
            raise ValueError("ONNX export currently supports only GaussianDistribution")
        distribution["class_name"] = "rsl_rl.modules.distribution:GaussianDistribution"
    weights = sorted((int(k.split(".")[1]), v) for k, v in state.items()
                     if k.startswith("mlp.") and k.endswith(".weight") and v.ndim == 2)
    if not weights:
        raise ValueError("checkpoint has no MLP weights")
    output_dim = weights[-1][1].shape[0]
    cls = classes[name]
    if cls is HIMActorModel:
        obs_dim = int(cfg["num_one_step_obs"]) * int(cfg["history_size"])
        contract = (saved.get("infos") or {}).get("him_numerics")
        if not contract:
            raise ValueError("HIM export requires the saved numerical contract; legacy checkpoint cannot be exported reliably")
        expected = {"action_clip": cfg.get("action_clip"), "observation_clip": cfg.get("observation_clip"),
                    "history_size": cfg["history_size"], "frame_size": cfg["num_one_step_obs"],
                    "action_observation_slice": list(cfg.get("action_observation_slice") or [])}
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("saved agent configuration differs from checkpoint HIM numerical contract")
    else:
        obs_dim = weights[0][1].shape[1]
        contract = {}
    if observations is None:
        observations = TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1])
    elif observations["actor"][0].numel() != obs_dim:
        raise ValueError(f"actor expects {obs_dim} observation values, got {observations['actor'][0].numel()}")
    actor = cls(observations,
                {"actor": ["actor"]}, "actor", output_dim, **cfg)
    actor.load_state_dict(state, strict=True)
    actor.to(device).eval().requires_grad_(False)
    return actor, obs_dim, output_dim, contract


def export_checkpoint(run: Run, checkpoint: Path, output: Path) -> dict:
    """Use the runner's ONNX wrapper with the saved architecture, without simulation."""
    import onnx
    import torch

    actor, obs_dim, output_dim, contract = load_checkpoint_actor(run, checkpoint)
    exported = actor.as_onnx(verbose=False).cpu().eval()
    kwargs = {"external_data": False} if "external_data" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(exported, exported.get_dummy_inputs(), str(output), export_params=True,
                      opset_version=18, input_names=exported.input_names, output_names=exported.output_names,
                      dynamic_axes={name: {0: "batch"} for name in (*exported.input_names, *exported.output_names)},
                      dynamo=False, **kwargs)
    metadata = {"run_path": str(run.path), "source_checkpoint": checkpoint.name,
                "source_checkpoint_sha256": sha256(checkpoint),
                "saved_config_sha256": json.dumps(run.fingerprints, sort_keys=True),
                "metadata_scope": "offline_actor_export; deployment joint configuration remains external"}
    metadata.update({f"him_{key}": str(value) for key, value in contract.items()})
    model = onnx.load(str(output))
    onnx.helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    return {"input_dim": obs_dim, "output_dim": output_dim, "metadata": metadata}


def publish_best(run: Run, result: dict, output: Path, export_onnx: bool = False) -> dict:
    iteration = result["best_current_stage"]
    if iteration is None:
        raise ValueError("no eligible checkpoint in the current stage; no best model was written")
    candidate = next(c for c in result["candidates"] if c["iteration"] == iteration)
    source = Path(candidate["path"])
    stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (candidate["size"], candidate["mtime_ns"]):
        raise ValueError("selected checkpoint changed after analysis; run the command again")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=output) as staging:
        staging = Path(staging)
        copied = staging / source.name
        shutil.copyfile(source, copied)
        digest = sha256(copied)
        if digest != sha256(source):
            raise ValueError("selected checkpoint changed while copying")
        exported = export_checkpoint(run, copied, staging / "policy.onnx") if export_onnx else None
        manifest = {"source_checkpoint": str(source), "iteration": iteration, "sha256": digest,
                    "selection_policy": result["selection_policy"], "segment": result["segment"],
                    "stage": candidate["stage"], "score": candidate["score"],
                    "model": str(output / "model_best.pt"),
                    "onnx": str(output / "policy.onnx") if exported else None,
                    "onnx_export": exported}
        os.replace(copied, output / "model_best.pt")
        if exported:
            os.replace(staging / "policy.onnx", output / "policy.onnx")
        else:
            # Never leave the previous best's ONNX beside newly selected weights.
            (output / "policy.onnx").unlink(missing_ok=True)
        _atomic_text(output / "model_manifest.json", lambda f: json.dump(manifest, f, indent=2, allow_nan=False))
    return manifest


def write_selection(result: dict, output: Path) -> None:
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_text(output / "selection.json", lambda f: json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False))


def write_reports(run: Run, result: dict, output: Path, plot: bool = True) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_selection(result, output)
    tags = sorted({tag for c in result["candidates"] for tag in c["metrics"]})
    fields = ["iteration", "stage", "eligible", "score", "window_start", "window_end", "samples", "reason", "path"]

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=fields + tags)
        writer.writeheader()
        for candidate in result["candidates"]:
            row = {key: candidate.get(key) for key in fields}
            row.update({tag: candidate["metrics"].get(tag, {}).get("median") for tag in tags})
            writer.writerow(row)

    _atomic_text(output / "candidates.csv", write_csv)
    if not plot:
        (output / "report.png").unlink(missing_ok=True)
        return
    import numpy as np
    from .evaluation_report import STATUSES, _plot_setup
    plt = _plot_setup()

    segment = next(s for s in run.segments if s.index == result["segment"])
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    labels = ("训练总回报 ↑", "平均回合长度（步）↑", "线速度跟踪奖励 ↑", "转向跟踪奖励 ↑", "地形课程等级", "动作平滑奖励")
    for axis, tag, label in zip(axes.flat, (REWARD, LENGTH, LINEAR, ANGULAR, TERRAIN, "Episode_Reward/smoothness"), labels):
        points = sorted((p for p in segment.series.get(tag, {}).values() if p.step <= result["as_of_iteration"]), key=lambda p: p.step)
        steps = [p.step for p in points]
        values = [p.value for p in points]
        axis.plot(steps, values, linewidth=0.6, alpha=0.3, label="原始值")
        median = [np.median(values[max(0, i-49):i+1]) for i in range(len(values))]
        axis.plot(steps, median, linewidth=1.4, label="最近 50 个样本中位数")
        axis.set_title(label, fontsize=11)
        if not points:
            axis.text(0.5, 0.5, "日志未记录此指标", ha="center", transform=axis.transAxes)
        for boundary in result["curriculum"]["boundaries"]:
            step = boundary["iteration"]
            if segment.first_step <= step <= segment.last_step:
                axis.axvline(step, color="gray", linestyle=":", alpha=0.5)
        for iteration in result["top_current_stage"]:
            axis.axvline(iteration, color="tab:green", alpha=0.65,
                         linestyle="-" if iteration == result["best_current_stage"] else "--")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("训练迭代 iteration")
    axes.flat[0].legend(fontsize=8, loc="best")
    status = STATUSES.get(result['convergence']['status'], result['convergence']['status'])
    figure.suptitle(f"{run.path.name} · {status}\n日志评分候选：{log_candidate_label(result)}（训练曲线参考）；灰线：课程切换；绿线：日志候选", fontsize=12)
    figure.tight_layout()
    descriptor, temporary = tempfile.mkstemp(suffix=".png", dir=output)
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=150)
        os.replace(temporary, output / "report.png")
    finally:
        plt.close(figure)
        if os.path.exists(temporary):
            os.unlink(temporary)
