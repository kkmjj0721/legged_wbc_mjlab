"""Resolve an explicit resume ancestry, preserving each run's own configuration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import yaml

from .training_logs import ConfigLoader, Run, model_key, read_run, sha256


class UnresolvedAncestry(ValueError):
    """Old metadata cannot identify the historical checkpoint unambiguously."""


def resolve_run_directory(directory):
    """Accept a run itself or select the newest immediate run in an experiment."""
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"run directory does not exist: {path}")
    if ((path / "params/agent.yaml").is_file() or (path / "params/lineage.json").is_file()
            or any(path.glob("events.out.tfevents.*"))
            or any(re.fullmatch(r"model_\d+\.pt", p.name) for p in path.glob("model_*.pt"))):
        return path
    candidates = [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")
                  and (p / "params/agent.yaml").is_file()]
    if not candidates:
        raise ValueError(f"no training runs found in {path}; --run expects a run or an experiment "
                         "directory whose immediate child runs contain params/agent.yaml")

    def started_at(candidate):
        try:
            # train.py names runs by their start time. Report creation or a
            # resumed writer touching an older directory must not reorder them.
            stamp = datetime.strptime(candidate.name[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
        except ValueError:
            stamp = (candidate / "params/agent.yaml").stat().st_mtime
        return stamp, candidate.name

    return max(candidates, key=started_at).resolve()


def _owning_run(source):
    return next((p for p in source.parents if (p / "params/agent.yaml").is_file()), source.parent)


def write_lineage(directory, task, checkpoint=None, checkpoint_iteration=None):
    """Record the resolved source, not a mutable 'latest checkpoint' pattern."""
    from .training_report import _atomic_text
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params").mkdir(exist_ok=True)
    source = Path(checkpoint).resolve() if checkpoint is not None else None
    parent = _owning_run(source) if source else None
    data = {"schema_version": 1, "task": task, "parent_run": str(parent) if source else None,
            "checkpoint": str(source.relative_to(parent)) if source else None,
            "checkpoint_iteration": checkpoint_iteration,
            "checkpoint_sha256": sha256(source) if source else None}
    _atomic_text(directory / "params" / "lineage.json", lambda f: json.dump(data, f, indent=2))


def parent_reference(directory: Path):
    recorded = directory / "params" / "lineage.json"
    if recorded.exists():
        data = json.loads(recorded.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"invalid lineage mapping: {recorded}")
        if not data.get("parent_run"):
            return None
        name = Path(str(data.get("checkpoint", "")))
        if name.is_absolute() or ".." in name.parts or name.suffix != ".pt":
            raise ValueError(f"invalid source checkpoint in {recorded}")
        parent = Path(data["parent_run"]).expanduser()
        if not parent.is_absolute():
            parent = directory.parent / parent
        if not parent.is_dir():
            # Logs may have been moved as a complete experiment directory.
            parent = directory.parent / parent.name
        return {"parent": parent.resolve(), "checkpoint": data["checkpoint"],
                "checkpoint_iteration": data.get("checkpoint_iteration"),
                "sha256": data.get("checkpoint_sha256"), "evidence": "lineage.json"}
    config = directory / "params" / "agent.yaml"
    if not config.exists():
        raise UnresolvedAncestry(f"cannot trace resume ancestry without {config}; use --single-run or --runs")
    agent = yaml.load(config.read_text(), Loader=ConfigLoader)
    if not isinstance(agent, dict):
        raise ValueError(f"invalid agent configuration mapping: {config}")
    if not agent.get("resume", False):
        return None
    name, checkpoint = str(agent.get("load_run", "")), str(agent.get("load_checkpoint", ""))
    parent = Path(name).expanduser()
    if not parent.is_absolute():
        parent = directory.parent / parent
    if not parent.is_dir() or not (re.fullmatch(r"model_\d+\.pt", checkpoint) or (parent / checkpoint).is_file()):
        raise UnresolvedAncestry(f"resume source in {config} is missing or a regex ({name!r}, {checkpoint!r}); "
                         "cannot reconstruct the historical 'latest' choice. Supply --runs ORIGINAL RESUME ... in order")
    source = (parent / checkpoint).resolve()
    parent = _owning_run(source)
    return {"parent": parent, "checkpoint": str(source.relative_to(parent)), "sha256": None, "evidence": "saved exact load_run/load_checkpoint"}


def read_training_chain(latest=None, explicit=None, single=False):
    if explicit:
        paths = [Path(p).expanduser().resolve() for p in explicit]
        if len(set(paths)) != len(paths):
            raise ValueError("--runs contains duplicate directories")
        # The supplied order is authoritative when old logs only saved regexes.
        links = []
        for index, path in enumerate(paths):
            try:
                ref = parent_reference(path)
            except UnresolvedAncestry:
                ref = None
            if index and ref and ref["parent"] != paths[index-1]:
                raise ValueError(f"{path} resumes {ref['parent']}, not the preceding supplied run")
            links.append(ref or {"evidence": "user-supplied order; ancestry not verified"})
    else:
        path = resolve_run_directory(latest)
        paths, links = [], []
        while True:
            if path in paths:
                raise ValueError(f"cycle in resume ancestry: {path}")
            if not path.is_dir():
                raise ValueError(f"missing ancestor run: {path}; restore it or specify --runs / --single-run")
            paths.append(path)
            ref = None if single else parent_reference(path)
            links.append(ref)
            if ref is None:
                break
            path = ref["parent"]
        paths.reverse(); links.reverse()
    runs = [read_run(p) for p in paths]
    if explicit and links[0].get("parent") not in (None, paths[0]):
        runs[0].warnings.append("the first supplied run itself resumes an earlier run outside --runs; this is a partial history")
    for run, ref in zip(runs, links):
        if ref and ref.get("checkpoint"):
            checkpoint = ref["parent"] / ref["checkpoint"]
            if not checkpoint.is_file():
                run.warnings.append(f"resume checkpoint is no longer present: {checkpoint}; ancestry metadata retained, model cannot be evaluated")
            elif ref.get("sha256") and sha256(checkpoint) != ref["sha256"]:
                raise ValueError(f"recorded resume checkpoint has changed: {checkpoint}")
    return runs, links


def combine_runs(runs: list[Run]) -> Run:
    if len(runs) == 1:
        return runs[0]
    checkpoints = []
    for index, run in enumerate(runs, 1):
        token = hashlib.sha256(str(run.path).encode()).hexdigest()[:16]
        for checkpoint in run.checkpoints:
            checkpoints.append({**checkpoint, "model_id": f"{token}:{checkpoint['iteration']}",
                                "label": f"R{index} / model_{checkpoint['iteration']}",
                                "source_run": str(run.path), "source_index": index})
    return replace(runs[-1], checkpoints=checkpoints, sources={str(r.path): r for r in runs},
                   fingerprints={str(r.path): r.fingerprints for r in runs})


def analyze_chain(runs, links, options):
    from .best_model import analyze
    latest = analyze(runs[-1], options)
    chain = []
    for index, (run, link) in enumerate(zip(runs, links), 1):
        segments = [analyze(run, replace(options, segment=s.index,
                                        env_step_offset=options.env_step_offset if run is runs[-1] and s.index == latest["segment"] else None)) for s in run.segments]
        chain.append({"index": index, "label": f"R{index}", "path": str(run.path),
                      "parent_checkpoint": str(link["parent"] / link["checkpoint"]) if link and link.get("checkpoint") else None,
                      "resume_iteration": (link.get("checkpoint_iteration") if link.get("checkpoint_iteration") is not None else
                                           int(re.fullmatch(r"model_(\d+)\.pt", Path(link.get("checkpoint", "")).name)[1])
                                           if re.fullmatch(r"model_(\d+)\.pt", Path(link.get("checkpoint", "")).name) else None) if link else None,
                      "link_evidence": link["evidence"] if link else "root / single run",
                      "config_changed": bool(index > 1 and run.env != runs[index-2].env),
                      "agent_changed": bool(index > 1 and run.agent != runs[index-2].agent),
                      "checkpoints": len(run.checkpoints), "valid_checkpoints": sum(c["valid"] for c in run.checkpoints),
                      "first_step": min(s.first_step for s in run.segments), "last_step": max(s.last_step for s in run.segments),
                      "config_sha256": run.fingerprints, "segments": segments})
    latest["chain"] = chain
    combined = combine_runs(runs)
    latest["top_current_model_ids"] = [model_key(c) for c in combined.checkpoints
                                       if c.get("source_run", str(combined.path)) == str(runs[-1].path)
                                       and c["iteration"] in latest["top_current_stage"]]
    latest["chain_warnings"] = [f"R{i}: {warning}" for i, run in enumerate(runs, 1) for warning in run.warnings]
    return combined, latest
