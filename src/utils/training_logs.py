"""Read immutable snapshots of RSL-RL logs without importing task code."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re

import yaml


REWARD = "Train/mean_reward"
LENGTH = "Train/mean_episode_length"


class ConfigLoader(yaml.SafeLoader):
    """Read Python-tagged dump_yaml output as inert data, never Python objects."""


def _python_data(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node) or suffix.removeprefix("name:")
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_data)


@dataclass(frozen=True)
class Point:
    step: int
    value: float
    wall_time: float


@dataclass
class Segment:
    index: int
    series: dict[str, dict[int, Point]] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def first_step(self):
        return min(self.series[REWARD])

    @property
    def last_step(self):
        return max(self.series[REWARD])


@dataclass
class Run:
    path: Path
    env: dict
    agent: dict
    segments: list[Segment]
    checkpoints: list[dict]
    warnings: list[str]
    fingerprints: dict[str, str]
    event_files: list[dict]
    sources: dict[str, "Run"] = field(default_factory=dict)


def model_key(row):
    """Checkpoint identity; an iteration alone is only unique within one run."""
    return row.get("model_id", row["iteration"])


def model_label(row):
    return row.get("label", f"model_{row['iteration']}")


def model_origin(row):
    return {key: row[key] for key in ("model_id", "label", "source_run", "source_index") if key in row}


def source_run(run, checkpoint):
    return run.sources.get(checkpoint.get("source_run"), run)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_records(records: list[tuple[str, Point]]) -> tuple[list[Segment], int]:
    """Separate step rewinds; latest wall-time wins duplicate tag/step pairs.

    Time-indexed and performance series must not drive iteration boundaries.
    Input records are ordered by wall time, with stable ordering for ties.
    """
    segments = [Segment(0)]
    previous_step = None
    duplicates = 0
    for tag, point in sorted(records, key=lambda record: record[1].wall_time):
        if tag.endswith("/time") or tag.startswith("Perf/"):
            continue
        if previous_step is not None and point.step < previous_step:
            segments.append(Segment(len(segments)))
        current = segments[-1]
        if not current.series:
            current.start_time = point.wall_time
        current.end_time = point.wall_time
        series = current.series.setdefault(tag, {})
        duplicates += int(point.step in series)
        series[point.step] = point
        previous_step = point.step
    return [s for s in segments if s.series.get(REWARD)], duplicates


def inspect_checkpoint(path: Path, iteration: int) -> dict:
    """Validate the saved state on CPU, retaining only small metadata."""
    import torch

    before = path.stat()
    result = {"path": str(path), "iteration": iteration, "mtime": before.st_mtime,
              "size": before.st_size, "mtime_ns": before.st_mtime_ns,
              "valid": False, "error": None, "env_steps": None}
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if int(saved.get("iter", -1)) != iteration:
            raise ValueError("checkpoint iter does not match its numeric filename")
        actor = saved.get("actor_state_dict")
        if not isinstance(actor, dict) or not actor:
            raise ValueError("missing actor_state_dict (legacy format is unsupported)")

        def check_tensors(value, prefix=""):
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError(f"non-finite tensor: {prefix}")
                if prefix.endswith(("._var", ".count")) and (value < 0).any():
                    raise ValueError(f"invalid normalizer: {prefix}")
                if prefix.endswith("._std") and (value <= 0).any():
                    raise ValueError(f"invalid normalizer: {prefix}")
            elif isinstance(value, dict):
                for key, item in value.items():
                    check_tensors(item, f"{prefix}.{key}")
            elif isinstance(value, (list, tuple)):
                for key, item in enumerate(value):
                    check_tensors(item, f"{prefix}.{key}")

        check_tensors(saved)
        infos = saved.get("infos") or {}
        result["env_steps"] = (infos.get("env_state") or {}).get("common_step_counter")
        if result["env_steps"] is not None:
            result["env_steps"] = int(result["env_steps"])
            if result["env_steps"] < 0:
                raise ValueError("negative environment step counter")
        contract = infos.get("him_numerics") or {}
        bounds = contract.get("action_observation_slice")
        clip = contract.get("action_clip")
        if bounds and clip is not None:
            for name in ("actor_state_dict", "critic_state_dict"):
                state = saved.get(name, {})
                for key in ("obs_normalizer._mean", "obs_normalizer._std"):
                    value = state.get(key)
                    if value is None:
                        continue
                    if name == "actor_state_dict":
                        value = value.reshape(contract["history_size"], -1)
                    if (value[..., bounds[0]:bounds[1]].abs() > clip + 1e-4).any():
                        raise ValueError(f"{name} last-action normalizer exceeds clipping contract")
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError("checkpoint changed while being read; retry after saving completes")
        result["valid"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_run(directory: str | Path) -> Run:
    from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.util.tensor_util import make_ndarray

    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"run directory does not exist: {path}")
    warnings, fingerprints = [], {}

    def config(name):
        source = path / "params" / f"{name}.yaml"
        if not source.exists():
            warnings.append(f"missing {source.name}; some conclusions will be unavailable")
            return {}
        raw = source.read_bytes()
        fingerprints[name] = hashlib.sha256(raw).hexdigest()
        value = yaml.load(raw, Loader=ConfigLoader)
        if not isinstance(value, dict):
            raise ValueError(f"{source} must contain a mapping")
        return value

    env, agent = config("env"), config("agent")
    records, event_files = [], []
    for source in sorted(path.glob("events.out.tfevents.*")):
        stat = source.stat()
        event_files.append({"name": source.name, "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns})
        consumed = 0
        try:
            # A TFRecord consists of a payload plus 16 bytes of length/CRC fields.
            # Bound reads by the size at open, even if training appends meanwhile.
            for payload in RawEventFileLoader(str(source)).Load():
                record_end = consumed + len(payload) + 16
                if record_end > stat.st_size:
                    break
                consumed = record_end
                event = Event.FromString(payload)
                for value in event.summary.value:
                    if value.HasField("simple_value"):
                        scalar = float(value.simple_value)
                    elif value.HasField("tensor"):
                        array = make_ndarray(value.tensor)
                        if array.size != 1 or array.dtype.kind not in "biuf":
                            continue
                        scalar = float(array.item())
                    else:
                        continue
                    records.append((value.tag, Point(int(event.step), scalar, event.wall_time)))
        except Exception as exc:
            warnings.append(f"event file {source.name}: {type(exc).__name__}: {exc}")
        event_files[-1]["complete_bytes_read"] = consumed
        if consumed < stat.st_size:
            warnings.append(f"ignored incomplete or corrupt tail in {source.name} ({stat.st_size-consumed} bytes)")
    segments, duplicates = split_records(records)
    if not segments:
        raise ValueError(f"no {REWARD} iteration scalars found in {path}")
    if duplicates:
        warnings.append(f"replaced {duplicates} duplicate tag/step samples by their latest values")
    if len(segments) > 1:
        warnings.append("iteration rewind detected; segments are analyzed separately, checkpoint association uses mtime")
    checkpoints = []
    for source in path.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", source.name)
        if match and source.is_file():
            checkpoint = inspect_checkpoint(source, int(match[1]))
            checkpoint["segment"] = None
            for segment in segments:
                if not segment.first_step <= checkpoint["iteration"] <= segment.last_step:
                    continue
                if len(segments) == 1:
                    checkpoint["segment"] = segment.index
                else:
                    next_times = [s.start_time for s in segments if s.index > segment.index]
                    end = min(next_times, default=math.inf)
                    if segment.start_time <= checkpoint["mtime"] < end:
                        checkpoint["segment"] = segment.index
            checkpoints.append(checkpoint)
    return Run(path, env, agent, segments, sorted(checkpoints, key=lambda c: c["iteration"]),
               warnings, fingerprints, event_files)
