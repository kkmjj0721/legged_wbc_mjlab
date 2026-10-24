"""Pure CPU task compilation; integer IDs are local to an ordered catalog."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, is_dataclass
from types import SimpleNamespace

import numpy as np


def plain(value):
    if isinstance(value, SimpleNamespace) or (
        is_dataclass(value) and not isinstance(value, type)
    ):
        return {k: plain(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(
            plain(value), sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def stable_seed(seed: int, key: str) -> int:
    return int(fingerprint([seed, key])[:16], 16)


def _finite(*values):
    if not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in values
    ):
        raise ValueError("Command values must be finite numbers")


@dataclass(frozen=True)
class AxisBin:
    name: str
    low: float
    high: float
    magnitude: bool = False
    positive_probability: float = 1.0
    sign: str = "within"


def compile_axis(axis: str, cfg) -> list[AxisBin]:
    modes = {"fixed", "continuous", "magnitude_bins", "signed_bins"}
    if cfg.mode not in modes:
        raise ValueError(f"{axis}: mode must be one of {sorted(modes)}")
    if cfg.signs not in {"both", "positive", "negative"}:
        raise ValueError(f"{axis}: invalid signs")
    _finite(cfg.positive_probability, cfg.value)
    if not 0 <= cfg.positive_probability <= 1:
        raise ValueError("positive_probability must lie in [0, 1]")
    if cfg.mode != "magnitude_bins" and (
        cfg.sign_as_task_axis or cfg.signs != "both" or cfg.positive_probability != 0.5
    ):
        raise ValueError(f"{axis}: sign rules require magnitude_bins")
    if cfg.mode == "fixed":
        if any(x is not None for x in (cfg.edges, cfg.bins, cfg.range)):
            raise ValueError(f"{axis}: fixed accepts value only")
        return [AxisBin("fixed", cfg.value, cfg.value)]
    if cfg.value != 0:
        raise ValueError(f"{axis}: value is only valid for fixed mode")
    if cfg.mode == "continuous":
        if (
            cfg.edges is not None
            or cfg.bins is not None
            or cfg.range is None
            or len(cfg.range) != 2
        ):
            raise ValueError(f"{axis}: continuous requires range only")
        lo, hi = cfg.range
        _finite(lo, hi)
        if lo >= hi:
            raise ValueError(f"{axis}: continuous range must have positive width")
        return [AxisBin("continuous", lo, hi)]
    if cfg.range is not None or (cfg.edges is None) == (cfg.bins is None):
        raise ValueError(f"{axis}: supply exactly one of edges or bins")
    if cfg.edges is not None:
        if len(cfg.edges) < 2:
            raise ValueError(f"{axis}: at least two edges required")
        bins = [
            (f"{axis}{i}", (lo, hi))
            for i, (lo, hi) in enumerate(zip(cfg.edges[:-1], cfg.edges[1:]))
        ]
    else:
        bins = list(cfg.bins.items())
    if not bins:
        raise ValueError(f"{axis}: bins cannot be empty")
    result, intervals = [], []
    for name, bounds in bins:
        if (
            not isinstance(name, str)
            or not name
            or any(c in name for c in "/=:")
            or len(bounds) != 2
        ):
            raise ValueError(f"{axis}: invalid bin name or interval")
        lo, hi = bounds
        _finite(lo, hi)
        if lo >= hi or (cfg.mode == "magnitude_bins" and lo < 0):
            raise ValueError(f"{axis}/{name}: invalid interval")
        if any(max(lo, a) < min(hi, b) for a, b in intervals):
            raise ValueError(f"{axis}: overlapping bins")
        intervals.append((lo, hi))
        if cfg.mode == "signed_bins":
            result.append(AxisBin(name, lo, hi))
            continue
        prob = {"positive": 1.0, "negative": 0.0, "both": cfg.positive_probability}[
            cfg.signs
        ]
        if cfg.sign_as_task_axis:
            signs = ["positive", "negative"] if cfg.signs == "both" else [cfg.signs]
            result.extend(
                AxisBin(name, lo, hi, True, float(s == "positive"), s) for s in signs
            )
        else:
            result.append(AxisBin(name, lo, hi, True, prob))
    return result


@dataclass(frozen=True)
class TaskRecord:
    key: str
    terrain: str
    level: str
    axes: tuple[AxisBin, AxisBin, AxisBin]
    mode: str = "fixed"

    def metadata(self):
        values = {"terrain": self.terrain, "level": self.level, "mode": self.mode}
        rotation_axis = "heading" if self.mode == "random_process" or "/heading=" in self.key else "wz"
        if self.mode == "random_process":
            values.update(Ax=self.axes[0].high, Ay=self.axes[1].high)
        for name, axis in zip(("vx", "vy", rotation_axis), self.axes):
            values[name] = axis.name
            values[name + "_sign"] = axis.sign
        return values


class TaskCatalog:
    def __init__(self, cfg):
        self.compact = getattr(cfg, "compact", False)
        self.heading_command = getattr(cfg, "heading_command", False)
        self.axis_names = ("vx", "vy", "heading" if self.heading_command else "wz")
        if set(cfg.command_axes) != set(self.axis_names):
            raise ValueError(f"command_axes must define {', '.join(self.axis_names)}")
        self.axis_bins = {
            axis: compile_axis(axis, cfg.command_axes[axis])
            for axis in self.axis_names
        }
        if self.heading_command and any(
            b.low < -math.pi or b.high > math.pi + 1e-12
            for b in self.axis_bins["heading"]
        ):
            raise ValueError("Heading offsets must be within [-pi, pi] radians")
        for terrain, family in cfg.terrains.items():
            if not family.levels:
                raise ValueError(f"{terrain}: no levels")
            for level in family.levels:
                if any(
                    not isinstance(x, str) or not x or any(c in x for c in "/=:")
                    for x in (terrain, level)
                ):
                    raise ValueError("Invalid terrain/level name")
        candidates = []
        fixed_axes = (
            itertools.product(*self.axis_bins.values())
            if getattr(cfg, "fixed_tasks", True) and not self.compact else ()
        )
        for axes in fixed_axes:
            for terrain, family in cfg.terrains.items():
                for level in family.levels:
                    parts = [
                        f"{name}={a.name}"
                        + (f":{a.sign}" if a.sign != "within" else "")
                        for name, a in zip(self.axis_names, axes)
                    ]
                    key = "/".join(parts + [f"terrain={terrain}", f"level={level}"])
                    candidates.append(TaskRecord(key, terrain, level, axes))

        self.random_process = getattr(cfg, "random_process", False)
        if self.random_process:
            for terrain, family in cfg.terrains.items():
                for level in family.levels:
                    for ax, ay in itertools.product(cfg.process_amplitudes, repeat=2):
                        key = f"mode=random_process/terrain={terrain}/level={level}/Ax={ax:g}/Ay={ay:g}"
                        axes = (AxisBin(f"Ax{ax:g}", -ax, ax), AxisBin(f"Ay{ay:g}", -ay, ay),
                                AxisBin("world_heading", *cfg.process_heading_range))
                        candidates.append(TaskRecord(key, terrain, level, axes, "random_process"))

        compact_families, self._sequences = {}, {}
        if self.compact:
            candidates, compact_families, self._sequences = self._compile_compact(cfg)
            self.sequence_hold_s = cfg.sequence_hold_s
            self.process_hold_s = tuple(cfg.process_hold_s)

        def matches(task, rule):
            meta = task.metadata()
            if self.compact:
                meta["family"] = compact_families[task.key]
            if not rule or any(k not in meta for k in rule):
                raise ValueError(f"Invalid filter: {rule}")
            return all(
                meta[k] in (v if isinstance(v, list) else [v]) for k, v in rule.items()
            )

        for rule in cfg.include + cfg.exclude:
            if not any(matches(t, rule) for t in candidates):
                raise ValueError(f"Filter matches no candidate: {rule}")
        self.records = tuple(
            t
            for t in candidates
            if (not cfg.include or any(matches(t, r) for r in cfg.include))
            and not any(matches(t, r) for r in cfg.exclude)
        )
        self._ids = {t.key: i for i, t in enumerate(self.records)}
        if not self.records or len(self._ids) != len(self.records):
            raise ValueError("Empty catalog or duplicate semantic keys")
        self.sample_bin_centers = getattr(cfg, "sample_bin_centers", False)
        signature = {"records": self.records, "axes": cfg.command_axes}
        self.task_families = tuple(compact_families.get(t.key, "legacy") for t in self.records)
        if self.compact:
            signature["compact"] = {
                "families": self.task_families,
                "sequences": {t.key: self._sequences[t.key] for t in self.records if t.mode == "sequence"},
                "sequence_hold_s": self.sequence_hold_s,
                "process_hold_s": self.process_hold_s,
            }
        # Centre sampling changes the distribution and therefore the identity.
        if self.sample_bin_centers:
            signature["sample_bin_centers"] = True
        self.fingerprint = fingerprint(signature)
        self._binned_axes = np.array([
            cfg.command_axes[axis].mode in {"magnitude_bins", "signed_bins"}
            for axis in self.axis_names
        ])
        self._lo = np.array(
            [[a.low for a in t.axes] for t in self.records], dtype=np.float64
        )
        self._hi = np.array(
            [[a.high for a in t.axes] for t in self.records], dtype=np.float64
        )
        self._mag = np.array([[a.magnitude for a in t.axes] for t in self.records])
        self._positive = np.array(
            [[a.positive_probability for a in t.axes] for t in self.records]
        )
        self.process_mask = np.array([t.mode in {"random_process", "random_body"} for t in self.records])
        self.sequence_mask = np.array([t.mode == "sequence" for t in self.records])

    @staticmethod
    def _compile_compact(cfg):
        """Compile the explicitly listed tasks; never form an axis product."""
        if cfg.heading_command or cfg.random_process:
            raise ValueError("Compact tasks require body-frame commands")
        if len(cfg.compact_tasks) == 0:
            raise ValueError("Compact requires a nonempty explicit task table")
        candidates, families, sequences = [], {}, {}
        for spec in cfg.compact_tasks:
            family, name = spec["family"], spec["name"]
            terrain, level, mode = spec["terrain"], spec["level"], spec["mode"]
            if family not in {"steady", "stairs", "sequence", "random"}:
                raise ValueError(f"Unknown Compact task family: {family}")
            if not isinstance(name, str) or not name or any(c in name for c in "/=:"):
                raise ValueError("Invalid Compact behavior name")
            # Evaluation can select a subset of the declared terrain conditions.
            if terrain not in cfg.terrains or level not in cfg.terrains[terrain].levels:
                continue
            expected = {"steady": "fixed", "stairs": "fixed", "sequence": "sequence", "random": "random_body"}
            if mode != expected[family]:
                raise ValueError(f"Compact {family} tasks require mode={expected[family]}")
            key = f"family={family}/behavior={name}/terrain={terrain}/level={level}"
            if mode == "random_body":
                bounds = np.asarray(spec["ranges"], dtype=float)
                if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
                    raise ValueError("Compact random ranges need three finite intervals")
                low, high = bounds.T
            else:
                commands = np.asarray(spec["commands"] if mode == "sequence" else [spec["command"]], dtype=float)
                if commands.ndim != 2 or commands.shape[1] != 3 or not len(commands) or not np.isfinite(commands).all():
                    raise ValueError("Compact commands must be a nonempty finite N x 3 sequence")
                low, high = commands.min(axis=0), commands.max(axis=0)
                if mode == "sequence":
                    sequences[key] = commands
            axes = tuple(AxisBin(axis, float(lo), float(hi)) for axis, lo, hi in zip(("vx", "vy", "wz"), low, high))
            candidates.append(TaskRecord(key, terrain, level, axes, mode))
            families[key] = family
        return candidates, families, sequences

    @property
    def num_tasks(self):
        return len(self.records)

    def encode(self, key: str) -> int:
        return self._ids[key]

    def decode(self, task_id: int) -> TaskRecord:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(task_id)
        return self.records[task_id]

    def sample_commands(self, ids, rng):
        """Sample task axes; the third value is a heading offset in heading mode."""
        ids = np.asarray(ids, dtype=np.int64)
        values = self._lo[ids] + rng.random((len(ids), 3)) * (
            self._hi[ids] - self._lo[ids]
        )
        if self.sample_bin_centers:
            mask = ~self.process_mask[ids, None] & self._binned_axes[None, :]
            values = np.where(mask, (self._lo[ids] + self._hi[ids]) / 2, values)
        signs = np.where(rng.random((len(ids), 3)) < self._positive[ids], 1.0, -1.0)
        values = np.where(self._mag[ids], values * signs, values)
        for row in np.flatnonzero(self.sequence_mask[ids]):
            values[row] = self._sequences[self.records[ids[row]].key][0]
        return values

    def sample_command_sequences(self, ids, command_count, rng):
        """Share training/evaluation semantics, preserving legacy random draws."""
        ids = np.asarray(ids, dtype=np.int64)
        if type(command_count) is not int or command_count <= 0:
            raise ValueError("command_count must be a positive integer")
        if not self.compact:
            return self.sample_commands(np.repeat(ids, command_count), rng).reshape(len(ids), command_count, 3)
        commands = np.empty((len(ids), command_count, 3), dtype=float)
        for row, tid in enumerate(ids):
            task = self.records[tid]
            if task.mode == "fixed":
                commands[row] = self.sample_commands([tid], rng)[0]
            elif task.mode == "sequence":
                script = self._sequences[task.key]
                commands[row] = script[np.minimum(np.arange(command_count), len(script) - 1)]
            else:
                commands[row] = self.sample_commands([tid] * command_count, rng)
        return commands

    def sample_command_durations(self, ids, command_count, rng, horizon_s):
        """Compact hold timers: whole episode, scripted segments or random holds."""
        if not self.compact:
            raise ValueError("Per-task duration sampling is only used by Compact")
        ids = np.asarray(ids, dtype=np.int64)
        if type(command_count) is not int or command_count <= 0 or not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("Positive command_count and horizon_s required")
        durations = np.full((len(ids), command_count), horizon_s, dtype=float)
        durations[self.sequence_mask[ids]] = self.sequence_hold_s
        random = self.process_mask[ids]
        durations[random] = rng.uniform(*self.process_hold_s, size=(int(random.sum()), command_count))
        return durations

    def initial_teacher_probabilities(self, seed_bin):
        """Lift FIBO's 2:1 vx seed-bin prior across all retained other axes."""
        available = [b for b in self.axis_bins["vx"]
                     if any(t.axes[0] == b for t in self.records)]
        if not 0 <= seed_bin < len(available):
            raise ValueError("seed_bin is outside the retained vx bins")
        weights = np.zeros(self.num_tasks)
        for index, mass in ((seed_bin, 1.0), (seed_bin + 1, 0.5)):
            if index < len(available):
                mask = np.array([t.axes[0] == available[index] for t in self.records])
                weights[mask] = mass / mask.sum()
        return weights / weights.sum()

    def classify_axis(self, axis: str, value: float) -> list[str]:
        _finite(value)
        bins = self.axis_bins[axis]
        maximum = max(b.high for b in bins)
        result = []
        for b in bins:
            x = abs(value) if b.magnitude else value
            inside = (
                x == b.low
                if b.low == b.high
                else b.low <= x < b.high or x == b.high == maximum
            )
            sign_ok = (
                not b.magnitude
                or value == 0
                or (value > 0 and b.positive_probability > 0)
                or (value < 0 and b.positive_probability < 1)
            )
            # Zero has one semantic sign when signs are separate task axes.
            if (
                value == 0
                and b.sign == "negative"
                and any(a.sign == "positive" and a.name == b.name for a in bins)
            ):
                sign_ok = False
            if inside and sign_ok:
                result.append(b.name + (":" + b.sign if b.sign != "within" else ""))
        if not result:
            raise ValueError(f"{axis}={value} lies outside the task space")
        return result


class TeacherGroups:
    """Group physical tasks by vx bin, terrain/level and actual travel direction.

    The ordered physical catalog is unchanged. Rotational bins (and any lateral
    bins) are sampled uniformly inside each group, independently of its LP.
    """

    def __init__(self, catalog):
        self.catalog = catalog
        grouped = {}
        self.task_direction_to_group = {}
        for tid, task in enumerate(catalog.records):
            axis = task.axes[0]
            if catalog.compact:
                key = fingerprint(["compact", task.key])
                grouped[key] = {"key": key, "mode": task.mode, "vx_bin": plain(axis),
                                "terrain": task.terrain, "level": task.level,
                                "family": catalog.task_families[tid], "direction": 0,
                                "task_ids": [tid]}
                continue
            if task.mode == "random_process":
                key = fingerprint([task.mode, task.terrain, task.level, task.axes[:2]])
                grouped[key] = {"key": key, "mode": task.mode, "vx_bin": plain(axis),
                                "terrain": task.terrain, "level": task.level,
                                "Ax": axis.high, "Ay": task.axes[1].high,
                                "direction": 0, "task_ids": [tid]}
                continue
            if axis.magnitude:
                directions = ([1] if axis.positive_probability > 0 else []) + (
                    [-1] if axis.positive_probability < 1 else [])
            else:
                directions = ([1] if axis.high > 0 else []) + ([-1] if axis.low < 0 else [])
                if not directions:
                    directions = [0]
            for direction in directions:
                key = fingerprint([axis, task.terrain, task.level, direction])
                if key not in grouped:
                    grouped[key] = {"key": key, "vx_bin": plain(axis),
                                    "terrain": task.terrain, "level": task.level,
                                    "direction": direction, "task_ids": []}
                grouped[key]["task_ids"].append(tid)
        self.records = tuple(grouped.values())
        self.num_groups = len(self.records)
        self.families = tuple(r.get("family", "legacy") for r in self.records)
        self.directions = np.array([r["direction"] for r in self.records], dtype=np.int64)
        self.members = tuple(np.array(r["task_ids"], dtype=np.int64) for r in self.records)
        self.member_counts = np.array([len(m) for m in self.members])
        self.member_table = np.zeros((self.num_groups, self.member_counts.max()), dtype=np.int64)
        for gid, members in enumerate(self.members):
            self.member_table[gid, :len(members)] = members
        for gid, row in enumerate(self.records):
            for tid in row["task_ids"]:
                self.task_direction_to_group[tid, row["direction"]] = gid
        if catalog.compact:
            self.task_to_group = np.array([
                self.task_direction_to_group[tid, 0] for tid in range(catalog.num_tasks)
            ], dtype=np.int64)

    def initial_probabilities(self, seed_bin):
        if self.catalog.random_process or self.catalog.compact:
            return np.full(self.num_groups, 1 / self.num_groups)
        task_prior = self.catalog.initial_teacher_probabilities(seed_bin)
        p = np.zeros(self.num_groups)
        for gid, row in enumerate(self.records):
            # Teacher directions are balanced even when the catalog samples signs
            # within a task. Lift the existing low-speed prior across these groups.
            signs = sum((row["task_ids"][0], d) in self.task_direction_to_group for d in (-1, 0, 1))
            p[gid] = task_prior[row["task_ids"]].sum() / signs
        return p / p.sum()

    def sample(self, probabilities, count, rng, command_count=1):
        gids = rng.choice(self.num_groups, count, p=probabilities)
        tids = self.member_table[gids, rng.integers(self.member_counts[gids])]
        commands = self.catalog.sample_command_sequences(tids, command_count, rng)
        if self.catalog.compact:
            return gids, tids, commands
        magnitude = self.catalog._mag[tids, 0]
        commands[magnitude, :, 0] = np.abs(commands[magnitude, :, 0]) * self.directions[gids[magnitude], None]
        for i in np.flatnonzero(~magnitude):
            tid, gid = tids[i], gids[i]
            if self.catalog.process_mask[tid]:
                continue
            axis = self.catalog.records[tid].axes[0]
            direction = self.directions[gid]
            if axis.low < 0 < axis.high:
                lo, hi = (0, axis.high) if direction > 0 else (axis.low, 0)
                if self.catalog.sample_bin_centers and self.catalog._binned_axes[0]:
                    commands[i, :, 0] = (lo + hi) / 2
                else:
                    commands[i, :, 0] = rng.uniform(lo, hi, command_count)
        return gids, tids, commands


def make_trial_table(catalog, cfg):
    trials = []
    interval = cfg.task_space.resampling_interval_s
    count = 1 if interval is None else math.ceil(cfg.evaluation.horizon_s / interval) + 1
    for tid, task in enumerate(catalog.records):
        for trial_id in range(cfg.evaluation.trials_per_task):
            rng = np.random.default_rng(
                stable_seed(cfg.evaluation.seed, task.key + f"/trial={trial_id}")
            )
            process = task.mode == "random_process"
            if catalog.compact:
                minimum_hold = min(cfg.task_space.process_hold_s[0], cfg.task_space.sequence_hold_s)
                n = math.ceil(cfg.evaluation.horizon_s / minimum_hold) + 1
                parameters = catalog.sample_command_sequences([tid], n, rng)[0]
                durations = catalog.sample_command_durations([tid], n, rng, cfg.evaluation.horizon_s)[0]
            else:
                n = math.ceil(cfg.evaluation.horizon_s / cfg.task_space.process_hold_s[0]) + 1 if process else count
                parameters = catalog.sample_command_sequences([tid], n, rng)[0]
                durations = rng.uniform(*cfg.task_space.process_hold_s, size=n) if process else None
            replica = int(rng.integers(cfg.geometry.replicas))
            yaw = float(rng.uniform(-math.pi, math.pi))
            commands = parameters.copy()
            heading = {}
            if catalog.heading_command:
                targets = ((0 if process else yaw) + parameters[:, 2] + math.pi) % (2 * math.pi) - math.pi
                errors = (targets - yaw + math.pi) % (2 * math.pi) - math.pi
                commands[:, 2] = np.clip(
                    cfg.task_space.heading_control_stiffness * errors,
                    *cfg.task_space.ang_vel_z_range,
                )
                heading = {
                    "heading_target": float(targets[0]),
                    "heading_offset": float(parameters[0, 2]),
                    "heading_sequence": targets.tolist(),
                }
            commands = commands.tolist()
            trials.append(
                {
                    "task_id": tid,
                    "task_key": task.key,
                    "trial_id": trial_id,
                    "command": commands[0],
                    "command_sequence": commands,
                    "replica": replica,
                    "yaw": yaw,
                    "task_mode": task.mode,
                    "spawn_choice": float(rng.random()),
                    **({"command_hold_s": durations.tolist()} if durations is not None else {}),
                    **({"task_family": catalog.task_families[tid]} if catalog.compact else {}),
                    **heading,
                }
            )
    return trials
