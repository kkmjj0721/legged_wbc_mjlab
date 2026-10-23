"""Signed learning progress, transactional updates, and reset-time task assignment."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.recorder_manager import RecorderTerm
from mjlab.utils.lab_api.math import wrap_to_pi

from .task_catalog import TeacherGroups, plain


def fixed_family_allocation(num_envs, fractions):
    """Largest-remainder quotas, fixed per local environment for the whole run.

    Every rank uses the same local rule; the runner separately checks world size.
    Reset frequency therefore cannot change a family's share of control steps.
    """
    names = tuple(sorted(fractions))
    weights = np.asarray([fractions[name] for name in names], dtype=np.float64)
    if (not names or not np.isfinite(weights).all() or (weights <= 0).any()
            or not np.isclose(weights.sum(), 1)):
        raise ValueError("Family fractions must be positive and sum to one")
    exact = num_envs * weights
    counts = np.floor(exact).astype(np.int64)
    remainder = num_envs - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1
    if (counts == 0).any():
        raise ValueError("Compact training needs enough environments to represent every family (at least 4)")
    return names, counts, np.repeat(np.arange(len(names)), counts)


class LPState:
    """Signed LP over normalized, episode-weighted scores for teacher groups.

    FIBO supplies signed LP/softmax/mixing. Sparse-group sample carry and the
    per-group baseline are local extensions, not upstream parity behavior.
    """

    VERSION = 7
    COMPACT_VERSION = 8
    SCORE_KIND = "linear_angular_tracking_failure_budget_v3"
    ARRAY_FIELDS = (
        "probabilities", "mean_scores", "lp", "counts", "pending_sums",
        "pending_counts", "stage_counts", "iteration_counts", "reported_counts",
        "baseline_valid", "comparison_valid",
        "baseline_updates",
    )
    CLOCK_FIELDS = ("completed_policy_updates", "last_stage_update", "stage_index")

    def __init__(self, num_tasks, cfg, fingerprints=None, initial_probabilities=None,
                 group_families=None):
        if num_tasks < 1 or not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
            raise ValueError("Positive group count and finite positive temperature required")
        if cfg.method not in {"lp", "uniform"} or not 0 <= cfg.uniform_mix < 1:
            raise ValueError("Invalid curriculum method/mixing coefficient")
        if cfg.method == "uniform" and cfg.uniform_mix:
            raise ValueError("Uniform requires zero mixing coefficient")
        for name in ("stage_length_updates", "min_episodes_per_group"):
            if type(getattr(cfg, name)) is not int or getattr(cfg, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(cfg.seed_bin) is not int or cfg.seed_bin < 0:
            raise ValueError("seed_bin must be a nonnegative integer")
        # num_tasks is retained as an API alias; its dimension now counts groups.
        self.num_tasks = self.num_groups = num_tasks
        self.cfg, self.fingerprints = cfg, dict(fingerprints or {})
        self.group_families = tuple(group_families) if group_families is not None else None
        self.version = self.COMPACT_VERSION if self.group_families is not None else self.VERSION
        if self.group_families is not None:
            if (len(self.group_families) != num_tasks
                    or set(self.group_families) != set(cfg.family_fractions)):
                raise ValueError("Compact teacher families must match the task groups")
            self.family_masks = {
                name: np.asarray([family == name for family in self.group_families])
                for name in sorted(cfg.family_fractions)
            }
            initial_probabilities = self._probabilities(np.zeros(num_tasks))
        elif cfg.method == "uniform":
            initial_probabilities = np.full(num_tasks, 1.0 / num_tasks)
        elif initial_probabilities is None:
            if cfg.seed_bin >= num_tasks:
                raise ValueError("seed_bin is outside the group table")
            initial_probabilities = np.zeros(num_tasks)
            initial_probabilities[cfg.seed_bin] = 1.0
            if cfg.seed_bin + 1 < num_tasks:
                initial_probabilities[cfg.seed_bin + 1] = 0.5
            initial_probabilities /= initial_probabilities.sum()
        initial = np.asarray(initial_probabilities, dtype=np.float64)
        if (initial.shape != (num_tasks,) or not np.isfinite(initial).all()
                or (initial < 0).any() or not np.isclose(initial.sum(), 1)):
            raise ValueError("Invalid teacher initialization")
        self.initial_probabilities = initial.copy()
        for name in self.ARRAY_FIELDS:
            dtype = (np.bool_ if name.endswith("valid") else
                     np.int64 if name.endswith(("counts", "updates")) else np.float64)
            setattr(self, name, np.zeros(num_tasks, dtype=dtype))
        self.probabilities = initial.copy()
        self.completed_policy_updates = self.last_stage_update = self.stage_index = 0
        self.probability_l1_change = 0.0

    def add_samples(self, score_sums, counts):
        sums, counts = np.asarray(score_sums, dtype=np.float64), np.asarray(counts)
        if (sums.shape != (self.num_groups,) or counts.shape != sums.shape
                or not np.issubdtype(counts.dtype, np.integer)
                or not np.isfinite(sums).all() or (counts < 0).any()
                or (sums < 0).any() or (sums > counts + 1e-6).any()
                or (sums[counts == 0] != 0).any()):
            raise ValueError("Invalid tracking samples")
        self.pending_sums += sums
        self.pending_counts += counts
        self.counts += counts

    def advance(self, update_count):
        if type(update_count) is not int or update_count != self.completed_policy_updates + 1:
            raise ValueError("Curriculum must consume every PPO iteration in order")
        self.iteration_counts = self.counts - self.reported_counts
        self.reported_counts = self.counts.copy()
        self.completed_policy_updates = update_count
        if update_count % self.cfg.stage_length_updates:
            return False
        ready = self.pending_counts >= self.cfg.min_episodes_per_group
        self.comparison_valid = ready & self.baseline_valid
        self.lp.fill(0)
        means = self.pending_sums[ready] / self.pending_counts[ready]
        self.lp[ready] = np.where(self.baseline_valid[ready], means - self.mean_scores[ready], 0)
        self.mean_scores[ready] = means
        self.baseline_valid[ready] = True
        self.baseline_updates[ready] = update_count
        self.stage_counts = np.where(ready, self.pending_counts, 0)
        self.pending_sums[ready] = 0
        self.pending_counts[ready] = 0
        self.last_stage_update = update_count
        self.stage_index += 1
        # The first stage establishes baselines; later stages give missing/new
        # groups zero LP, never a fabricated zero competence score.
        if self.stage_index > 1:
            previous = self.probabilities
            self.probabilities = self._probabilities(self.lp)
            self.probability_l1_change = float(np.abs(self.probabilities - previous).sum())
        return True

    def _probabilities(self, delta):
        if self.group_families is not None:
            result = np.zeros(self.num_groups)
            for name, mask in self.family_masks.items():
                count = int(mask.sum())
                if self.cfg.method == "uniform" or name in self.cfg.uniform_families:
                    conditional = np.full(count, 1.0 / count)
                else:
                    scaled = delta[mask] / max(self.cfg.temperature, 1e-8)
                    exp = np.exp(scaled - scaled.max())
                    conditional = ((1 - self.cfg.uniform_mix) * exp / exp.sum()
                                   + self.cfg.uniform_mix / count)
                result[mask] = self.cfg.family_fractions[name] * conditional
            return result
        if self.cfg.method == "uniform":
            return np.full(self.num_groups, 1.0 / self.num_groups)
        scaled = delta / max(self.cfg.temperature, 1e-8)
        exp = np.exp(scaled - scaled.max())
        return ((1 - self.cfg.uniform_mix) * exp / exp.sum()
                + self.cfg.uniform_mix / self.num_groups)

    def state_dict(self):
        return {
            "version": self.version, "score_kind": self.SCORE_KIND,
            "num_groups": self.num_groups, "fingerprints": dict(self.fingerprints),
            "initial_probabilities": self.initial_probabilities.copy(),
            "curriculum_config": plain(self.cfg),
            "probability_l1_change": self.probability_l1_change,
            **{name: getattr(self, name).copy() for name in self.ARRAY_FIELDS},
            **{name: getattr(self, name) for name in self.CLOCK_FIELDS},
        }

    def load_state_dict(self, state):
        if (state.get("version") != self.version
                or state.get("score_kind") != self.SCORE_KIND
                or state.get("num_groups") != self.num_groups
                or state.get("fingerprints") != self.fingerprints
                or state.get("curriculum_config") != plain(self.cfg)
                or not np.array_equal(state.get("initial_probabilities"), self.initial_probabilities)):
            raise ValueError("Incompatible teacher state; use explicit warm-start for older curricula")
        arrays = {}
        probability_change = state.get("probability_l1_change", 0.0)
        if not math.isfinite(probability_change) or not 0 <= probability_change <= 2 + 1e-12:
            raise ValueError("Invalid checkpoint probability change")
        for name in self.ARRAY_FIELDS:
            a = np.asarray(state[name])
            if a.shape != (self.num_groups,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid checkpoint {name}")
            if name.endswith("valid"):
                if a.dtype != np.bool_:
                    raise ValueError(f"Invalid checkpoint {name}")
                dtype = np.bool_
            elif name.endswith(("counts", "updates")):
                if not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
                    raise ValueError(f"Invalid checkpoint {name}")
                dtype = np.int64
            else:
                dtype = np.float64
            arrays[name] = a.astype(dtype, copy=True)
        clocks = {name: state[name] for name in self.CLOCK_FIELDS}
        completed, last, stage = (clocks[name] for name in self.CLOCK_FIELDS)
        if any(type(v) is not int for v in clocks.values()):
            raise ValueError("Invalid checkpoint stage clock")
        expected_stage = completed // self.cfg.stage_length_updates
        if completed < 0 or stage != expected_stage or last != expected_stage * self.cfg.stage_length_updates:
            raise ValueError("Invalid checkpoint stage clock")
        expected_p = self.initial_probabilities if stage < 2 else self._probabilities(arrays["lp"])
        if (not np.allclose(arrays["probabilities"], expected_p, atol=1e-12, rtol=1e-10)
                or (arrays["probabilities"] < 0).any()
                or not np.isclose(arrays["probabilities"].sum(), 1)
                or ((arrays["mean_scores"] < 0) | (arrays["mean_scores"] > 1)).any()
                or (arrays["mean_scores"][~arrays["baseline_valid"]] != 0).any()
                or (np.abs(arrays["lp"]) > 1).any()
                or (arrays["lp"][~arrays["comparison_valid"]] != 0).any()
                or (arrays["comparison_valid"] & ~arrays["baseline_valid"]).any()
                or (arrays["baseline_valid"] & (arrays["counts"] < self.cfg.min_episodes_per_group)).any()
                or (arrays["comparison_valid"] & (arrays["stage_counts"] == 0)).any()
                or (stage < 2 and arrays["comparison_valid"].any())
                or (stage == 0 and arrays["baseline_valid"].any())
                or (arrays["baseline_updates"] > completed).any()
                or (arrays["baseline_updates"][~arrays["baseline_valid"]] != 0).any()
                or (arrays["pending_sums"] < 0).any()
                or (arrays["pending_sums"] > arrays["pending_counts"] + 1e-6).any()
                or (arrays["pending_sums"][arrays["pending_counts"] == 0] != 0).any()
                or (arrays["pending_counts"] > arrays["counts"]).any()
                or (arrays["stage_counts"] > arrays["counts"]).any()
                or (arrays["pending_counts"] + arrays["stage_counts"] > arrays["counts"]).any()
                or ((arrays["stage_counts"] > 0) & (arrays["stage_counts"] < self.cfg.min_episodes_per_group)).any()
                or (arrays["reported_counts"] > arrays["counts"]).any()
                or (arrays["iteration_counts"] > arrays["reported_counts"]).any()):
            raise ValueError("Inconsistent teacher checkpoint statistics")
        for name, value in {**arrays, **clocks}.items():
            setattr(self, name, value)
        self.probability_l1_change = probability_change

    def diagnostics(self):
        p = self.probabilities
        positive = p[p > 0]
        entropy = -np.sum(positive * np.log(positive))
        return {
            "entropy_normalized": float(entropy / math.log(self.num_groups)) if self.num_groups > 1 else 0.0,
            "max_probability": float(p.max()), "min_probability": float(p.min()),
            "probability_l1_change": self.probability_l1_change,
            "effective_groups": float(1.0 / np.square(p).sum()), "num_groups": self.num_groups,
            "lp_min": float(self.lp.min()), "lp_max": float(self.lp.max()), "lp_std": float(self.lp.std()),
            "stage_index": self.stage_index, "completed_episodes": int(self.counts.sum()),
            "iteration_episodes": int(self.iteration_counts.sum()),
            "sampled_groups": int((self.iteration_counts > 0).sum()),
            "baseline_groups": int(self.baseline_valid.sum()),
            "comparison_groups": int(self.comparison_valid.sum()),
            "pending_episodes": int(self.pending_counts.sum()),
            "minimum_pending_samples": int(self.pending_counts.min()),
            "maximum_baseline_age_updates": int((self.completed_policy_updates - self.baseline_updates[self.baseline_valid]).max()) if self.baseline_valid.any() else 0,
            "mean_tracking_score": float(self.mean_scores[self.baseline_valid].mean()) if self.baseline_valid.any() else 0.0,
            "temperature": float(self.cfg.temperature), "uniform_mix": float(self.cfg.uniform_mix),
        }


def episode_mean_tracking_kernel(env, env_ids, name, weight):
    """Recover a bounded tracking kernel, undoing reward weight and dt once."""
    steps = env.episode_length_buf[env_ids].clamp(min=1).float()
    terms = env.reward_manager._episode_sums[name][env_ids]
    dt_scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    if not math.isfinite(dt_scale) or dt_scale <= 0 or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Tracking score needs positive finite weight and dt")
    raw = terms / steps / float(weight) / dt_scale
    return torch.where(torch.isfinite(raw), raw.clamp(0, 1), torch.full_like(raw, float("nan")))


def episode_tracking_scores(env, env_ids, linear_weight, angular_weight):
    """Joint tracking with a full-horizon budget for physical failures.

    Following the reference terrain curriculum, use both accumulated tracking
    rewards. Normalize their positive weights to keep the LP score in [0, 1].
    For a fall, unexecuted steps contribute zero; for a normal timeout or terrain
    boundary, use the observed segment mean. A boundary is not a fall, and a
    simultaneous physical failure takes precedence over truncation.

    This is a reward score, not the paper's EPTE-SP error metric. Its changed
    units/semantics require a new teacher state; old score baselines cannot resume.
    """
    linear = episode_mean_tracking_kernel(env, env_ids, "track_linear_velocity", linear_weight)
    angular = episode_mean_tracking_kernel(env, env_ids, "track_angular_velocity", angular_weight)
    score = (linear_weight * linear + angular_weight * angular) / (linear_weight + angular_weight)
    horizon = env.max_episode_length
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Teacher score requires a positive episode horizon")
    fraction = (env.episode_length_buf[env_ids].float() / horizon).clamp(0, 1)
    failed = env.termination_manager.terminated[env_ids]
    return score * torch.where(failed, fraction, torch.ones_like(fraction))


def get_controller(env):
    return env.curriculum_manager.get_term_cfg("lpacrl").func


class CurriculumController(ManagerTermBase):
    START_COUNT_FIELDS = ("starts", "group_starts", "coverage_starts")

    def __init__(self, cfg, env):
        super().__init__(env)
        from ..config.curriculum_cfg import (
            build_curriculum,
            environment_signature,
            finalize_fingerprints,
        )

        self.mode = cfg.params["mode"]
        expected = cfg.params["expected_signature"]
        if self.mode != "play" and expected != environment_signature(env.cfg):
            raise ValueError(
                "LP task structure changed after assembly. Edit the central LPACRL "
                "config or env_cfg factory so training and evaluation stay consistent."
            )
        self.setup = build_curriculum(cfg.params["task_cfg"])
        finalize_fingerprints(self.setup, env.cfg)
        if (
            self.mode != "play"
            and cfg.params["fingerprints"] != self.setup.fingerprints
        ):
            raise ValueError("LP task parameters changed after environment assembly")
        self.cfg = self.setup.cfg
        self.cfg.training.num_envs = (
            env.num_envs if self.mode != "evaluation" else self.cfg.training.num_envs
        )
        self.cfg.training.seed = env.cfg.seed
        self.catalog = self.setup.catalog
        self.groups = TeacherGroups(self.catalog)
        self.compact = getattr(self.catalog, "compact", False)
        self.terrain_map = self.setup.terrain_map
        self.terrain_map.bind(env.scene.terrain.terrain_origins)
        self.state = LPState(
            self.groups.num_groups, self.cfg.curriculum, cfg.params.get("fingerprints"),
            (np.full(self.groups.num_groups, 1 / self.groups.num_groups) if self.terrain_map.connected
             else self.groups.initial_probabilities(self.cfg.curriculum.seed_bin)),
            group_families=self.groups.families if self.compact and self.mode == "train" else None,
        )
        if self.mode == "train" and self.cfg.task_space.resampling_interval_s is not None:
            raise ValueError("Training commands must stay fixed for each episode")
        self.rng = np.random.default_rng(self.cfg.training.seed)
        n, device = env.num_envs, env.device
        self.family_names = tuple(sorted(self.cfg.curriculum.family_fractions)) if self.compact else ()
        self.family_counts = None
        self.env_families = None
        if self.compact and self.mode == "train":
            self.family_names, self.family_counts, self.env_families = fixed_family_allocation(
                n, self.cfg.curriculum.family_fractions)
        self.episode_task_id = torch.full((n,), -1, device=device, dtype=torch.long)
        self.episode_group_id = torch.full_like(self.episode_task_id, -1)
        # 0 = regular teacher distribution; 1 = fixed forward stair coverage.
        self.episode_sampling_mode = torch.zeros_like(self.episode_task_id)
        self.episode_geometry_id = torch.full_like(self.episode_task_id, -1)
        self.commands = torch.zeros(n, 3, device=device)
        interval_s = self.cfg.task_space.resampling_interval_s
        self.command_interval_steps = None
        self.command_count = 1
        if interval_s is not None:
            interval = interval_s / env.step_dt
            if (
                not math.isclose(interval, round(interval), abs_tol=1e-6)
                or interval < 1
            ):
                raise ValueError(
                    "Command interval must be a positive multiple of step_dt"
                )
            self.command_interval_steps = round(interval)
            self.command_count = (
                math.ceil(
                    max(self.cfg.training.horizon_s, self.cfg.evaluation.horizon_s)
                    / interval_s
                )
                + 1
            )
        self.command_sequences = torch.zeros(n, self.command_count, 3, device=device)
        if self.catalog.random_process or self.compact:
            minimum_hold = self.cfg.task_space.process_hold_s[0]
            if self.compact:
                minimum_hold = min(minimum_hold, self.cfg.task_space.sequence_hold_s)
            self.command_count = math.ceil(max(self.cfg.training.horizon_s, self.cfg.evaluation.horizon_s)
                                           / minimum_hold) + 1
            self.command_sequences = torch.zeros(n, self.command_count, 3, device=device)
        self.process_tasks = torch.as_tensor(self.catalog.process_mask, device=device)
        self.scheduled_tasks = (torch.ones_like(self.process_tasks) if self.compact else self.process_tasks)
        self.command_deadlines = torch.full((n, self.command_count), torch.iinfo(torch.long).max,
                                            dtype=torch.long, device=device)
        self.command_index = torch.zeros(n, dtype=torch.long, device=device)
        self.command_event = torch.zeros(n, dtype=torch.bool, device=device)
        self.command_event_step = torch.full((n,), -1, dtype=torch.long, device=device)
        self.command_event_delta = torch.zeros(n, 3, device=device)
        self.command_heading_delta = torch.zeros(n, device=device)
        self.switch_counts = torch.zeros(n, dtype=torch.long, device=device)
        self._last_command_step = torch.full((n,), -1, dtype=torch.long, device=device)
        self.heading_sequences = torch.zeros(n, self.command_count, device=device)
        self.heading_targets = torch.zeros(n, device=device)
        self.heading_error = torch.zeros(n, device=device)
        self._heading_reference_pending = torch.zeros(n, dtype=torch.bool, device=device)
        # 评估出界后继续同一试验；指令时钟不能随物理 reset 回到零。
        self.evaluation_step_offsets = torch.zeros(n, dtype=torch.long, device=device)
        self.yaws = torch.zeros(n, device=device)
        self._natural = torch.zeros(n, dtype=torch.bool, device=device)
        self._discard_depth = 0
        self.starts = np.zeros(self.catalog.num_tasks, dtype=np.int64)
        self.group_starts = np.zeros(self.groups.num_groups, dtype=np.int64)
        self.coverage_starts = np.zeros(self.catalog.num_tasks, dtype=np.int64)
        self._synced_starts = {
            name: getattr(self, name).copy() for name in self.START_COUNT_FIELDS
        }
        self.coverage_counts = np.zeros(self.catalog.num_tasks, dtype=np.int64)
        self._coverage_counts = torch.zeros(self.catalog.num_tasks, dtype=torch.long, device=device)
        self.coverage_tasks = self._coverage_task_ids()
        self._score_sums = torch.zeros(
            self.groups.num_groups, dtype=torch.float32, device=device
        )
        self._score_counts = torch.zeros(
            self.groups.num_groups, dtype=torch.long, device=device
        )
        self.statistics_host_seconds = 0.0
        self._diagnostics = self.state.diagnostics()
        if self.mode == "train":
            self._tracking_weight = env.reward_manager.get_term_cfg("track_linear_velocity").weight
            self._angular_tracking_weight = env.reward_manager.get_term_cfg("track_angular_velocity").weight
            if any(not math.isfinite(w) or w <= 0 for w in (self._tracking_weight, self._angular_tracking_weight)):
                raise ValueError("Teacher needs positive linear and angular tracking reward weights")
        self.trials_by_slot = []
        self.episode_metrics = None
        self._saved_diagnostics = None
        self._behavior_monitor = None
        if self.mode == "evaluation":
            trials = self.setup.trials
            self.trials_by_slot = [trials[i % len(trials)] for i in range(n)]
        # Interactive selections are runtime state, never part of the training
        # catalog, teacher statistics, or checkpoint fingerprints.
        self.play = None
        if self.mode == "play":
            from .play_control import PlayController

            self.play = PlayController(self)

    def __call__(self, env, env_ids, **kwargs):
        ids = self._ids(env_ids)
        valid = (
            self._natural[ids]
            & (self.episode_task_id[ids] >= 0)
            & (env.episode_length_buf[ids] > 0)
        )
        if self.mode == "train" and not self._discard_depth:
            start = time.perf_counter()
            ended = ids[valid]
            covered = ended[self.episode_sampling_mode[ended] == 1]
            self._coverage_counts.index_add_(0, self.episode_task_id[covered], torch.ones_like(covered))
            ended = ended[self.episode_sampling_mode[ended] == 0]
            task_ids = self.episode_group_id[ended]
            scores = episode_tracking_scores(env, ended, self._tracking_weight, self._angular_tracking_weight)
            self._score_sums.index_add_(0, task_ids, scores)
            self._score_counts.index_add_(0, task_ids, torch.ones_like(task_ids))
            self.statistics_host_seconds += time.perf_counter() - start
            if self.episode_metrics is not None:
                self.episode_metrics.finish(ids[valid])
        self._natural[ids] = False
        self.assign(ids)
        return self._diagnostics

    def _ids(self, ids):
        if ids is None or isinstance(ids, slice):
            return torch.arange(self.num_envs, device=self.device)[
                slice(None) if ids is None else ids
            ]
        return ids

    def _coverage_task_ids(self):
        """One representative catalog task per eligible stair geometry.

        Coverage commands override heading/vy, so the physical task ID is a
        geometry/catalog reference, never a regular teacher score sample.
        Filtered/flat evaluation catalogs simply have no coverage branch.
        """
        selected = {}
        velocity = self.cfg.curriculum.forward_stairs_velocity
        for tid, task in enumerate(self.catalog.records):
            height = self.cfg.task_space.terrains[task.terrain].levels[task.level].get("step_height_m")
            if (not task.terrain.startswith("stairs_") or not task.terrain.endswith("_up")
                    or height is None or not any(math.isclose(height, h, abs_tol=1e-8)
                        for h in self.cfg.curriculum.forward_stairs_heights_m)):
                continue
            vx, vy, rotation = task.axes
            if ((tid, 1) not in self.groups.task_direction_to_group
                    or not vx.low <= velocity <= vx.high
                    or not vy.low <= 0 <= vy.high or not rotation.low <= 0 <= rotation.high):
                continue
            selected.setdefault((task.terrain, task.level), tid)
        return np.array(list(selected.values()), dtype=np.int64)

    def assign(self, env_ids):
        """在 reset 时联合确定任务、地形和命令，避免三者分别采样。"""
        ids = self._ids(env_ids)
        if not len(ids):
            return
        if self.play is not None:
            self.play.assign(ids)
            if self._behavior_monitor is not None:
                self._behavior_monitor.reset(ids)
            return
        if self.mode == "evaluation":
            rows = [self.trials_by_slot[i] for i in ids.cpu().tolist()]
            tasks = np.array([r["task_id"] for r in rows])
            replicas = np.array([r["replica"] for r in rows])
            sequences = []
            heading_sequences = []
            for row in rows:
                sequence = row.get("command_sequence", [row["command"]])
                if not sequence or len(sequence) > self.command_count:
                    raise ValueError("Invalid evaluation command sequence length")
                sequences.append(
                    sequence + [sequence[-1]] * (self.command_count - len(sequence))
                )
                if self.catalog.heading_command:
                    targets = row.get("heading_sequence", [row.get("heading_target")])
                    if (not targets or len(targets) != len(sequence)
                            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in targets)):
                        raise ValueError("Invalid evaluation heading sequence")
                    heading_sequences.append(
                        targets + [targets[-1]] * (self.command_count - len(targets))
                    )
            commands = np.asarray(sequences)
            group_ids = np.array([self.groups.task_direction_to_group[t, 0 if self.compact or self.catalog.process_mask[t] else int(np.sign(c[0, 0]))]
                                  for t, c in zip(tasks, commands)])
            sampling_modes = np.zeros(len(ids), dtype=np.int64)
            self.yaws[ids] = torch.as_tensor(
                [r["yaw"] for r in rows], dtype=self.yaws.dtype, device=self.device
            )
        else:
            if self.compact:
                slot_families = self.env_families[ids.cpu().numpy()]
                group_ids = np.empty(len(ids), dtype=np.int64)
                tasks = np.empty_like(group_ids)
                commands = np.empty((len(ids), self.command_count, 3))
                for family_id, name in enumerate(self.family_names):
                    selected = slot_families == family_id
                    if not selected.any():
                        continue
                    conditional = self.state.probabilities * self.state.family_masks[name]
                    conditional /= conditional.sum()
                    group_ids[selected], tasks[selected], commands[selected] = self.groups.sample(
                        conditional, int(selected.sum()), self.rng, self.command_count)
            else:
                group_ids, tasks, commands = self.groups.sample(
                    self.state.probabilities, len(ids), self.rng, self.command_count)
            sampling_modes = np.zeros(len(ids), dtype=np.int64)
            if self.mode == "train" and len(self.coverage_tasks):
                covered = self.rng.random(len(ids)) < self.cfg.curriculum.forward_stairs_fraction
                sampling_modes[covered] = 1
                tasks[covered] = self.rng.choice(self.coverage_tasks, int(covered.sum()))
                commands[covered] = (self.cfg.curriculum.forward_stairs_velocity, 0, 0)
                group_ids[covered] = [self.groups.task_direction_to_group[t, 1] for t in tasks[covered]]
            replicas = None
            self.yaws[ids] = torch.as_tensor(
                self.rng.uniform(-math.pi, math.pi, len(ids)),
                dtype=self.yaws.dtype,
                device=self.device,
            )
            if self.mode == "train":
                np.add.at(self.starts, tasks, 1)
                np.add.at(self.group_starts, group_ids[sampling_modes == 0], 1)
                np.add.at(self.coverage_starts, tasks[sampling_modes == 1], 1)
        choices = (([r.get("spawn_choice", 0.0) for r in rows] if self.terrain_map.connected else replicas)
                   if self.mode == "evaluation" else None)
        rows_index, columns, geometry_ids = self.terrain_map.choose_spawns(tasks, self.rng, choices)
        self.episode_task_id[ids] = torch.as_tensor(tasks, device=self.device)
        self.episode_group_id[ids] = torch.as_tensor(group_ids, device=self.device)
        self.episode_sampling_mode[ids] = torch.as_tensor(sampling_modes, device=self.device)
        self.episode_geometry_id[ids] = torch.as_tensor(
            geometry_ids, device=self.device
        )
        self.command_sequences[ids] = torch.as_tensor(
            commands, dtype=self.commands.dtype, device=self.device
        )
        process = self.catalog.process_mask[tasks] & (sampling_modes == 0)
        scheduled = np.ones_like(process) if self.compact else process
        self.command_deadlines[ids] = torch.iinfo(torch.long).max
        self.command_index[ids] = 0
        self.command_event[ids] = False
        self.command_event_step[ids] = -1
        self.switch_counts[ids] = 0
        self._last_command_step[ids] = 0
        if scheduled.any():
            if self.mode == "evaluation":
                durations = np.array([r["command_hold_s"] + [self.cfg.task_space.process_hold_s[1]] *
                                      (self.command_count - len(r["command_hold_s"]))
                                      for r, selected in zip(rows, scheduled) if selected])
            elif self.compact:
                durations = self.catalog.sample_command_durations(
                    tasks, self.command_count, self.rng, self.cfg.training.horizon_s)
            else:
                durations = self.rng.uniform(*self.cfg.task_space.process_hold_s,
                                             size=(int(process.sum()), self.command_count))
            deadlines = np.ceil(durations / self._env.step_dt).astype(np.int64).cumsum(axis=1)
            self.command_deadlines[ids[torch.as_tensor(scheduled, device=self.device)]] = torch.as_tensor(deadlines, device=self.device)
        if self.catalog.heading_command:
            if self.mode == "evaluation":
                targets = torch.as_tensor(heading_sequences, dtype=self.commands.dtype, device=self.device)
            else:
                # The sampled third axis is an offset from the NEW spawn yaw,
                # before the physics reset updates robot.data.heading_w.
                offsets = self.command_sequences[ids, :, 2].clone()
                targets = torch.where(torch.as_tensor(process, device=self.device)[:, None],
                                      offsets, self.yaws[ids, None] + offsets)
                self.command_sequences[ids, :, 2] = torch.clamp(
                    self.cfg.task_space.heading_control_stiffness * wrap_to_pi(offsets),
                    *self.cfg.task_space.ang_vel_z_range,
                )
            self.heading_sequences[ids] = wrap_to_pi(targets)
            # Train reset randomizes yaw after task assignment. Anchor once to
            # the actual post-reset yaw before observations reach the policy.
            self._heading_reference_pending[ids] = (self.mode != "evaluation") & ~torch.as_tensor(process, device=self.device)
        self.update_commands(ids, reset=True)
        if not self.terrain_map.plane:
            terrain = self._env.scene.terrain
            row = torch.as_tensor(rows_index, device=self.device)
            col = torch.as_tensor(columns, device=self.device)
            terrain.terrain_levels[ids], terrain.terrain_types[ids] = row, col
            terrain.env_origins[ids] = self.terrain_map.origins[row, col]
        if self.episode_metrics is not None:
            self.episode_metrics.reset(ids)
        if self._behavior_monitor is not None:
            self._behavior_monitor.reset(ids)

    def update_commands(self, env_ids=None, *, reset=False, dt=None):
        """Hold the task target; convert heading error to policy yaw velocity."""
        ids = self._ids(env_ids)
        if self.play is not None:
            self.play.update_commands(ids, reset=reset)
            return
        steps = torch.zeros_like(ids) if reset else self._env.episode_length_buf[ids]
        if self.mode == "evaluation":
            steps = steps + self.evaluation_step_offsets[ids]
        advance = steps > self._last_command_step[ids]
        if dt is not None:
            advance &= (dt[ids] > 0 if isinstance(dt, torch.Tensor) and dt.ndim else dt > 0)
        if not reset and hasattr(self._env, "termination_manager"):
            # No new segment on the final physical step, including the terminal observation hook.
            advance &= ~self._env.termination_manager.terminated[ids]
            if hasattr(self._env, "reset_buf"):
                advance &= ~self._env.reset_buf[ids]
        indices = (torch.zeros_like(ids) if self.command_interval_steps is None else
                   (steps // self.command_interval_steps).clamp_max(self.command_count - 1))
        process = self.scheduled_tasks[self.episode_task_id[ids].clamp_min(0)]
        process_indices = (steps[:, None] >= self.command_deadlines[ids]).sum(-1).clamp_max(self.command_count - 1)
        indices = torch.where(process, process_indices, indices)
        indices = torch.where(advance, indices, self.command_index[ids])
        changed = indices != self.command_index[ids]
        old_commands = self.commands[ids].clone()
        old_targets = self.heading_targets[ids].clone()
        self.command_index[ids] = indices
        self._last_command_step[ids] = torch.where(advance, steps, self._last_command_step[ids])
        self.commands[ids] = self.command_sequences[ids, indices]
        if self.catalog.heading_command:
            if not reset:
                fresh = ids[self._heading_reference_pending[ids]]
                if len(fresh):
                    current_spawn = self._env.scene["robot"].data.heading_w[fresh]
                    offsets = wrap_to_pi(self.heading_sequences[fresh] - self.yaws[fresh, None])
                    self.yaws[fresh] = current_spawn
                    self.heading_sequences[fresh] = wrap_to_pi(current_spawn[:, None] + offsets)
                    self._heading_reference_pending[fresh] = False
            self.heading_targets[ids] = self.heading_sequences[ids, indices]
            current = self.yaws[ids] if reset else self._env.scene["robot"].data.heading_w[ids]
            self.heading_error[ids] = wrap_to_pi(self.heading_targets[ids] - current)
            self.commands[ids, 2] = torch.clamp(
                self.cfg.task_space.heading_control_stiffness * self.heading_error[ids],
                *self.cfg.task_space.ang_vel_z_range,
            )
        # An explicit sampler event remains attached to the next action/observation.
        # Repeated dt=0/terminal calls neither create nor erase it.
        self.command_event[ids] = torch.where(advance, changed, self.command_event[ids])
        changed_ids = ids[changed]
        self.command_event_step[changed_ids] = steps[changed]
        self.command_event_delta[changed_ids] = (self.commands[ids] - old_commands)[changed]
        self.command_heading_delta[changed_ids] = wrap_to_pi(self.heading_targets[ids] - old_targets)[changed]
        self.switch_counts[changed_ids] += 1

    def flush_samples(self, *, synchronize=False):
        """Flush new samples; collective calls belong only at shared update/save boundaries."""
        if synchronize and torch.distributed.is_initialized():
            if self.episode_metrics is not None:
                self.episode_metrics.synchronize()
            for values in (self._score_sums, self._score_counts, self._coverage_counts):
                torch.distributed.all_reduce(values)
            # Counts include the previous global total plus this rank's new starts.
            # Reduce only the delta so old samples are never counted again.
            for name, previous in self._synced_starts.items():
                delta = torch.as_tensor(getattr(self, name) - previous, device=self.device)
                torch.distributed.all_reduce(delta)
                previous += delta.cpu().numpy()
                getattr(self, name)[:] = previous
        self.state.add_samples(
            self._score_sums.cpu().numpy(), self._score_counts.cpu().numpy()
        )
        self._score_sums.zero_()
        self._score_counts.zero_()
        self.coverage_counts += self._coverage_counts.cpu().numpy()
        self._coverage_counts.zero_()

    def advance(self, update_count):
        self.flush_samples(synchronize=True)
        stage_closed = self.state.advance(update_count)
        self._diagnostics = self.state.diagnostics()
        return stage_closed

    @contextmanager
    def discard_episodes(self):
        self._discard_depth += 1
        try:
            yield
        finally:
            self._discard_depth -= 1

    def reset(self, env_ids=None):
        # Manager reset clears per-episode terms; global course state persists.
        pass

    def state_dict(self):
        self.flush_samples()
        return {
            **self.state.state_dict(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "task_keys": [r.key for r in self.catalog.records],
            "task_config": plain(self.cfg),
            "starts": self.starts.copy(),
            "teacher_groups": deepcopy(self.groups.records),
            "group_starts": self.group_starts.copy(),
            "coverage_starts": self.coverage_starts.copy(),
            "coverage_counts": self.coverage_counts.copy(),
            "map_manifest": deepcopy(self.terrain_map.manifest),
            "interaction_statistics": (self.episode_metrics.state_dict() if self.episode_metrics is not None
                                       else deepcopy(self._saved_diagnostics)),
            **({"family_assignment": self.family_assignment()} if self.compact else {}),
        }

    def family_assignment(self):
        return {
            "protocol": "fixed_local_env_largest_remainder_v1",
            "num_envs": self.num_envs,
            "names": list(self.family_names),
            "counts": self.family_counts.tolist() if self.family_counts is not None else None,
        }

    def load_state_dict(self, state):
        if state.get("version") != self.state.version:
            raise ValueError("Incompatible teacher state; use explicit warm-start for older curricula")
        if self.compact and state.get("family_assignment") != self.family_assignment():
            raise ValueError("Compact family allocation/environment count differs; use warm-start")
        if state["task_keys"] != [r.key for r in self.catalog.records]:
            raise ValueError("Checkpoint task table differs")
        if state.get("map_manifest") != self.terrain_map.manifest:
            raise ValueError("Checkpoint map/spawn protocol differs")
        starts = np.asarray(state["starts"])
        if (
            starts.shape != (self.catalog.num_tasks,)
            or not np.issubdtype(starts.dtype, np.integer)
            or (starts < 0).any()
        ):
            raise ValueError("Invalid checkpoint task starts")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        if plain(state.get("teacher_groups")) != plain(self.groups.records):
            raise ValueError("Checkpoint teacher group mapping differs")
        sampling_counts = {}
        for name, size in (("group_starts", self.groups.num_groups),
                           ("coverage_starts", self.catalog.num_tasks),
                           ("coverage_counts", self.catalog.num_tasks)):
            a = np.asarray(state[name])
            if a.shape != (size,) or not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
                raise ValueError(f"Invalid checkpoint {name}")
            sampling_counts[name] = a.copy()
        if ((sampling_counts["coverage_counts"] > sampling_counts["coverage_starts"]).any()
                or (sampling_counts["coverage_starts"] > starts).any()
                or sampling_counts["group_starts"].sum() + sampling_counts["coverage_starts"].sum() != starts.sum()):
            raise ValueError("Inconsistent checkpoint sampling counts")
        self.state.load_state_dict(state)
        self.rng, self.starts = rng, starts.copy()
        for name, values in sampling_counts.items():
            setattr(self, name, values)
        self._synced_starts = {
            name: getattr(self, name).copy() for name in self.START_COUNT_FIELDS
        }
        self._score_sums.zero_()
        self._score_counts.zero_()
        self._diagnostics = self.state.diagnostics()
        self.episode_task_id.fill_(-1)
        self.episode_group_id.fill_(-1)
        self.episode_sampling_mode.zero_()
        self._coverage_counts.zero_()
        self._natural.zero_()
        self._heading_reference_pending.zero_()
        self.command_index.zero_()
        self.command_event.zero_()
        self.command_event_step.fill_(-1)
        self._last_command_step.fill_(-1)
        self.command_deadlines.fill_(torch.iinfo(torch.long).max)
        self.episode_metrics = None
        self._saved_diagnostics = deepcopy(state.get("interaction_statistics"))
        if self._behavior_monitor is not None:
            self._behavior_monitor.reset(torch.arange(self.num_envs, device=self.device))


class NaturalEpisodeRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        get_controller(self._env)._natural[env_ids] = True
