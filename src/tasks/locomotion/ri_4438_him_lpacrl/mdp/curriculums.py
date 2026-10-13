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

from .task_catalog import plain


class LPState:
    """FIBO teacher stages over per-PPO means, with resumable state.

    FIBO-Physical-AI/curriculum_training@5813fc3: teacher_guided.py and mdp.py.
    Empty iteration bins are zero; iteration means are equally weighted.
    First baseline at update 51 and first probability update at 101 (length 50).
    """

    VERSION = 4
    SCORE_KIND = "fibo_linear_tracking_per_step_v1"
    ARRAY_FIELDS = (
        "probabilities", "mean_scores", "lp", "counts", "pending_sums",
        "pending_counts", "stage_sums", "iteration_scores", "iteration_counts",
    )
    CLOCK_FIELDS = (
        "completed_policy_updates", "last_stage_update", "stage_count", "stage_index",
    )

    def __init__(self, num_tasks, cfg, fingerprints=None, initial_probabilities=None):
        if num_tasks < 1 or not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
            raise ValueError("Positive task count and finite positive temperature required")
        if cfg.method not in {"lp", "uniform"} or not 0 <= cfg.uniform_mix < 1:
            raise ValueError("Invalid curriculum method/mixing coefficient")
        if cfg.method == "uniform" and cfg.uniform_mix:
            raise ValueError("Uniform requires zero mixing coefficient")
        if type(cfg.stage_length_updates) is not int or cfg.stage_length_updates < 1:
            raise ValueError("stage_length_updates must be a positive integer")
        if type(cfg.seed_bin) is not int or cfg.seed_bin < 0:
            raise ValueError("seed_bin must be a nonnegative integer")
        self.num_tasks, self.cfg = num_tasks, cfg
        self.fingerprints = dict(fingerprints or {})
        if cfg.method == "uniform":
            initial_probabilities = np.full(num_tasks, 1.0 / num_tasks)
        elif initial_probabilities is None:
            if cfg.seed_bin >= num_tasks:
                raise ValueError("seed_bin is outside the task table")
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
            dtype = np.int64 if name.endswith("counts") else np.float64
            setattr(self, name, np.zeros(num_tasks, dtype=dtype))
        self.probabilities = initial.copy()
        self.completed_policy_updates = 0
        self.last_stage_update = -1
        self.stage_count = self.stage_index = 0

    def add_samples(self, score_sums, counts):
        # Pending samples belong only to this PPO iteration; advance always clears them.
        sums = np.asarray(score_sums, dtype=np.float64)
        counts = np.asarray(counts)
        if (sums.shape != (self.num_tasks,) or counts.shape != sums.shape
                or not np.issubdtype(counts.dtype, np.integer)
                or not np.isfinite(sums).all() or (counts < 0).any()
                or (sums < 0).any() or (sums > counts + 1e-6).any()):
            raise ValueError("Invalid tracking samples")
        self.pending_sums += sums
        self.pending_counts += counts
        self.counts += counts

    def advance(self, update_count):
        """Consume exactly one PPO iteration; return whether a stage closed."""
        if type(update_count) is not int or update_count != self.completed_policy_updates + 1:
            raise ValueError("Curriculum must consume every PPO iteration in order")
        self.iteration_scores = self.pending_sums / np.maximum(self.pending_counts, 1)
        self.iteration_counts = self.pending_counts.copy()
        self.pending_sums.fill(0)
        self.pending_counts.fill(0)
        self.stage_sums += self.iteration_scores
        self.stage_count += 1
        self.completed_policy_updates = update_count
        if self.last_stage_update < 0:
            self.last_stage_update = update_count
            return False
        if update_count < self.last_stage_update + self.cfg.stage_length_updates:
            return False
        means = self.stage_sums / self.stage_count
        self.stage_sums.fill(0)
        self.stage_count = 0
        self.last_stage_update = update_count
        if self.stage_index:
            self.lp = means - self.mean_scores
            self.probabilities = self._probabilities(self.lp)
        self.mean_scores = means
        self.stage_index += 1
        return True

    def _probabilities(self, delta):
        if self.cfg.method == "uniform":
            return np.full(self.num_tasks, 1.0 / self.num_tasks)
        scaled = delta / max(self.cfg.temperature, 1e-8)
        exp = np.exp(scaled - scaled.max())
        return ((1 - self.cfg.uniform_mix) * exp / exp.sum()
                + self.cfg.uniform_mix / self.num_tasks)

    def state_dict(self):
        return {
            "version": self.VERSION, "score_kind": self.SCORE_KIND,
            "num_tasks": self.num_tasks, "fingerprints": dict(self.fingerprints),
            "initial_probabilities": self.initial_probabilities.copy(),
            "curriculum_config": plain(self.cfg),
            **{name: getattr(self, name).copy() for name in self.ARRAY_FIELDS},
            **{name: getattr(self, name) for name in self.CLOCK_FIELDS},
        }

    def load_state_dict(self, state):
        if (state.get("version") != self.VERSION
                or state.get("score_kind") != self.SCORE_KIND
                or state.get("num_tasks") != self.num_tasks
                or state.get("fingerprints") != self.fingerprints
                or state.get("curriculum_config") != plain(self.cfg)
                or not np.array_equal(state.get("initial_probabilities"), self.initial_probabilities)):
            raise ValueError("Incompatible teacher state; use explicit warm-start for older curricula")
        arrays = {}
        for name in self.ARRAY_FIELDS:
            a = np.asarray(state[name])
            if a.shape != (self.num_tasks,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid checkpoint {name}")
            if name.endswith("counts") and (not np.issubdtype(a.dtype, np.integer) or (a < 0).any()):
                raise ValueError(f"Invalid checkpoint {name}")
            arrays[name] = a.astype(np.int64 if name.endswith("counts") else np.float64, copy=True)
        clocks = {name: state[name] for name in self.CLOCK_FIELDS}
        completed, last, count, stage = (clocks[name] for name in self.CLOCK_FIELDS)
        if any(type(value) is not int for value in clocks.values()):
            raise ValueError("Invalid checkpoint stage clock")
        expected_stage = max(0, (completed - 1) // self.cfg.stage_length_updates)
        expected_last = -1 if completed == 0 else 1 + expected_stage * self.cfg.stage_length_updates
        expected_count = completed if expected_stage == 0 else completed - expected_last
        if (completed < 0 or stage != expected_stage or last != expected_last
                or count != expected_count):
            raise ValueError("Invalid checkpoint stage clock")
        expected_p = self.initial_probabilities if stage < 2 else self._probabilities(arrays["lp"])
        if (not np.allclose(arrays["probabilities"], expected_p, atol=1e-12, rtol=1e-10)
                or (arrays["probabilities"] < 0).any()
                or not np.isclose(arrays["probabilities"].sum(), 1)
                or ((arrays["mean_scores"] < 0) | (arrays["mean_scores"] > 1)).any()
                or (np.abs(arrays["lp"]) > 1).any()
                or (stage < 2 and arrays["lp"].any())
                or (stage == 0 and arrays["mean_scores"].any())
                or (arrays["pending_sums"] < 0).any()
                or (arrays["pending_sums"] > arrays["pending_counts"] + 1e-6).any()
                or (arrays["pending_counts"] > arrays["counts"]).any()
                or (arrays["iteration_counts"] > arrays["counts"]).any()
                or ((arrays["iteration_scores"] < 0) | (arrays["iteration_scores"] > 1)).any()
                or (arrays["iteration_scores"][arrays["iteration_counts"] == 0] != 0).any()
                or (arrays["stage_sums"] < 0).any()
                or (arrays["stage_sums"] > count + 1e-6).any()):
            raise ValueError("Inconsistent teacher checkpoint statistics")
        for name, value in {**arrays, **clocks}.items():
            setattr(self, name, value)

    def diagnostics(self):
        p = self.probabilities
        positive = p[p > 0]
        entropy = -np.sum(positive * np.log(positive))
        return {
            "entropy_normalized": float(entropy / math.log(self.num_tasks)) if self.num_tasks > 1 else 0.0,
            "max_probability": float(p.max()), "min_probability": float(p.min()),
            "effective_tasks": float(1.0 / np.square(p).sum()),
            "lp_min": float(self.lp.min()), "lp_max": float(self.lp.max()),
            "lp_std": float(self.lp.std()), "stage_index": self.stage_index,
            "completed_episodes": int(self.counts.sum()),
            "iteration_episodes": int(self.iteration_counts.sum()),
            "sampled_tasks": int((self.iteration_counts > 0).sum()),
            "mean_tracking_score": float(self.mean_scores.mean()),
            "temperature": float(self.cfg.temperature), "uniform_mix": float(self.cfg.uniform_mix),
        }


def episode_tracking_scores(env, env_ids, weight):
    """FIBO mdp.py: linear episode sum / actual steps / weight, retaining dt.

    No yaw blend or dt normalization. Clipping follows the reference; nonfinite
    input remains invalid so the training numerical guard cannot be bypassed.
    """
    steps = env.episode_length_buf[env_ids].clamp(min=1).float()
    terms = env.reward_manager._episode_sums["track_linear_velocity"][env_ids]
    raw = terms / steps / max(float(weight), 1e-6)
    return torch.where(torch.isfinite(raw), raw.clamp(0, 1), torch.full_like(raw, float("nan")))


def get_controller(env):
    return env.curriculum_manager.get_term_cfg("lpacrl").func


class CurriculumController(ManagerTermBase):
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
        self.terrain_map = self.setup.terrain_map
        self.terrain_map.bind(env.scene.terrain.terrain_origins)
        self.state = LPState(
            self.catalog.num_tasks, self.cfg.curriculum, cfg.params.get("fingerprints"),
            self.catalog.initial_teacher_probabilities(self.cfg.curriculum.seed_bin),
        )
        if self.mode == "train" and self.cfg.task_space.resampling_interval_s is not None:
            raise ValueError("Training commands must stay fixed for each episode")
        self.rng = np.random.default_rng(self.cfg.training.seed)
        n, device = env.num_envs, env.device
        self.episode_task_id = torch.full((n,), -1, device=device, dtype=torch.long)
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
        self._score_sums = torch.zeros(
            self.catalog.num_tasks, dtype=torch.float32, device=device
        )
        self._score_counts = torch.zeros(
            self.catalog.num_tasks, dtype=torch.long, device=device
        )
        self.statistics_host_seconds = 0.0
        self._diagnostics = self.state.diagnostics()
        if self.mode == "train":
            self._tracking_weight = env.reward_manager.get_term_cfg("track_linear_velocity").weight
            if not math.isfinite(self._tracking_weight) or self._tracking_weight <= 0:
                raise ValueError("Teacher needs a positive linear tracking reward weight")
        self.trials_by_slot = []
        if self.mode == "evaluation":
            trials = self.setup.trials
            self.trials_by_slot = [trials[i % len(trials)] for i in range(n)]

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
            task_ids = self.episode_task_id[ended]
            scores = episode_tracking_scores(env, ended, self._tracking_weight)
            self._score_sums.index_add_(0, task_ids, scores)
            self._score_counts.index_add_(0, task_ids, torch.ones_like(task_ids))
            self.statistics_host_seconds += time.perf_counter() - start
        self._natural[ids] = False
        self.assign(ids)
        return self._diagnostics

    def _ids(self, ids):
        if ids is None or isinstance(ids, slice):
            return torch.arange(self.num_envs, device=self.device)[
                slice(None) if ids is None else ids
            ]
        return ids

    def assign(self, env_ids):
        """在 reset 时联合确定任务、地形和命令，避免三者分别采样。"""
        ids = self._ids(env_ids)
        if not len(ids):
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
            self.yaws[ids] = torch.as_tensor(
                [r["yaw"] for r in rows], dtype=self.yaws.dtype, device=self.device
            )
        else:
            tasks = self.rng.choice(
                self.catalog.num_tasks, len(ids), p=self.state.probabilities
            )
            replicas = self.rng.integers(self.terrain_map.replicas, size=len(ids))
            commands = self.catalog.sample_commands(
                np.repeat(tasks, self.command_count), self.rng
            ).reshape(len(ids), self.command_count, 3)
            self.yaws[ids] = torch.as_tensor(
                self.rng.uniform(-math.pi, math.pi, len(ids)),
                dtype=self.yaws.dtype,
                device=self.device,
            )
            if self.mode == "train":
                np.add.at(self.starts, tasks, 1)
        columns = self.terrain_map.task_columns[tasks]
        self.episode_task_id[ids] = torch.as_tensor(tasks, device=self.device)
        self.episode_geometry_id[ids] = torch.as_tensor(
            columns * self.terrain_map.replicas + replicas, device=self.device
        )
        self.command_sequences[ids] = torch.as_tensor(
            commands, dtype=self.commands.dtype, device=self.device
        )
        if self.catalog.heading_command:
            if self.mode == "evaluation":
                targets = torch.as_tensor(heading_sequences, dtype=self.commands.dtype, device=self.device)
            else:
                # The sampled third axis is an offset from the NEW spawn yaw,
                # before the physics reset updates robot.data.heading_w.
                offsets = self.command_sequences[ids, :, 2].clone()
                targets = self.yaws[ids, None] + offsets
                self.command_sequences[ids, :, 2] = torch.clamp(
                    self.cfg.task_space.heading_control_stiffness * wrap_to_pi(offsets),
                    *self.cfg.task_space.ang_vel_z_range,
                )
            self.heading_sequences[ids] = wrap_to_pi(targets)
            # Train reset randomizes yaw after task assignment. Anchor once to
            # the actual post-reset yaw before observations reach the policy.
            self._heading_reference_pending[ids] = self.mode != "evaluation"
        self.update_commands(ids, reset=True)
        if not self.terrain_map.plane:
            terrain = self._env.scene.terrain
            row = torch.as_tensor(replicas, device=self.device)
            col = torch.as_tensor(columns, device=self.device)
            terrain.terrain_levels[ids], terrain.terrain_types[ids] = row, col
            terrain.env_origins[ids] = self.terrain_map.origins[row, col]

    def update_commands(self, env_ids=None, *, reset=False):
        """Hold the task target; convert heading error to policy yaw velocity."""
        ids = self._ids(env_ids)
        if self.command_interval_steps is None:
            indices = torch.zeros_like(ids)
        else:
            steps = torch.zeros_like(ids) if reset else self._env.episode_length_buf[ids]
            if self.mode == "evaluation":
                steps = steps + self.evaluation_step_offsets[ids]
            indices = (steps // self.command_interval_steps).clamp_max(self.command_count - 1)
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

    def flush_samples(self):
        """Transfer completed episode statistics once per PPO update/checkpoint."""
        self.state.add_samples(
            self._score_sums.cpu().numpy(), self._score_counts.cpu().numpy()
        )
        self._score_sums.zero_()
        self._score_counts.zero_()

    def advance(self, update_count):
        self.flush_samples()
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
        }

    def load_state_dict(self, state):
        if state["task_keys"] != [r.key for r in self.catalog.records]:
            raise ValueError("Checkpoint task table differs")
        starts = np.asarray(state["starts"])
        if (
            starts.shape != (self.catalog.num_tasks,)
            or not np.issubdtype(starts.dtype, np.integer)
            or (starts < 0).any()
        ):
            raise ValueError("Invalid checkpoint task starts")
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        self.state.load_state_dict(state)
        self.rng, self.starts = rng, starts.copy()
        self._score_sums.zero_()
        self._score_counts.zero_()
        self._diagnostics = self.state.diagnostics()
        self.episode_task_id.fill_(-1)
        self._natural.zero_()
        self._heading_reference_pending.zero_()


class NaturalEpisodeRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        get_controller(self._env)._natural[env_ids] = True
