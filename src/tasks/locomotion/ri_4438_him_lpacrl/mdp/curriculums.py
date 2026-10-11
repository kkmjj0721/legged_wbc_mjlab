"""Signed learning progress, transactional updates, and reset-time task assignment."""

from __future__ import annotations

import math
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.recorder_manager import RecorderTerm

from .task_catalog import plain


class LPState:
    """纯课程统计：评估成功后一次性提交回报、学习进步和采样概率。"""

    VERSION = 1

    def __init__(self, num_tasks, cfg, fingerprints=None):
        if num_tasks < 1 or not math.isfinite(cfg.temperature) or cfg.temperature <= 0:
            raise ValueError(
                "Positive task count and finite positive temperature required"
            )
        if cfg.method not in {"lp", "alp", "uniform"} or not 0 <= cfg.uniform_mix < 1:
            raise ValueError("Invalid curriculum method/mixing coefficient")
        if cfg.method == "uniform" and cfg.uniform_mix:
            raise ValueError("Uniform requires zero mixing coefficient")
        self.num_tasks = num_tasks
        self.cfg = cfg
        self.fingerprints = fingerprints or {}
        self.probabilities = np.full(num_tasks, 1.0 / num_tasks, dtype=np.float64)
        self.mean_returns = None
        self.lp = np.zeros(num_tasks, dtype=np.float64)
        self.counts = np.zeros(num_tasks, dtype=np.int64)
        self.completed_policy_updates = 0
        self.last_evaluated_update = -1
        self.consecutive_failures = 0

    def commit_evaluation(self, result, update_count, trials_per_task):
        means = np.asarray(result["mean_returns"], dtype=np.float64)
        counts = np.asarray(result["counts"])
        if means.shape != (self.num_tasks,) or counts.shape != means.shape:
            raise ValueError("Evaluation must cover the complete ordered task catalog")
        if not np.isfinite(means).all() or not np.all(counts == trials_per_task):
            raise ValueError(
                "Incomplete/nonfinite evaluation; previous curriculum retained"
            )
        if update_count < 0 or (
            self.mean_returns is not None and update_count <= self.last_evaluated_update
        ):
            raise ValueError("Evaluation clock must advance")
        delta = (
            np.zeros_like(means)
            if self.mean_returns is None
            else means - self.mean_returns
        )
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite learning progress")
        if self.cfg.method == "uniform" or self.mean_returns is None:
            probabilities = np.full(self.num_tasks, 1.0 / self.num_tasks)
        else:
            score = np.abs(delta) if self.cfg.method == "alp" else delta
            with np.errstate(over="ignore", under="ignore"):
                exponentials = np.exp((score - score.max()) / self.cfg.temperature)
            probabilities = exponentials / exponentials.sum()
            eta = self.cfg.uniform_mix
            probabilities = (1 - eta) * probabilities + eta / self.num_tasks
        if not np.isfinite(probabilities).all() or not np.isclose(
            probabilities.sum(), 1
        ):
            raise ValueError("Invalid sampling distribution")
        # Commit only after every validation/calculation succeeds.
        self.mean_returns, self.counts = means.copy(), counts.astype(np.int64)
        self.lp, self.probabilities = delta, probabilities
        self.last_evaluated_update = update_count
        self.consecutive_failures = 0

    def state_dict(self):
        return {
            "version": self.VERSION,
            "num_tasks": self.num_tasks,
            "fingerprints": dict(self.fingerprints),
            "probabilities": self.probabilities.copy(),
            "mean_returns": None
            if self.mean_returns is None
            else self.mean_returns.copy(),
            "lp": self.lp.copy(),
            "counts": self.counts.copy(),
            "completed_policy_updates": self.completed_policy_updates,
            "last_evaluated_update": self.last_evaluated_update,
        }

    def load_state_dict(self, state):
        if (
            state["version"] != self.VERSION
            or state["num_tasks"] != self.num_tasks
            or state["fingerprints"] != self.fingerprints
        ):
            raise ValueError("Incompatible LP-ACRL state; use explicit warm-start")
        arrays = {}
        for name in ("probabilities", "lp", "counts", "mean_returns"):
            value = state[name]
            if value is None and name == "mean_returns":
                arrays[name] = None
                continue
            a = np.asarray(value)
            if a.shape != (self.num_tasks,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid checkpoint {name}")
            arrays[name] = a.copy()
        p = arrays["probabilities"]
        if (p < 0).any() or not np.isclose(p.sum(), 1) or (arrays["counts"] < 0).any():
            raise ValueError("Invalid checkpoint probabilities/counts")
        completed, evaluated = (
            state["completed_policy_updates"],
            state["last_evaluated_update"],
        )
        if (
            not isinstance(completed, int)
            or not isinstance(evaluated, int)
            or not -1 <= evaluated <= completed
        ):
            raise ValueError("Invalid checkpoint clock")
        if (arrays["mean_returns"] is None) != (evaluated == -1):
            raise ValueError("Inconsistent evaluation baseline")
        for name, value in arrays.items():
            setattr(self, name, value)
        self.completed_policy_updates, self.last_evaluated_update = completed, evaluated
        self.consecutive_failures = 0

    def diagnostics(self):
        p = self.probabilities
        positive = p[p > 0]
        entropy = -np.sum(positive * np.log(positive))
        return {
            "entropy_normalized": float(entropy / math.log(self.num_tasks))
            if self.num_tasks > 1
            else 0.0,
            "zero_probability_tasks": int(np.count_nonzero(p == 0)),
            "max_probability": float(p.max()),
        }


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
            self.catalog.num_tasks, self.cfg.curriculum, cfg.params.get("fingerprints")
        )
        self.rng = np.random.default_rng(self.cfg.training.seed)
        n, device = env.num_envs, env.device
        self.episode_task_id = torch.full((n,), -1, device=device, dtype=torch.long)
        self.episode_geometry_id = torch.full_like(self.episode_task_id, -1)
        self.commands = torch.zeros(n, 3, device=device)
        interval = self.cfg.task_space.resampling_interval_s / env.step_dt
        if not math.isclose(interval, round(interval), abs_tol=1e-6) or interval < 1:
            raise ValueError("Command interval must be a positive multiple of step_dt")
        self.command_interval_steps = round(interval)
        self.command_count = math.ceil(
            max(self.cfg.training.horizon_s, self.cfg.evaluation.horizon_s)
            / self.cfg.task_space.resampling_interval_s
        ) + 1
        self.command_sequences = torch.zeros(n, self.command_count, 3, device=device)
        # 评估出界后继续同一试验；指令时钟不能随物理 reset 回到零。
        self.evaluation_step_offsets = torch.zeros(n, dtype=torch.long, device=device)
        self.yaws = torch.zeros(n, device=device)
        self._natural = torch.zeros(n, dtype=torch.bool, device=device)
        self._discard_depth = 0
        self.starts = np.zeros(self.catalog.num_tasks, dtype=np.int64)
        self.completed = np.zeros_like(self.starts)
        self.slot_steps = np.zeros_like(self.starts)
        self.return_sums = np.zeros(self.catalog.num_tasks)
        self.trials = self.setup.trials
        self.trials_by_slot = [self.trials[i % len(self.trials)] for i in range(n)]

    def __call__(self, env, env_ids, **kwargs):
        ids = self._ids(env_ids)
        valid = (
            self._natural[ids]
            & (self.episode_task_id[ids] >= 0)
            & (env.episode_length_buf[ids] > 0)
        )
        if self.mode == "train" and not self._discard_depth and valid.any():
            ended = ids[valid]
            task_ids = self.episode_task_id[ended].cpu().numpy()
            total = sum(
                value[ended] for value in env.reward_manager._episode_sums.values()
            )
            rewards = total.detach().cpu().numpy()
            if not np.isfinite(rewards).all():
                raise ValueError("Nonfinite raw episode return")
            np.add.at(self.completed, task_ids, 1)
            np.add.at(self.return_sums, task_ids, rewards)
        self._natural[ids] = False
        self.assign(ids)
        return self.state.diagnostics()

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
            for row in rows:
                sequence = row.get("command_sequence", [row["command"]])
                if not sequence or len(sequence) > self.command_count:
                    raise ValueError("Invalid evaluation command sequence length")
                sequences.append(
                    sequence + [sequence[-1]] * (self.command_count - len(sequence))
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
        self.update_commands(ids, reset=True)
        if not self.terrain_map.plane:
            terrain = self._env.scene.terrain
            row = torch.as_tensor(replicas, device=self.device)
            col = torch.as_tensor(columns, device=self.device)
            terrain.terrain_levels[ids], terrain.terrain_types[ids] = row, col
            terrain.env_origins[ids] = self.terrain_map.origins[row, col]

    def update_commands(self, env_ids=None, *, reset=False):
        """按控制步索引预采样指令；重复调用不推进 RNG，也不跨任务分箱。"""
        ids = self._ids(env_ids)
        steps = torch.zeros_like(ids) if reset else self._env.episode_length_buf[ids]
        if self.mode == "evaluation":
            steps = steps + self.evaluation_step_offsets[ids]
        indices = (steps // self.command_interval_steps).clamp_max(
            self.command_count - 1
        )
        self.commands[ids] = self.command_sequences[ids, indices]

    def note_step(self):
        ids = self.episode_task_id.cpu().numpy()
        if (ids < 0).any():
            raise RuntimeError("Step before initial task assignment")
        np.add.at(self.slot_steps, ids, 1)

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

    def commit_evaluation(self, result, update_count):
        self.state.commit_evaluation(
            result, update_count, self.cfg.evaluation.trials_per_task
        )

    def state_dict(self):
        return {
            **self.state.state_dict(),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "task_keys": [r.key for r in self.catalog.records],
            "trials": deepcopy(self.trials),
            "task_config": plain(self.cfg),
            "starts": self.starts.copy(),
            "completed": self.completed.copy(),
            "slot_steps": self.slot_steps.copy(),
            "return_sums": self.return_sums.copy(),
        }

    def load_state_dict(self, state):
        if (
            state["task_keys"] != [r.key for r in self.catalog.records]
            or state["trials"] != self.trials
        ):
            raise ValueError("Checkpoint task/trial table differs")
        self.state.load_state_dict(state)
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        for name in ("starts", "completed", "slot_steps", "return_sums"):
            value = np.asarray(state[name])
            if value.shape != (self.catalog.num_tasks,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint statistics: {name}")
            setattr(self, name, value.copy())
        self.episode_task_id.fill_(-1)
        self._natural.zero_()


class NaturalEpisodeRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        get_controller(self._env)._natural[env_ids] = True
