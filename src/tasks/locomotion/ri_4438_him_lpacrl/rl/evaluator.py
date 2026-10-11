"""Independent nominal evaluation, paired across curriculum stages and methods."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from ..config.env_cfgs import build_env_cfg
from ..mdp.curriculums import get_controller
from ..mdp.task_catalog import plain


def freeze_actor(actor, device):
    """Copy every module, detaching transient distribution/forward caches.

    PPO leaves a Normal distribution with non-leaf loc/scale tensors on the
    actor. PyTorch forbids deepcopy of those tensors. Supplying detached copies
    through deepcopy's memo preserves the complete module without mutating the
    live actor or its optimizer/normalizer state.
    """
    memo, seen = {}, set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if not value.is_leaf:
                memo[id(value)] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif not isinstance(value, type) and hasattr(value, "__dict__"):
            visit(vars(value))

    with torch.no_grad():
        visit(actor)
        frozen = deepcopy(actor, memo).to(device).eval()
        frozen.requires_grad_(False)
    return frozen


@contextmanager
def preserve_rng():
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def export_result(result, output_dir, catalog, fingerprints):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        **result,
        "task_catalog": plain(catalog.records),
        "fingerprints": fingerprints,
    }
    temporary = path / ".evaluation.json.tmp"
    temporary.write_text(
        json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path / "evaluation.json")
    with (path / "per_task.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_task"][0]))
        writer.writeheader()
        writer.writerows(result["per_task"])


class TaskEvaluator:
    def __init__(self, setup, device, env_factory=None):
        self.setup, self.device = setup, device
        self.env = None
        self.env_factory = env_factory

    def _get_env(self):
        if self.env is None:
            if self.env_factory is None:
                from mjlab.envs import ManagerBasedRlEnv

                self.env_factory = ManagerBasedRlEnv
            self.env = self.env_factory(
                cfg=build_env_cfg(self.setup.cfg, "evaluation"), device=self.device
            )
        return self.env

    def evaluate(self, actor):
        start = time.monotonic()
        with preserve_rng():
            frozen = freeze_actor(actor, self.device)
            with torch.inference_mode():
                result = self._evaluate_frozen(frozen)
        result["wall_seconds"] = time.monotonic() - start
        return result

    def _evaluate_frozen(self, actor):
        env = self._get_env()
        controller = get_controller(env)
        from ..mdp.terrains import GeometryMonitor, TraversalMetrics

        monitor = (
            GeometryMonitor(env, controller)
            if hasattr(controller, "terrain_map")
            else None
        )
        table = self.setup.trials
        e = env.num_envs
        device = env.device
        rows, slot_steps = [], 0
        max_steps = math.ceil(self.setup.cfg.evaluation.horizon_s / env.step_dt)
        for offset in range(0, len(table), e):
            batch = table[offset : offset + e]
            controller.trials_by_slot = batch + [batch[-1]] * (e - len(batch))
            controller.evaluation_step_offsets.zero_()
            obs, _ = env.reset()
            obs = TensorDict(obs, batch_size=[e], device=device)
            active = torch.arange(e, device=device) < len(batch)
            returns = torch.zeros(e, dtype=torch.float64, device=device)
            lengths = torch.zeros(e, dtype=torch.int64, device=device)
            errors = torch.zeros(e, 3, dtype=torch.float64, device=device)
            boundary_counts = torch.zeros(e, dtype=torch.long, device=device)
            traversal = TraversalMetrics(env, controller)
            for _ in range(max_steps):
                actions = actor(obs)
                # step() 返回时指令可能已更新；误差必须对应刚执行动作的指令。
                commands = controller.commands.clone()
                next_obs, rewards, terminated, truncated, _extras = env.step(actions)
                raw_bad = getattr(env, "_him_numerics_bad", None)
                if raw_bad is not None and raw_bad[active].any():
                    raise ValueError(
                        "Raw evaluation observations violated the HIM numerical contract"
                    )
                if (
                    not torch.isfinite(rewards[active]).all()
                    or not torch.isfinite(actions[active]).all()
                ):
                    raise ValueError("Nonfinite evaluation transition")
                actual = torch.cat(
                    (
                        env.scene["robot"].data.root_link_lin_vel_b[:, :2],
                        env.scene["robot"].data.root_link_ang_vel_b[:, 2:3],
                    ),
                    dim=1,
                )
                difference = actual - commands
                returns[active] += rewards[active].double()
                lengths[active] += 1
                errors[active] += difference[active].double().square()
                slot_steps += e
                traversal.observe(active, commands)
                done = terminated | truncated
                if monitor is not None:
                    monitor.observe(done | ~active)
                boundary = (
                    env.termination_manager.get_term("boundary_truncation")
                    if "boundary_truncation" in env.termination_manager.active_terms
                    else torch.zeros_like(done)
                )
                boundary_counts[active] += boundary[active].long()
                time_limit = lengths >= max_steps
                # 出界只结束当前物理片段，不缩短试验的评分预算。物理失败仍结束试验。
                finished = ((terminated | time_limit) & active).nonzero(
                    as_tuple=False
                ).flatten()
                for slot in finished.tolist():
                    rows.append(
                        {
                            **batch[slot],
                            "return": float(returns[slot]),
                            "steps": int(lengths[slot]),
                            "squared_error": errors[slot].cpu().tolist(),
                            "terminated": bool(terminated[slot]),
                            "truncated": bool(time_limit[slot] and not terminated[slot]),
                            "boundary": bool(boundary_counts[slot] > 0),
                            "boundary_crossings": int(boundary_counts[slot]),
                            "time_limit": bool(time_limit[slot]),
                            **traversal.trial(slot),
                        }
                    )
                active[finished] = False
                if not active.any():
                    break
                # All done slots, including inactive/padding slots, must reset.
                obs = TensorDict(next_obs, batch_size=[e], device=device)
                reset_ids = done.nonzero(as_tuple=False).flatten()
                if len(reset_ids):
                    controller.evaluation_step_offsets[reset_ids] = lengths[reset_ids]
                    reset_obs, _ = env.reset(env_ids=reset_ids)
                    traversal.reset_segments(reset_ids)
                    # TensorDict iteration is explicit; this is not a plain dict.
                    for key in obs.keys():  # noqa: SIM118
                        obs[key][reset_ids] = reset_obs[key][reset_ids]
            else:
                raise RuntimeError("Evaluation horizon exhausted with active trials")
        result = summarize_trials(
            rows, self.setup.catalog, self.setup.cfg.evaluation.trials_per_task
        )
        result["simulated_slot_steps"] = slot_steps
        result["geometry_checks"] = monitor.summary() if monitor is not None else {}
        result["protocol"] = (
            "v2: nominal, deterministic policy, fixed trial horizon; "
            "boundary resets continue the same task and command clock; "
            "physical failure ends the trial; not paper EPTE-SP"
        )
        return result

    def close(self):
        if self.env is not None:
            try:
                with preserve_rng():
                    self.env.close()
            finally:
                self.env = None


def summarize_trials(rows, catalog, trials_per_task):
    if len(rows) != catalog.num_tasks * trials_per_task:
        raise ValueError("Missing evaluation trials")
    grouped = [[] for _ in catalog.records]
    seen = set()
    for row in rows:
        key = (row["task_key"], row["trial_id"])
        if key in seen or catalog.decode(row["task_id"]).key != row["task_key"]:
            raise ValueError("Duplicate/misattributed evaluation trial")
        seen.add(key)
        if (
            row["steps"] <= 0
            or not np.isfinite([row["return"], *row["squared_error"]]).all()
        ):
            raise ValueError("Nonfinite or empty trial")
        grouped[row["task_id"]].append(row)
    per_task = []
    for tid, group in enumerate(grouped):
        if len(group) != trials_per_task or {r["trial_id"] for r in group} != set(
            range(trials_per_task)
        ):
            raise ValueError("Incomplete per-task trial coverage")
        returns = [r["return"] for r in group]
        steps = np.array([r["steps"] for r in group])
        rmse = np.sqrt(
            np.sum([r["squared_error"] for r in group], axis=0) / steps.sum()
        )
        per_task.append(
            {
                "task_id": tid,
                "task_key": catalog.decode(tid).key,
                "count": len(group),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "mean_length": float(steps.mean()),
                "vx_rmse": float(rmse[0]),
                "vy_rmse": float(rmse[1]),
                "wz_rmse": float(rmse[2]),
                "failure_rate": float(np.mean([r["terminated"] for r in group])),
                "boundary_rate": float(
                    np.mean([r["boundary"] and not r["terminated"] for r in group])
                ),
                "time_limit_rate": float(
                    np.mean(
                        [
                            r["time_limit"]
                            and not r["terminated"]
                            for r in group
                        ]
                    )
                ),
                "raw_boundary_rate": float(np.mean([r["boundary"] for r in group])),
                # boundary_rate 是曾到达边界的比例，可与 time_limit_rate 重叠。
                "mean_boundary_crossings": float(
                    np.mean([r["boundary_crossings"] for r in group])
                ),
                "mean_distance_m": float(np.mean([r["distance_m"] for r in group])),
                "mean_stair_contact_seconds": float(
                    np.mean([r["stair_contact_seconds"] for r in group])
                ),
                "mean_stalled_seconds": float(
                    np.mean([r["stalled_seconds"] for r in group])
                ),
            }
        )
    numeric = [k for k in per_task[0] if k not in {"task_id", "task_key", "count"}]
    return {
        "mean_returns": [r["mean_return"] for r in per_task],
        "counts": [r["count"] for r in per_task],
        "per_task": per_task,
        "trials": rows,
        "macro": {k: float(np.mean([r[k] for r in per_task])) for k in numeric},
    }
