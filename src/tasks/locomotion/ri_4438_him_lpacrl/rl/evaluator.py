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
from mjlab.utils.lab_api.math import wrap_to_pi

from ..config.env_cfgs import build_env_cfg
from ..mdp.curriculums import get_controller
from ..mdp.task_catalog import plain
from .metrics import epte_sp, relative_tracking_error, reward_terms_for_step
from .deployment import DeploymentMetrics, deployment_summary


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
    def __init__(self, setup, device, env_factory=None, step_callback=None):
        self.setup, self.device = setup, device
        self.env = None
        self.env_factory = env_factory
        self.step_callback = step_callback

    def _get_env(self):
        if self.env is None:
            if self.env_factory is None:
                from src.envs import ManagerBasedRlEnv

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
        evaluation_cfg = self.setup.cfg.evaluation
        reward_names = tuple(env.reward_manager.active_terms)
        floors = torch.tensor(evaluation_cfg.relative_error_floors, device=device)
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
            relative_errors = torch.zeros_like(errors)
            term_returns = torch.zeros(
                e, len(reward_names), dtype=torch.float64, device=device
            )
            boundary_counts = torch.zeros(e, dtype=torch.long, device=device)
            traversal = TraversalMetrics(env, controller)
            deployment = DeploymentMetrics(e, device, env.step_dt, evaluation_cfg)
            for step in range(max_steps):
                actions = actor(obs)
                # step() 返回时指令可能已更新；误差必须对应刚执行动作的指令。
                commands = controller.commands.clone()
                command_event = controller.command_event.clone()
                command_delta = controller.command_event_delta.clone()
                heading_delta = controller.command_heading_delta.clone()
                heading_target = controller.heading_targets.clone() if self.setup.catalog.heading_command else None
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
                # Only physical failures end the alive portion; a boundary
                # reset or normal time limit is not a fall.
                alive = active & ~terminated
                relative_errors[alive] += relative_tracking_error(
                    actual, commands, floors
                )[alive].double()
                term_returns[active] += reward_terms_for_step(env)[active]
                slot_steps += e
                traversal.observe(active, commands, actions)
                deployment.observe(actual, commands, active, terminated,
                                   event=command_event, event_delta=command_delta,
                                   heading_delta=heading_delta,
                                   support_tile=(traversal.connected.support_tiles if traversal.connected is not None else None),
                                   support_surface=(traversal.connected.support_surface if traversal.connected is not None else None),
                                   heading_error=(wrap_to_pi(heading_target - env.scene["robot"].data.heading_w) if heading_target is not None else None))
                if self.step_callback is not None:
                    self.step_callback(env, batch, active, step)
                done = terminated | truncated
                if monitor is not None:
                    monitor.observe(done | ~active)
                boundary_name = "map_boundary_truncation" if controller.terrain_map.connected else "boundary_truncation"
                boundary = (
                    env.termination_manager.get_term(boundary_name)
                    if boundary_name in env.termination_manager.active_terms
                    else torch.zeros_like(done)
                )
                boundary_counts[active] += boundary[active].long()
                time_limit = lengths >= max_steps
                # 出界只结束当前物理片段，不缩短试验的评分预算。物理失败仍结束试验。
                finished = ((terminated | time_limit | (boundary if controller.terrain_map.connected else torch.zeros_like(boundary))) & active).nonzero(
                    as_tuple=False
                ).flatten()
                deployment.finish(finished)
                for slot in finished.tolist():
                    rows.append(
                        {
                            **batch[slot],
                            "return": float(returns[slot]),
                            "steps": int(lengths[slot]),
                            "squared_error": errors[slot].cpu().tolist(),
                            "relative_error_sum": relative_errors[slot].cpu().tolist(),
                            "alive_steps": int(lengths[slot]) - int(terminated[slot]),
                            "horizon_steps": max_steps,
                            "reward_terms": dict(
                                zip(reward_names, term_returns[slot].cpu().tolist())
                            ),
                            "terminated": bool(terminated[slot]),
                            "truncated": bool((time_limit[slot] or boundary[slot]) and not terminated[slot]),
                            "boundary": bool(boundary_counts[slot] > 0),
                            "boundary_crossings": int(boundary_counts[slot]),
                            "time_limit": bool(time_limit[slot]),
                            **traversal.trial(slot),
                            **deployment.trial(slot),
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
                    deployment.reset_segments(reset_ids)
                    # TensorDict iteration is explicit; this is not a plain dict.
                    for key in obs.keys():  # noqa: SIM118
                        obs[key][reset_ids] = reset_obs[key][reset_ids]
            else:
                raise RuntimeError("Evaluation horizon exhausted with active trials")
        result = summarize_trials(
            rows, self.setup.catalog, evaluation_cfg.trials_per_task,
            success_alive_fraction=evaluation_cfg.success_alive_fraction,
            success_tracking_error=evaluation_cfg.success_tracking_error,
            deployment_stall_s=evaluation_cfg.deployment_stall_s,
        )
        result["simulated_slot_steps"] = slot_steps
        result["geometry_checks"] = monitor.summary() if monitor is not None else {}
        result["protocol"] = (
            "v7: nominal, deterministic policy; connected maps end trials at the outer support boundary; "
            "legacy_v6 maps alone continue trials after patch boundary resets; explicit sampler events; "
            "physical failure ends the trial; EPTE-SP stability aggregation from Eq. (8), "
            "project-relative-error convention for changing/near-zero commands; "
            "paper success retained; separate all-axis, stall and switch deployment gates"
        )
        result["metric_settings"] = {
            "traversal_version": 7 if controller.terrain_map.connected else 1,
            "stair_progress": ">=2 outward treads; >=2 supported feet and root vertical progress for >=0.1 s; survives trial",
            "terrain_traversal": ("four feet beyond actual exit plus margin; >=2 real ground contacts for confirmation duration; latched even after failure"
                                  if controller.terrain_map.connected else "legacy all internal treads; survives trial"),
            "pass_margin_m": self.setup.cfg.geometry.pass_margin_m,
            "pass_confirm_s": self.setup.cfg.geometry.pass_confirm_s,
            "heading_convergence_tolerance_rad": .15,
            "action_delta_rms": "unscaled policy actions; boundary-reset transitions excluded",
            "relative_error_floors": list(evaluation_cfg.relative_error_floors),
            "success_alive_fraction": evaluation_cfg.success_alive_fraction,
            "success_tracking_error": evaluation_cfg.success_tracking_error,
            "success": "legacy alias of paper_success: alive fraction, vx and wz EPTE-SP",
            "tracking_success": "alive fraction and vx/vy/wz EPTE-SP",
            "deployment_success": "all axes within deployment absolute/relative tolerances, no physical failure, no sustained stall, all observed switches pass",
            "deployment_tracking_error": "mean_executed(abs(error)/max(abs_tolerance,rel_tolerance*abs(command))); each axis <1",
            "deployment": {
                key: plain(value) for key, value in vars(evaluation_cfg).items()
                if key.startswith("deployment_")
            },
            "transition_coverage": "zero observed switches/reversals -> null success, not a successful test",
            "reward_terms": "weighted contributions, same dt scaling as total return",
            "relative_error": "clip(mean_alive(abs(error)/max(abs(command),floor)),0,1)",
        }
        return result

    def close(self):
        if self.env is not None:
            try:
                with preserve_rng():
                    self.env.close()
            finally:
                self.env = None


def summarize_trials(
    rows, catalog, trials_per_task, *,
    success_alive_fraction=0.9, success_tracking_error=0.3,
    deployment_stall_s=2.0,
):
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
    per_task, trial_returns = [], []
    reward_names = tuple(rows[0]["reward_terms"])
    for tid, group in enumerate(grouped):
        if len(group) != trials_per_task or {r["trial_id"] for r in group} != set(
            range(trials_per_task)
        ):
            raise ValueError("Incomplete per-task trial coverage")
        group.sort(key=lambda r: r["trial_id"])
        returns = [r["return"] for r in group]
        trial_returns.append(returns)
        for row in group:
            if (
                row["steps"] > row["horizon_steps"]
                or row["alive_steps"] != row["steps"] - int(row["terminated"])
                or (not row["terminated"] and not row["boundary"] and row["steps"] != row["horizon_steps"])
                or set(row["reward_terms"]) != set(reward_names)
                or not np.isfinite(list(row["reward_terms"].values())).all()
                or not np.isclose(
                    sum(row["reward_terms"].values()), row["return"],
                    rtol=1e-5,
                    # The manager sums float32 terms before we accumulate in
                    # float64; tolerate cancellation relative to component size.
                    atol=1e-6 * max(1.0, sum(abs(v) for v in row["reward_terms"].values())),
                )
            ):
                raise ValueError("Invalid trial duration or reward decomposition")
            scores = epte_sp(
                row["relative_error_sum"], row["alive_steps"], row["horizon_steps"]
            )
            row.update(zip(("vx_epte_sp", "vy_epte_sp", "wz_epte_sp"), scores.tolist()))
            row["paper_success"] = bool(
                row["alive_steps"] > success_alive_fraction * row["horizon_steps"]
                and scores[0] < success_tracking_error
                and scores[2] < success_tracking_error
            )
            row["success"] = row["paper_success"]  # Historical result consumers retain the paper metric.
            row["tracking_success"] = bool(
                row["alive_steps"] > success_alive_fraction * row["horizon_steps"]
                and (scores < success_tracking_error).all()
            )
            if "max_stall_seconds" in row and "deployment_tracking_error" in row:
                deployment_error = np.asarray(row["deployment_tracking_error"])
                if deployment_error.shape != (3,) or not np.isfinite(deployment_error).all() or (deployment_error < 0).any():
                    raise ValueError("Invalid deployment tracking errors")
                counters = ("transition_count", "transition_completed", "transition_passed",
                            "reversal_count", "reversal_passed")
                if (any(type(row[k]) is not int or row[k] < 0 for k in counters)
                        or not 0 <= row["transition_passed"] <= row["transition_completed"] <= row["transition_count"]
                        or not 0 <= row["reversal_passed"] <= row["reversal_count"] <= row["transition_count"]
                        or row["reversal_passed"] > row["transition_passed"]
                        or not math.isfinite(row["max_stall_seconds"]) or row["max_stall_seconds"] < 0):
                    raise ValueError("Invalid deployment temporal diagnostics")
                row["deployment_tracking_success"] = bool((deployment_error < 1.).all())
                row["transition_success"] = bool(
                    not row["terminated"] and row["transition_passed"] == row["transition_count"]
                ) if row["transition_count"] else None
                row["reversal_success"] = bool(
                    not row["terminated"] and row["reversal_passed"] == row["reversal_count"]
                ) if row["reversal_count"] else None
                row["deployment_success"] = bool(
                    row["deployment_tracking_success"] and not row["terminated"]
                    and row["max_stall_seconds"] + 1e-9 < deployment_stall_s
                    and row["transition_passed"] == row["transition_count"]
                    and row["steps"] == row["horizon_steps"]
                )
                row["process_success"] = bool(row["deployment_success"] and row["transition_count"] > 0)
                row["process_incomplete"] = bool(row.get("transition_unobserved", 0) > 0)
            else:
                # Old saved trials have no time-resolved evidence. Do not
                # invent a successful stall/transition test from episode means.
                row["deployment_success"] = None
                row["deployment_tracking_success"] = None
                row["transition_success"] = row["reversal_success"] = None
            if "stair_steps_crossed" in row:
                stairs = row["stair_step_count"] > 0
                row["stair_progress_success"] = bool(stairs and row["stair_steps_crossed"] >= 2 and not row["terminated"])
                row["terrain_traversal_success"] = bool(stairs and row["stair_steps_crossed"] >= row["stair_step_count"] and not row["terminated"])
                row["legacy_terrain_traversal_success"] = row["terrain_traversal_success"]
            if "terrain_passed" in row:
                row["terrain_traversal_success"] = row["terrain_passed"]
                row["terrain_passed_without_failure"] = bool(row["terrain_passed"] and not row["terminated"] and row["steps"] == row["horizon_steps"])
                row["obstacle_traversal_success"] = row["terrain_passed"] if row["obstacle_eligible"] else None
        steps = np.array([r["steps"] for r in group])
        rmse = np.sqrt(
            np.sum([r["squared_error"] for r in group], axis=0) / steps.sum()
        )
        per_task.append(
            {
                "task_id": tid,
                "task_key": catalog.decode(tid).key,
                "terrain": catalog.decode(tid).terrain,
                "level": catalog.decode(tid).level,
                "task_mode": getattr(catalog.decode(tid), "mode", "fixed"),
                "count": len(group),
                "mean_return": float(np.mean(returns)),
                "std_return": float(np.std(returns)),
                "return_standard_error": float(np.std(returns, ddof=1) / np.sqrt(len(group)))
                if len(group) > 1 else None,
                "mean_length": float(steps.mean()),
                "vx_rmse": float(rmse[0]),
                "vy_rmse": float(rmse[1]),
                "wz_rmse": float(rmse[2]),
                **{
                    axis + "_epte_sp": float(np.mean([r[axis + "_epte_sp"] for r in group]))
                    for axis in ("vx", "vy", "wz")
                },
                "success_rate": float(np.mean([r["success"] for r in group])),
                "paper_success_rate": float(np.mean([r["paper_success"] for r in group])),
                "tracking_success_rate": float(np.mean([r["tracking_success"] for r in group])),
                **{
                    "reward/" + name: float(np.mean([r["reward_terms"][name] for r in group]))
                    for name in reward_names
                },
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
        deployed = deployment_summary(group)
        per_task[-1].update(
            deployment_success_rate=deployed["success_rate"],
            deployment_evaluated_fraction=deployed["evaluated_trials"] / len(group),
        )
        # These new diagnostics are optional for historical saved results.
        for key in ("net_displacement_m", "action_delta_rms", "foot_slip_m_s", "stair_steps_crossed"):
            if key in group[0]:
                per_task[-1]["mean_" + key] = float(np.mean([r[key] for r in group]))
        if "integrated_yaw_error_rad" in group[0]:
            per_task[-1]["mean_abs_integrated_yaw_error_rad"] = float(np.mean([abs(r["integrated_yaw_error_rad"]) for r in group]))
        if "terrain_traversal_success" in group[0]:
            for key in ("stair_progress_success", "terrain_traversal_success"):
                per_task[-1][key + "_rate"] = float(np.mean([r[key] for r in group]))
        if "terrain_passed_without_failure" in group[0]:
            per_task[-1]["terrain_passed_without_failure_rate"] = float(np.mean([r["terrain_passed_without_failure"] for r in group]))
            eligible = [r["obstacle_traversal_success"] for r in group if r["obstacle_traversal_success"] is not None]
            per_task[-1]["obstacle_traversal_success_rate"] = float(np.mean(eligible)) if eligible else None
    numeric = [
        k for k, v in per_task[0].items()
        if k not in {"task_id", "count"} and (v is None or isinstance(v, (int, float)))
        and any(r[k] is not None for r in per_task)
    ]
    def mean_present(tasks, key):
        values = [r[key] for r in tasks if r[key] is not None]
        return float(np.mean(values)) if values else None
    by_terrain = {}
    for terrain in sorted({r["terrain"] for r in per_task}):
        tasks = [r for r in per_task if r["terrain"] == terrain]
        by_terrain[terrain] = {
            k: mean_present(tasks, k) for k in numeric
        }
    return {
        "mean_returns": [r["mean_return"] for r in per_task],
        "counts": [r["count"] for r in per_task],
        "trial_returns": trial_returns,
        "per_task": per_task,
        "trials": rows,
        "macro": {k: mean_present(per_task, k) for k in numeric},
        "by_terrain": by_terrain,
        "by_task_mode": {
            mode: {"episodes": len(selected), "environment_steps": sum(r["steps"] for r in selected),
                   "zero_switch_rate": float(np.mean([r.get("resampling_count", 0) == 0 for r in selected])),
                   "process_success_rate": float(np.mean([r.get("process_success", False) for r in selected])),
                   "traversal_incidence": float(np.mean([r.get("terrain_passed", False) for r in selected])),
                   "deployment": deployment_summary(selected)}
            for mode in sorted({getattr(t, "mode", "fixed") for t in catalog.records})
            if (selected := [r for r in rows if getattr(catalog.decode(r["task_id"]), "mode", "fixed") == mode])
        },
        "by_switch_surface": {
            surface: {"attempts": len(events),
                      **{outcome: sum(e["result"] == outcome for e in events)
                         for outcome in ("passed", "failed", "unobserved")},
                      "complete_windows_per_million_steps": sum(e["window_complete"] for e in events) * 1e6 / sum(r["steps"] for r in rows)}
            for surface in ("flat", "rough", "stairs", "slope", "mixed", "unsupported")
            if (events := [event for r in rows for event in r.get("switch_events", [])
                           if event["significant"] and event["support_surface"] == surface])
        },
        "deployment": deployment_summary(rows),
        "by_profile": {
            profile: deployment_summary([r for r in rows if r.get("profile") == profile])
            for profile in sorted({r["profile"] for r in rows if "profile" in r})
        },
    }
