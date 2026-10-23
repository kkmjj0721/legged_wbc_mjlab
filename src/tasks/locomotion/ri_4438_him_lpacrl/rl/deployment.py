"""Deployment diagnostics and fixed command sequences, separate from training.

All-axis tolerances/progress monitoring follow the Go2 LP-ACRL reference's
command diagnostics. Sustained-stall and post-switch gates are local extensions;
they are not the paper's success criterion.
Reference: github.com/lcy31415/Unitree-Go2-Adaptive-Energy-Locomotion,
commit c527f4325029, mdp/commands/lp_acrl_velocity_command.py.
"""

from copy import deepcopy
import math
from types import SimpleNamespace as NS

import numpy as np
import torch

ZERO = (0.0, 0.0, 0.0)
DEPLOYMENT_SEQUENCE_VERSION = 2
PROFILES = {
    "forward_reverse": [ZERO, (.5, 0, 0), (-.5, 0, 0), (.5, 0, 0), (-.5, 0, 0), ZERO],
    "backward_forward": [ZERO, (-.5, 0, 0), (.5, 0, 0), (-.5, 0, 0), (.5, 0, 0), ZERO],
    "lateral_reverse": [ZERO, (0, .5, 0), (0, -.5, 0), (0, .5, 0), (0, -.5, 0), ZERO],
    "yaw_reverse": [ZERO, (0, 0, .5), (0, 0, -.5), (0, 0, .5), (0, 0, -.5), ZERO],
    "stop_restart": [ZERO, (.5, 0, 0), ZERO, (-.5, 0, 0), ZERO, (.5, 0, 0)],
    "diagonal_reverse": [ZERO, (.5, .3, .5), (-.5, -.3, -.5), (.5, -.3, .5), (-.5, .3, -.5), ZERO],
    "lateral_acceleration": [ZERO, (0, .2, 0), (0, .5, 0), (0, .9, 0), (0, -.5, 0), ZERO],
    "low_speed_01": [ZERO, (.1, 0, 0), ZERO, (.1, 0, 0), (-.1, 0, 0), ZERO],
    "low_speed_02": [ZERO, (.2, 0, 0), ZERO, (.2, 0, 0), (-.2, 0, 0), ZERO],
    "low_speed_03": [ZERO, (.3, 0, 0), ZERO, (.3, 0, 0), (-.3, 0, 0), ZERO],
}


def clear_compact_task_mode(source_cfg):
    """Copy a configuration for explicit evaluation commands, outside Compact.

    Fixed evaluation profiles have their own task table; training environment
    family quotas must not constrain that independent table.
    """
    cfg = deepcopy(source_cfg)
    if hasattr(cfg.task_space, "compact"):
        cfg.task_space.compact = False
    for name in ("family_fractions", "uniform_families"):
        if hasattr(cfg.curriculum, name):
            delattr(cfg.curriculum, name)
    return cfg


def make_deployment_setup(source_cfg):
    """Build a small flat suite with direct body-frame commands and no resets.

    The initial zero command belongs to one catalog task. The profile and trial
    identify the subsequent sequence; evaluation never feeds these into LP.
    Source configuration, training sampler and terrain catalog are untouched.
    """
    from ..config.curriculum_cfg import build_curriculum

    cfg = clear_compact_task_mode(source_cfg)
    cfg.task_space.random_process = False
    cfg.task_space.fixed_tasks = True
    repeats = cfg.evaluation.trials_per_task
    cfg.task_space.heading_command = False
    default_axis = vars(cfg.task_space.command_axes["vx"])
    cfg.task_space.command_axes = {
        name: NS(**{**default_axis, "mode": "fixed", "value": 0.,
                    "edges": None, "bins": None, "range": None})
        for name in ("vx", "vy", "wz")
    }
    cfg.task_space.terrains = {"flat": NS(levels={"base": {}})}
    cfg.task_space.include, cfg.task_space.exclude = [], []
    cfg.geometry.replicas = 1
    cfg.curriculum.seed_bin = 0
    cfg.curriculum.forward_stairs_fraction = 0.0
    # Version 2 adds low-speed trials. Their 0.1 m/s starts must be observable
    # as switches and movement commands, rather than filtered by a 0.2 gate.
    cfg.evaluation.deployment_sequence_version = DEPLOYMENT_SEQUENCE_VERSION
    change = cfg.evaluation.deployment_change_thresholds
    cfg.evaluation.deployment_change_thresholds = (
        min(change[0], .05), min(change[1], .05), change[2],
    )
    moving, turning = cfg.evaluation.deployment_stall_command
    cfg.evaluation.deployment_stall_command = (min(moving, .05), turning)
    tolerance = cfg.evaluation.deployment_tracking_abs
    cfg.evaluation.deployment_tracking_abs = (
        min(tolerance[0], .05), min(tolerance[1], .05), tolerance[2],
    )
    cfg.task_space.resampling_interval_s = cfg.evaluation.deployment_segment_s
    cfg.evaluation.horizon_s = len(next(iter(PROFILES.values()))) * cfg.evaluation.deployment_segment_s
    cfg.evaluation.trials_per_task = repeats * len(PROFILES)
    cfg.evaluation.num_envs = min(cfg.evaluation.num_envs, cfg.evaluation.trials_per_task)
    setup = build_curriculum(cfg)
    for row in setup.trials:
        index = row["trial_id"]
        profile = list(PROFILES)[index // repeats]
        row.update(profile=profile, profile_trial_id=index % repeats,
                   command=list(ZERO), command_sequence=[list(c) for c in PROFILES[profile]])
    return setup


class DeploymentMetrics:
    """Streaming metrics on pre-reset transitions; bounded memory per batch."""

    def __init__(self, num_envs, device, dt, cfg):
        self.dt, self.cfg = dt, cfg
        self.window_steps = max(1, math.ceil(cfg.deployment_stall_window_s / dt))
        self.grace_steps = math.ceil(cfg.deployment_transition_grace_s / dt)
        self.check_steps = max(1, math.ceil(cfg.deployment_transition_window_s / dt))
        self.abs_tolerance = torch.tensor(cfg.deployment_tracking_abs, device=device)
        self.rel_tolerance = torch.tensor(cfg.deployment_tracking_rel, device=device)
        self.change_thresholds = torch.tensor(cfg.deployment_change_thresholds, device=device)
        self.last_command = torch.zeros(num_envs, 3, device=device)
        self.seen = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.history = torch.zeros(self.window_steps, num_envs, 3, device=device)
        self.history_sum = torch.zeros(num_envs, 3, device=device)
        self.history_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cursor = 0
        for name in ("stall_run", "max_stall", "stalled_steps", "transition_count",
                     "transition_completed", "transition_passed", "reversal_count",
                     "reversal_passed", "elapsed", "error_count", "transition_failed",
                     "transition_unobserved", "resampling_count"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long, device=device))
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reversing = torch.zeros_like(self.pending)
        self.error_sum = torch.zeros(num_envs, 3, dtype=torch.float64, device=device)
        self.trial_error_sum = torch.zeros_like(self.error_sum)
        self.trial_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.event_capacity = max(2, math.ceil(cfg.horizon_s / 2) + 2)
        self.event_data = torch.zeros(num_envs, self.event_capacity, 14, device=device)
        # time, delta vx/vy/wz, heading delta, pre-switch vx/vy/wz,
        # support tile, significant, result (0 incomplete, 1 pass, -1 fail), window steps.
        self.pending_event = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_actual = torch.zeros_like(self.last_command)
        self.heading_error_sum = torch.zeros(num_envs, device=device)
        self.heading_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.heading_elapsed = torch.zeros_like(self.heading_steps)
        self.heading_pending = torch.zeros_like(self.pending)
        self.heading_event = torch.zeros_like(self.heading_steps)

    def _clear_motion_history(self, ids):
        self.history[:, ids] = 0
        self.history_sum[ids] = 0
        self.history_count[ids] = 0
        self.stall_run[ids] = 0

    def reset_segments(self, ids):
        # A teleport cannot complete a physical transition. Its attempted count
        # remains, so an interrupted switch cannot silently pass the gate.
        self._clear_motion_history(ids)
        self.seen[ids] = False
        self.pending[ids] = False

    def reset(self, ids):
        """Discard only new-episode transient state, never on command changes."""
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and value.ndim and name not in {
                "abs_tolerance", "rel_tolerance", "change_thresholds", "history"}:
                value[ids] = 0
        self.history[:, ids] = 0

    def finish(self, ids):
        self.transition_unobserved[ids] += self.pending[ids].long()
        self.pending[ids] = False

    def observe(self, actual, commands, active, terminated, *, event=None,
                event_delta=None, heading_delta=None, support_tile=None, heading_error=None, support_surface=None):
        delta = commands - self.last_command if event_delta is None else event_delta
        changed = (active & self.seen
                   & (delta.abs() >= self.change_thresholds - 1e-6).any(-1))
        if event is not None:
            changed &= event
        sampled = changed if event is None else active & event
        self.transition_unobserved += (changed & self.pending).long()
        slots = sampled.nonzero(as_tuple=False).flatten()
        indexes = self.resampling_count[slots].clamp_max(self.event_capacity - 1)
        self.event_data[slots, indexes, 0] = self.trial_steps[slots].float() * self.dt
        self.event_data[slots, indexes, 1:4] = delta[slots]
        if heading_delta is not None:
            self.event_data[slots, indexes, 4] = heading_delta[slots]
        self.event_data[slots, indexes, 5:8] = self.last_actual[slots]
        if support_tile is not None:
            self.event_data[slots, indexes, 8] = support_tile[slots].float()
        if support_surface is not None:
            self.event_data[slots, indexes, 13] = support_surface[slots].float()
        self.event_data[slots, indexes, 9] = changed[slots].float()
        self.event_data[slots, indexes, 12] = -1
        self.heading_event[slots] = indexes
        self.heading_elapsed[slots] = 0
        self.heading_pending[slots] = True
        self.pending_event[slots] = indexes
        self.resampling_count += sampled.long()
        reversed_axis = ((commands * self.last_command < 0)
                         & (commands.abs() >= self.change_thresholds)
                         & (self.last_command.abs() >= self.change_thresholds))
        reversal = changed & reversed_axis.any(-1)
        # Starting a new switch leaves an unfinished earlier switch unsuccessful.
        self.transition_count += changed.long()
        self.reversal_count += reversal.long()
        self.pending[changed] = True
        self.reversing[changed] = reversal[changed]
        self.elapsed[changed] = 0
        self.error_sum[changed] = 0
        self.error_count[changed] = 0
        self._clear_motion_history(changed)
        self.last_command[active] = commands[active]
        self.last_actual[active] = actual[active]
        self.seen[active] = True
        alive = active & ~terminated
        tolerance = torch.maximum(self.abs_tolerance, self.rel_tolerance * commands.abs())
        error_ratio = (actual - commands).abs() / tolerance
        self.trial_error_sum[active] += error_ratio[active].double()
        self.trial_steps += active.long()
        if heading_error is not None:
            self.heading_error_sum[active] += heading_error[active].abs()
            self.heading_steps += active.long()
            self.heading_elapsed[self.heading_pending & active] += 1
            converged = self.heading_pending & active & ~terminated & (heading_error.abs() < .15)
            slots = converged.nonzero().flatten()
            self.event_data[slots, self.heading_event[slots], 12] = self.heading_elapsed[slots].float() * self.dt
            self.heading_pending[converged | terminated] = False

        self.history_sum[alive] -= self.history[self.cursor, alive]
        self.history[self.cursor, alive] = actual[alive]
        self.history_sum[alive] += actual[alive]
        self.history_count[alive] = (self.history_count[alive] + 1).clamp(max=self.window_steps)
        self.cursor = (self.cursor + 1) % self.window_steps
        velocity = self.history_sum / self.history_count.clamp(min=1)[:, None]
        speed = commands[:, :2].norm(dim=-1)
        progress = (velocity[:, :2] * commands[:, :2]).sum(-1) / speed.clamp(min=1e-6)
        yaw_progress = velocity[:, 2] * commands[:, 2].sign()
        moving, turning = self.cfg.deployment_stall_command
        min_linear, min_angular = self.cfg.deployment_stall_speed
        stalled = (alive & (self.history_count == self.window_steps) & (
            ((speed > moving) & (progress < min_linear))
            | ((commands[:, 2].abs() > turning) & (yaw_progress < min_angular))))
        self.stall_run.copy_(torch.where(stalled, self.stall_run + 1, 0))
        self.max_stall.copy_(torch.maximum(self.max_stall, self.stall_run))
        self.stalled_steps += stalled.long()

        self.elapsed[self.pending & active] += 1
        checking = self.pending & alive & (self.elapsed > self.grace_steps)
        self.error_sum[checking] += error_ratio[checking].double()
        self.error_count += checking.long()
        complete = self.pending & active & (self.elapsed >= self.grace_steps + self.check_steps)
        error = self.error_sum / self.error_count.clamp(min=1)[:, None]
        passed = (complete & alive & (self.error_count >= self.check_steps)
                  & (error < 1.).all(-1))
        self.transition_completed += complete.long()
        self.transition_passed += passed.long()
        failed = (complete & ~passed) | (self.pending & active & terminated)
        self.transition_failed += failed.long()
        self.reversal_passed += (passed & self.reversing).long()
        slots = (complete | failed).nonzero(as_tuple=False).flatten()
        self.event_data[slots, self.pending_event[slots], 10] = torch.where(passed[slots], 1., -1.)
        self.event_data[slots, self.pending_event[slots], 11] = self.error_count[slots].float()
        self.pending[complete | terminated] = False

    def trial(self, slot):
        events = self.event_data[slot, :min(int(self.resampling_count[slot]), self.event_capacity)].tolist()
        return {
            "deployment_tracking_error": (self.trial_error_sum[slot] / self.trial_steps[slot].clamp(min=1)).tolist(),
            "max_stall_seconds": float(self.max_stall[slot]) * self.dt,
            "deployment_stalled_seconds": float(self.stalled_steps[slot]) * self.dt,
            **{name: int(getattr(self, name)[slot]) for name in (
                "transition_count", "transition_completed", "transition_passed",
                "reversal_count", "reversal_passed", "transition_failed", "resampling_count")},
            "transition_unobserved": int(self.transition_unobserved[slot]) + int(self.pending[slot]),
            "heading_error_mean_rad": float(self.heading_error_sum[slot] / self.heading_steps[slot].clamp_min(1)) if self.heading_steps[slot] else None,
            "switch_events": [{"time_s": e[0], "command_delta": e[1:4], "heading_delta_rad": e[4],
                               "velocity_before": e[5:8], "support_tile": int(e[8]),
                               "significant": bool(e[9]),
                               "result": ({1: "passed", -1: "failed", 0: "unobserved"}[int(e[10])] if e[9] else "not_significant"),
                               "check_steps": int(e[11]),
                               "window_complete": e[11] >= self.check_steps,
                               "heading_convergence_s": e[12] if e[12] >= 0 else None,
                               "support_surface": ("flat", "rough", "stairs", "slope", "mixed", "unsupported")[int(e[13])]}
                              for e in events],
        }


def deployment_summary(rows):
    """Report coverage explicitly: missing switch evidence is never a pass."""
    evaluated = [r for r in rows if r.get("deployment_success") is not None]
    transitions = [r for r in evaluated if r["transition_count"] > 0]
    reversals = [r for r in evaluated if r["reversal_count"] > 0]
    return {
        "evaluated_trials": len(evaluated),
        "success_rate": float(np.mean([r["deployment_success"] for r in evaluated])) if evaluated else None,
        "transition_trials": len(transitions),
        "transition_success_rate": float(np.mean([r["transition_success"] for r in transitions])) if transitions else None,
        "reversal_trials": len(reversals),
        "reversal_success_rate": float(np.mean([r["reversal_success"] for r in reversals])) if reversals else None,
        "resamplings": sum(r.get("resampling_count", 0) for r in rows),
        "significant_switches": sum(r.get("transition_count", 0) for r in rows),
        "switch_passed": sum(r.get("transition_passed", 0) for r in rows),
        "switch_failed": sum(r.get("transition_failed", 0) for r in rows),
        "switch_unobserved": sum(r.get("transition_unobserved", 0) for r in rows),
    }
