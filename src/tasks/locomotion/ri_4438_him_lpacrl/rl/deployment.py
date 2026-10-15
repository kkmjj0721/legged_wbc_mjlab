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
PROFILES = {
    "forward_reverse": [ZERO, (.5, 0, 0), (-.5, 0, 0), (.5, 0, 0), (-.5, 0, 0), ZERO],
    "backward_forward": [ZERO, (-.5, 0, 0), (.5, 0, 0), (-.5, 0, 0), (.5, 0, 0), ZERO],
    "lateral_reverse": [ZERO, (0, .5, 0), (0, -.5, 0), (0, .5, 0), (0, -.5, 0), ZERO],
    "yaw_reverse": [ZERO, (0, 0, .5), (0, 0, -.5), (0, 0, .5), (0, 0, -.5), ZERO],
    "stop_restart": [ZERO, (.5, 0, 0), ZERO, (-.5, 0, 0), ZERO, (.5, 0, 0)],
    "diagonal_reverse": [ZERO, (.5, .3, .5), (-.5, -.3, -.5), (.5, -.3, .5), (-.5, .3, -.5), ZERO],
    "lateral_acceleration": [ZERO, (0, .2, 0), (0, .5, 0), (0, .9, 0), (0, -.5, 0), ZERO],
}


def make_deployment_setup(source_cfg):
    """Build a small flat suite with direct body-frame commands and no resets.

    The initial zero command belongs to one catalog task. The profile and trial
    identify the subsequent sequence; evaluation never feeds these into LP.
    Source configuration, training sampler and terrain catalog are untouched.
    """
    from ..config.curriculum_cfg import build_curriculum

    cfg = deepcopy(source_cfg)
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
                     "reversal_passed", "elapsed", "error_count"):
            setattr(self, name, torch.zeros(num_envs, dtype=torch.long, device=device))
        self.pending = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.reversing = torch.zeros_like(self.pending)
        self.error_sum = torch.zeros(num_envs, 3, dtype=torch.float64, device=device)
        self.trial_error_sum = torch.zeros_like(self.error_sum)
        self.trial_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

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

    def observe(self, actual, commands, active, terminated):
        changed = (active & self.seen
                   & ((commands - self.last_command).abs() >= self.change_thresholds - 1e-6).any(-1))
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
        self.seen[active] = True
        alive = active & ~terminated
        tolerance = torch.maximum(self.abs_tolerance, self.rel_tolerance * commands.abs())
        error_ratio = (actual - commands).abs() / tolerance
        self.trial_error_sum[alive] += error_ratio[alive].double()
        self.trial_steps += alive.long()

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
        self.stall_run = torch.where(stalled, self.stall_run + 1, 0)
        self.max_stall = torch.maximum(self.max_stall, self.stall_run)
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
        self.reversal_passed += (passed & self.reversing).long()
        self.pending[complete | terminated] = False

    def trial(self, slot):
        return {
            "deployment_tracking_error": (self.trial_error_sum[slot] / self.trial_steps[slot].clamp(min=1)).tolist(),
            "max_stall_seconds": float(self.max_stall[slot]) * self.dt,
            "deployment_stalled_seconds": float(self.stalled_steps[slot]) * self.dt,
            **{name: int(getattr(self, name)[slot]) for name in (
                "transition_count", "transition_completed", "transition_passed",
                "reversal_count", "reversal_passed")},
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
    }
