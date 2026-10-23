"""Shared, once-per-control-step motion monitoring for LP-ACRL.

This state belongs to the training task, independently of deployment metrics.
Thresholds are experiment parameters, not calibrated robot limits. Consumers
must finish an old episode before calling ``reset`` for its environment IDs.
"""

from __future__ import annotations

import math

import torch


class BehaviorMonitor:
    """Measure sustained lack of signed progress using simulator velocities.

    ``ensure_step`` may be called by termination, reward and diagnostics. The
    first caller updates the state; subsequent callers read the same tensors.
    It does not consult the termination manager's partially computed result.
    """

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.num_envs, self.device = env.num_envs, env.device
        self.dt = float(env.step_dt)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Behavior monitoring requires positive finite step_dt")

        self.enabled = self._option("enabled", False)
        self.termination_enabled = self._option("termination_enabled", False)
        if type(self.enabled) is not bool or type(self.termination_enabled) is not bool:
            raise ValueError("Behavior monitoring switches must be boolean")
        defaults = {
            "linear_command_threshold": 0.08,
            "angular_command_threshold": 0.1,
            "linear_progress_floor": 0.02,
            "angular_progress_floor": 0.02,
            "progress_ratio": 0.2,
            "smoothing_s": 0.3,
            "grace_s": 0.8,
            "penalty_sustain_s": 0.4,
            "termination_sustain_s": 3.0,
            "heading_change_threshold": 0.2,
        }
        for name, default in defaults.items():
            value = float(self._option(name, default))
            if not math.isfinite(value) or value < 0 or (name != "grace_s" and value == 0):
                raise ValueError(f"Invalid behavior-monitor parameter {name}")
            setattr(self, name, value)
        if self.progress_ratio > 1:
            raise ValueError("Behavior progress_ratio must be in (0, 1]")
        if self.termination_sustain_s < self.penalty_sustain_s:
            raise ValueError("Stall termination must wait at least as long as its penalty")
        changes = self._option("command_change_thresholds", (0.08, 0.08, 0.1))
        if len(changes) != 3 or any(not math.isfinite(v) or v <= 0 for v in changes):
            raise ValueError("command_change_thresholds needs three positive finite values")

        def steps(seconds):
            return math.ceil(seconds / self.dt - 1e-9)

        self.window_steps = max(1, steps(self.smoothing_s))
        self.grace_steps = steps(self.grace_s)
        self.penalty_steps_required = max(1, steps(self.penalty_sustain_s))
        self.termination_steps_required = max(1, steps(self.termination_sustain_s))
        # PPO creates monitors inside inference_mode; reset/checkpoint callers
        # also mutate these buffers outside inference_mode.
        with torch.inference_mode(False):
            self.change_thresholds = torch.tensor(changes, device=self.device)
            self._history = torch.zeros(self.window_steps, self.num_envs, 2, device=self.device)
            self._history_sum = torch.zeros(self.num_envs, 2, device=self.device)
            for name in (
                "cost", "linear_progress", "yaw_progress", "cost_sum",
            ):
                setattr(self, name, torch.zeros(self.num_envs, device=self.device))
            for name in (
                "steps", "linear_active_steps", "yaw_active_steps", "grace_observed_steps",
                "linear_stall_steps", "yaw_stall_steps", "penalty_steps", "significant_events",
                "linear_stall_run", "yaw_stall_run", "max_linear_stall_steps", "max_yaw_stall_steps",
                "grace_remaining_steps", "_history_count", "_cursor",
            ):
                setattr(self, name, torch.zeros(self.num_envs, dtype=torch.long, device=self.device))
            for name in ("terminated", "linear_stalled", "yaw_stalled", "grace_active", "_seen"):
                setattr(self, name, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            self._last_tick = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            self._last_event_step = torch.full_like(self._last_tick, -1)

    def _option(self, name, default):
        if isinstance(self.cfg, dict):
            return self.cfg.get(name, default)
        return getattr(self.cfg, name, default)

    def _ids(self, env_ids):
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[
                slice(None) if env_ids is None else env_ids
            ]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _clear_window(self, ids):
        self._history[:, ids] = 0
        self._history_sum[ids] = 0
        self._history_count[ids] = 0
        self._cursor[ids] = 0
        self.linear_stall_run[ids] = 0
        self.yaw_stall_run[ids] = 0

    def reset(self, env_ids=None):
        """Clear selected episodes, without allowing this reset to consume a step."""
        ids = self._ids(env_ids)
        for name, value in vars(self).items():
            if (isinstance(value, torch.Tensor) and value.ndim == 1
                    and name != "change_thresholds"):
                value[ids] = 0
        self._clear_window(ids)
        self._last_event_step[ids] = -1
        # Controller.assign runs before episode_length_buf is zeroed. Mark the
        # current tick consumed so a late caller cannot score the old state twice.
        self._last_tick[ids] = int(self.env.common_step_counter)

    def ensure_step(self):
        """Update on real, nonempty episode transitions and return this monitor."""
        if not self.enabled:
            return self
        tick = int(self.env.common_step_counter)
        ready = (self._last_tick != tick) & (self.env.episode_length_buf > 0)
        ids = ready.nonzero(as_tuple=False).flatten()
        if not len(ids):
            return self

        from .curriculums import get_controller

        controller = get_controller(self.env)
        command = self.env.command_manager.get_command("twist")[ids]
        robot = self.env.scene["robot"].data
        linear_velocity = robot.root_link_lin_vel_b[ids, :2]
        angular_velocity = robot.root_link_ang_vel_b[ids, 2]

        event_step = controller.command_event_step[ids]
        event = controller.command_event[ids] & (event_step != self._last_event_step[ids])
        delta = controller.command_event_delta[ids]
        significant = event & (
            (delta.abs() >= self.change_thresholds).any(-1)
            | (controller.command_heading_delta[ids].abs() >= self.heading_change_threshold)
        )
        restart = significant | ~self._seen[ids]
        restarted_ids = ids[restart]
        self._clear_window(restarted_ids)
        self.grace_remaining_steps[restarted_ids] = self.grace_steps
        self.significant_events[ids] += significant.long()
        self._last_event_step[ids[event]] = event_step[event]

        speed = command[:, :2].norm(dim=-1)
        linear_active = speed >= self.linear_command_threshold
        yaw_active = command[:, 2].abs() >= self.angular_command_threshold
        # Project each sample before smoothing: body axes may rotate during a
        # window, and averaging old body-frame vectors mixes different frames.
        projected = (linear_velocity * command[:, :2]).sum(-1) / speed.clamp_min(1e-6)
        yaw_projected = angular_velocity * command[:, 2].sign()
        sample = torch.stack((projected, yaw_projected), dim=-1)
        cursor = self._cursor[ids]
        self._history_sum[ids] -= self._history[cursor, ids]
        self._history[cursor, ids] = sample
        self._history_sum[ids] += sample
        self._history_count[ids] = (self._history_count[ids] + 1).clamp_max(self.window_steps)
        self._cursor[ids] = (cursor + 1) % self.window_steps
        progress = self._history_sum[ids] / self._history_count[ids, None].clamp_min(1)
        self.linear_progress[ids], self.yaw_progress[ids] = progress.unbind(-1)

        in_grace = self.grace_remaining_steps[ids] > 0
        self.grace_active[ids] = in_grace
        eligible = ~in_grace & (self._history_count[ids] >= self.window_steps)
        self.grace_remaining_steps[ids] = (self.grace_remaining_steps[ids] - 1).clamp_min(0)
        linear_minimum = (self.progress_ratio * speed).clamp_min(self.linear_progress_floor)
        yaw_minimum = (self.progress_ratio * command[:, 2].abs()).clamp_min(self.angular_progress_floor)
        linear_bad = eligible & linear_active & (progress[:, 0] < linear_minimum)
        yaw_bad = eligible & yaw_active & (progress[:, 1] < yaw_minimum)
        self.linear_stalled[ids], self.yaw_stalled[ids] = linear_bad, yaw_bad
        self.linear_stall_run[ids] = torch.where(linear_bad, self.linear_stall_run[ids] + 1, 0)
        self.yaw_stall_run[ids] = torch.where(yaw_bad, self.yaw_stall_run[ids] + 1, 0)
        linear_penalty = self.linear_stall_run[ids] >= self.penalty_steps_required
        yaw_penalty = self.yaw_stall_run[ids] >= self.penalty_steps_required
        linear_cost = (1 - progress[:, 0] / linear_minimum).clamp(0, 1) * linear_penalty
        yaw_cost = (1 - progress[:, 1] / yaw_minimum).clamp(0, 1) * yaw_penalty
        self.cost[ids] = torch.maximum(linear_cost, yaw_cost)
        self.terminated[ids] = self.termination_enabled & (
            (self.linear_stall_run[ids] >= self.termination_steps_required)
            | (self.yaw_stall_run[ids] >= self.termination_steps_required)
        )

        self.steps[ids] += 1
        self.linear_active_steps[ids] += linear_active.long()
        self.yaw_active_steps[ids] += yaw_active.long()
        self.grace_observed_steps[ids] += in_grace.long()
        self.linear_stall_steps[ids] += linear_bad.long()
        self.yaw_stall_steps[ids] += yaw_bad.long()
        self.penalty_steps[ids] += (self.cost[ids] > 0).long()
        self.cost_sum[ids] += self.cost[ids]
        self.max_linear_stall_steps[ids] = torch.maximum(self.max_linear_stall_steps[ids], self.linear_stall_run[ids])
        self.max_yaw_stall_steps[ids] = torch.maximum(self.max_yaw_stall_steps[ids], self.yaw_stall_run[ids])
        self._seen[ids] = True
        self._last_tick[ids] = tick
        return self

    def episode_summary(self, env_ids=None):
        """Copy per-episode tensors before reset; this does not advance clocks."""
        ids = self._ids(env_ids)
        result = {
            name: getattr(self, name)[ids].clone()
            for name in (
                "steps", "linear_active_steps", "yaw_active_steps", "grace_observed_steps",
                "linear_stall_steps", "yaw_stall_steps", "penalty_steps", "significant_events",
                "cost_sum", "cost", "terminated",
            )
        }
        result["max_linear_stall_s"] = self.max_linear_stall_steps[ids] * self.dt
        result["max_yaw_stall_s"] = self.max_yaw_stall_steps[ids] * self.dt
        return result

    def summary(self):
        """Small current-batch diagnostics suitable for scalar logging."""
        values = {
            "cost_mean": self.cost.mean(),
            "linear_stalled_fraction": self.linear_stalled.float().mean(),
            "yaw_stalled_fraction": self.yaw_stalled.float().mean(),
            "termination_fraction": self.terminated.float().mean(),
            "observed_steps": self.steps.sum().float(),
            "linear_stall_steps": self.linear_stall_steps.sum().float(),
            "yaw_stall_steps": self.yaw_stall_steps.sum().float(),
            "penalty_steps": self.penalty_steps.sum().float(),
        }
        return dict(zip(values, torch.stack(tuple(values.values())).detach().cpu().tolist()))


def get_behavior_monitor(env):
    """Lazily attach ordinary mutable monitor buffers to the task controller."""
    from .curriculums import get_controller

    controller = get_controller(env)
    monitor = getattr(controller, "_behavior_monitor", None)
    if monitor is None:
        monitor = BehaviorMonitor(env, getattr(controller.cfg, "behavior_monitor", None))
        controller._behavior_monitor = monitor
    return monitor
