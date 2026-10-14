"""Play-only scene selection and command overrides; no training state is restored.

GUI callbacks enqueue immutable requests. The simulation thread consumes them
before reset/observation computation, keeping terrain, pose and HIM history in
the same reset. A sampled scene retains its exact commands, signs and spawn yaw.
"""

from dataclasses import dataclass, replace
import math
from threading import Lock
import time

import numpy as np
import torch
from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(frozen=True)
class PlaySelection:
    column: int = 0
    replica: int = 0
    yaw: float = 0.0
    command: tuple[float, float, float] = (0.0, 0.0, 0.0)
    heading_hold: bool = True
    mode: str = "manual"
    resample_on_reset: bool = False


@dataclass(frozen=True)
class SampledScene:
    task: int
    group: int
    column: int
    replica: int
    yaw: float
    sequence: np.ndarray  # Third axis is an offset in heading-mode catalogs.


class PlayController:
    def __init__(self, controller):
        if controller.mode != "play":
            raise ValueError("Interactive control is only available in play mode")
        self.controller = controller
        n, device = controller.num_envs, controller.device
        self.selections = [PlaySelection() for _ in range(n)]
        self.scenes: list[SampledScene | None] = [None] * n
        self.checkpoint_probabilities = None
        self._lock = Lock()
        self._pending_resets = {}
        self._pending_commands = {}
        self._manual = torch.ones(n, dtype=torch.bool, device=device)
        self._raw_commands = torch.zeros(n, 3, device=device)
        self._hold_enabled = torch.ones_like(self._manual)
        self._holding = torch.zeros_like(self._manual)
        self._heading_reference = torch.zeros(n, device=device)
        self.status_callback = None
        self._last_status = 0.0
        # Manual commands may be between bins; the task ID only references the
        # selected geometry and must not be advertised as a sampled task.
        self._geometry_tasks = [
            int(np.flatnonzero(controller.terrain_map.task_columns == col)[0])
            for col in range(len(controller.terrain_map.columns))
        ]
        self.command_limits = tuple(
            max(abs(value) for b in controller.catalog.axis_bins[axis] for value in (b.low, b.high))
            for axis in ("vx", "vy")
        ) + (max(abs(x) for x in controller.cfg.task_space.ang_vel_z_range),)

    def set_checkpoint_distribution(self, state):
        """Accept a compatible grouped teacher, without loading its clocks/RNG."""
        self.checkpoint_probabilities = None
        if not state or tuple(state.get("teacher_groups", ())) != self.controller.groups.records:
            return
        p = np.asarray(state.get("probabilities"), dtype=np.float64)
        if (p.shape == (self.controller.groups.num_groups,) and np.isfinite(p).all()
                and (p >= 0).all() and np.isclose(p.sum(), 1.0)):
            self.checkpoint_probabilities = p.copy() / p.sum()

    def _validate_command(self, command):
        if (len(command) != 3 or not all(math.isfinite(x) for x in command)
                or any(abs(x) > limit + 1e-6 for x, limit in zip(command, self.command_limits))):
            raise ValueError("Play command is outside the configured velocity limits")

    def request_reset(self, index, selection=None):
        """None replays the current scene, even when resampling is enabled."""
        if not 0 <= index < len(self.selections):
            raise ValueError("Invalid play environment index")
        if selection is not None:
            c = self.controller
            if (selection.mode not in ("manual", "uniform", "checkpoint")
                    or not 0 <= selection.column < len(c.terrain_map.columns)
                    or not 0 <= selection.replica < c.terrain_map.replicas
                    or not math.isfinite(selection.yaw)):
                raise ValueError("Invalid play selection")
            if selection.mode == "checkpoint" and self.checkpoint_probabilities is None:
                raise ValueError("Checkpoint has no compatible teacher distribution")
            self._validate_command(selection.command)
        with self._lock:
            self._pending_resets[index] = selection

    def request_command(self, index, command, heading_hold=True):
        self._validate_command(command)
        if not 0 <= index < len(self.selections):
            raise ValueError("Invalid play environment index")
        with self._lock:
            self._pending_commands[index] = (tuple(command), bool(heading_hold))

    def _consume_commands(self, indices):
        with self._lock:
            updates = {i: self._pending_commands.pop(i) for i in indices if i in self._pending_commands}
        for i, (command, hold) in updates.items():
            if self.selections[i].mode != "manual":
                continue
            if hold != self.selections[i].heading_hold:
                self._holding[i] = False
            self.selections[i] = replace(self.selections[i], command=command, heading_hold=hold)
            self._raw_commands[i] = torch.tensor(command, device=self.controller.device)
            self._hold_enabled[i] = hold

    def _sample(self, selection):
        c = self.controller
        if selection.mode == "manual":
            return SampledScene(
                self._geometry_tasks[selection.column], -1, selection.column,
                selection.replica, selection.yaw,
                np.tile(selection.command, (c.command_count, 1)),
            )
        if selection.mode == "checkpoint":
            groups, tasks, sequences = c.groups.sample(
                self.checkpoint_probabilities, 1, c.rng, c.command_count)
            task, group, sequence = int(tasks[0]), int(groups[0]), sequences[0]
        else:
            task = int(c.rng.integers(c.catalog.num_tasks))
            sequence = c.catalog.sample_commands(np.full(c.command_count, task), c.rng)
            group = c.groups.task_direction_to_group.get((task, int(np.sign(sequence[0, 0]))), -1)
        return SampledScene(
            task, group, int(c.terrain_map.task_columns[task]),
            int(c.rng.integers(c.terrain_map.replicas)),
            float(c.rng.uniform(-math.pi, math.pi)), sequence.copy(),
        )

    def assign(self, ids):
        c = self.controller
        indices = ids.cpu().tolist()
        self._consume_commands(indices)
        with self._lock:
            requests = {i: self._pending_resets.pop(i) for i in indices if i in self._pending_resets}
        for i in indices:
            selection = requests.get(i)
            if selection is not None:
                self.selections[i] = selection
            selection = self.selections[i]
            if (selection.mode == "manual" or self.scenes[i] is None
                    or requests.get(i) is not None
                    or (i not in requests and selection.resample_on_reset)):
                self.scenes[i] = self._sample(selection)
            scene = self.scenes[i]
            c.episode_task_id[i] = scene.task
            c.episode_group_id[i] = scene.group
            c.episode_sampling_mode[i] = 2 if selection.mode == "manual" else 0
            c.episode_geometry_id[i] = scene.column * c.terrain_map.replicas + scene.replica
            c.yaws[i] = scene.yaw
            c.command_sequences[i] = torch.as_tensor(
                scene.sequence, dtype=c.commands.dtype, device=c.device)
            if c.catalog.heading_command:
                c.heading_sequences[i] = wrap_to_pi(c.yaws[i] + c.command_sequences[i, :, 2])
            c._heading_reference_pending[i] = False
            self._manual[i] = selection.mode == "manual"
            self._raw_commands[i] = torch.tensor(selection.command, device=c.device)
            self._hold_enabled[i] = selection.heading_hold
            self._heading_reference[i] = scene.yaw
            self._holding[i] = selection.heading_hold and any(abs(x) > 1e-4 for x in selection.command[:2])
            if not c.terrain_map.plane:
                terrain = c._env.scene.terrain
                terrain.terrain_levels[i] = scene.replica
                terrain.terrain_types[i] = scene.column
                terrain.env_origins[i] = c.terrain_map.origins[scene.replica, scene.column]
        self.update_commands(ids, reset=True)

    def update_commands(self, ids, *, reset=False):
        c = self.controller
        # Avoid a device synchronization on every step when the GUI is idle.
        with self._lock:
            pending = bool(self._pending_commands)
        if pending:
            self._consume_commands(ids.cpu().tolist())
        steps = torch.zeros_like(ids) if reset else c._env.episode_length_buf[ids]
        indices = (torch.zeros_like(ids) if c.command_interval_steps is None else
                   (steps // c.command_interval_steps).clamp_max(c.command_count - 1))
        c.commands[ids] = c.command_sequences[ids, indices]
        yaw = c.yaws[ids] if reset else c._env.scene["robot"].data.heading_w[ids]
        if c.catalog.heading_command:
            c.heading_targets[ids] = c.heading_sequences[ids, indices]
            c.heading_error[ids] = wrap_to_pi(c.heading_targets[ids] - yaw)
            c.commands[ids, 2] = torch.clamp(
                c.cfg.task_space.heading_control_stiffness * c.heading_error[ids],
                *c.cfg.task_space.ang_vel_z_range)
        manual_ids = ids[self._manual[ids]]
        raw = self._raw_commands[manual_ids]
        manual_yaw = yaw[self._manual[ids]]
        hold = (self._hold_enabled[manual_ids] & (raw[:, :2].abs().amax(dim=1) > 1e-4)
                & (raw[:, 2].abs() <= 1e-4) & torch.isfinite(manual_yaw))
        start = hold & ~self._holding[manual_ids]
        self._heading_reference[manual_ids] = torch.where(
            start, manual_yaw, self._heading_reference[manual_ids])
        self._holding[manual_ids] = hold
        error = wrap_to_pi(self._heading_reference[manual_ids] - manual_yaw)
        correction = torch.clamp(c.cfg.task_space.heading_control_stiffness * error,
                                 *c.cfg.task_space.ang_vel_z_range)
        c.commands[manual_ids] = raw
        c.commands[manual_ids, 2] = torch.where(hold, correction, raw[:, 2])
        c.heading_targets[manual_ids] = self._heading_reference[manual_ids]
        c.heading_error[manual_ids] = torch.where(hold, error, 0.0)
        now = time.monotonic()
        if self.status_callback is not None and now - self._last_status >= 0.2:
            self._last_status = now
            self.status_callback()
