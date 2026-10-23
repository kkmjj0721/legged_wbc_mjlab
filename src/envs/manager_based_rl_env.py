"""Reset control isolation for the project's mjlab 1.6.0 environments.

mjlab writes controls once after both explicit and automatic resets. Its normal
scene writer appends to every actuator delay queue and advances every lag clock,
even for worlds that did not reset. Route just that write to a scoped refresh;
the physics-substep writer and the upstream reset/step ordering stay intact.

This adapter uses mjlab 1.6.0's builtin actuator groups (the actuators used by
this repository). Keep the dependency pinned until its reset writer accepts
environment indices. Custom/stateful control laws need their own scoped refresh
implementation and are rejected instead of silently advancing their state.
"""

from __future__ import annotations

import torch

from mjlab.actuator.builtin_group import _TARGET_TENSOR_MAP
from mjlab.envs import ManagerBasedRlEnv as MjlabManagerBasedRlEnv
from mjlab.utils.buffers import DelayBuffer

from .reproducibility import configure_simulation, validate_simulation


def _backfill_reset_targets(
    delay: DelayBuffer, targets: torch.Tensor, env_ids: torch.Tensor
) -> None:
    """Seed reset rows' history without sampling lags or advancing their clocks."""
    if not delay.is_initialized:
        # CircularBuffer.backfill requires allocated storage. At this point no
        # world has any history: allocate it, then clear the allocation frame's
        # push counts. Do not call DelayBuffer.reset(), which resamples phases.
        delay.append(torch.zeros_like(targets))
        delay._buffer.reset()
    delay.backfill(targets, env_ids)


def _validate_reset_actuators(scene) -> None:
    for name, entity in scene.entities.items():
        if entity._custom_actuators or entity._fused_actuator_group._groups:
            raise NotImplementedError(
                f"Reset control isolation for entity {name!r} requires builtin "
                "actuators; add a scoped refresh for its custom control law."
            )


def _refresh_reset_controls(scene, env_ids: torch.Tensor) -> None:
    """Write only reset worlds' targets; retain every other world's controls.

    Entity.reset has already cleared these rows' histories/lags. Backfilling
    their current targets gives the next physics tick a valid reset frame.
    There is no time advance or lag resampling during this refresh.
    """
    if env_ids.numel() == 0:
        return
    for entity in scene.entities.values():
        group = entity._builtin_group
        data = entity.data
        for key, (target_ids, ctrl_ids) in group._index_groups.items():
            targets = getattr(data, _TARGET_TENSOR_MAP[key])
            data.write_ctrl(targets[env_ids[:, None], target_ids], ctrl_ids, env_ids)
        # Each fused delay group is handled once, even if several actuators
        # alias its buffer (e.g. RI-4438 hip/thigh/calf).
        for delayed in group._delayed_groups:
            assert delayed.delay_buffer is not None
            targets = getattr(data, delayed.target_attr)[:, delayed.target_ids]
            _backfill_reset_targets(delayed.delay_buffer, targets, env_ids)
            data.write_ctrl(targets[env_ids], delayed.ctrl_ids, env_ids)


class ManagerBasedRlEnv(MjlabManagerBasedRlEnv):
    """mjlab environment whose reset writes cannot tick other actuator queues."""

    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("cfg", args[0] if args else None)
        device = kwargs.get("device", args[1] if len(args) > 1 else "cuda:0")
        configure_simulation(cfg.sim, device)
        self._pending_control_reset: torch.Tensor | None = None
        super().__init__(*args, **kwargs)
        try:
            _validate_reset_actuators(self.scene)
        except Exception:
            self.close()
            raise
        self._write_all_controls = self.scene.write_data_to_sim
        # Instance-local hook: upstream reset() and step() both call this
        # immediately after _reset_idx(). No global mjlab monkey patch and no
        # copy of its stepping loop is needed.
        self.scene.write_data_to_sim = self._write_controls
        validate_simulation(self.sim)

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        validate_simulation(self.sim)
        return result

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        validate_simulation(self.sim)
        return result

    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        super()._reset_idx(env_ids)
        self._pending_control_reset = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None else env_ids
        )

    def _write_controls(self) -> None:
        env_ids = self._pending_control_reset
        if env_ids is None:
            self._write_all_controls()
            return
        self._pending_control_reset = None
        _refresh_reset_controls(self.scene, env_ids)
