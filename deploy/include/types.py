from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np


class ControlMode(str, enum.Enum):
    """Backend-independent low-level control modes.

    The modes intentionally mirror the small contract a real robot adapter is
    expected to expose.  In particular, ``EMERGENCY_STOP`` is a latched,
    torque-off state and is not another pose target.
    """

    DAMPING = "damping"
    POSITION = "position"
    EMERGENCY_STOP = "emergency_stop"
    DISABLED = "disabled"


Gain: TypeAlias = float | np.ndarray


@dataclass
class JointCommand:
    """One low-level command in configured joint order.

    Position control implements the usual joint-space law
    ``tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff``.  Gains may be a
    scalar or one value per joint.  Optional fields let damping-only backends
    use the same message without inventing a position target.
    """

    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    kp: Gain | None = None
    kd: Gain | None = None
    feedforward_effort: np.ndarray | None = None

    @staticmethod
    def _vector(value: object, joint_count: int, name: str, *, scalar: bool = False) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if scalar and array.ndim == 0:
            array = np.full(joint_count, float(array), dtype=np.float64)
        if array.shape != (joint_count,):
            raise ValueError(f"JointCommand.{name} must have shape ({joint_count},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"JointCommand.{name} must contain only finite values")
        return array.copy()

    def validate(self, joint_count: int, *, require_position: bool = False) -> JointCommand:
        """Return a normalized copy, rejecting malformed or unsafe values."""

        if not isinstance(joint_count, int) or isinstance(joint_count, bool) or joint_count <= 0:
            raise ValueError("joint_count must be a positive integer")
        if require_position and self.position is None:
            raise ValueError("JointCommand.position is required in position mode")
        position = None if self.position is None else self._vector(self.position, joint_count, "position")
        velocity = None if self.velocity is None else self._vector(self.velocity, joint_count, "velocity")
        kp = None if self.kp is None else self._vector(self.kp, joint_count, "kp", scalar=True)
        kd = None if self.kd is None else self._vector(self.kd, joint_count, "kd", scalar=True)
        effort = (
            None
            if self.feedforward_effort is None
            else self._vector(self.feedforward_effort, joint_count, "feedforward_effort")
        )
        if kp is not None and np.any(kp < 0.0):
            raise ValueError("JointCommand.kp must be non-negative")
        if kd is not None and np.any(kd < 0.0):
            raise ValueError("JointCommand.kd must be non-negative")
        return JointCommand(position, velocity, kp, kd, effort)


@dataclass
class RobotState:
    """State common to a simulated or real robot backend."""

    time: float
    position: np.ndarray
    quaternion: np.ndarray  # MuJoCo/Unitree convention: w, x, y, z.
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    angular_velocity_body: np.ndarray
    # Defaults preserve all pre-V5 positional and keyword construction sites.
    valid: bool = True
    connected: bool = True
    monotonic_time: float | None = None
    sequence: int | None = None


class RobotBackend(Protocol):
    timestep: float

    def read_state(self) -> RobotState: ...

    def set_control_mode(self, mode: ControlMode, damping_kd: Gain | None = None) -> None: ...

    def write_joint_command(self, command: JointCommand) -> None: ...

    def emergency_stop(self, reason: str | None = None) -> None: ...

    # Compatibility adapter for older FSMs.  New code should use an explicit
    # POSITION mode and ``write_joint_command``.
    def set_joint_target(self, target: np.ndarray) -> None: ...

    # Simulation-only reset hook.  Runtime state transitions must never use
    # this to teleport the base.
    def set_base_pose(self, position: np.ndarray, quaternion: np.ndarray) -> None: ...

    def step(self) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
