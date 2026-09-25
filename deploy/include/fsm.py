from __future__ import annotations

import enum
import math

import numpy as np

from .config import SimConfig
from .math_utils import rotate_inverse
from .policy import OnnxPolicy
from .types import ControlMode, JointCommand, RobotBackend, RobotState
from .inputs import CommandBus


class State(enum.Enum):
    DAMPING = "DAMPING"
    PASSIVE = "PASSIVE"
    GETUP = "GETUP"
    STAND_SETTLE = "STAND_SETTLE"
    READY = "READY"
    STAND = "STAND"
    RL = "RL"
    GETDOWN = "GETDOWN"
    FAULT = "FAULT"


class LocomotionFSM:
    """Measured-state lifecycle controller shared by simulation and real backends."""

    def __init__(self, cfg: SimConfig, backend: RobotBackend, policy: OnnxPolicy, commands: CommandBus):
        self.cfg, self.backend, self.policy, self.commands = cfg, backend, policy, commands
        self.state = State.PASSIVE
        self.state_time = 0.0
        self.step_index = 0
        self.command = np.zeros(cfg.command_dim, dtype=np.float64)
        self.last_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self.target = cfg.lying_joint.copy()
        self._transition_start: RobotState | None = None
        self._trajectory_destination = self.target.copy()
        self._stable_time = 0.0
        self._observation_history: list[np.ndarray] = []
        self._rl_phase_time = 0.0
        self._rl_step_index = 0
        self._auto_stand_armed = bool(cfg.auto_stand)
        # Auto-enable is a startup-only continuation of auto-stand.  Keep it
        # separate from ``_auto_stand_armed`` so leaving RL for STAND cannot
        # immediately re-enter policy control on the next FSM tick.
        self._auto_enable_armed = bool(cfg.auto_stand and cfg.auto_enable_rl)
        self._last_state_time: float | None = None
        self._last_sequence: int | None = None
        self._stale_elapsed = 0.0
        self._fault_reason: str | None = None
        self._heading_ref: float | None = None
        self._set_backend_mode("DAMPING")

    def _set_backend_mode(self, mode_name: str) -> None:
        method = getattr(self.backend, "set_control_mode", None) or getattr(self.backend, "set_mode", None)
        if method is None:
            return
        # STAND and RL are high-level states; both use the backend's explicit
        # low-level position mode.
        low_level_name = "POSITION" if mode_name in ("STAND", "RL") else mode_name
        value: object = getattr(ControlMode, low_level_name, low_level_name)
        try:
            method(value)
        except Exception:
            try:
                method(low_level_name.lower())
            except Exception:
                pass

    def _tilt(self, state: RobotState) -> float:
        gravity = rotate_inverse(state.quaternion, np.array([0.0, 0.0, -1.0]))
        return math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))

    def fallen(self, state: RobotState) -> bool:
        try:
            return bool(state.position[2] < self.cfg.fall_height or self._tilt(state) > self.cfg.fall_tilt)
        except (IndexError, ValueError, FloatingPointError):
            return True

    def _apply_heading_hold(
        self, state: RobotState, operator_command: np.ndarray
    ) -> np.ndarray:
        """Hold the current yaw while the operator commands forward motion."""

        command = np.asarray(operator_command, dtype=np.float64).copy()
        if self.state is not State.RL:
            self._heading_ref = None
            return command

        vx = float(command[0])
        operator_wz = float(command[2])
        if abs(operator_wz) > 1.0e-4:
            self._heading_ref = None
            return command

        min_forward_speed = 0.10 if self._heading_ref is not None else 0.15
        if vx < min_forward_speed:
            self._heading_ref = None
            return command

        quaternion = np.asarray(state.quaternion, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            self._heading_ref = None
            return command
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(norm) or norm <= 1.0e-9:
            self._heading_ref = None
            return command

        w, x, y, z = quaternion / norm
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        if not np.isfinite(yaw):
            self._heading_ref = None
            return command

        if self._heading_ref is None:
            self._heading_ref = yaw
            command[2] = 0.0
            return command

        error = math.atan2(
            math.sin(self._heading_ref - yaw),
            math.cos(self._heading_ref - yaw),
        )
        yaw_limit = min(0.4, float(self.cfg.command_limit[2]))
        command[2] = float(np.clip(error, -yaw_limit, yaw_limit))
        return command

    @staticmethod
    def _snapshot(state: RobotState) -> RobotState:
        return RobotState(state.time, state.position.copy(), state.quaternion.copy(), state.joint_position.copy(), state.joint_velocity.copy(), state.angular_velocity_body.copy())

    def _request_stand(self, state: RobotState) -> None:
        # A latched FAULT may only be cleared by reset; do not attempt to
        # re-enable a real actuator from an emergency-stop state.
        if self.state is not State.PASSIVE:
            return
        self._auto_stand_armed = False
        self.state, self.state_time = State.GETUP, 0.0
        self._transition_start = self._snapshot(state)
        self._trajectory_destination = self.cfg.default_joint.copy()
        self.target = state.joint_position.copy()
        self._stable_time = 0.0
        self._set_backend_mode("POSITION")

    def _request_passive(self, state: RobotState) -> None:
        self._heading_ref = None
        if self.state in (State.PASSIVE, State.DAMPING, State.GETDOWN, State.FAULT):
            return
        self.state, self.state_time = State.GETDOWN, 0.0
        self._transition_start = self._snapshot(state)
        self._trajectory_destination = self.cfg.lying_joint.copy()
        self.target = state.joint_position.copy()
        self.last_action.fill(0.0)
        self._set_backend_mode("POSITION")

    def _enter_passive(self) -> None:
        self._heading_ref = None
        self.state, self.state_time = State.PASSIVE, 0.0
        self._transition_start = None
        self._trajectory_destination = self.cfg.lying_joint.copy()
        self.target = self.cfg.lying_joint.copy()
        self.last_action.fill(0.0)
        self.command.fill(0.0)
        self._observation_history.clear()
        self._rl_phase_time = 0.0
        self._rl_step_index = 0
        self._stable_time = 0.0
        self._set_backend_mode("DAMPING")
        self._auto_stand_armed = False
        self._auto_enable_armed = False

    def _enter_fault(self, reason: str, state: RobotState | None = None) -> None:
        self._heading_ref = None
        if self.state == State.FAULT:
            return
        self.state, self.state_time = State.FAULT, 0.0
        self._fault_reason = reason
        self._transition_start = None
        self.last_action.fill(0.0)
        self.command.fill(0.0)
        self._observation_history.clear()
        self._rl_phase_time = 0.0
        self._rl_step_index = 0
        self.target = state.joint_position.copy() if state is not None and np.all(np.isfinite(state.joint_position)) else self.cfg.lying_joint.copy()
        emergency_stop = getattr(self.backend, "emergency_stop", None)
        if emergency_stop is not None:
            try:
                emergency_stop(reason)
            except Exception:
                self._set_backend_mode("EMERGENCY_STOP")
        else:
            self._set_backend_mode("EMERGENCY_STOP")
        self.commands.stop()
        print(f"[WARN] FSM FAULT: {reason}")

    def _enter_rl(self) -> int:
        """Enter policy control with no velocity inherited from another mode."""

        self._heading_ref = None
        stop_generation = self.commands.begin_stop()
        self.command = self.commands.command()
        self._auto_enable_armed = False
        self.state, self.state_time = State.RL, 0.0
        self._rl_phase_time = 0.0
        self._rl_step_index = 0
        self._observation_history.clear()
        self._set_backend_mode("RL")
        return stop_generation

    def _leave_rl(self) -> None:
        """Return to a zero-velocity standing hold."""

        self._heading_ref = None
        self.commands.stop()
        self.command = self.commands.command()
        self._auto_enable_armed = False
        self.state, self.state_time = State.STAND, 0.0
        self.last_action.fill(0.0)
        self.target = self.cfg.default_joint.copy()
        self._set_backend_mode("STAND")

    def _state_stale(self, state: RobotState) -> bool:
        if not bool(getattr(state, "valid", True)) or not bool(getattr(state, "connected", True)):
            return True
        sequence = getattr(state, "sequence", None)
        monotonic = getattr(state, "monotonic_time", None)
        timestamp = float(state.time if monotonic is None else monotonic)
        if not np.isfinite(timestamp):
            return True
        dt = float(getattr(self.backend, "timestep", self.cfg.timestep))
        progressed = (
            self._last_sequence is None
            or sequence is None
            or int(sequence) > self._last_sequence
        ) if sequence is not None else (
            self._last_state_time is None or timestamp > self._last_state_time + 1e-9
        )
        if progressed:
            self._stale_elapsed = 0.0
        else:
            self._stale_elapsed += max(dt, 0.0)
        self._last_state_time = timestamp
        self._last_sequence = None if sequence is None else int(sequence)
        return self._stale_elapsed > self.cfg.stale_timeout

    def _write_command(self) -> None:
        writer = getattr(self.backend, "write_joint_command", None)
        if writer is None:
            self.backend.set_joint_target(self.target)
            return
        if self.state == State.FAULT:
            return
        if self.state in (State.PASSIVE, State.DAMPING):
            writer(JointCommand(kp=0.0, kd=self.cfg.kd))
        else:
            writer(
                JointCommand(
                    position=self.target,
                    velocity=np.zeros(self.cfg.action_dim, dtype=np.float64),
                    kp=self.cfg.kp,
                    kd=self.cfg.kd,
                )
            )

    def _stable_for_stand(self, state: RobotState) -> bool:
        if not (np.all(np.isfinite(state.position)) and np.all(np.isfinite(state.quaternion))):
            return False
        return bool(
            abs(float(state.position[2]) - float(self.cfg.standing_pos[2])) <= self.cfg.settle_height_tolerance
            and self._tilt(state) <= self.cfg.settle_tilt_tolerance
            and np.max(np.abs(state.joint_position - self.cfg.default_joint)) <= self.cfg.settle_joint_tolerance
            and np.max(np.abs(state.joint_velocity)) <= self.cfg.settle_velocity_tolerance
        )

    def _advance_joint_trajectory(self, speed: float) -> bool:
        duration = max(float(self.cfg.stand_duration if self.state == State.GETUP else self.cfg.getdown_duration), 1e-3)
        amount = min(1.0, self.state_time / duration)
        origin = self._transition_start.joint_position if self._transition_start is not None else self.target
        desired = origin + amount * (self._trajectory_destination - origin)
        max_delta = max(float(speed), 1e-6) * float(getattr(self.backend, "timestep", self.cfg.timestep))
        self.target = self.target + np.clip(desired - self.target, -max_delta, max_delta)
        if amount >= 1.0 and np.max(np.abs(self.target - self._trajectory_destination)) <= max_delta + 1e-6:
            self.target = self._trajectory_destination.copy()
            return True
        return False

    def _lifecycle_timeout(self, name: str, fallback: float) -> float:
        """Return a positive lifecycle timeout with legacy-config fallback.

        A few integrations construct a light-weight config object instead of
        calling :func:`load_config`.  Keeping the fallback here preserves that
        pre-timeout interface while making the deadline available to the FSM.
        Invalid values are treated as absent; validated YAML values still pass
        through unchanged.
        """
        try:
            value = float(getattr(self.cfg, name))
        except (AttributeError, TypeError, ValueError):
            value = float(fallback)
        if not np.isfinite(value) or value <= 0.0:
            value = float(fallback)
        return value

    def observation(self, state: RobotState) -> np.ndarray:
        gravity_body = rotate_inverse(state.quaternion, np.array([0.0, 0.0, -1.0]))
        phase = np.zeros(2, dtype=np.float64)
        # Match the training task and real-robot observation builder.
        if np.linalg.norm(self.command[:2]) + abs(self.command[2]) > 0.1:
            period = max(float(self.cfg.phase_period), 1e-6)
            angle = 2 * math.pi * ((self._rl_phase_time % period) / period)
            phase[:] = [math.sin(angle), math.cos(angle)]
        values = {"base_ang_vel": state.angular_velocity_body, "projected_gravity": gravity_body, "command": self.command, "phase": phase, "joint_pos_rel": state.joint_position - self.cfg.default_joint, "joint_vel": state.joint_velocity, "last_action": self.last_action}
        try:
            frame = np.concatenate([values[name] for name in self.cfg.observation_terms])
        except KeyError as exc:
            raise ValueError(f"Unsupported observation term in config: {exc.args[0]}") from exc
        if self.cfg.observation_dim % self.cfg.history_size != 0:
            raise RuntimeError("observation_dim must be divisible by history_size")
        expected_frame_dim = self.cfg.observation_dim // self.cfg.history_size
        if frame.shape != (expected_frame_dim,):
            raise RuntimeError(f"Observation frame must have {expected_frame_dim} values, got {frame.shape[0]}")
        if not self._observation_history:
            self._observation_history = [frame.copy() for _ in range(self.cfg.history_size)]
        else:
            self._observation_history.append(frame.copy())
            self._observation_history = self._observation_history[-self.cfg.history_size:]
        history = reversed(self._observation_history) if self.cfg.history_order == "current-first" else iter(self._observation_history)
        observation = np.concatenate(tuple(history))
        if observation.shape != (self.cfg.observation_dim,) or not np.all(np.isfinite(observation)):
            raise RuntimeError("Invalid policy observation shape or non-finite value")
        return observation

    def step(self) -> None:
        state = self.backend.read_state()
        dt = float(getattr(self.backend, "timestep", self.cfg.timestep))
        self.command = self.commands.command()
        events, stop_generation = self.commands.consume_many(
            "zero_velocity",
            "estop",
            "reset",
            "get_down",
            "disable_rl",
            "stand",
            "enable_rl",
        )
        estop_requested = events["estop"]
        reset_requested = events["reset"]
        passive_requested = events["get_down"]
        disable_rl = events["disable_rl"]
        stand_requested = events["stand"]
        enable_rl = events["enable_rl"]
        safety_event = estop_requested or reset_requested or passive_requested or disable_rl
        transition_stop_generation: int | None = None

        # Global safety priority is strict and independent of the current FSM
        # state: estop > reset > get-down > disable-RL > stand/enable/auto.
        if estop_requested:
            self._enter_fault("operator emergency stop", state)
        elif reset_requested:
            self.commands.stop(); self.backend.reset(); self._last_state_time = None; self._stale_elapsed = 0.0; self._fault_reason = None
            self._last_sequence = None
            self._enter_passive(); self._auto_stand_armed = bool(self.cfg.auto_stand); self._auto_enable_armed = bool(self.cfg.auto_stand and self.cfg.auto_enable_rl); self.step_index = 0; state = self.backend.read_state(); self.command = self.commands.command()
        elif passive_requested:
            self.commands.stop()
            if self.state not in (State.PASSIVE, State.FAULT):
                self._request_passive(state)
        elif disable_rl:
            self.commands.stop()
            if self.state is State.RL:
                self._leave_rl()
        elif stand_requested:
            # A manual stand is deliberately a two-step operator flow: first
            # reach STAND, then wait for an explicit enable_rl request.
            self._auto_enable_armed = False
            self._request_stand(state)
        elif self.cfg.auto_stand and self._auto_stand_armed and self.state == State.PASSIVE and self.state_time > 0.2:
            self._request_stand(state)

        # A simultaneous reset+disable/get-down keeps reset's backend recovery
        # semantics while preserving the safer lower-priority cancellation.
        if disable_rl or passive_requested:
            self._auto_enable_armed = False
        if passive_requested:
            self._auto_stand_armed = False
        operator_command = self.commands.command()
        self.command = self._apply_heading_hold(state, operator_command)

        stale = self._state_stale(state)
        finite = bool(np.all(np.isfinite(state.joint_position)) and np.all(np.isfinite(state.joint_velocity)))
        if self.state not in (State.PASSIVE, State.DAMPING, State.FAULT) and (stale or not finite):
            self._enter_fault("stale or non-finite backend state", state if finite else None)
        self.state_time += dt
        if self.state in (State.PASSIVE, State.DAMPING):
            self.target = self.cfg.lying_joint.copy(); self.last_action.fill(0.0)
        elif self.state == State.GETUP:
            timeout = self._lifecycle_timeout(
                "getup_timeout", float(getattr(self.cfg, "stand_duration", 3.0)) + 2.0
            )
            if self._advance_joint_trajectory(self.cfg.getup_joint_speed):
                self.state, self.state_time = State.STAND_SETTLE, 0.0; self._transition_start = None; self._stable_time = 0.0; self.target = self.cfg.default_joint.copy(); self._set_backend_mode("POSITION")
            elif self.state_time >= timeout:
                self._enter_fault(
                    f"GETUP timeout after {self.state_time:.3f}s "
                    f"(limit {timeout:.3f}s)",
                    state,
                )
            self.last_action.fill(0.0)
        elif self.state == State.STAND_SETTLE:
            timeout = self._lifecycle_timeout(
                "settle_timeout", max(3.0 * float(getattr(self.cfg, "settle_duration", 0.5)), 2.0)
            )
            self.target = self.cfg.default_joint.copy()
            if self.fallen(state): self._enter_fault("fall detected while settling", state)
            elif self._stable_for_stand(state):
                self._stable_time += dt
                if self._stable_time >= self.cfg.settle_duration: self.state, self.state_time = State.READY, 0.0
                elif self.state_time >= timeout:
                    self._enter_fault(
                        f"STAND_SETTLE timeout after {self.state_time:.3f}s "
                        f"(limit {timeout:.3f}s)",
                        state,
                    )
            else:
                self._stable_time = 0.0
                if self.state_time >= timeout:
                    self._enter_fault(
                        f"STAND_SETTLE timeout after {self.state_time:.3f}s "
                        f"(limit {timeout:.3f}s)",
                        state,
                    )
            self.last_action.fill(0.0)
        elif self.state == State.READY:
            self.target = self.cfg.default_joint.copy()
            if self.fallen(state): self._enter_fault("fall detected before ready", state)
            elif self._stable_for_stand(state): self.state, self.state_time = State.STAND, 0.0; self._set_backend_mode("STAND")
            else: self.state, self._stable_time = State.STAND_SETTLE, 0.0
            self.last_action.fill(0.0)
        elif self.state == State.STAND:
            self.target = self.cfg.default_joint.copy()
            if self.fallen(state): self._enter_fault("fall detected while standing", state)
            elif not safety_event and (enable_rl or self._auto_enable_armed):
                transition_stop_generation = self._enter_rl()
            self.last_action.fill(0.0)
        elif self.state == State.RL:
            if self.fallen(state): self._enter_fault("fall detected during RL", state)
            elif self._rl_step_index % max(1, int(round(self.cfg.control_dt / max(dt, 1e-9)))) == 0:
                try:
                    self.last_action = np.asarray(self.policy(self.observation(state)), dtype=np.float64)
                    if self.last_action.shape != (self.cfg.action_dim,) or not np.all(np.isfinite(self.last_action)): raise ValueError("policy produced invalid action")
                except Exception as exc: self._enter_fault(f"policy failure: {exc}", state)
                else:
                    self.target = self.cfg.default_joint + self.cfg.action_scale * self.last_action; self._rl_phase_time += float(self.cfg.control_dt)
            if self.state == State.RL: self._rl_step_index += 1
        elif self.state == State.GETDOWN:
            timeout = self._lifecycle_timeout(
                "getdown_timeout",
                float(getattr(self.cfg, "getdown_duration", getattr(self.cfg, "stand_duration", 3.0))) + 2.0,
            )
            if self._advance_joint_trajectory(self.cfg.getdown_joint_speed): self._enter_passive()
            elif self.state_time >= timeout:
                self._enter_fault(
                    f"GETDOWN timeout after {self.state_time:.3f}s "
                    f"(limit {timeout:.3f}s)",
                    state,
                )
            self.last_action.fill(0.0)
        elif self.state == State.FAULT:
            self.last_action.fill(0.0); self.target = np.where(np.isfinite(self.target), self.target, self.cfg.lying_joint)
        if not np.all(np.isfinite(self.target)): self._enter_fault("non-finite joint target", state if finite else None)
        self._write_command(); self.backend.step(); self.step_index += 1
        self.commands.acknowledge_stop(
            transition_stop_generation
            if transition_stop_generation is not None
            else stop_generation
        )
