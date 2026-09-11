from __future__ import annotations

import enum
import math

import numpy as np

from .config import SimConfig
from .math_utils import quaternion_slerp, rotate_inverse
from .policy import OnnxPolicy
from .types import RobotBackend, RobotState
from .inputs import CommandBus


class State(enum.Enum):
    PASSIVE = "PASSIVE"
    GETUP = "GETUP"
    RL = "RL"
    GETDOWN = "GETDOWN"


class LocomotionFSM:
    """Backend-independent safety and locomotion state machine.

    ``MujocoBackend`` and a future DDS/SDK backend expose the same small
    interface.  Only the simulation backend implements ``set_base_pose`` as a
    scripted get-up aid; a real backend can replace that phase with a measured
    whole-body stand controller without changing this FSM's public contract.
    """

    def __init__(self, cfg: SimConfig, backend: RobotBackend, policy: OnnxPolicy, commands: CommandBus):
        self.cfg = cfg
        self.backend = backend
        self.policy = policy
        self.commands = commands
        self.state = State.PASSIVE
        self.state_time = 0.0
        self.step_index = 0
        self.command = np.zeros(cfg.command_dim, dtype=np.float64)
        self.last_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self.target = cfg.lying_joint.copy()
        # A pose snapshot is captured whenever a scripted transition starts.
        # Keeping a snapshot (rather than a live backend state) makes an
        # interrupted transition reversible from the pose actually measured
        # at the time of the request.
        self._transition_start: RobotState | None = None

    def _tilt(self, state: RobotState) -> float:
        gravity = rotate_inverse(state.quaternion, np.array([0.0, 0.0, -1.0]))
        return math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))

    def fallen(self, state: RobotState) -> bool:
        return bool(state.position[2] < self.cfg.fall_height or self._tilt(state) > self.cfg.fall_tilt)

    def _request_stand(self, state: RobotState) -> None:
        if self.state not in (State.PASSIVE, State.GETDOWN):
            return
        self.state = State.GETUP
        self.state_time = 0.0
        self._transition_start = self._snapshot(state)

    def _request_passive(self, state: RobotState) -> None:
        if self.state not in (State.GETUP, State.RL):
            return
        self.state = State.GETDOWN
        self.state_time = 0.0
        self._transition_start = self._snapshot(state)
        self.last_action.fill(0.0)

    @staticmethod
    def _snapshot(state: RobotState) -> RobotState:
        """Copy the measured pose used as the beginning of a transition."""

        return RobotState(
            time=state.time,
            position=state.position.copy(),
            quaternion=state.quaternion.copy(),
            joint_position=state.joint_position.copy(),
            joint_velocity=state.joint_velocity.copy(),
            angular_velocity_body=state.angular_velocity_body.copy(),
        )

    def _enter_passive(self) -> None:
        self.state = State.PASSIVE
        self.state_time = 0.0
        self._transition_start = None
        self.last_action.fill(0.0)
        self.target = self.cfg.lying_joint.copy()

    def observation(self, state: RobotState) -> np.ndarray:
        gravity_body = rotate_inverse(state.quaternion, np.array([0.0, 0.0, -1.0]))
        phase = np.zeros(2, dtype=np.float64)
        if np.linalg.norm(self.command) >= 0.1:
            angle = 2 * math.pi * ((state.time % self.cfg.phase_period) / self.cfg.phase_period)
            phase[:] = [math.sin(angle), math.cos(angle)]
        values = {
            "base_ang_vel": state.angular_velocity_body,
            "projected_gravity": gravity_body,
            "command": self.command,
            "phase": phase,
            "joint_pos_rel": state.joint_position - self.cfg.default_joint,
            "joint_vel": state.joint_velocity,
            "last_action": self.last_action,
        }
        try:
            observation = np.concatenate([values[name] for name in self.cfg.observation_terms])
        except KeyError as exc:
            raise ValueError(f"Unsupported observation term in config: {exc.args[0]}") from exc
        if observation.shape != (self.cfg.observation_dim,):
            raise RuntimeError(
                f"Configured observation_dim={self.cfg.observation_dim}, "
                f"but terms produce {observation.shape[0]} values"
            )
        return observation

    def step(self) -> None:
        state = self.backend.read_state()
        self.command = self.commands.command()
        if self.commands.consume("reset"):
            self.commands.stop()
            self.backend.reset()
            self._enter_passive()
            self.step_index = 0
            state = self.backend.read_state()
            self.command = self.commands.command()
        else:
            if self.commands.consume("passive"):
                self.commands.stop()
                self._request_passive(state)
            if self.commands.consume("stand"):
                self._request_stand(state)
            if self.cfg.auto_stand and self.state == State.PASSIVE and state.time > 0.2:
                self._request_stand(state)

        self.state_time += self.backend.timestep
        if self.state == State.PASSIVE:
            self.target = state.joint_position.copy()
            self.last_action.fill(0.0)
        elif self.state == State.GETUP:
            assert self._transition_start is not None
            amount = min(1.0, self.state_time / max(self.cfg.stand_duration, 1e-3))
            position = (1 - amount) * self._transition_start.position + amount * self.cfg.standing_pos
            quaternion = quaternion_slerp(self._transition_start.quaternion, self.cfg.standing_quat, amount)
            self.target = (1 - amount) * self._transition_start.joint_position + amount * self.cfg.default_joint
            self.backend.set_base_pose(position, quaternion)
            self.last_action.fill(0.0)
            if amount >= 1.0:
                self.state = State.RL
                self.state_time = 0.0
                self._transition_start = None
        elif self.state == State.GETDOWN:
            assert self._transition_start is not None
            getdown_duration = getattr(self.cfg, "getdown_duration", getattr(self.cfg, "stand_duration", 3.0))
            amount = min(1.0, self.state_time / max(getdown_duration, 1e-3))
            position = (1 - amount) * self._transition_start.position + amount * self.cfg.lying_pos
            quaternion = quaternion_slerp(self._transition_start.quaternion, self.cfg.lying_quat, amount)
            self.target = (1 - amount) * self._transition_start.joint_position + amount * self.cfg.lying_joint
            self.backend.set_base_pose(position, quaternion)
            self.last_action.fill(0.0)
            if amount >= 1.0:
                self._enter_passive()
        else:
            if not getattr(self.backend, "finite", lambda: True)():
                # A non-finite backend cannot provide a valid interpolation
                # start pose. Keep the old safety fallback without using
                # height/tilt as an automatic state transition trigger.
                self._enter_passive()
            elif self.step_index % self.cfg.decimation == 0:
                try:
                    self.last_action = self.policy(self.observation(state))
                except Exception as exc:
                    print(f"[WARN] policy failure, entering PASSIVE: {exc}")
                    self._request_passive(state)
                self.target = self.cfg.default_joint + self.cfg.action_scale * self.last_action
        self.backend.set_joint_target(self.target)
        self.backend.step()
        self.step_index += 1
