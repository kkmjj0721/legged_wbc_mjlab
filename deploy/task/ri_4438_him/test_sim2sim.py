"""Regression tests for the RI-4438 HIM deployment stack.

The tests intentionally keep the FSM checks independent of MuJoCo.  The
MuJoCo and ONNX checks are marked as optional because ``deploy`` can be used
for a backend-only smoke test without installing either runtime dependency.
Run from the repository root with ``python -m unittest deploy.task.ri_4438_him.test_sim2sim``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.include.config import load_config
from deploy.include.fsm import LocomotionFSM, State
from deploy.include.inputs import (
    AxisCalibrator,
    CommandBus,
    GamepadAxisProcessor,
    JoystickInput,
    KeyboardInput,
)
from deploy.include.policy import OnnxPolicy
from deploy.include.types import RobotState


def _optional_mujoco():
    """Import MuJoCo without turning its optional dependency error into a test failure."""

    try:
        import mujoco  # type: ignore
    except BaseException:  # MuJoCo bridge raises SystemExit when unavailable.
        return None
    return mujoco


MUJOCO = _optional_mujoco()
try:
    import onnxruntime  # type: ignore  # noqa: F401

    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ONNXRUNTIME_AVAILABLE = False


CONFIG_PATH = ROOT / "deploy/task/ri_4438_him/config/sim2sim.yaml"


class _RecordingPolicy:
    """Deterministic policy double that records every observation it receives."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.observations: list[np.ndarray] = []

    @property
    def available(self) -> bool:
        return False

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation, dtype=np.float64).copy())
        return np.zeros(self.action_dim, dtype=np.float64)


class _FakeBackend:
    """Small in-memory RobotBackend implementation for FSM unit tests."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.timestep = cfg.timestep
        self._time = 0.0
        self.position = cfg.lying_pos.copy()
        self.quaternion = cfg.lying_quat.copy()
        self.joint_position = cfg.lying_joint.copy()
        self.joint_velocity = np.zeros(cfg.action_dim, dtype=np.float64)
        self.angular_velocity_body = np.zeros(3, dtype=np.float64)
        self.targets: list[np.ndarray] = []
        self.mode_commands: list[object] = []
        self.joint_commands: list[object] = []
        self.emergency_reasons: list[str | None] = []
        self.reset_count = 0

    def read_state(self) -> RobotState:
        return RobotState(
            time=self._time,
            position=self.position.copy(),
            quaternion=self.quaternion.copy(),
            joint_position=self.joint_position.copy(),
            joint_velocity=self.joint_velocity.copy(),
            angular_velocity_body=self.angular_velocity_body.copy(),
        )

    def set_control_mode(self, mode, damping_kd=None) -> None:
        del damping_kd
        self.mode_commands.append(mode)

    def write_joint_command(self, command) -> None:
        self.joint_commands.append(command)
        if command.position is None:
            self.joint_velocity.fill(0.0)
            return
        target = np.asarray(command.position, dtype=np.float64).copy()
        self.targets.append(target)
        self.joint_velocity = target - self.joint_position
        self.joint_position = target
        # Simulate a position-controlled base settling at the corresponding
        # scripted pose.  Crucially, there is no set_base_pose method: runtime
        # transitions must be achieved by low-level joint commands only.
        if np.max(np.abs(target - self.cfg.default_joint)) < 1.0e-6:
            self.position = self.cfg.standing_pos.copy()
            self.quaternion = self.cfg.standing_quat.copy()
        elif np.max(np.abs(target - self.cfg.lying_joint)) < 1.0e-6:
            self.position = self.cfg.lying_pos.copy()
            self.quaternion = self.cfg.lying_quat.copy()

    def set_joint_target(self, target: np.ndarray) -> None:
        # Legacy compatibility path used by a few bridge tests.
        from deploy.include.types import JointCommand
        self.write_joint_command(JointCommand(position=target, kp=self.cfg.kp, kd=self.cfg.kd))

    def emergency_stop(self, reason=None) -> None:
        self.emergency_reasons.append(reason)

    def step(self) -> None:
        self._time += self.timestep

    def reset(self) -> None:
        self.reset_count += 1
        self._time = 0.0
        self.position = self.cfg.lying_pos.copy()
        self.quaternion = self.cfg.lying_quat.copy()
        self.joint_position = self.cfg.lying_joint.copy()
        self.joint_velocity.fill(0.0)
        self.angular_velocity_body.fill(0.0)

    def close(self) -> None:
        return None

    def finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.position))
            and np.all(np.isfinite(self.quaternion))
            and np.all(np.isfinite(self.joint_position))
            and np.all(np.isfinite(self.joint_velocity))
        )


class Sim2SimTest(unittest.TestCase):
    def make_config(self):
        self.assertTrue(CONFIG_PATH.exists(), f"missing HIM config: {CONFIG_PATH}")
        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        # Keep transition tests quick while retaining the production values in
        # the YAML itself.
        cfg.stand_duration = min(float(cfg.stand_duration), 0.05)
        # New configs expose a separate getdown duration.  The fallback keeps
        # this test useful with older deploy checkouts during migration.
        if not hasattr(cfg, "getdown_duration"):
            cfg.getdown_duration = cfg.stand_duration
        else:
            cfg.getdown_duration = min(float(cfg.getdown_duration), 0.05)
        # Keep unit transitions short while preserving the production timing
        # in YAML.  The FSM applies a joint-speed safety limit in addition to
        # the nominal duration, so the fake backend uses a high speed to reach
        # each scripted endpoint deterministically.
        cfg.getup_joint_speed = 100.0
        cfg.getdown_joint_speed = 100.0
        cfg.settle_duration = min(float(cfg.settle_duration), 0.01)
        return cfg

    def make_stack(self, *, auto_stand: bool = False):
        cfg = self.make_config()
        cfg.auto_stand = auto_stand
        backend = _FakeBackend(cfg)
        commands = CommandBus(cfg.command_limit, cfg.command_deadzone)
        policy = _RecordingPolicy(cfg.action_dim)
        fsm = LocomotionFSM(cfg, backend, policy, commands)
        return cfg, backend, commands, fsm, policy

    def _advance(self, fsm: LocomotionFSM, backend: _FakeBackend, seconds: float) -> None:
        count = int(np.ceil(seconds / backend.timestep)) + 2
        for _ in range(count):
            fsm.step()

    def _enter_rl(self, cfg, backend, commands, fsm) -> None:
        """Drive GETUP -> STAND_SETTLE -> READY -> STAND -> RL."""
        commands.request("stand")
        fsm.step()
        self.assertEqual(fsm.state, State.GETUP)
        self._advance(fsm, backend, cfg.stand_duration)
        self.assertIn(fsm.state, (State.STAND_SETTLE, State.READY, State.STAND))
        self._advance(fsm, backend, cfg.settle_duration + 0.05)
        if fsm.state is State.READY:
            fsm.step()
        self.assertEqual(fsm.state, State.STAND)
        commands.request("enable_rl")
        fsm.step()
        self.assertEqual(fsm.state, State.RL)
        fsm.step()  # first policy invocation occurs on the next control tick

    @staticmethod
    def _force_safe_stand(cfg, backend, fsm) -> None:
        backend.position = cfg.standing_pos.copy()
        backend.quaternion = cfg.standing_quat.copy()
        backend.joint_position = cfg.default_joint.copy()
        backend.joint_velocity.fill(0.0)
        fsm.state = State.STAND
        fsm.state_time = 0.0

    def test_config_contract_and_policy_path(self) -> None:
        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        self.assertEqual(cfg.history_size, 6)
        self.assertEqual(cfg.history_order, "current-first")
        self.assertEqual(cfg.action_dim, 12)
        self.assertEqual(cfg.observation_dim, 282)
        self.assertEqual(cfg.command_dim, 3)
        self.assertEqual(len(cfg.joint_names), cfg.action_dim)
        self.assertEqual(cfg.observation_dim, cfg.history_size * 47)
        self.assertEqual(
            cfg.observation_terms,
            (
                "base_ang_vel",
                "projected_gravity",
                "command",
                "phase",
                "joint_pos_rel",
                "joint_vel",
                "last_action",
            ),
        )
        np.testing.assert_allclose(cfg.action_scale, [0.25, 0.5, 0.5] * 4)
        self.assertIsNone(cfg.action_clip)
        self.assertAlmostEqual(cfg.standing_pos[2], 0.27, places=6)
        self.assertTrue(cfg.auto_stand)
        self.assertIsNotNone(cfg.policy)
        self.assertEqual(cfg.policy.suffix.casefold(), ".onnx")
        self.assertTrue(cfg.policy.is_file(), f"configured HIM policy does not exist: {cfg.policy}")
        # Export run directories are intentionally timestamped.  Validate the
        # deploy contract instead of pinning this test to one dated training
        # run, so replacing a policy does not make an otherwise valid config
        # stale.
        self.assertGreater(cfg.policy.stat().st_size, 0)
        self.assertEqual(cfg.gamepad_options["axis_config"]["left_x"], "LEFTX")
        self.assertEqual(cfg.gamepad_options["button_config"]["enable_rl"], "X")
        self.assertAlmostEqual(cfg.gamepad_options["deadzone"], 0.18)
        self.assertAlmostEqual(cfg.gamepad_options["deadzone_exit"], 0.15)
        self.assertAlmostEqual(cfg.gamepad_options["calibration_center_limit"], 0.15)
        self.assertAlmostEqual(cfg.gamepad_options["calibration_release_margin"], 0.005)
        self.assertEqual(cfg.settle_timeout, max(3.0 * cfg.settle_duration, 2.0))
        self.assertEqual(cfg.getup_timeout, cfg.stand_duration + 2.0)
        self.assertEqual(cfg.getdown_timeout, cfg.getdown_duration + 2.0)

    def test_getup_timeout_latches_fault_and_emergency_stop(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack()
        cfg.getup_joint_speed = 1.0e-6
        cfg.getup_timeout = 2.0 * cfg.timestep
        commands.request("stand")

        fsm.step()
        self.assertEqual(fsm.state, State.GETUP)
        fsm.step()

        self.assertEqual(fsm.state, State.FAULT)
        self.assertTrue(backend.emergency_reasons)
        self.assertIn("GETUP timeout", backend.emergency_reasons[-1])

    def test_stand_settle_timeout_cannot_enter_rl(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        cfg.settle_timeout = 2.0 * cfg.timestep
        backend.position = cfg.standing_pos.copy()
        backend.position[2] += cfg.settle_height_tolerance + 0.01
        backend.quaternion = cfg.standing_quat.copy()
        backend.joint_position = cfg.default_joint.copy()
        backend.joint_velocity.fill(0.0)
        fsm.state = State.STAND_SETTLE
        commands.request("enable_rl")

        fsm.step()
        fsm.step()

        self.assertEqual(fsm.state, State.FAULT)
        self.assertEqual(policy.observations, [])
        self.assertTrue(backend.emergency_reasons)
        self.assertIn("STAND_SETTLE timeout", backend.emergency_reasons[-1])

    def test_getdown_timeout_latches_fault_and_emergency_stop(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack()
        cfg.getdown_joint_speed = 1.0e-6
        cfg.getdown_timeout = 2.0 * cfg.timestep
        backend.position = cfg.standing_pos.copy()
        backend.quaternion = cfg.standing_quat.copy()
        backend.joint_position = cfg.default_joint.copy()
        fsm.state = State.STAND
        commands.request("passive")

        fsm.step()
        self.assertEqual(fsm.state, State.GETDOWN)
        fsm.step()

        self.assertEqual(fsm.state, State.FAULT)
        self.assertTrue(backend.emergency_reasons)
        self.assertIn("GETDOWN timeout", backend.emergency_reasons[-1])

    def test_observation_first_frame_backfill(self) -> None:
        cfg, backend, _, fsm, _ = self.make_stack()
        first = fsm.observation(backend.read_state())
        self.assertEqual(first.shape, (cfg.observation_dim,))
        frames = first.reshape(cfg.history_size, 47)
        for frame in frames[1:]:
            np.testing.assert_array_equal(frame, frames[0])

    def test_observation_history_is_current_first(self) -> None:
        cfg, backend, _, fsm, _ = self.make_stack()
        observations: list[np.ndarray] = []
        # Use a sentinel in base angular velocity, which is the first term of
        # each 47-value frame, so frame ordering is unambiguous.
        for sample in range(cfg.history_size + 1):
            backend.angular_velocity_body[:] = float(sample)
            observations.append(fsm.observation(backend.read_state()))
        latest = observations[-1].reshape(cfg.history_size, 47)
        self.assertEqual(latest[:, 0].tolist(), list(range(cfg.history_size, 0, -1)))

    def test_phase_gate_matches_training_for_mixed_commands_and_boundary(self) -> None:
        cfg, backend, _, fsm, _ = self.make_stack()
        fsm._rl_phase_time = cfg.phase_period / 4.0
        cases = (
            ((0.0, 0.0, 0.0), False),
            ((0.1, 0.0, 0.0), False),
            ((0.0, 0.0, -0.1), False),
            ((0.0, -0.101, 0.0), True),
            ((0.06, 0.0, 0.06), True),
            ((-0.06, 0.0, -0.06), True),
            ((0.03, 0.04, 0.04), False),
            ((0.03, 0.04, -0.06), True),
        )
        for command, moving in cases:
            with self.subTest(command=command):
                fsm.command = np.asarray(command, dtype=np.float64)
                frame = fsm.observation(backend.read_state()).reshape(
                    cfg.history_size, 47
                )[0]
                np.testing.assert_allclose(
                    frame[9:11], [1.0, 0.0] if moving else [0.0, 0.0], atol=1e-12
                )

    def test_zero_bus_command_zeroes_current_command_phase_and_bounds_history(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack()
        commands.set_command(0.5, -0.2, 0.1, source="keyboard")
        fsm.command = commands.command()
        for _ in range(cfg.history_size):
            fsm.observation(backend.read_state())

        commands.stop()
        fsm.command = commands.command()
        observation = fsm.observation(backend.read_state())
        frames = observation.reshape(cfg.history_size, 47)

        # base_ang_vel(3) + projected_gravity(3) precede command(3) and
        # phase(2) in each HIM frame.
        np.testing.assert_allclose(frames[0, 6:11], 0.0)
        historical_commands = frames[1:, 6:9]
        self.assertLessEqual(
            int(np.count_nonzero(np.linalg.norm(historical_commands, axis=1))),
            cfg.history_size - 1,
        )

        for _ in range(cfg.history_size - 1):
            observation = fsm.observation(backend.read_state())
        frames = observation.reshape(cfg.history_size, 47)
        np.testing.assert_allclose(frames[:, 6:11], 0.0)

    def test_fsm_lifecycle_passive_getup_rl_getdown_passive(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        self.assertEqual(fsm.state, State.PASSIVE)

        commands.request("stand")
        fsm.step()
        self.assertEqual(fsm.state, State.GETUP)
        self.assertEqual(policy.observations, [])
        self._advance(fsm, backend, cfg.stand_duration)
        self.assertIn(fsm.state, (State.STAND_SETTLE, State.READY, State.STAND))
        self._advance(fsm, backend, cfg.settle_duration + 0.05)
        if fsm.state is State.READY:
            fsm.step()
        self.assertEqual(fsm.state, State.STAND)
        commands.request("enable_rl")
        fsm.step()
        self.assertEqual(fsm.state, State.RL)
        fsm.step()
        # Policy is called only while RL is active (and at decimation cadence).
        self.assertGreaterEqual(len(policy.observations), 1)

        commands.request("passive")
        calls_before_getdown = len(policy.observations)
        fsm.step()
        self.assertEqual(fsm.state, State.GETDOWN)
        self._advance(fsm, backend, cfg.getdown_duration)
        self.assertEqual(fsm.state, State.PASSIVE)
        self.assertTrue(np.allclose(fsm.last_action, 0.0))
        self.assertEqual(len(policy.observations), calls_before_getdown)

    def test_simultaneous_passive_and_stand_keeps_getdown(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack()
        self._enter_rl(cfg, backend, commands, fsm)

        # Passive has higher priority than stand.  The stand request must also
        # be consumed, otherwise it would reverse GETDOWN on the next tick.
        commands.request("passive")
        commands.request("stand")
        fsm.step()
        self.assertEqual(fsm.state, State.GETDOWN)
        fsm.step()
        self.assertEqual(fsm.state, State.GETDOWN)

    def test_same_poll_disable_dominates_enable_rl(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        self._force_safe_stand(cfg, backend, fsm)

        events = [
            types.SimpleNamespace(type=2, button=2, instance_id=42),  # X: enable
            types.SimpleNamespace(type=2, button=1, instance_id=42),  # B: disable
        ]
        joystick = JoystickInput(commands, prefer_controller=False)
        device = _MutableJoystick()
        joystick._device = device
        joystick._controller = False
        joystick._instance_id = device.get_instance_id()
        joystick.available = True
        joystick._pygame = _fake_pygame(events)
        joystick._owner_thread = threading.get_ident()
        joystick._running = True

        joystick.poll()
        fsm.step()

        self.assertEqual(fsm.state, State.STAND)
        self.assertEqual(policy.observations, [])
        self.assertFalse(commands.consume("enable_rl"))
        self.assertFalse(commands.consume("disable_rl"))

    def test_same_poll_getdown_dominates_stand_while_passive(self) -> None:
        _, _, commands, fsm, _ = self.make_stack()
        self.assertEqual(fsm.state, State.PASSIVE)

        events = [
            types.SimpleNamespace(type=2, button=3, instance_id=42),  # Y: getdown
            types.SimpleNamespace(type=2, button=0, instance_id=42),  # A: stand
        ]
        joystick = JoystickInput(commands, prefer_controller=False)
        device = _MutableJoystick()
        joystick._device = device
        joystick._controller = False
        joystick._instance_id = device.get_instance_id()
        joystick.available = True
        joystick._pygame = _fake_pygame(events)
        joystick._owner_thread = threading.get_ident()
        joystick._running = True

        joystick.poll()
        fsm.step()

        self.assertEqual(fsm.state, State.PASSIVE)
        self.assertFalse(commands.consume("stand"))
        self.assertFalse(commands.consume("get_down"))

    def test_disable_or_getdown_cancels_armed_auto_enable(self) -> None:
        for stop_event, expected_state in (
            ("disable_rl", State.STAND),
            ("get_down", State.GETDOWN),
        ):
            with self.subTest(stop_event=stop_event):
                cfg, backend, commands, fsm, policy = self.make_stack()
                self._force_safe_stand(cfg, backend, fsm)
                fsm._auto_enable_armed = True
                commands.request(stop_event)

                fsm.step()

                self.assertEqual(fsm.state, expected_state)
                self.assertFalse(fsm._auto_enable_armed)
                self.assertEqual(policy.observations, [])

    def test_fsm_consumes_zero_velocity_and_allows_later_explicit_control(self) -> None:
        _, _, commands, fsm, _ = self.make_stack()
        keyboard = KeyboardInput(commands, keyboard_step=0.2)
        commands.set_command(0.8, 0.0, 0.0, source="keyboard")
        commands.request("zero_velocity")

        fsm.step()

        self.assertFalse(commands.pending("zero_velocity"))
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(commands.command(), [0.2, 0.0, 0.0])

    def test_reset_consumes_conflicting_stand_and_passive_events(self) -> None:
        _, backend, commands, fsm, _ = self.make_stack()
        commands.request("stand")
        commands.request("passive")
        commands.request("reset")

        fsm.step()

        self.assertEqual(fsm.state, State.PASSIVE)
        self.assertEqual(backend.reset_count, 1)
        self.assertFalse(commands.consume("stand"))
        self.assertFalse(commands.consume("passive"))
        # No stale transition request may fire on the following tick.
        fsm.step()
        self.assertEqual(fsm.state, State.PASSIVE)

    def test_auto_stand_is_one_shot_after_getdown(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack(auto_stand=True)
        # Advance past the startup auto-stand delay and complete GETUP.
        self._advance(fsm, backend, 0.25 + cfg.stand_duration + cfg.settle_duration + 0.1)
        if fsm.state is State.READY:
            fsm.step()
        self.assertIn(fsm.state, (State.STAND, State.RL))
        if fsm.state is State.STAND:
            commands.request("enable_rl")
            fsm.step()
        fsm.step()

        commands.request("passive")
        fsm.step()
        self._advance(fsm, backend, cfg.getdown_duration)
        self.assertEqual(fsm.state, State.PASSIVE)
        # Staying passive must not trigger another automatic stand.
        self._advance(fsm, backend, 0.5)
        self.assertEqual(fsm.state, State.PASSIVE)

    def test_reset_clears_history_action_and_command(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack()
        fsm.observation(backend.read_state())
        self.assertTrue(getattr(fsm, "_observation_history", []))
        fsm.last_action[:] = 1.0
        commands.set_command(0.5, -0.25, 0.1)
        commands.request("reset")
        fsm.step()
        self.assertEqual(fsm.state, State.PASSIVE)
        self.assertEqual(backend.reset_count, 1)
        np.testing.assert_allclose(fsm.last_action, 0.0)
        np.testing.assert_allclose(fsm.command, 0.0)
        self.assertEqual(getattr(fsm, "_observation_history", []), [])

    def test_reset_rearms_startup_auto_stand(self) -> None:
        cfg, backend, commands, fsm, _ = self.make_stack(auto_stand=True)
        self._advance(fsm, backend, 0.25 + cfg.stand_duration + cfg.settle_duration + 0.1)
        if fsm.state is State.READY:
            fsm.step()
        self.assertIn(fsm.state, (State.STAND, State.RL))
        if fsm.state is State.STAND:
            commands.request("enable_rl")
            fsm.step()
        fsm.step()
        commands.request("passive")
        fsm.step()
        self._advance(fsm, backend, cfg.getdown_duration + 0.3)
        self.assertEqual(fsm.state, State.PASSIVE)

        commands.request("reset")
        fsm.step()
        self._advance(fsm, backend, 0.25)
        self.assertIn(fsm.state, (State.GETUP, State.STAND_SETTLE, State.READY, State.STAND, State.RL))

    def test_rl_phase_starts_at_zero_instead_of_global_sim_time(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        # Give global simulation time a large offset before entering RL.
        backend._time = 12.345
        commands.request("stand")
        fsm.step()
        self._advance(fsm, backend, cfg.stand_duration)
        self._advance(fsm, backend, cfg.settle_duration + 0.05)
        if fsm.state is State.READY:
            fsm.step()
        commands.request("enable_rl")
        fsm.step()
        self.assertEqual(fsm.state, State.RL)
        # Lifecycle transitions intentionally clear stale velocity.  Issue an
        # explicit post-transition command before the first policy tick.
        commands.set_command(0.5, 0.0, 0.0, source="keyboard")
        fsm.step()
        self.assertTrue(policy.observations)
        current_frame = policy.observations[0].reshape(cfg.history_size, 47)[0]
        np.testing.assert_allclose(current_frame[9:11], [0.0, 1.0], atol=1e-12)

    def test_rl_phase_advances_once_per_decimated_policy_step(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        self.assertEqual(cfg.timestep, 0.005)
        self.assertEqual(cfg.decimation, 4)
        control_dt = cfg.timestep * cfg.decimation

        # Enter RL directly to isolate policy cadence from the scripted GETUP
        # duration.  The pose is made safe so fall detection cannot pre-empt
        # the policy step under test.
        backend.position = cfg.standing_pos.copy()
        backend.quaternion = cfg.standing_quat.copy()
        backend.joint_position = cfg.default_joint.copy()
        commands.set_command(0.5, 0.0, 0.0)
        fsm.state = State.RL
        fsm.state_time = 0.0
        fsm._rl_phase_time = 0.0
        fsm._rl_step_index = 0

        # The first policy observation is at phase zero.  Completing that
        # control step advances phase by env.step_dt = 0.005 * 4 = 0.02 s.
        fsm.step()
        self.assertEqual(len(policy.observations), 1)
        self.assertAlmostEqual(fsm._rl_phase_time, control_dt)
        first_frame = policy.observations[0].reshape(cfg.history_size, 47)[0]
        np.testing.assert_allclose(first_frame[9:11], [0.0, 1.0], atol=1e-12)

        # Physics-only ticks must not change control-step phase.
        for _ in range(cfg.decimation - 1):
            fsm.step()
        self.assertEqual(len(policy.observations), 1)
        self.assertAlmostEqual(fsm._rl_phase_time, control_dt)

        # The next policy input sees exactly one 20 ms phase increment.
        fsm.step()
        self.assertEqual(len(policy.observations), 2)
        second_frame = policy.observations[1].reshape(cfg.history_size, 47)[0]
        angle = 2.0 * np.pi * control_dt / cfg.phase_period
        np.testing.assert_allclose(
            second_frame[9:11],
            [np.sin(angle), np.cos(angle)],
            atol=1e-12,
        )
        self.assertAlmostEqual(fsm._rl_phase_time, 2.0 * control_dt)

    def test_fall_safety_transitions_rl_to_getdown(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        commands.request("stand")
        fsm.step()
        self._advance(fsm, backend, cfg.stand_duration)
        self._advance(fsm, backend, cfg.settle_duration + 0.05)
        if fsm.state is State.READY:
            fsm.step()
        commands.request("enable_rl")
        fsm.step()
        fsm.step()
        self.assertEqual(fsm.state, State.RL)
        calls_before = len(policy.observations)
        backend.position[2] = cfg.fall_height - 0.01
        fsm.step()
        # Safety policy may latch FAULT on a real backend; both paths must
        # prevent another policy invocation after the unsafe measurement.
        self.assertIn(fsm.state, (State.GETDOWN, State.FAULT))
        # The unsafe frame must not be sent to the actor after the transition
        # request; subsequent GETDOWN steps are scripted only.
        self.assertEqual(len(policy.observations), calls_before)

    @unittest.skipUnless(MUJOCO is not None, "MuJoCo is not installed")
    def test_runtime_actuator_contract(self) -> None:
        from deploy.include.mujoco_bridge import MujocoBackend

        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        backend = MujocoBackend(cfg)
        try:
            self.assertEqual(backend.model.nu, cfg.action_dim)
            ids = backend.actuator_ids
            self.assertEqual(ids.shape, (cfg.action_dim,))
            from deploy.include.types import ControlMode
            backend.set_control_mode(ControlMode.POSITION)
            np.testing.assert_allclose(backend.model.actuator_gainprm[ids, 0], cfg.kp, rtol=0, atol=1e-5)
            np.testing.assert_allclose(backend.model.actuator_biasprm[ids, 2], -cfg.kd, rtol=0, atol=1e-5)
            np.testing.assert_allclose(
                backend.model.actuator_forcerange[ids],
                np.tile([-cfg.effort_limit, cfg.effort_limit], (cfg.action_dim, 1)),
                atol=1e-5,
            )
        finally:
            backend.close()

    @unittest.skipUnless(MUJOCO is not None, "MuJoCo is not installed")
    def test_joint_targets_are_clipped_to_individual_joint_limits(self) -> None:
        from deploy.include.mujoco_bridge import MujocoBackend

        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        backend = MujocoBackend(cfg)
        try:
            joint_range = backend.model.jnt_range[backend.joint_ids]
            self.assertTrue(np.all(backend.model.jnt_limited[backend.joint_ids]))

            backend.set_joint_target(np.full(cfg.action_dim, 1.0e6))
            np.testing.assert_allclose(backend._target, joint_range[:, 1])
            np.testing.assert_allclose(backend.data.ctrl[backend.actuator_ids], joint_range[:, 1])

            backend.set_joint_target(np.full(cfg.action_dim, -1.0e6))
            np.testing.assert_allclose(backend._target, joint_range[:, 0])
            np.testing.assert_allclose(backend.data.ctrl[backend.actuator_ids], joint_range[:, 0])

            self.assertTrue(np.all(backend._target >= joint_range[:, 0]))
            self.assertTrue(np.all(backend._target <= joint_range[:, 1]))
        finally:
            backend.close()

    @unittest.skipUnless(ONNXRUNTIME_AVAILABLE, "onnxruntime is not installed")
    def test_real_onnx_policy_smoke(self) -> None:
        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        if cfg.policy is None or not cfg.policy.exists():
            self.skipTest("configured HIM ONNX export is unavailable")
        policy = OnnxPolicy(cfg.policy, cfg.observation_dim, cfg.action_dim, cfg.action_clip)
        action = policy(np.zeros(cfg.observation_dim, dtype=np.float64))
        self.assertEqual(action.shape, (cfg.action_dim,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_action_clip_none_preserves_policy_output(self) -> None:
        class Session:
            @staticmethod
            def run(_outputs, _inputs):
                return [np.array([[2.5, -3.0, 0.25]], dtype=np.float32)]

        policy = OnnxPolicy(None, observation_dim=4, action_dim=3, action_clip=None)
        policy._session = Session()
        policy._input_name = "observation"
        np.testing.assert_allclose(policy(np.zeros(4)), [2.5, -3.0, 0.25])

    def test_gamepad_config_and_command_bus_deadzone(self) -> None:
        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        self.assertEqual(cfg.gamepad_options["axis_config"]["yaw"], "RIGHTX")
        self.assertEqual(cfg.gamepad_options["button_config"]["stand"], "A")
        # CommandBus applies the configured deadzone and physical limits to
        # normalized SDL axis values.
        bus = CommandBus(cfg.command_limit, cfg.command_deadzone)
        bus.set_axes(0.01, -0.5, 2.0)
        shaped_y = -(0.5 - cfg.command_deadzone) / (1.0 - cfg.command_deadzone)
        np.testing.assert_allclose(
            bus.command(),
            [0.0, shaped_y * cfg.command_limit[1], cfg.command_limit[2]],
        )

    def test_gamepad_left_stick_axis_semantics(self) -> None:
        """SDL/XInput left-stick X is lateral and Y is fore/aft motion.

        The physical convention used by the Beitong Asura 2 Pro is
        ``left_x -> vy`` and ``left_y -> -vx`` (SDL reports positive Y when
        the stick is pushed down).  This test uses a raw-joystick double so
        it does not require pygame or a physical controller.
        """

        class _FakeJoystick:
            def __init__(self, axes: tuple[float, float, float]):
                self._axes = axes

            def get_numaxes(self) -> int:
                return len(self._axes)

            def get_axis(self, index: int) -> float:
                return self._axes[index]

        limits = np.array([2.0, 1.0, 0.5], dtype=np.float64)
        bus = CommandBus(limits, deadzone=0.0)
        joystick = JoystickInput(bus, axes=(0, 1, 2), prefer_controller=False)
        joystick._device = _FakeJoystick((0.4, -0.6, 0.25))
        joystick._controller = False

        joystick._set_axes_from_device()

        # Forward (negative Y) maps to positive vx.  Right on either stick
        # follows the keyboard D/E convention: negative vy / negative wz.
        shaped = GamepadAxisProcessor().process(
            (0.4, -0.6, 0.25),
            (0.0, 0.0, 0.0),
        ).processed
        np.testing.assert_allclose(
            bus.command(),
            [-shaped[1] * limits[0], -shaped[0] * limits[1], -shaped[2] * limits[2]],
        )

        keyboard_bus = CommandBus(limits, deadzone=0.0)
        keyboard = KeyboardInput(keyboard_bus, keyboard_step=0.1)
        keyboard.handle_viewer_key(ord("d"))
        keyboard.handle_viewer_key(ord("e"))
        self.assertEqual(np.sign(bus.command()[1]), np.sign(keyboard_bus.command()[1]))
        self.assertEqual(np.sign(bus.command()[2]), np.sign(keyboard_bus.command()[2]))

    def test_raw_joystick_semantic_button_aliases(self) -> None:
        """Raw pygame.Joystick events must understand YAML semantic names.

        A Beitong/XInput-compatible pad may not be recognized by SDL's
        GameController database, in which case ``JOYBUTTONDOWN.button`` is a
        numeric index while the YAML still contains ``A``/``START``/``BACK``.
        Verify the unambiguous raw Xbox/XInput fallback used by the HIM task.
        Other firmware layouts must be configured with explicit numeric
        bindings after inspecting them with ``tools/gamepad_diagnostic.py``.
        """
        bus = CommandBus(np.ones(3), 0.0)
        joystick = JoystickInput(
            bus,
            button_config={
                "stand": "A",
                "enable_rl": "X",
                "disable_rl": "B",
                "get_down": "Y",
                "reset": "START",
                "back": "BACK",
            },
            prefer_controller=False,
        )
        joystick._controller = False

        self.assertTrue(joystick._button_value(0, "stand"))
        self.assertTrue(joystick._button_value(2, "enable_rl"))
        self.assertTrue(joystick._button_value(1, "disable_rl"))
        self.assertTrue(joystick._button_value(3, "get_down"))
        self.assertTrue(joystick._button_value(7, "reset"))
        self.assertFalse(joystick._button_value(6, "reset"))
        self.assertTrue(joystick._button_value(6, "back"))
        self.assertFalse(joystick._button_value(7, "back"))

        # Explicit numeric and tuple configurations are still honored as-is.
        custom = JoystickInput(
            bus,
            button_config={"stand": 11, "reset": (12, 13)},
            prefer_controller=False,
        )
        custom._controller = False
        self.assertTrue(custom._button_value(11, "stand"))
        self.assertFalse(custom._button_value(0, "stand"))
        self.assertTrue(custom._button_value(12, "reset"))
        self.assertTrue(custom._button_value(13, "reset"))

    def test_keyboard_manual_rl_lifecycle_keys(self) -> None:
        """Keyboard aliases mirror the gamepad RL lifecycle events."""
        bus = CommandBus(np.ones(3), 0.0)
        keyboard = KeyboardInput(bus)
        for key, event in ((ord("l"), "enable_rl"), (ord("b"), "disable_rl")):
            keyboard.handle_key(key, 1)
            keyboard.handle_key(key, 0)
            self.assertTrue(bus.consume(event), event)

    @unittest.skipUnless(MUJOCO is not None, "MuJoCo is not installed")
    def test_headless_no_policy_smokes(self) -> None:
        command = [
            sys.executable,
            "deploy/main.py",
            "--task",
            "ri_4438_him",
            "--no-policy",
            "--headless",
            "--steps",
            "10",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    @unittest.skipUnless(MUJOCO is not None, "MuJoCo is not installed")
    def test_headless_auto_stand_smokes(self) -> None:
        command = [
            sys.executable,
            "deploy/main.py",
            "--task",
            "ri_4438_him",
            "--no-policy",
            "--headless",
            "--auto-stand",
            "--steps",
            "800",
        ]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)


class _MutableJoystick:
    """Raw pygame.Joystick double with mutable axis state."""

    def __init__(
        self,
        axes=(0.0, 0.0, -1.0, 0.0),
        *,
        instance_id: int = 42,
    ):
        self.axes = list(axes)
        self.instance_id = instance_id
        self.closed = False

    def get_numaxes(self) -> int:
        return len(self.axes)

    def get_axis(self, index: int) -> float:
        return float(self.axes[index])

    def get_attached(self) -> bool:
        return not self.closed

    def get_instance_id(self) -> int:
        return self.instance_id

    def get_name(self) -> str:
        return "Fake Raw Gamepad"

    def get_guid(self) -> str:
        return "03000000feedface"

    def quit(self) -> None:
        self.closed = True


class _MutableController:
    """pygame._sdl2 Controller double returning signed-int16 axis counts."""

    def __init__(
        self,
        axes=(0, 0, 0),
        *,
        joystick_view: _MutableJoystick | None = None,
        controller_id: int = 2,
    ):
        self.axes = list(axes)
        self._joystick_view = joystick_view
        self.id = controller_id
        self.closed = False

    def init(self) -> None:
        return None

    def quit(self) -> None:
        self.closed = True

    def as_joystick(self):
        return self._joystick_view

    def get_attached(self) -> bool:
        return not self.closed

    def get_name(self) -> str:
        return "Fake SDL Controller"

    def get_axis(self, index: int) -> int:
        value = self.axes[index]
        if not isinstance(value, (int, np.integer)):
            raise TypeError("SDL Controller axis double requires integer counts")
        value = int(value)
        if not -32768 <= value <= 32767:
            raise ValueError("SDL Controller axis count is outside signed-int16 range")
        return value


class _EventQueue:
    def __init__(self, events=()):
        self.events = list(events)

    def pump(self) -> None:
        return None

    def get(self):
        events, self.events = self.events, []
        return events


def _fake_pygame(events=()):
    return types.SimpleNamespace(
        event=_EventQueue(events),
        JOYAXISMOTION=1,
        JOYBUTTONDOWN=2,
        JOYDEVICEADDED=3,
        JOYDEVICEREMOVED=4,
        CONTROLLERAXISMOTION=5,
        CONTROLLERBUTTONDOWN=6,
        CONTROLLERDEVICEADDED=7,
        CONTROLLERDEVICEREMOVED=8,
    )


class GamepadInputRegressionTest(unittest.TestCase):
    """Safety and arbitration contract for keyboard/gamepad coexistence."""

    @staticmethod
    def make_bus() -> CommandBus:
        return CommandBus(np.asarray([2.0, 1.0, 0.5]), deadzone=0.1)

    @staticmethod
    def attach_raw(joystick: JoystickInput, device: _MutableJoystick) -> None:
        joystick._device = device
        joystick._controller = False
        joystick._instance_id = device.get_instance_id()
        joystick.available = True
        joystick._last_seen = time.monotonic()

    @staticmethod
    def attach_controller(
        joystick: JoystickInput,
        device: _MutableController,
    ) -> None:
        joystick._device = device
        joystick._controller = True
        joystick._instance_id = 42
        joystick.available = True
        joystick._last_seen = time.monotonic()

    def calibrate_raw(
        self,
        axes: tuple[float, float, float, float],
        **options,
    ) -> tuple[CommandBus, JoystickInput, _MutableJoystick, list[float]]:
        now = [0.0]
        bus = self.make_bus()
        settings = {
            "prefer_controller": False,
            "calibration_duration": 0.25,
            "neutral_rearm_samples": 2,
            "neutral_rearm_duration": 0.0,
            "clock": lambda: now[0],
        }
        settings.update(options)
        joystick = JoystickInput(bus, **settings)
        device = _MutableJoystick(axes)
        self.attach_raw(joystick, device)
        joystick.begin_calibration(now=0.0)
        for index in range(25):
            now[0] = index * 0.02
            joystick._set_axes_from_device(now=now[0])
        return bus, joystick, device, now

    def test_default_center_limit_rejects_minus_point_two_and_keeps_zero(self) -> None:
        bus, joystick, _, _ = self.calibrate_raw((0.0, -0.2, -1.0, 0.0))
        snapshot = joystick.debug_snapshot()

        self.assertFalse(snapshot.calibration_ready)
        self.assertRegex(snapshot.calibration_reason or "", "safe limit")
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertIsNone(bus.active_source)

    def test_small_center_bias_calibrates_to_exact_neutral(self) -> None:
        bus, joystick, _, _ = self.calibrate_raw((0.0, -0.03, -1.0, 0.0))
        snapshot = joystick.debug_snapshot()

        self.assertTrue(snapshot.calibration_ready)
        self.assertEqual(snapshot.center, (0.0, -0.03, 0.0))
        np.testing.assert_allclose(snapshot.calibrated, 0.0)
        np.testing.assert_allclose(snapshot.processed, 0.0)
        np.testing.assert_allclose(bus.command(), 0.0)

    def assert_safe_release_sequence(
        self,
        calibrated_axes: tuple[float, float, float, float],
    ) -> None:
        bus, joystick, device, now = self.calibrate_raw(calibrated_axes)
        self.assertTrue(joystick.debug_snapshot().calibration_ready)

        # Complete the post-calibration neutral hold at the learned center.
        now[0] += 0.02
        joystick._set_axes_from_device(now=now[0])

        # First release after calibration must already be inside the *exit*
        # threshold, not merely below enter, so it is exactly neutral.
        device.axes[:] = (0.0, 0.0, -1.0, 0.0)
        now[0] += 0.02
        joystick._set_axes_from_device(now=now[0])
        snapshot = joystick.debug_snapshot()
        translation_residual = float(np.linalg.norm(snapshot.calibrated[:2]))
        yaw_residual = abs(snapshot.calibrated[2])
        self.assertLess(
            translation_residual + joystick._calibrator.release_margin,
            joystick.deadzone_exit,
        )
        self.assertLess(
            yaw_residual + joystick._calibrator.release_margin,
            joystick.deadzone_exit,
        )
        self.assertTrue(snapshot.neutral)
        np.testing.assert_array_equal(snapshot.processed, np.zeros(3))
        np.testing.assert_array_equal(bus.command(), np.zeros(3))
        self.assertIsNone(bus.active_source)

        # Take ownership with a deliberate large input, then release to raw
        # zero again.  Hysteresis must clear the active state and owner.
        device.axes[:] = (0.8, -0.8, -1.0, 0.8)
        now[0] += 0.02
        joystick._set_axes_from_device(now=now[0])
        self.assertEqual(bus.active_source, "gamepad")
        self.assertTrue(np.any(bus.command() != 0.0))

        device.axes[:] = (0.0, 0.0, -1.0, 0.0)
        now[0] += 0.02
        joystick._set_axes_from_device(now=now[0])
        snapshot = joystick.debug_snapshot()
        self.assertTrue(snapshot.neutral)
        np.testing.assert_array_equal(snapshot.processed, np.zeros(3))
        np.testing.assert_array_equal(bus.command(), np.zeros(3))
        self.assertIsNone(bus.active_source)

    def test_single_axis_center_boundary_is_neutral_on_every_release(self) -> None:
        self.assert_safe_release_sequence((0.15, 0.0, -1.0, 0.0))

    def test_two_axis_center_boundary_is_neutral_on_every_release(self) -> None:
        # 0.11 on each axis is close to the accepted radial release boundary:
        # hypot(0.11/1.11, 0.11/1.11) + 0.005 < exit=0.15.
        self.assert_safe_release_sequence((0.11, 0.11, -1.0, 0.0))

    def test_diagonal_held_at_connect_is_rejected_by_radial_release_check(self) -> None:
        bus, joystick, _, _ = self.calibrate_raw((0.12, 0.12, -1.0, 0.0))
        snapshot = joystick.debug_snapshot()

        # Each component is below center_limit=0.15, but releasing both axes
        # would lie outside the neutral/exit boundary.
        self.assertFalse(snapshot.calibration_ready)
        self.assertRegex(snapshot.calibration_reason or "", "release|held-at-connect")
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertIsNone(bus.active_source)

    def test_unsafe_center_limit_without_matching_exit_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "center.*exit|release|deadzone"):
            JoystickInput(
                self.make_bus(),
                deadzone_enter=0.18,
                deadzone_exit=0.13,
                calibration_center_limit=0.15,
            )

    def test_explicit_safe_large_center_configuration_can_calibrate(self) -> None:
        bus, joystick, _, _ = self.calibrate_raw(
            (0.0, -0.20, -1.0, 0.0),
            deadzone_enter=0.22,
            deadzone_exit=0.18,
            calibration_center_limit=0.20,
        )
        snapshot = joystick.debug_snapshot()

        self.assertTrue(snapshot.calibration_ready)
        self.assertEqual(snapshot.center, (0.0, -0.2, 0.0))
        np.testing.assert_allclose(snapshot.processed, 0.0)
        np.testing.assert_allclose(bus.command(), 0.0)

    def test_center_corrected_deflection_is_continuous_and_reaches_full_scale(self) -> None:
        processor = GamepadAxisProcessor()
        center = (0.0, -0.03, 0.0)
        forward_commands = []
        for raw_y in (-0.23, -0.53, -1.0):
            processed = processor.process((0.0, raw_y, 0.0), center)
            forward_commands.append(-processed.processed[1])

        self.assertGreater(forward_commands[0], 0.0)
        self.assertLess(forward_commands[0], forward_commands[1])
        self.assertLess(forward_commands[1], forward_commands[2])
        self.assertAlmostEqual(forward_commands[-1], 1.0)

    def test_stable_axis_near_negative_one_is_rejected_as_wrong_mapping(self) -> None:
        calibrator = AxisCalibrator(min_duration=0.25)
        calibrator.reset(0.0)
        update = None
        for index in range(25):
            update = calibrator.add_sample((0.0, 0.0, -0.98), now=index * 0.02)

        self.assertIsNotNone(update)
        self.assertFalse(update.ready)
        self.assertTrue(update.stable)
        self.assertIsNone(calibrator.center)
        self.assertRegex(update.reason or "", "trigger|wrong axis")

    def test_deadzone_hysteresis_does_not_chatter_command_owner(self) -> None:
        bus = CommandBus(np.ones(3), deadzone=0.0)
        processor = GamepadAxisProcessor()
        owners = []
        for left_x in (0.17, 0.18, 0.17, 0.151, 0.15, 0.16, 0.179):
            processed = processor.process((left_x, 0.0, 0.0), (0.0, 0.0, 0.0))
            command = processed.processed
            bus.set_command(
                -command[1],
                -command[0],
                -command[2],
                source="gamepad",
                active=processed.translation_active or processed.yaw_active,
            )
            owners.append(bus.snapshot().owner)

        self.assertEqual(
            owners,
            [None, "gamepad", "gamepad", "gamepad", None, None, None],
        )

    def test_sdl_display_and_joystick_are_initialized_before_event_pump(self) -> None:
        calls: list[str] = []

        class Display:
            @staticmethod
            def get_init():
                return False

            @staticmethod
            def init():
                calls.append("display.init")

            @staticmethod
            def get_surface():
                return None

            @staticmethod
            def set_mode(_size, *, flags=0):
                del flags
                calls.append("display.set_mode")
                return object()

        class Event:
            @staticmethod
            def pump():
                calls.append("event.pump")

        class JoystickSubsystem:
            @staticmethod
            def get_init():
                return False

            @staticmethod
            def init():
                calls.append("joystick.init")

        fake_pygame = types.ModuleType("pygame")
        fake_pygame.__path__ = []
        fake_pygame.HIDDEN = 1
        fake_pygame.display = Display()
        fake_pygame.event = Event()
        fake_pygame.joystick = JoystickSubsystem()

        fake_sdl2 = types.ModuleType("pygame._sdl2")
        fake_sdl2.__path__ = []
        fake_controller = types.ModuleType("pygame._sdl2.controller")
        fake_controller.get_init = lambda: False
        fake_controller.init = lambda: calls.append("controller.init")
        fake_sdl2.controller = fake_controller
        fake_pygame._sdl2 = fake_sdl2

        bus = self.make_bus()
        joystick = JoystickInput(bus)
        joystick._pygame = fake_pygame
        with mock.patch.dict(
            sys.modules,
            {
                "pygame": fake_pygame,
                "pygame._sdl2": fake_sdl2,
                "pygame._sdl2.controller": fake_controller,
            },
        ):
            joystick._initialize_sdl()

        self.assertEqual(calls[-1], "event.pump")
        self.assertLess(calls.index("display.init"), calls.index("event.pump"))
        self.assertLess(calls.index("display.set_mode"), calls.index("event.pump"))
        self.assertLess(calls.index("joystick.init"), calls.index("event.pump"))
        self.assertLess(calls.index("controller.init"), calls.index("event.pump"))

    def test_runtime_sets_background_event_hint_before_sdl_initialization(self) -> None:
        observed: list[str | None] = []

        class Display:
            @staticmethod
            def get_init():
                return False

            @staticmethod
            def init():
                observed.append(os.environ.get("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"))

            @staticmethod
            def get_surface():
                return object()

            @staticmethod
            def quit():
                return None

        class JoystickSubsystem:
            @staticmethod
            def get_init():
                return False

            @staticmethod
            def init():
                observed.append(os.environ.get("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"))

            @staticmethod
            def get_count():
                return 0

            @staticmethod
            def quit():
                return None

        fake_pygame = types.ModuleType("pygame")
        fake_pygame.__path__ = []
        fake_pygame.HIDDEN = 1
        fake_pygame.display = Display()
        fake_pygame.joystick = JoystickSubsystem()
        fake_pygame.event = _EventQueue()
        fake_controller = types.ModuleType("pygame._sdl2.controller")
        fake_controller.get_init = lambda: False
        fake_controller.init = lambda: observed.append(
            os.environ.get("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS")
        )
        fake_controller.quit = lambda: None
        fake_sdl2 = types.ModuleType("pygame._sdl2")
        fake_sdl2.__path__ = []
        fake_sdl2.controller = fake_controller
        fake_pygame._sdl2 = fake_sdl2

        bus = self.make_bus()
        joystick = JoystickInput(bus)
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules,
            {
                "pygame": fake_pygame,
                "pygame._sdl2": fake_sdl2,
                "pygame._sdl2.controller": fake_controller,
            },
        ):
            joystick.start()
            joystick.close()

        self.assertEqual(observed, ["1", "1", "1"])

    def test_poll_runs_on_owner_thread_and_rejects_worker_thread(self) -> None:
        class Event:
            @staticmethod
            def pump():
                return None

            @staticmethod
            def get():
                return []

        fake_pygame = types.SimpleNamespace(event=Event())
        bus = self.make_bus()
        joystick = JoystickInput(bus, reconnect_interval=1000.0)
        self.attach_raw(joystick, _MutableJoystick())
        joystick._pygame = fake_pygame
        joystick._running = True
        joystick._owner_thread = threading.get_ident()

        joystick.poll()

        failures: list[BaseException] = []

        def poll_from_worker() -> None:
            try:
                joystick.poll()
            except BaseException as exc:  # Preserve the exact cross-thread failure.
                failures.append(exc)

        worker = threading.Thread(target=poll_from_worker)
        worker.start()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], RuntimeError)
        self.assertIn("owner thread", str(failures[0]))

    def test_sdl_controller_axis_counts_use_signed_int16_normalization(self) -> None:
        bus = self.make_bus()
        joystick = JoystickInput(bus)
        controller = _MutableController()
        self.attach_controller(joystick, controller)

        cases = (
            (-32768, -1.0),
            (-16384, -0.5),
            (-1, -1.0 / 32768.0),
            (0, 0.0),
            (1, 1.0 / 32767.0),
            (6, 6.0 / 32767.0),
            (8192, 8192.0 / 32767.0),
            (16384, 16384.0 / 32767.0),
            (32767, 1.0),
        )
        for count, expected in cases:
            with self.subTest(count=count):
                controller.axes[0] = count
                self.assertAlmostEqual(
                    joystick._axis_value("left_x"),
                    expected,
                    places=15,
                )

    def test_sdl_controller_rejects_values_outside_int16_contract(self) -> None:
        bus = self.make_bus()
        joystick = JoystickInput(bus)
        controller = _MutableController()
        self.attach_controller(joystick, controller)

        cases = (
            (True, "not a signed-int16 integer"),
            (1.0, "not a signed-int16 integer"),
            (-32769, "outside signed-int16 range"),
            (32768, "outside signed-int16 range"),
            (np.nan, "non-finite"),
            (np.inf, "non-finite"),
        )
        for invalid, reason in cases:
            with self.subTest(value=invalid), mock.patch.object(
                controller,
                "get_axis",
                return_value=invalid,
            ):
                with self.assertRaisesRegex(RuntimeError, reason):
                    joystick._axis_value("left_x")

    def test_sdl_controller_idle_counts_calibrate_as_neutral(self) -> None:
        now = [0.0]
        bus = self.make_bus()
        joystick = JoystickInput(
            bus,
            calibration_duration=0.25,
            neutral_rearm_samples=2,
            neutral_rearm_duration=0.0,
            clock=lambda: now[0],
        )
        controller = _MutableController((0, -1, 6))
        self.attach_controller(joystick, controller)
        joystick.begin_calibration(now=0.0)

        for index in range(25):
            now[0] = index * 0.02
            joystick._set_axes_from_device(now=now[0])

        expected_idle = (0.0, -1.0 / 32768.0, 6.0 / 32767.0)
        snapshot = joystick.debug_snapshot(now=now[0])
        self.assertTrue(snapshot.calibration_ready, snapshot.calibration_reason)
        np.testing.assert_allclose(snapshot.raw, expected_idle, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(snapshot.center, expected_idle, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(snapshot.calibrated, 0.0, rtol=0.0, atol=1.0e-15)
        np.testing.assert_array_equal(snapshot.processed, np.zeros(3))
        np.testing.assert_array_equal(bus.command(), np.zeros(3))
        self.assertIsNone(bus.active_source)

    def test_raw_joystick_axis_values_remain_normalized_floats(self) -> None:
        bus = self.make_bus()
        joystick = JoystickInput(bus, prefer_controller=False)
        device = _MutableJoystick((0.25, -0.5, -1.0, 0.125))
        self.attach_raw(joystick, device)

        np.testing.assert_array_equal(
            joystick._read_raw_axes(),
            (0.25, -0.5, 0.125),
        )

    def test_controller_events_use_raw_joystick_instance_id(self) -> None:
        raw = _MutableJoystick(instance_id=91)
        now = [0.0]
        # Deliberately model SDL's device index/id as different from the event
        # instance ID exposed by Controller.as_joystick().
        controller = _MutableController(joystick_view=raw, controller_id=2)

        class ControllerModule:
            CONTROLLER_AXIS = None

            @staticmethod
            def is_controller(index):
                return index == 2

            @staticmethod
            def Controller(index):
                self.assertEqual(index, 2)
                return controller

        events = [
            types.SimpleNamespace(type=5, axis=0, instance_id=91),
            types.SimpleNamespace(type=6, button=0, instance_id=91),
        ]
        pygame = _fake_pygame(events)
        bus = self.make_bus()
        joystick = JoystickInput(
            bus,
            calibration_duration=0.0,
            neutral_rearm_samples=2,
            neutral_rearm_duration=0.0,
            clock=lambda: now[0],
        )
        joystick._pygame = pygame
        joystick._controller_mod = ControllerModule
        joystick._owner_thread = threading.get_ident()
        joystick._running = True

        self.assertTrue(joystick._open_controller(2))
        self.assertEqual(joystick._instance_id, 91)

        # A real open path is fail-safe calibrated before motion is accepted.
        # Feed centered samples, complete the neutral hold, then verify that an
        # axis/button event keyed by the *backing joystick* instance is handled.
        for index in range(25):
            now[0] = index * 0.01
            joystick._set_axes_from_device(now=now[0])
        now[0] += 0.01
        joystick._set_axes_from_device(now=now[0])
        controller.axes[:] = (16384, -16384, 8192)
        now[0] += 0.01
        joystick.poll()

        self.assertTrue(bus.consume("stand"))
        normalized = (16384.0 / 32767.0, -0.5, 8192.0 / 32767.0)
        shaped = GamepadAxisProcessor().process(
            normalized,
            (0.0, 0.0, 0.0),
        ).processed
        np.testing.assert_allclose(
            bus.command(),
            [-shaped[1] * 2.0, -shaped[0], -shaped[2] * 0.5],
        )

    def test_nonfinite_command_values_fail_safe_to_zero(self) -> None:
        invalid_calls = (
            lambda bus: bus.set_command(np.nan, 0.0, 0.0, source="gamepad"),
            lambda bus: bus.set_command(0.0, np.inf, 0.0, source="keyboard"),
            lambda bus: bus.set_axes(0.0, 0.0, -np.inf, source="gamepad"),
            lambda bus: bus.set_axes(np.nan, 0.0, 0.0, source="gamepad"),
            lambda bus: bus.increment_command(
                np.asarray((0.0, np.nan, 0.0)), source="keyboard"
            ),
        )
        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                bus = self.make_bus()
                bus.set_command(1.0, 0.5, 0.25, source="gamepad")
                try:
                    invalid_call(bus)
                except (TypeError, ValueError, RuntimeError):
                    pass
                self.assertTrue(np.all(np.isfinite(bus.command())))
                np.testing.assert_allclose(bus.command(), 0.0)
                self.assertIsNone(bus.active_source)

    def test_nonfinite_device_axes_disconnect_without_full_speed_command(self) -> None:
        for axes, controller in (
            ((np.nan, 0.0, -1.0, 0.0), False),
            ((0.0, 0.0, -1.0, np.inf), False),
            ((0.0, np.nan, 0.0), True),
        ):
            with self.subTest(axes=axes, controller=controller):
                bus = self.make_bus()
                bus.set_command(1.0, 0.5, 0.25, source="gamepad")
                joystick = JoystickInput(bus, prefer_controller=False)
                device = _MutableJoystick(axes)
                self.attach_raw(joystick, device)
                joystick._controller = controller
                joystick._pygame = _fake_pygame()
                joystick._owner_thread = threading.get_ident()
                joystick._running = True

                joystick.poll()

                self.assertTrue(np.all(np.isfinite(bus.command())))
                np.testing.assert_allclose(bus.command(), 0.0)
                self.assertFalse(joystick.available)

    def test_keyboard_takes_clean_ownership_from_multi_axis_gamepad(self) -> None:
        bus = self.make_bus()
        device = _MutableJoystick((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(bus, prefer_controller=False)
        self.attach_raw(joystick, device)
        joystick._set_axes_from_device()
        self.assertEqual(bus.active_source, "gamepad")
        self.assertTrue(np.all(np.abs(bus.command()) > 0.0))

        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), [0.2, 0.0, 0.0])
        self.assertEqual(bus.active_source, "keyboard")

        device.axes[:] = (0.0, 0.0, -1.0, 0.0)
        joystick._set_axes_from_device()
        np.testing.assert_allclose(bus.command(), [0.2, 0.0, 0.0])
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), [0.4, 0.0, 0.0])

    def test_pending_stop_blocks_keyboard_viewer_and_gamepad_writes(self) -> None:
        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        device = _MutableJoystick((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(bus, prefer_controller=False)
        self.attach_raw(joystick, device)

        bus.set_command(1.0, 0.5, 0.25, source="gamepad")
        bus.request("reset")
        keyboard.handle_key(ord("w"), 1)
        keyboard.handle_key(ord("w"), 0)
        keyboard.handle_viewer_key(ord("d"))
        joystick._set_axes_from_device()

        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertIsNone(bus.active_source)

    def test_gamepad_requires_neutral_after_stop_before_rearming(self) -> None:
        bus = self.make_bus()
        now = [0.0]
        device = _MutableJoystick((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(
            bus,
            prefer_controller=False,
            neutral_rearm_samples=3,
            neutral_rearm_duration=0.10,
            clock=lambda: now[0],
        )
        self.attach_raw(joystick, device)
        joystick._set_axes_from_device(now=now[0])
        self.assertEqual(bus.active_source, "gamepad")

        bus.request("zero_velocity")
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertTrue(bus.consume("zero_velocity"))

        # A stick still held away from center must not immediately undo STOP.
        now[0] = 0.01
        joystick._set_axes_from_device(now=now[0])
        np.testing.assert_allclose(bus.command(), 0.0)
        device.axes[:] = (0.0, 0.0, -1.0, 0.0)
        for timestamp in (0.02, 0.07, 0.13):
            now[0] = timestamp
            joystick._set_axes_from_device(now=timestamp)
            np.testing.assert_allclose(bus.command(), 0.0)

        device.axes[:] = (0.5, -0.5, -1.0, 0.5)
        now[0] = 0.14
        joystick._set_axes_from_device(now=now[0])
        self.assertEqual(bus.active_source, "gamepad")
        self.assertTrue(np.all(np.abs(bus.command()) > 0.0))

    def test_held_stick_polling_refreshes_watchdog_and_timeout_fails_safe(self) -> None:
        now = [0.0]
        bus = self.make_bus()
        device = _MutableJoystick((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(
            bus,
            prefer_controller=False,
            reconnect_interval=10.0,
            liveness_timeout=0.10,
            clock=lambda: now[0],
        )
        self.attach_raw(joystick, device)
        joystick._pygame = _fake_pygame()
        joystick._owner_thread = threading.get_ident()
        joystick._running = True

        for timestamp in (0.0, 0.05, 0.10, 0.15):
            now[0] = timestamp
            joystick.poll()
            self.assertTrue(joystick.available)
            self.assertEqual(joystick.debug_snapshot().last_successful_poll, timestamp)
        self.assertEqual(bus.active_source, "gamepad")

        # No polling occurred for longer than the deadline.  The next control
        # tick must stop before accepting another device sample.
        now[0] = 0.30
        joystick.poll()
        self.assertFalse(joystick.available)
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertTrue(bus.pending("estop"))
        self.assertTrue(bus.pending("quit"))

    def test_axis_read_crossing_watchdog_deadline_cannot_publish(self) -> None:
        now = [0.0]

        class SlowAxisDevice(_MutableJoystick):
            def get_axis(self, index: int) -> float:
                now[0] += 0.03
                return super().get_axis(index)

        bus = self.make_bus()
        bus.set_command(0.4, 0.0, 0.0, source="gamepad")
        device = SlowAxisDevice((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(
            bus,
            prefer_controller=False,
            reconnect_interval=10.0,
            liveness_timeout=0.10,
            clock=lambda: now[0],
        )
        self.attach_raw(joystick, device)
        joystick._last_successful_poll = 0.0
        joystick._pygame = _fake_pygame()
        joystick._owner_thread = threading.get_ident()
        joystick._running = True
        now[0] = 0.05

        with mock.patch.object(
            bus,
            "set_command",
            wraps=bus.set_command,
        ) as publish:
            joystick.poll()

        publish.assert_not_called()
        self.assertFalse(joystick.available)
        self.assertFalse(joystick._running)
        self.assertTrue(device.closed)
        self.assertEqual(joystick.debug_snapshot().last_successful_poll, 0.0)
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertTrue(bus.pending("estop"))
        self.assertTrue(bus.pending("quit"))

    def test_short_axis_read_updates_completion_heartbeat_and_publishes(self) -> None:
        now = [0.0]

        class ShortAxisDevice(_MutableJoystick):
            def get_axis(self, index: int) -> float:
                now[0] += 0.005
                return super().get_axis(index)

        bus = self.make_bus()
        device = ShortAxisDevice((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(
            bus,
            prefer_controller=False,
            reconnect_interval=10.0,
            liveness_timeout=0.10,
            clock=lambda: now[0],
        )
        self.attach_raw(joystick, device)
        joystick._last_successful_poll = 0.0
        joystick._pygame = _fake_pygame()
        joystick._owner_thread = threading.get_ident()
        joystick._running = True
        now[0] = 0.05

        with mock.patch.object(
            bus,
            "set_command",
            wraps=bus.set_command,
        ) as publish:
            joystick.poll()

        publish.assert_called_once()
        self.assertTrue(joystick.available)
        self.assertTrue(joystick._running)
        self.assertAlmostEqual(
            joystick.debug_snapshot().last_successful_poll,
            0.065,
        )
        self.assertEqual(bus.active_source, "gamepad")
        self.assertTrue(np.all(np.abs(bus.command()) > 0.0))

        # A native get_axis call that never returns cannot be interrupted from
        # this same synchronous SDL thread.  That permanent-block case belongs
        # to process/OS supervision and is intentionally not executed here.

    def test_debug_snapshot_separates_keyboard_command_from_gamepad_bias(self) -> None:
        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        keyboard.handle_viewer_key(ord("w"))
        device = _MutableJoystick((0.05, 0.0, -1.0, 0.0))
        joystick = JoystickInput(bus, prefer_controller=False)
        self.attach_raw(joystick, device)

        joystick._set_axes_from_device()
        snapshot = joystick.debug_snapshot()

        self.assertEqual(snapshot.mode, "raw")
        self.assertEqual(snapshot.axis_bindings, (0, 1, 3))
        self.assertEqual(snapshot.raw, (0.05, 0.0, 0.0))
        np.testing.assert_allclose(snapshot.calibrated, [0.05, 0.0, 0.0])
        np.testing.assert_allclose(snapshot.processed, 0.0)
        self.assertTrue(snapshot.neutral)
        self.assertEqual(snapshot.owner, "keyboard")
        np.testing.assert_allclose(snapshot.command, [0.2, 0.0, 0.0])

    def test_consumed_zero_velocity_allows_explicit_keyboard_control(self) -> None:
        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        bus.set_command(1.0, 0.0, 0.0, source="keyboard")
        bus.request("zero_velocity")
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), 0.0)

        self.assertTrue(bus.consume("zero_velocity"))
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), [0.2, 0.0, 0.0])

    def test_newer_stop_cannot_be_unlocked_by_stale_control_tick(self) -> None:
        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        bus.request("reset")
        first_events, first_generation = bus.consume_many("reset")
        self.assertTrue(first_events["reset"])
        self.assertTrue(bus.stop_pending)

        # Simulate another input thread publishing a newer STOP after the FSM
        # took its first event snapshot but before that tick acknowledged it.
        bus.request("disable_rl")
        self.assertFalse(bus.acknowledge_stop(first_generation))
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), 0.0)

        second_events, second_generation = bus.consume_many("disable_rl")
        self.assertTrue(second_events["disable_rl"])
        self.assertTrue(bus.acknowledge_stop(second_generation))
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), [0.2, 0.0, 0.0])

    def test_neutral_gamepad_does_not_overwrite_keyboard_command(self) -> None:
        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        keyboard.handle_viewer_key(ord("w"))
        before = bus.command()
        self.assertEqual(bus.active_source, "keyboard")

        joystick = JoystickInput(bus, prefer_controller=False)
        self.attach_raw(joystick, _MutableJoystick())
        joystick._set_axes_from_device()

        np.testing.assert_allclose(bus.command(), before)
        self.assertEqual(bus.active_source, "keyboard")

    def test_active_gamepad_takes_ownership_then_neutral_releases_once(self) -> None:
        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        keyboard.handle_viewer_key(ord("w"))

        device = _MutableJoystick((0.5, -0.5, -1.0, 0.5))
        joystick = JoystickInput(bus, prefer_controller=False)
        self.attach_raw(joystick, device)
        joystick._set_axes_from_device()

        self.assertEqual(bus.active_source, "gamepad")
        self.assertTrue(np.all(np.abs(bus.command()) > 0.0))

        device.axes[:] = (0.0, 0.0, -1.0, 0.0)
        joystick._set_axes_from_device()
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertIsNone(bus.active_source)

        # A fresh keyboard command owns the bus.  Repeated neutral joystick
        # polls must not resurrect the old gamepad command or erase keyboard.
        keyboard.handle_viewer_key(ord("w"))
        expected = bus.command()
        joystick._set_axes_from_device()
        joystick._set_axes_from_device()
        np.testing.assert_allclose(bus.command(), expected)
        self.assertEqual(bus.active_source, "keyboard")

    def test_raw_back6_estop_has_priority_and_start7_is_reset(self) -> None:
        bus = self.make_bus()
        joystick = JoystickInput(bus, prefer_controller=False)
        joystick._controller = False
        bus.set_command(1.0, 0.5, 0.25, source="gamepad")

        joystick._dispatch_button(6)

        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertTrue(bus.consume("estop"))
        self.assertTrue(bus.consume("quit"))
        self.assertFalse(bus.consume("reset"))

        bus.set_command(0.5, 0.0, 0.0, source="gamepad")
        joystick._dispatch_button(7)
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertTrue(bus.consume("reset"))
        self.assertFalse(bus.consume("estop"))
        self.assertFalse(bus.consume("quit"))

    def test_reset_and_back_button_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reset and back/estop overlap"):
            JoystickInput(
                self.make_bus(),
                button_config={"reset": 6, "back": 6},
                prefer_controller=False,
            )

    def test_disconnect_zeros_velocity_and_latches_estop_quit(self) -> None:
        bus = self.make_bus()
        joystick = JoystickInput(bus, prefer_controller=False)
        self.attach_raw(joystick, _MutableJoystick((0.0, -1.0, -1.0, 0.0)))
        joystick._set_axes_from_device()
        self.assertGreater(bus.command()[0], 0.0)

        joystick._close_device(disconnected=True)

        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertFalse(joystick.available)
        self.assertTrue(bus.consume("estop"))
        self.assertTrue(bus.consume("quit"))

    def test_stop_events_and_escape_clear_velocity_without_stale_revival(self) -> None:
        for event in ("disable_rl", "get_down", "reset"):
            with self.subTest(event=event):
                bus = self.make_bus()
                bus.set_command(1.0, 0.5, 0.25, source="keyboard")
                bus.request(event)
                np.testing.assert_allclose(bus.command(), 0.0)
                self.assertIsNone(bus.active_source)

        bus = self.make_bus()
        keyboard = KeyboardInput(bus, keyboard_step=0.2)
        for _ in range(3):
            keyboard.handle_viewer_key(ord("w"))
        self.assertAlmostEqual(bus.command()[0], 0.6)
        keyboard.handle_viewer_key(27)
        np.testing.assert_allclose(bus.command(), 0.0)
        self.assertTrue(bus.consume("estop"))
        self.assertTrue(bus.consume("quit"))

        # The keyboard's legacy cache must not revive the pre-stop 0.6 m/s.
        keyboard.handle_viewer_key(ord("w"))
        np.testing.assert_allclose(bus.command(), [0.2, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
