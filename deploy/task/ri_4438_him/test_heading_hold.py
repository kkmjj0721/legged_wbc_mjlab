"""Focused regression tests for deployment heading hold.

These tests use the deployment stack's in-memory backend and never start
MuJoCo, a viewer, or robot hardware.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from deploy.include.config import load_config
from deploy.include.fsm import LocomotionFSM, State
from deploy.include.inputs import CommandBus
from deploy.task.ri_4438_him.test_sim2sim import (
    CONFIG_PATH,
    _FakeBackend,
    _RecordingPolicy,
)


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float64,
    )


class HeadingHoldTest(unittest.TestCase):
    def make_stack(self):
        cfg = load_config(CONFIG_PATH, "ri_4438_him")
        cfg.auto_stand = False
        backend = _FakeBackend(cfg)
        commands = CommandBus(cfg.command_limit, cfg.command_deadzone)
        policy = _RecordingPolicy(cfg.action_dim)
        fsm = LocomotionFSM(cfg, backend, policy, commands)
        backend.position = cfg.standing_pos.copy()
        backend.quaternion = cfg.standing_quat.copy()
        backend.joint_position = cfg.default_joint.copy()
        backend.joint_velocity.fill(0.0)
        return cfg, backend, commands, fsm, policy

    @staticmethod
    def set_yaw(backend: _FakeBackend, yaw: float) -> None:
        backend.quaternion = _yaw_quaternion(yaw)

    def test_signed_correction_saturation_and_wrapping(self) -> None:
        _, backend, _, fsm, _ = self.make_stack()
        fsm.state = State.RL
        command = np.array([0.5, 0.1, 0.0])

        first = fsm._apply_heading_hold(backend.read_state(), command)
        self.assertAlmostEqual(first[2], 0.0)
        self.assertAlmostEqual(fsm._heading_ref, 0.0)

        self.set_yaw(backend, 0.2)
        corrected = fsm._apply_heading_hold(backend.read_state(), command)
        self.assertAlmostEqual(corrected[2], -0.2)

        fsm._heading_ref = 0.0
        self.set_yaw(backend, -0.25)
        corrected = fsm._apply_heading_hold(backend.read_state(), command)
        self.assertAlmostEqual(corrected[2], 0.25)

        fsm._heading_ref = 0.0
        self.set_yaw(backend, 1.0)
        saturated = fsm._apply_heading_hold(backend.read_state(), command)
        self.assertAlmostEqual(saturated[2], -0.4)

        fsm._heading_ref = math.pi - 0.05
        self.set_yaw(backend, -math.pi + 0.05)
        wrapped = fsm._apply_heading_hold(backend.read_state(), command)
        self.assertAlmostEqual(wrapped[2], -0.1)

    def test_manual_turn_has_priority_and_does_not_alias_input(self) -> None:
        _, backend, _, fsm, _ = self.make_stack()
        fsm.state = State.RL
        fsm._heading_ref = 0.3
        operator_command = np.array([0.6, -0.2, 0.25])
        original = operator_command.copy()

        output = fsm._apply_heading_hold(backend.read_state(), operator_command)

        np.testing.assert_array_equal(output, original)
        np.testing.assert_array_equal(operator_command, original)
        self.assertFalse(np.shares_memory(output, operator_command))
        self.assertIsNone(fsm._heading_ref)
        output[0] = -1.0
        np.testing.assert_array_equal(operator_command, original)

    def test_turn_release_reanchors_at_current_yaw(self) -> None:
        _, backend, _, fsm, _ = self.make_stack()
        fsm.state = State.RL
        fsm._heading_ref = 0.0
        self.set_yaw(backend, 0.6)

        turning = fsm._apply_heading_hold(
            backend.read_state(), np.array([0.4, 0.2, 0.3])
        )
        np.testing.assert_allclose(turning, [0.4, 0.2, 0.3])
        self.assertIsNone(fsm._heading_ref)

        released = fsm._apply_heading_hold(
            backend.read_state(), np.array([0.4, 0.2, 1.0e-4])
        )
        self.assertAlmostEqual(released[2], 0.0)
        self.assertAlmostEqual(fsm._heading_ref, 0.6)

        self.set_yaw(backend, 0.7)
        corrected = fsm._apply_heading_hold(
            backend.read_state(), np.array([0.4, 0.2, 0.0])
        )
        self.assertAlmostEqual(corrected[2], -0.1)

    def test_forward_hysteresis_and_disable_conditions(self) -> None:
        _, backend, _, fsm, _ = self.make_stack()
        fsm.state = State.RL

        fsm._apply_heading_hold(backend.read_state(), np.array([0.149, 0.0, 0.0]))
        self.assertIsNone(fsm._heading_ref)
        fsm._apply_heading_hold(backend.read_state(), np.array([0.15, 0.0, 0.0]))
        self.assertIsNotNone(fsm._heading_ref)

        self.set_yaw(backend, 0.1)
        held = fsm._apply_heading_hold(
            backend.read_state(), np.array([0.10, 0.0, 0.0])
        )
        self.assertAlmostEqual(held[2], -0.1)
        self.assertIsNotNone(fsm._heading_ref)
        released = fsm._apply_heading_hold(
            backend.read_state(), np.array([0.099, 0.0, 0.0])
        )
        np.testing.assert_allclose(released, [0.099, 0.0, 0.0])
        self.assertIsNone(fsm._heading_ref)

        for state, vx in ((State.RL, -0.3), (State.RL, 0.0), (State.STAND, 0.5)):
            with self.subTest(state=state, vx=vx):
                fsm.state = state
                fsm._heading_ref = 0.2
                command = np.array([vx, 0.1, 0.0])
                output = fsm._apply_heading_hold(backend.read_state(), command)
                np.testing.assert_array_equal(output, command)
                self.assertIsNone(fsm._heading_ref)

    def test_invalid_orientation_releases_hold_and_existing_fault_path_runs(self) -> None:
        _, backend, commands, fsm, _ = self.make_stack()
        fsm.state = State.RL
        command = np.array([0.4, 0.1, 0.0])
        invalid_quaternions = (
            np.array([np.nan, 0.0, 0.0, 1.0]),
            np.zeros(4),
            np.ones(3),
        )
        for quaternion in invalid_quaternions:
            with self.subTest(quaternion=quaternion):
                state = backend.read_state()
                state.quaternion = quaternion
                fsm._heading_ref = 0.2
                output = fsm._apply_heading_hold(state, command)
                np.testing.assert_array_equal(output, command)
                self.assertIsNone(fsm._heading_ref)

        backend.quaternion = np.array([np.nan, 0.0, 0.0, 1.0])
        commands.set_command(0.4, 0.0, 0.0)
        fsm.step()
        self.assertEqual(fsm.state, State.FAULT)
        self.assertIsNone(fsm._heading_ref)

    def test_configuration_can_lower_yaw_limit(self) -> None:
        cfg, backend, _, fsm, _ = self.make_stack()
        cfg.command_limit[2] = 0.17
        fsm.state = State.RL
        fsm._heading_ref = 0.0
        self.set_yaw(backend, 1.0)

        output = fsm._apply_heading_hold(
            backend.read_state(), np.array([0.5, 0.0, 0.0])
        )

        self.assertAlmostEqual(output[2], -0.17)

    def test_step_sends_corrected_command_to_policy_and_reset_clears_hold(self) -> None:
        cfg, backend, commands, fsm, policy = self.make_stack()
        fsm.state = State.RL
        commands.set_command(0.5, 0.2, 0.0)

        fsm.step()
        self.assertAlmostEqual(fsm._heading_ref, 0.0)
        self.assertGreaterEqual(len(policy.observations), 1)

        self.set_yaw(backend, 0.2)
        calls_before = len(policy.observations)
        decimation = max(1, int(round(cfg.control_dt / backend.timestep)))
        for _ in range(decimation):
            # The shared fake backend snaps its scripted pose to standing on
            # every position command. Reapply the measured yaw so the sensor
            # input remains offset through the policy decimation interval.
            self.set_yaw(backend, 0.2)
            fsm.step()
            if len(policy.observations) > calls_before:
                break

        self.assertGreater(len(policy.observations), calls_before)
        newest_frame = policy.observations[-1].reshape(cfg.history_size, -1)[0]
        np.testing.assert_allclose(newest_frame[6:9], [0.5, 0.2, -0.2])

        commands.request("reset")
        policy_calls_before_reset = len(policy.observations)
        fsm.step()
        self.assertEqual(fsm.state, State.PASSIVE)
        self.assertEqual(backend.reset_count, 1)
        self.assertIsNone(fsm._heading_ref)
        np.testing.assert_allclose(fsm.command, 0.0)
        self.assertEqual(len(policy.observations), policy_calls_before_reset)
        self.assertFalse(commands.stop_pending)


if __name__ == "__main__":
    unittest.main()
