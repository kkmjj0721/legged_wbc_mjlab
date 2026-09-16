"""Regression tests for the RI-4438 HIM deployment stack.

The tests intentionally keep the FSM checks independent of MuJoCo.  The
MuJoCo and ONNX checks are marked as optional because ``deploy`` can be used
for a backend-only smoke test without installing either runtime dependency.
Run from the repository root with ``python -m unittest deploy.task.ri_4438_him.test_sim2sim``.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deploy.include.config import load_config
from deploy.include.fsm import LocomotionFSM, State
from deploy.include.inputs import CommandBus, JoystickInput, KeyboardInput
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
        self.assertTrue(cfg.policy.exists(), f"configured HIM policy does not exist: {cfg.policy}")
        # The current export is the 15-52-14 run.  Keep this assertion
        # explicit so a stale/broken YAML path cannot silently pass smoke tests.
        self.assertIn("ri_4438_him/2026-09-15_15-52-14/policy.onnx", str(cfg.policy))
        self.assertEqual(cfg.gamepad_options["axis_config"]["left_x"], "LEFTX")
        self.assertEqual(cfg.gamepad_options["button_config"]["enable_rl"], "X")
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
        commands.set_command(0.5, 0.0, 0.0)
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
        np.testing.assert_allclose(bus.command(), [0.0, -0.5, 1.0])

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
        joystick = JoystickInput(bus, axes=(0, 1, 2), deadzone=0.0, prefer_controller=False)
        joystick._device = _FakeJoystick((0.4, -0.6, 0.25))
        joystick._controller = False

        joystick._set_axes_from_device()

        # Forward (negative Y) maps to positive vx; right (positive X) maps
        # to positive vy.  Yaw retains its configured sign and scaling.
        np.testing.assert_allclose(bus.command(), [1.2, 0.4, 0.125])

    def test_raw_joystick_semantic_button_aliases(self) -> None:
        """Raw pygame.Joystick events must understand YAML semantic names.

        A Beitong/XInput-compatible pad may not be recognized by SDL's
        GameController database, in which case ``JOYBUTTONDOWN.button`` is a
        numeric index while the YAML still contains ``A``/``START``/``BACK``.
        Verify the aliases used by the HIM task, including the common
        firmware-dependent Start/Back candidates.
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
        for index in (6, 7, 9):
            self.assertTrue(joystick._button_value(index, "reset"), index)
        for index in (4, 6, 8):
            self.assertTrue(joystick._button_value(index, "back"), index)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
