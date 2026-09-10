"""Fast regression checks for the modular RI-4438 deployment stack."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deploy.include.config import load_config
from deploy.include.fsm import LocomotionFSM, State
from deploy.include.inputs import CommandBus
from deploy.include.mujoco_bridge import MujocoBackend
from deploy.include.policy import OnnxPolicy


class Sim2SimTest(unittest.TestCase):
    def make_stack(self):
        config = load_config(
            ROOT / "deploy/task/ri_4438_ppo/config/sim2sim.yaml",
            "ri_4438_ppo",
            policy_override=None,
        )
        config.policy = None
        backend = MujocoBackend(config)
        commands = CommandBus(config.command_limit, config.command_deadzone)
        fsm = LocomotionFSM(config, backend, OnnxPolicy(None, config.observation_dim, config.action_dim), commands)
        return config, backend, commands, fsm

    def test_runtime_model_and_observation(self) -> None:
        config, backend, _, fsm = self.make_stack()
        try:
            self.assertEqual(backend.model.nu, config.action_dim)
            self.assertEqual(fsm.observation(backend.read_state()).shape, (config.observation_dim,))
            self.assertTrue(np.all(np.isfinite(fsm.observation(backend.read_state()))))
            self.assertEqual(fsm.state, State.PASSIVE)
        finally:
            backend.close()

    def test_getup_transition(self) -> None:
        config, backend, commands, fsm = self.make_stack()
        try:
            commands.request("stand")
            for _ in range(int(config.stand_duration / config.timestep) + 20):
                fsm.step()
            self.assertEqual(fsm.state, State.RL)
            self.assertTrue(np.all(np.isfinite(backend.data.qpos)))
        finally:
            backend.close()

    def test_command_deadzone_and_limits(self) -> None:
        commands = CommandBus(np.array([2.0, 1.0, 1.0]), 0.1)
        commands.set_axes(0.05, 2.0, -2.0)
        np.testing.assert_allclose(commands.command(), [0.0, 1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
