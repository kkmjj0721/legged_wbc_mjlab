"""GPU checks for fresh terminal supervision and partial automatic resets."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict


@unittest.skipUnless(torch.cuda.is_available(), "MuJoCo Warp CUDA integration test")
class TerminalEstimatorTests(unittest.TestCase):
    terrain = "rough"

    @classmethod
    def setUpClass(cls):
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from src.tasks.locomotion.ri_4438_him.config.env_cfgs import ri_4438_rough_env_cfg, ri_4438_flat_env_cfg
        from src.tasks.locomotion.ri_4438_him.config.rl_cfg import ri_4438_him_runner_cfg
        from src.tasks.locomotion.ri_4438_him.rl.runner import HIMOnPolicyRunner

        cfg = ri_4438_rough_env_cfg() if cls.terrain == "rough" else ri_4438_flat_env_cfg()
        cfg.scene.num_envs = 8
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.num_rows = 2
            cfg.scene.terrain.terrain_generator.seed = 41
        cfg.seed = 41
        cfg.episode_length_s = 2.0
        cls.env = ManagerBasedRlEnv(cfg, device="cuda:0")
        cls.wrapped = RslRlVecEnvWrapper(cls.env)
        agent = ri_4438_him_runner_cfg()
        agent.num_steps_per_env = 2
        cls.runner = HIMOnPolicyRunner(cls.wrapped, asdict(agent), None, "cuda:0")
        cls.recorder = cls.env.recorder_manager.get_term("him_terminal")

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        inference = torch.inference_mode()
        inference.__enter__()
        self.addCleanup(inference.__exit__, None, None, None)
        self.env.cfg.auto_reset = True
        self.env.reset(seed=41)
        self.runner.alg.storage.clear()

    def test_compact_observation_matches_full_critic(self):
        for _ in range(3):
            obs, *_ = self.env.step(torch.randn(8, 12, device="cuda:0") * 0.2)
            torch.testing.assert_close(obs["estimator"], obs["critic"][:, :50], atol=0, rtol=0)

    def test_terminal_refresh_matches_full_forward_after_velocity_push(self):
        self.env.cfg.auto_reset = False
        robot = self.env.scene["robot"]
        for _ in range(3):
            self.env.step(torch.randn(8, 12, device="cuda:0") * 0.2)
            robot.write_root_link_velocity_to_sim(torch.randn(8, 6, device="cuda:0"))
            before = self.env.observation_manager.compute_group("estimator").clone()
            self.recorder._refresh_terminal_state()
            actual = self.env.observation_manager.compute_group("estimator").clone()
            self.env.sim.forward()
            self.env.sim.sense()
            expected = self.env.observation_manager.compute_group("critic", update_history=True)[:, :50]
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            self.assertGreater((actual[:, 47:50] - before[:, 47:50]).abs().max().item(), 0.01)

    def test_partial_reset_preserves_targets_and_advances_live_history_once(self):
        for _ in range(2):
            self.env.step(torch.zeros(8, 12, device="cuda:0"))
        buffers = self.env.observation_manager._group_obs_term_history_buffer["actor"]
        histories = {key: value.buffer.clone() for key, value in buffers.items()}
        delays = self.env.observation_manager._group_obs_term_delay_buffer["actor"]
        delay_steps = {key: value._step_count.clone() for key, value in delays.items()}
        self.env.episode_length_buf.zero_()
        self.env.episode_length_buf[:2] = self.env.max_episode_length - 1
        command = self.env.command_manager.get_term("twist")
        command.time_left.fill_(5.0)
        before_timers = command.time_left.clone()
        obs = self.wrapped.get_observations()
        with torch.inference_mode():
            action = self.runner.alg.act(obs)
            with patch.object(self.env.sim, "forward", wraps=self.env.sim.forward) as forward, \
                 patch.object(self.env.sim, "sense", wraps=self.env.sim.sense) as sense:
                obs, rewards, dones, extras = self.wrapped.step(action)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(sense.call_count, 1)
            self.assertTrue(dones[:2].all())
            live = ~dones.bool()
            self.assertTrue(live.any())
            terminal = extras["terminal_observations"]["estimator"].clone()
            self.assertGreater((terminal - obs["estimator"][~live]).abs().max().item(), 0.01)
            self.runner.alg.process_env_step(obs, rewards, dones, extras)
            saved = self.runner.alg.storage.next_observations[0]["estimator"]
            torch.testing.assert_close(saved[~live], terminal, atol=0, rtol=0)
            torch.testing.assert_close(saved[live], obs["estimator"][live], atol=0, rtol=0)
            self.assertTrue(self.runner.alg.storage.next_observations_valid[0].all())
            for key, buffer in buffers.items():
                torch.testing.assert_close(buffer.buffer[live, :-1], histories[key][live, 1:], atol=0, rtol=0)
            for key, delay in delays.items():
                torch.testing.assert_close(delay._step_count[live], delay_steps[key][live] + 1)
            torch.testing.assert_close(command.time_left[live], before_timers[live] - self.env.step_dt)
            _, _, _, _, extras = self.env.step(torch.zeros(8, 12, device="cuda:0"))
            self.assertNotIn("terminal_observations", extras)
            self.env.reset()
            self.assertNotIn("terminal_observations", self.env.extras)
            torch.testing.assert_close(saved[~live], terminal, atol=0, rtol=0)
            _, _, _, _, extras = self.env.step(torch.zeros(8, 12, device="cuda:0"))
            self.assertNotIn("terminal_observations", extras)

    def test_graph_is_recreated_after_model_recapture(self):
        self.recorder._refresh_terminal_state()
        old_graph = self.recorder._refresh_graph
        self.env.sim.create_graph()
        self.recorder._refresh_terminal_state()
        self.assertIsNot(self.recorder._refresh_graph, old_graph)
        self.assertIs(self.recorder._source_graph, self.env.sim.step_graph)


class FlatTerminalEstimatorTests(TerminalEstimatorTests):
    terrain = "flat"


class CompactSupervisionTests(unittest.TestCase):
    def test_compact_targets_preserve_updates_and_checkpoint_shapes(self):
        from src.tasks.locomotion.ri_4438_him.config.rl_cfg import ri_4438_him_runner_cfg
        from src.tasks.locomotion.ri_4438_him.rl.runner import HIMOnPolicyRunner
        from types import SimpleNamespace

        torch.manual_seed(42)
        frames = []
        for _ in range(3):
            critic = torch.randn(3, 244)
            frames.append(TensorDict({
                "actor": torch.randn(3, 6, 47), "critic": critic,
                "estimator": critic[:, :50].clone(),
            }, batch_size=[3]))
        env = SimpleNamespace(num_envs=3, num_actions=12, cfg={}, get_observations=lambda: frames[0])
        algorithms = []
        for group in ("critic", "estimator"):
            cfg = asdict(ri_4438_him_runner_cfg())
            cfg["num_steps_per_env"] = 2
            cfg["actor"]["hidden_dims"] = (16,)
            cfg["critic"]["hidden_dims"] = (16,)
            cfg["algorithm"].update(estimator_obs_groups=(group,), num_learning_epochs=1, num_mini_batches=1)
            torch.manual_seed(43)
            algorithms.append(HIMOnPolicyRunner(env, cfg, None, "cpu").alg)
        reference, compact = algorithms
        compact.load(reference.save(), load_cfg=None, strict=True)
        with torch.inference_mode():
            for step in range(2):
                for algorithm in algorithms:
                    torch.manual_seed(44 + step)
                    algorithm.act(frames[step])
                    algorithm.process_env_step(
                        frames[step + 1], torch.ones(3), torch.tensor([0, 1, 0]),
                        {"terminal_observations": frames[step + 1]},
                    )
            for algorithm in algorithms:
                algorithm.compute_returns(frames[-1])
        losses = []
        for algorithm in algorithms:
            torch.manual_seed(46)
            losses.append(algorithm.update())
        self.assertEqual(losses[0], losses[1])
        for left, right in zip(reference._all_parameters(), compact._all_parameters(), strict=True):
            torch.testing.assert_close(left, right, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
