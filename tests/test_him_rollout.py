"""Regression checks for HIM observation ownership and episode boundaries."""

from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.him_ppo import HIMPPO
from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner
from rsl_rl.storage.him_rollout_storage import HIMRolloutStorage


def observations(offset: float = 0.0) -> TensorDict:
    return TensorDict(
        {
            "actor": torch.arange(36, dtype=torch.float32).reshape(3, 2, 6) + offset,
            "critic": torch.arange(24, dtype=torch.float32).reshape(3, 8) + offset,
        },
        batch_size=[3],
    )


def make_algorithm() -> HIMPPO:
    algorithm = HIMPPO.__new__(HIMPPO)
    algorithm.device = "cpu"
    algorithm.estimator_obs_groups = ["critic"]
    algorithm.actor = SimpleNamespace(obs_groups=["actor"])
    algorithm.storage = HIMRolloutStorage(
        "rl", 3, 1, observations(), [2], next_observation_groups=["critic"]
    )
    return algorithm


def store_transition(storage, obs, next_obs, valid):
    transition = HIMRolloutStorage.Transition()
    transition.observations = obs
    transition.next_observations = next_obs
    transition.next_observations_valid = valid
    transition.actions = torch.zeros(3, 2)
    transition.rewards = torch.arange(3, dtype=torch.float32)
    transition.dones = torch.tensor([False, True, False])
    transition.values = torch.zeros(3, 1)
    transition.actions_log_prob = torch.zeros(3)
    transition.distribution_params = (torch.zeros(3, 2), torch.ones(3, 2))
    storage.add_transition(transition)


class NextObservationTests(unittest.TestCase):
    def setUp(self):
        self.algorithm = make_algorithm()
        self.obs = observations()
        self.dones = torch.tensor([False, True, False])

    def test_manual_reset_targets_are_valid_and_storage_owns_the_copy(self):
        resolved, valid = self.algorithm._resolve_next_observations(
            self.obs, self.dones, {"terminal_observations": self.obs}
        )
        self.assertEqual(set(resolved.keys()), {"critic"})
        self.assertTrue(valid.all())
        self.assertEqual(resolved["critic"].data_ptr(), self.obs["critic"].data_ptr())
        expected = self.obs.clone()
        storage = self.algorithm.storage
        store_transition(storage, self.obs, resolved, valid)
        self.obs["actor"].fill_(-100)
        self.obs["critic"].fill_(-100)
        torch.testing.assert_close(storage.observations[0]["actor"], expected["actor"])
        torch.testing.assert_close(storage.next_observations[0]["critic"], expected["critic"])
        self.assertEqual(set(storage.next_observations.keys()), {"critic"})
        batch = next(storage.mini_batch_generator(1, 1))
        self.assertEqual(set(batch.observations.keys()), {"actor", "critic"})
        self.assertEqual(set(batch.next_observations.keys()), {"critic"})
        self.assertTrue(batch.next_observations_valid.all())
        row_ids = batch.observations["critic"][:, 0] / 8
        torch.testing.assert_close(
            batch.next_observations["critic"], expected["critic"][row_ids.long()]
        )

    def test_auto_reset_without_terminal_observations_excludes_done_rows(self):
        resolved, valid = self.algorithm._resolve_next_observations(self.obs, self.dones, {})
        torch.testing.assert_close(valid.flatten(), ~self.dones)
        torch.testing.assert_close(resolved["critic"], self.obs["critic"])

    def test_full_and_done_only_terminal_batches_preserve_reset_observations(self):
        terminal = observations(100)
        for done_only in (False, True):
            with self.subTest(done_only=done_only):
                supplied = terminal[self.dones] if done_only else terminal
                before = self.obs.clone()
                resolved, valid = self.algorithm._resolve_next_observations(
                    self.obs, self.dones, {"terminal_observations": supplied}
                )
                expected = before["critic"].clone()
                expected[self.dones] = terminal["critic"][self.dones]
                torch.testing.assert_close(resolved["critic"], expected)
                torch.testing.assert_close(self.obs["critic"], before["critic"])
                self.assertTrue(valid.all())

    def test_missing_estimator_group_does_not_validate_terminal_rows(self):
        terminal = observations(100).select("actor")
        resolved, valid = self.algorithm._resolve_next_observations(
            self.obs, self.dones, {"terminal_observations": terminal}
        )
        torch.testing.assert_close(valid.flatten(), ~self.dones)
        torch.testing.assert_close(resolved["critic"], self.obs["critic"])

    def test_no_done_and_all_done(self):
        for done in (False, True):
            with self.subTest(done=done):
                mask = torch.full((3,), done)
                resolved, valid = self.algorithm._resolve_next_observations(
                    self.obs, mask, {"terminal_observations": observations(100)[mask]}
                )
                self.assertTrue(valid.all())
                expected = observations(100 if done else 0)["critic"]
                torch.testing.assert_close(resolved["critic"], expected)

    def test_malformed_terminal_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "terminal observations"):
            self.algorithm._resolve_next_observations(
                self.obs, self.dones, {"terminal_observations": observations(100)[:2]}
            )

    def test_nonstandard_estimator_group_is_supported(self):
        self.algorithm.storage.next_observation_groups = ("privileged",)
        self.algorithm.estimator_obs_groups = ["privileged"]
        obs = TensorDict({"privileged": self.obs["critic"]}, batch_size=[3])
        resolved, valid = self.algorithm._resolve_next_observations(
            obs, self.dones, {"terminal_observations": obs}
        )
        self.assertEqual(set(resolved.keys()), {"privileged"})
        self.assertTrue(valid.all())


class RunnerResetTests(unittest.TestCase):
    def test_partial_reset_keeps_live_rows_and_stored_terminal_state(self):
        algorithm = make_algorithm()
        terminal = observations()
        expected = terminal.clone()
        resolved, valid = algorithm._resolve_next_observations(
            terminal, torch.tensor([False, True, False]), {"terminal_observations": terminal}
        )
        store_transition(algorithm.storage, terminal, resolved, valid)
        reset_obs = observations(100)
        unwrapped = SimpleNamespace(
            cfg=SimpleNamespace(auto_reset=False), device="cpu",
            reset=Mock(return_value=(reset_obs, {"log": {"episode": 1}})),
        )
        runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
        runner.device = "cpu"
        runner.env = SimpleNamespace(unwrapped=unwrapped, num_envs=3)
        merged, extras = runner._reset_done_envs(terminal, torch.tensor([1]))
        self.assertIs(merged, terminal)
        for key in merged.keys():
            torch.testing.assert_close(merged[key][[0, 2]], expected[key][[0, 2]])
            torch.testing.assert_close(merged[key][1], reset_obs[key][1])
        torch.testing.assert_close(algorithm.storage.next_observations[0]["critic"], expected["critic"])
        self.assertEqual(extras["log"], {"episode": 1})

    def test_learn_only_labels_manual_reset_output_as_terminal(self):
        for auto_reset in (False, True):
            with self.subTest(auto_reset=auto_reset):
                runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
                runner.device = "cpu"
                runner.cfg = {"num_steps_per_env": 1, "check_for_nan": False}
                runner.is_distributed = False
                runner.current_learning_iteration = 0
                obs = observations()
                dones = torch.tensor([0, 1, 0])
                runner.env = SimpleNamespace(
                    device="cpu", unwrapped=SimpleNamespace(cfg=SimpleNamespace(auto_reset=auto_reset)),
                    get_observations=lambda: obs,
                    step=lambda _: (obs, torch.ones(3), dones, {}),
                )
                runner.alg = SimpleNamespace(
                    train_mode=Mock(), act=lambda _: torch.zeros(3, 2),
                    process_env_step=Mock(), compute_returns=Mock(), update=lambda: {},
                    intrinsic_rewards=None, learning_rate=0.001,
                    get_policy=lambda: SimpleNamespace(output_std=torch.ones(2)),
                )
                runner.logger = SimpleNamespace(
                    init_logging_writer=Mock(), process_env_step=Mock(), log=Mock(), writer=None,
                )
                runner._reset_done_envs = Mock(return_value=(obs, {}))
                runner.learn(1)
                extras = runner.alg.process_env_step.call_args.args[3]
                self.assertEqual("terminal_observations" in extras, not auto_reset)
                self.assertEqual(runner._reset_done_envs.call_count, int(not auto_reset))


class LearningParityTests(unittest.TestCase):
    def test_critic_only_storage_preserves_ppo_and_estimator_updates(self):
        from src.tasks.locomotion.ri_4438_him.config.rl_cfg import ri_4438_him_runner_cfg
        from src.tasks.locomotion.ri_4438_him.rl.runner import HIMOnPolicyRunner as TaskRunner

        cfg = asdict(ri_4438_him_runner_cfg())
        cfg["num_steps_per_env"] = 2
        cfg["actor"]["hidden_dims"] = (16,)
        cfg["critic"]["hidden_dims"] = (16,)
        cfg["algorithm"]["estimator_obs_groups"] = "critic"
        cfg["algorithm"]["num_learning_epochs"] = 1
        cfg["algorithm"]["num_mini_batches"] = 1
        torch.manual_seed(10)
        frames = [
            TensorDict({"actor": torch.randn(3, 6, 47), "critic": torch.randn(3, 244)}, batch_size=[3])
            for _ in range(3)
        ]
        env = SimpleNamespace(num_envs=3, num_actions=12, cfg={}, get_observations=lambda: frames[0])
        algorithms = []
        for _ in range(2):
            torch.manual_seed(11)
            algorithms.append(TaskRunner(env, cfg, None, "cpu").alg)
        reference, optimized = algorithms
        reference.storage = HIMRolloutStorage("rl", 3, 2, frames[0], [12])
        self.assertEqual(set(optimized.storage.next_observations.keys()), {"critic"})
        with torch.inference_mode():
            for step in range(2):
                for algorithm in algorithms:
                    torch.manual_seed(20 + step)
                    algorithm.act(frames[step])
                    algorithm.process_env_step(
                        frames[step + 1], torch.ones(3), torch.tensor([0, 1, 0]),
                        {"terminal_observations": frames[step + 1]},
                    )
            for algorithm in algorithms:
                algorithm.compute_returns(frames[-1])
        losses = []
        for algorithm in algorithms:
            torch.manual_seed(30)
            losses.append(algorithm.update())
        self.assertEqual(losses[0], losses[1])
        for left, right in zip(reference._all_parameters(), optimized._all_parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
