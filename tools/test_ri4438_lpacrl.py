"""CPU regression checks. Run: .venv/bin/python -m unittest tools.test_ri4438_lpacrl -v"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

from scripts.train import resolve_warm_start
from src.tasks.locomotion.ri_4438_him_lpacrl.config.curriculum_cfg import (
    build_curriculum, make_lpacrl_cfg,
)
from src.tasks.locomotion.ri_4438_him_lpacrl.config.env_cfgs import build_env_cfg
from src.tasks.locomotion.ri_4438_him_lpacrl.mdp.curriculums import LPState
from src.tasks.locomotion.ri_4438_him_lpacrl.mdp.terrains import _base
from src.tasks.locomotion.ri_4438_him_lpacrl.rl import evaluator as evaluator_module
from src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluator import TaskEvaluator, summarize_trials
from src.tasks.locomotion.ri_4438_him_lpacrl.rl.metrics import (
    annotate_progress, epte_sp, relative_tracking_error, reward_terms_for_step,
)
from src.tasks.locomotion.ri_4438_him_lpacrl.rl.runner import LPACRLRunner


def evaluation(returns):
    a = np.asarray(returns, dtype=float)
    return {
        "mean_returns": a.mean(axis=1).tolist(),
        "counts": [a.shape[1]] * a.shape[0],
        "trial_returns": a.tolist(),
        "per_task": [{"task_id": i} for i in range(len(a))],
        "macro": {},
    }


def state(n=3, **overrides):
    cfg = dict(method="lp", temperature=1.0, uniform_mix=0.0)
    cfg.update(overrides)
    return LPState(n, NS(**cfg), {"test": "fixed"})


class CurriculumTests(unittest.TestCase):
    def test_signed_lp_and_mixture(self):
        s = state(uniform_mix=0.15)
        s.commit_evaluation(evaluation([[10, 10]] * 3), 0, 2)
        np.testing.assert_allclose(s.probabilities, 1 / 3)
        s.commit_evaluation(evaluation([[12, 12], [9, 9], [10, 10]]), 50, 2)
        np.testing.assert_array_equal(s.lp, [2, -1, 0])
        softmax = np.exp([2, -1, 0]); softmax /= softmax.sum()
        np.testing.assert_allclose(s.probabilities, 0.85 * softmax + 0.05)

    def test_uniform_baseline_and_alp_remain_available(self):
        for method, expected in (("uniform", [1 / 3] * 3), ("alp", None)):
            s = state(method=method)
            s.commit_evaluation(evaluation([[10, 10]] * 3), 0, 2)
            s.commit_evaluation(evaluation([[12, 12], [7, 7], [10, 10]]), 1, 2)
            if expected is None:
                expected = np.exp([2, 3, 0]); expected /= expected.sum()
            np.testing.assert_allclose(s.probabilities, expected)
            self.assertEqual(s.diagnostics()["lp_logit_range"], 0 if method == "uniform" else 3)

    def test_failed_evaluation_does_not_mutate_baseline(self):
        s = state()
        s.commit_evaluation(evaluation([[10, 10]] * 3), 0, 2)
        original = deepcopy(s.state_dict())
        for corrupt in ("nonfinite", "count", "mean", "shape"):
            r = evaluation([[11, 11]] * 3)
            if corrupt == "nonfinite": r["trial_returns"][0][0] = float("nan")
            if corrupt == "count": r["counts"][0] = 1
            if corrupt == "mean": r["mean_returns"][0] = 12
            if corrupt == "shape": r["trial_returns"].pop()
            with self.assertRaises(ValueError): s.commit_evaluation(r, 50, 2)
            np.testing.assert_array_equal(s.trial_returns, original["trial_returns"])
            np.testing.assert_array_equal(s.probabilities, original["probabilities"])
            self.assertEqual(s.last_evaluated_update, 0)

    def test_checkpoint_preserves_paired_baseline(self):
        s = state()
        s.commit_evaluation(evaluation([[1, 3], [4, 6], [-3, 3]]), 0, 2)
        restored = state()
        restored.load_state_dict(s.state_dict())
        r = evaluation([[2, 4], [3, 5], [0, 0]])
        annotate_progress(r, restored.trial_returns)
        self.assertEqual(r["per_task"][0]["lp_standard_error"], 0)
        self.assertAlmostEqual(r["per_task"][2]["lp_standard_error"], 3)
        restored.commit_evaluation(r, 50, 2)
        np.testing.assert_array_equal(restored.lp, [1, -1, 0])

    def test_corrupt_or_old_checkpoint_rejected_atomically(self):
        s = state()
        s.commit_evaluation(evaluation([[1, 3]] * 3), 0, 2)
        for field in ("version", "trial_returns", "fingerprints"):
            saved = deepcopy(s.state_dict())
            saved[field] = {"version": 1, "trial_returns": [[0, 0]] * 3,
                            "fingerprints": {"test": "changed"}}[field]
            with self.assertRaises(ValueError): s.load_state_dict(saved)
            np.testing.assert_array_equal(s.trial_returns, [[1, 3]] * 3)

    def test_outlier_diagnostic_does_not_change_raw_lp(self):
        old = np.zeros((1, 8)); old[0, 7] = -800
        r = evaluation(np.zeros((1, 8)))
        annotate_progress(r, old)
        row = r["per_task"][0]
        self.assertEqual(row["lp"], 100)
        self.assertAlmostEqual(row["lp_standard_error"], 100)
        self.assertEqual(row["lp_largest_trial_abs_share"], 1)
        self.assertEqual(row["lp_without_largest_trial"], 0)
        self.assertEqual(r["macro"]["lp_positive_outlier_sensitive_tasks"], 1)
        s = state(1); s.commit_evaluation(evaluation(old), 0, 8)
        s.commit_evaluation(r, 50, 8)
        self.assertEqual(s.lp[0], 100)

    def test_uncertainty_is_unknown_with_only_one_trial(self):
        r = evaluation([[1]])
        annotate_progress(r, None)
        self.assertFalse(r["paired_progress_available"])
        annotate_progress(r, [[0]])
        self.assertIsNone(r["per_task"][0]["lp_standard_error"])
        self.assertIsNone(r["per_task"][0]["lp_without_largest_trial"])

    def test_temperature_and_floor_reduce_synthetic_concentration(self):
        cfg = make_lpacrl_cfg().curriculum
        s = LPState(600, cfg)
        baseline = np.zeros((600, 8))
        s.commit_evaluation(evaluation(baseline), 0, 8)
        current = baseline.copy(); current[0] = 90
        s.commit_evaluation(evaluation(current), 50, 8)
        self.assertLess(s.probabilities.max(), 0.06)
        self.assertGreaterEqual(s.probabilities.min(), cfg.uniform_mix / 600)
        self.assertGreater(s.diagnostics()["effective_tasks"], 200)


class MetricTests(unittest.TestCase):
    def test_perfect_timeout_has_no_survival_penalty(self):
        np.testing.assert_array_equal(epte_sp([0, 0, 0], 1000, 1000), [0, 0, 0])

    def test_immediate_fall_cannot_look_like_perfect_tracking(self):
        np.testing.assert_array_equal(epte_sp([0, 0, 0], 0, 1000), [1, 1, 1])
        np.testing.assert_allclose(epte_sp([0, 0, 0], 10, 1000), 0.99)

    def test_tracking_and_survival_penalties_combine(self):
        np.testing.assert_allclose(epte_sp([100, 5000, 0], 500, 1000), [0.6, 1, 0.5])

    def test_variable_sign_and_zero_commands_use_floors(self):
        command = torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
        actual = torch.tensor([[0.5, 0.05, -0.5], [-0.5, 0.0, 0.1]])
        error = relative_tracking_error(actual, command, torch.tensor([0.1] * 3))
        torch.testing.assert_close(error, torch.tensor([[0.5, 0.5, 0.5], [0.5, 0, 1.0]]))

    def test_weighted_reward_rate_is_scaled_exactly_once(self):
        env = NS(step_dt=0.02, cfg=NS(scale_rewards_by_dt=True),
                 reward_manager=NS(_step_reward=torch.tensor([[2.0, -100.0]])))
        torch.testing.assert_close(reward_terms_for_step(env), torch.tensor([[0.04, -2.0]], dtype=torch.float64))
        env.cfg.scale_rewards_by_dt = False
        torch.testing.assert_close(reward_terms_for_step(env), torch.tensor([[2.0, -100.0]], dtype=torch.float64))

    def test_installed_reward_manager_matches_decomposition(self):
        for use_dt in (False, True):
            env = NS(num_envs=2, device="cpu", step_dt=0.02,
                     cfg=NS(scale_rewards_by_dt=use_dt))
            terms = {
                "tracking": RewardTermCfg(func=lambda env: torch.tensor([0.75, 1.0]), weight=2.0),
                "penalty": RewardTermCfg(func=lambda env: torch.tensor([1.0, 0.0]), weight=-100.0),
            }
            env.reward_manager = RewardManager(terms, env, scale_by_dt=use_dt)
            reward = env.reward_manager.compute(env.step_dt)
            parts = reward_terms_for_step(env)
            torch.testing.assert_close(parts.sum(dim=1), reward.double())
            torch.testing.assert_close(parts[:, 0], torch.tensor([1.5, 2.0], dtype=torch.float64) * (0.02 if use_dt else 1))


class GeometryAndConfigTests(unittest.TestCase):
    def test_real_slope_geometry_has_meaningful_monotonic_height(self):
        setup = build_curriculum(make_lpacrl_cfg())
        self.assertEqual(setup.catalog.num_tasks, 600)
        for family in ("slope_outward_up", "slope_outward_down"):
            heights = []
            for level, gradient in (("l1", 0.1), ("l2", 0.2), ("l3", 0.3)):
                i = setup.terrain_map.columns.index((family, level))
                params = setup.terrain_map.parameters[i]
                self.assertEqual(params["gradient"], gradient)
                spec = mujoco.MjSpec(); spec.worldbody.add_body(name="terrain")
                terrain = _base(family, params, (8.0, 8.0)).function(0.0, spec, np.random.default_rng(1))
                heights.append(float(terrain.geometries[0].hfield.size[2]))
            self.assertGreater(heights[0], 0.15)
            self.assertGreater(heights[2], 0.6)
            self.assertTrue(all(a < b for a, b in zip(heights, heights[1:])))

    def test_actor_remains_blind(self):
        cfg = build_env_cfg(make_lpacrl_cfg(), "evaluation")
        self.assertNotIn("height_scan", cfg.observations["actor"].terms)
        self.assertIn("height_scan", cfg.observations["critic"].terms)
        self.assertNotIn("height_scan", cfg.observations["estimator"].terms)
        self.assertEqual(cfg.observations["actor"].history_length, 6)
        self.assertFalse(cfg.auto_reset)

    def test_evaluation_and_geometry_changes_invalidate_resume(self):
        cfg = make_lpacrl_cfg(); original = build_curriculum(cfg).fingerprints
        cfg.task_space.terrains["slope_outward_up"].levels["l3"]["gradient"] = 0.005
        changed = build_curriculum(cfg).fingerprints
        self.assertNotEqual(original["geometry"], changed["geometry"])
        cfg = make_lpacrl_cfg(); cfg.evaluation.relative_error_floors = (0.2, 0.1, 0.1)
        self.assertNotEqual(original["evaluation"], build_curriculum(cfg).fingerprints["evaluation"])

    def test_invalid_metric_config_rejected(self):
        for name, value in (("relative_error_floors", (0, 0.1, 0.1)),
                            ("relative_error_floors", (0.1,)),
                            ("success_alive_fraction", 1), ("success_tracking_error", float("nan")),
                            ("protocol_version", 2)):
            cfg = make_lpacrl_cfg(); setattr(cfg.evaluation, name, value)
            with self.assertRaises(ValueError): build_curriculum(cfg)


class FakeTraversal:
    def __init__(self, *args): pass
    def observe(self, *args): pass
    def reset_segments(self, *args): pass
    def trial(self, slot):
        return dict(distance_m=0.1, stair_contact_seconds=0.0, stalled_seconds=0.0)


class FakeEnv:
    """Real evaluator loop: boundary reset, physical failure, timeout, padding."""
    num_envs = 2
    device = "cpu"
    step_dt = 0.02

    def __init__(self):
        self.cfg = NS(scale_rewards_by_dt=True)
        self.controller = NS(evaluation_step_offsets=torch.zeros(2, dtype=torch.long),
                             commands=torch.tensor([[1.0, 0, 0]] * 2))
        self.robot = NS(data=NS(root_link_lin_vel_b=torch.zeros(2, 3),
                               root_link_ang_vel_b=torch.zeros(2, 3)))
        self.scene = {"robot": self.robot}
        self.reward_manager = NS(active_terms=["tracking", "termination"],
                                 _step_reward=torch.zeros(2, 2))
        self.boundary = torch.zeros(2, dtype=torch.bool)
        self.termination_manager = NS(active_terms=["boundary_truncation"],
                                      get_term=lambda _: self.boundary)
        self.partial_resets = 0

    def obs(self): return {"actor": torch.zeros(2, 1)}

    def reset(self, env_ids=None):
        if env_ids is None: self.batch_step = 0
        else: self.partial_resets += 1
        # Poison reset state; metrics must already have consumed pre-reset data.
        self.robot.data.root_link_lin_vel_b.fill_(-999)
        self.reward_manager._step_reward.zero_()
        return self.obs(), {}

    def step(self, actions):
        self.batch_step += 1
        self.robot.data.root_link_lin_vel_b.copy_(self.controller.commands)
        self.boundary[:] = torch.tensor([self.batch_step == 1, False])
        terminated = torch.tensor([False, self.batch_step == 2])
        truncated = self.boundary | (torch.ones(2, dtype=torch.bool) & (self.batch_step == 3))
        self.reward_manager._step_reward[:] = torch.tensor([[1.0, 0.0]] * 2)
        self.reward_manager._step_reward[terminated, 1] = -5
        rewards = self.reward_manager._step_reward.sum(dim=1) * self.step_dt
        return self.obs(), rewards, terminated, truncated, {}


class EvaluationLoopTests(unittest.TestCase):
    def run_evaluation(self):
        cfg = make_lpacrl_cfg()
        cfg.task_space.include = [{"terrain": "slope_outward_up", "level": "l0", "vx": "vx0", "wz": "wz0"}]
        cfg.evaluation.trials_per_task = 3; cfg.evaluation.horizon_s = 0.06
        setup = build_curriculum(cfg)
        evaluator = TaskEvaluator(setup, "cpu"); evaluator.env = FakeEnv()
        with patch.object(evaluator_module, "get_controller", lambda env: env.controller), patch(
            "src.tasks.locomotion.ri_4438_him_lpacrl.mdp.terrains.TraversalMetrics", FakeTraversal
        ):
            result = evaluator._evaluate_frozen(lambda obs: torch.zeros(2, 12))
        return result, setup, evaluator.env

    def test_full_evaluation_boundary_failure_timeout_and_padding(self):
        result, _, env = self.run_evaluation()
        self.assertGreater(env.partial_resets, 0)
        self.assertEqual(result["counts"], [3])
        np.testing.assert_allclose(result["trial_returns"], [[0.06, -0.06, 0.06]], atol=1e-7)
        rows = sorted(result["trials"], key=lambda r: r["trial_id"])
        self.assertEqual([r["alive_steps"] for r in rows], [3, 1, 3])
        self.assertEqual([r["vx_epte_sp"] for r in rows], [0, 2 / 3, 0])
        self.assertAlmostEqual(result["macro"]["success_rate"], 2 / 3)
        self.assertEqual(rows[0]["boundary_crossings"], 1)
        self.assertAlmostEqual(rows[0]["reward_terms"]["tracking"], 0.06)
        self.assertAlmostEqual(rows[1]["reward_terms"]["termination"], -0.1)

    def test_completion_order_does_not_change_pairing(self):
        result, setup, _ = self.run_evaluation()
        reordered = summarize_trials(list(reversed(result["trials"])), setup.catalog, 3)
        self.assertEqual(reordered["trial_returns"], result["trial_returns"])
        annotate_progress(reordered, result["trial_returns"])
        self.assertEqual(reordered["per_task"][0]["lp"], 0)
        self.assertEqual(reordered["per_task"][0]["lp_standard_error"], 0)

    def test_invalid_or_incomplete_results_rejected(self):
        result, setup, _ = self.run_evaluation()
        with self.assertRaises(ValueError): summarize_trials(result["trials"][:-1], setup.catalog, 3)
        rows = deepcopy(result["trials"]); rows[0]["reward_terms"]["tracking"] += 1
        with self.assertRaises(ValueError): summarize_trials(rows, setup.catalog, 3)


class WarmStartTests(unittest.TestCase):
    def test_cli_rejects_resume_conflict_unsupported_runner_and_missing_file(self):
        cfg = NS(warm_start="missing.pt", agent=NS(resume=True))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"): resolve_warm_start(cfg, LPACRLRunner)
        cfg.agent.resume = False
        with self.assertRaisesRegex(ValueError, "does not support"): resolve_warm_start(cfg, object)
        with self.assertRaises(FileNotFoundError): resolve_warm_start(cfg, LPACRLRunner)
        cfg.warm_start = None
        self.assertIsNone(resolve_warm_start(cfg, object))

    def test_warm_start_loads_weights_without_optimizer_or_lp_history(self):
        runner = LPACRLRunner.__new__(LPACRLRunner)
        runner.device = "cpu"; runner.controller = NS(mode="train", state=state(1))
        runner._numerical_contract = lambda: {"test": 1}
        runner._validate_numerics = Mock()
        runner.alg = NS(load=Mock())
        runner.env = NS(unwrapped=NS(common_step_counter=99), reset=Mock())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save({"infos": {"him_numerics": {"test": 1}, "lpacrl": {"old": True}},
                        "actor_state_dict": {}, "critic_state_dict": {},
                        "optimizer_state_dict": {"old": True},
                        "estimator_optimizer_state_dict": {"old": True}}, path)
            cfg = NS(warm_start=str(path), agent=NS(resume=False))
            self.assertEqual(resolve_warm_start(cfg, LPACRLRunner), path.resolve())
            runner.warm_start(path)
        checkpoint, load_cfg = runner.alg.load.call_args.args
        self.assertNotIn("optimizer_state_dict", checkpoint)
        self.assertNotIn("estimator_optimizer_state_dict", checkpoint)
        self.assertFalse(load_cfg["optimizer"]); self.assertFalse(load_cfg["iteration"])
        self.assertTrue(load_cfg["actor"]); self.assertTrue(load_cfg["estimator"])
        self.assertTrue(load_cfg["critic"])
        self.assertIsNone(runner.controller.state.trial_returns)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(runner.env.unwrapped.common_step_counter, 0)
        runner.env.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
