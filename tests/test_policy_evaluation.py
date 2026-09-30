"""Benchmark accounting, candidate selection and report publication tests (CPU)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from tensordict import TensorDict

from src.utils.policy_evaluation import (
    EpisodeMetrics, EvalOptions, ParallelActors, make_env_cfg, plain, select_candidates, summarize, validate_interface,
)
from src.utils.evaluation_report import publish_evaluation_best
from src.utils.training_logs import Run, sha256


class AccountingTests(unittest.TestCase):
    def update(self, metrics, velocity=0, terminated=(False, False), truncated=(False, False), actions=0):
        metrics.update(np.full((2, 2), velocity), np.zeros(2), -np.ones(2),
                       np.full((2, 12), actions), np.zeros((2, 3)),
                       np.array(terminated), np.array(truncated), 10)

    def test_terminal_state_included_and_later_episodes_excluded(self):
        metrics = EpisodeMetrics(2, 0.02)
        self.update(metrics, velocity=1, terminated=(True, False))
        self.update(metrics, velocity=3)
        rows = metrics.rows(2)
        self.assertFalse(rows[0]["success"])
        self.assertAlmostEqual(rows[0]["linear_rmse"], np.sqrt(2))
        self.assertEqual(rows[0]["survival_s"], 0.02)
        self.assertTrue(rows[1]["success"])
        self.assertAlmostEqual(rows[1]["linear_rmse"], np.sqrt(10))

    def test_failure_on_last_step_still_fails(self):
        metrics = EpisodeMetrics(2, 0.02)
        self.update(metrics)
        self.update(metrics, terminated=(True, False), truncated=(True, True))
        rows = metrics.rows(2)
        self.assertFalse(rows[0]["success"])
        self.assertTrue(rows[1]["success"])

    def test_early_truncation_is_not_complete(self):
        metrics = EpisodeMetrics(2, 0.02)
        self.update(metrics, truncated=(True, False))
        self.update(metrics)
        self.assertFalse(metrics.rows(2)[0]["success"])

    def test_action_rate_excludes_artificial_first_step_jump(self):
        metrics = EpisodeMetrics(2, 0.02)
        self.update(metrics, actions=3)
        self.update(metrics, actions=4)
        self.assertAlmostEqual(metrics.rows(2)[0]["action_rate_rms"], 50)

    def test_nonfinite_state_aborts(self):
        with self.assertRaises(ValueError):
            self.update(EpisodeMetrics(2, 0.02), velocity=float("nan"))

    def test_survival_precedes_tracking_in_ranking(self):
        metrics = EpisodeMetrics(2, 0.02)
        self.update(metrics, terminated=(True, False))
        self.update(metrics, velocity=1)
        rows = metrics.rows(2)
        for i, row in enumerate(rows):
            row["iteration"] = i
        self.assertEqual(summarize(rows)[0]["iteration"], 1)


class SelectionTests(unittest.TestCase):
    def make(self):
        checkpoints = [{"iteration": i, "valid": True, "segment": 0} for i in (500, 600, 700, 800, 900)]
        return Run(Path("/run"), {}, {}, [], checkpoints, [], {}, [])

    def test_more_candidates_are_selected_by_default(self):
        run = self.make()
        result = {"segment": 0, "top_current_stage": [900]}
        self.assertEqual([c["iteration"] for c in select_candidates(run, result)], [500, 600, 700, 800, 900])
        self.assertEqual([c["iteration"] for c in select_candidates(run, result, count=3)], [500, 700, 900])

    def test_invalid_models_excluded_but_earlier_segments_are_simulated(self):
        run = self.make()
        run.checkpoints[0]["valid"] = False
        run.checkpoints[-1]["segment"] = 1
        result = {"segment": 0, "top_current_stage": []}
        with self.assertRaises(ValueError):
            select_candidates(run, result, [500])
        self.assertEqual([c["iteration"] for c in select_candidates(run, result)], [600, 700, 800, 900])

    def test_count_limit_all_and_explicit_override(self):
        run = self.make()
        run.checkpoints = [{"iteration": i, "valid": True, "segment": 0} for i in range(30)]
        result = {"segment": 0, "top_current_stage": [27, 26, 25]}
        selected = [c["iteration"] for c in select_candidates(run, result)]
        self.assertEqual(selected, list(range(30)))
        self.assertEqual(EvalOptions().count, 0)
        self.assertEqual(len(select_candidates(run, result, count=10)), 10)
        self.assertEqual([c["iteration"] for c in select_candidates(run, result, requested=[4, 4, 9], count=1)], [4, 9])

    def test_invalid_protocols_rejected(self):
        for options in (EvalOptions(num_envs=7), EvalOptions(seeds=(0, 0, 1)), EvalOptions(seeds=(0,)),
                        EvalOptions(seeds=(0, 1, 2, 3)), EvalOptions(duration=float("nan")),
                        EvalOptions(count=-1), EvalOptions(parallel=17), EvalOptions(terrains=("stairs_up_8cm",)), EvalOptions(tilt_limit=float("nan"))):
            with self.assertRaises(ValueError):
                options.validate()

    def test_changed_winner_does_not_replace_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "model_500.pt"
            source.write_bytes(b"checkpoint")
            winner = output / "model_best_eval.pt"
            winner.write_bytes(b"previous")
            model = {"iteration": 500, "path": str(source), "sha256": sha256(source)}
            source.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                publish_evaluation_best(self.make(), {"models": [model], "best_iteration": 500}, output)
            self.assertEqual(winner.read_bytes(), b"previous")


class EnvironmentProtocolTests(unittest.TestCase):
    def setUp(self):
        import src.tasks  # noqa: F401 -- register tasks, no environment or GPU allocation
        from mjlab.tasks.registry import load_env_cfg
        self.original = load_env_cfg("Ri-4438-HIM-Rough", play=False)
        self.saved = plain(self.original)
        self.run = Run(Path("/run"), self.saved, {"experiment_name": "ri_4438_him"}, [], [], [], {}, [])

    def test_nominal_protocol_removes_random_delays_without_editing_training_config(self):
        cfg, _ = make_env_cfg(self.run, EvalOptions(), "flat")
        self.assertEqual(list(cfg.events), ["paired_reset"])
        self.assertFalse(cfg.auto_reset)
        self.assertFalse(cfg.observations["actor"].enable_corruption)
        self.assertFalse(cfg.curriculum)
        for actuator in cfg.scene.entities["robot"].articulation.actuators:
            self.assertEqual(actuator.delay_max_lag, 0)
        for term in cfg.observations["actor"].terms.values():
            self.assertEqual(term.delay_max_lag, 0)
        self.assertEqual(plain(self.original), self.saved)

    def test_reordered_actor_terms_are_rejected_before_simulation(self):
        terms = self.saved["observations"]["actor"]["terms"]
        self.saved["observations"]["actor"]["terms"] = dict(reversed(list(terms.items())))
        with self.assertRaisesRegex(ValueError, "term_order"):
            validate_interface(self.original, self.saved)

    def test_fixed_stair_scenarios_reach_actual_generator(self):
        from src.utils.evaluation_scenarios import expand_terrains
        scenarios = expand_terrains(EvalOptions().terrains)
        self.assertEqual(len(scenarios), 8)
        self.assertEqual(set(scenarios), {"flat", "rough", "stairs_up_5cm", "stairs_up_10cm", "stairs_up_15cm",
                                          "stairs_down_5cm", "stairs_down_10cm", "stairs_down_15cm"})
        for cm in (5, 10, 15):
            for direction in ("up", "down"):
                terrain = f"stairs_{direction}_{cm}cm"
                self.assertIn(terrain, expand_terrains((f"stairs_{direction}",)))
                cfg, _ = make_env_cfg(self.run, EvalOptions(), terrain)
                stairs = cfg.scene.terrain.terrain_generator.sub_terrains["stairs"]
                self.assertEqual(stairs.step_height, cm / 100)
                self.assertEqual(stairs.descending, direction == "down")
                self.assertIn("left_stair_lane", cfg.terminations)
        self.assertEqual(plain(self.original), self.saved)


class ParallelTests(unittest.TestCase):
    def test_auto_parallel_uses_eight_models_when_memory_allows(self):
        from unittest.mock import patch
        from src.utils.policy_evaluation import parallel_count
        with patch("torch.cuda.mem_get_info", return_value=(7 * 2**30, 8 * 2**30)):
            self.assertEqual(parallel_count(EvalOptions(), 30), 8)
            self.assertEqual(parallel_count(EvalOptions(), 3), 3)
        with patch("torch.cuda.mem_get_info", return_value=(2 * 2**30, 8 * 2**30)):
            self.assertEqual(parallel_count(EvalOptions(), 30), 1)

    def test_vectorized_actors_keep_independent_weights_and_normalizers(self):
        class Actor(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.linear = torch.nn.Linear(4, 2)
                self.register_buffer("mean", torch.full((4,), value))

            def forward(self, observations):
                return self.linear(observations["actor"] - self.mean)

        actors = [Actor(float(i)).eval().requires_grad_(False) for i in range(3)]
        observations = torch.randn(3, 6, 4)
        expected = torch.stack([a(TensorDict({"actor": o}, batch_size=[6])) for a, o in zip(actors, observations)])
        actual = ParallelActors(actors)(observations)
        torch.testing.assert_close(actual, expected)

    def test_reset_samples_independent_of_model_batch_size_and_order(self):
        from types import SimpleNamespace
        from src.utils.evaluation_protocol import reset_paired_robot

        class Robot:
            def __init__(self, total):
                root = torch.zeros(total, 13); root[:, 3] = 1
                self.data = SimpleNamespace(default_root_state=root, default_joint_pos=torch.zeros(total, 12),
                    default_joint_vel=torch.zeros(total, 12), soft_joint_pos_limits=torch.tensor([-3, 3]).expand(total, 12, 2))
                self.poses, self.positions = torch.zeros(total, 7), torch.zeros(total, 12)

            def write_root_link_pose_to_sim(self, poses, env_ids):
                self.poses[env_ids] = poses

            def write_root_link_velocity_to_sim(self, velocities, env_ids):
                pass

            def write_joint_state_to_sim(self, positions, velocities, env_ids):
                self.positions[env_ids] = positions

        class Scene(dict):
            pass

        def make(total):
            robot = Robot(total)
            scene = Scene(robot=robot); scene.env_origins = torch.zeros(total, 3)
            return SimpleNamespace(num_envs=total, device="cpu", _evaluation_seed=7, scene=scene), robot

        baseline_env, baseline = make(24)
        reset_paired_robot(baseline_env, None, 24, True)
        for total in (48, 96):
            env, robot = make(total)
            ids = torch.tensor([total-1, 3, 28, total-12])
            reset_paired_robot(env, ids, 24, True)
            torch.testing.assert_close(robot.poses[ids], baseline.poses[ids % 24], rtol=0, atol=0)
            torch.testing.assert_close(robot.positions[ids], baseline.positions[ids % 24], rtol=0, atol=0)


class StairAndStabilityTests(unittest.TestCase):
    def test_stair_geometry_has_exact_risers_and_correct_spawn_height(self):
        import mujoco
        from src.utils.evaluation_protocol import BenchmarkStairsCfg, stair_sections
        for height, down in ((cm / 100, down) for cm in (5, 10, 15) for down in (False, True)):
            spec = mujoco.MjSpec()
            spec.worldbody.add_body(name="terrain")
            output = BenchmarkStairsCfg(size=(16, 8), step_height=height, descending=down).function(1, spec, None)
            self.assertAlmostEqual(output.origin[2], 6 * height if down else 0)
            sections = stair_sections(16, height, down)
            for i in range(1, 7):
                self.assertAlmostEqual(sections[i][2] - sections[i-1][2], -height if down else height)
                self.assertAlmostEqual(sections[i][0], sections[i-1][1])
            model = spec.compile()
            self.assertEqual(model.ngeom, 8)

    def test_standing_or_crossing_at_wrong_height_does_not_pass_stairs(self):
        metrics = EpisodeMetrics(3, 0.02, stair_goal=3.15, stair_gain=0.48)
        metrics.initial_positions = np.zeros((3, 3))
        metrics.update(np.zeros((3, 2)), np.zeros(3), -np.ones(3), np.zeros((3, 12)), np.zeros((3, 3)),
                       np.zeros(3, bool), np.zeros(3, bool), 10,
                       position=np.array([[0, 0, 0], [3.2, 0, 0], [3.2, 0, 0.48]]))
        rows = metrics.rows(1)
        self.assertTrue(all(r["success"] for r in rows))
        self.assertEqual([r["task_success"] for r in rows], [False, False, True])

    def test_completed_stairs_then_falling_still_fails(self):
        metrics = EpisodeMetrics(1, 0.02, stair_goal=3.15, stair_gain=-0.48)
        metrics.initial_positions = np.zeros((1, 3))
        args = (np.zeros((1, 2)), np.zeros(1), -np.ones(1), np.zeros((1, 12)), np.zeros((1, 3)))
        metrics.update(*args, np.array([False]), np.array([False]), 10, position=np.array([[3.2, 0, -0.48]]))
        metrics.update(*args, np.array([True]), np.array([False]), 10, position=np.array([[3.3, 0, -0.48]]))
        self.assertTrue(metrics.rows(2)[0]["stair_completed"])
        self.assertFalse(metrics.rows(2)[0]["task_success"])

    def test_stability_percentiles_ignore_post_failure_states(self):
        metrics = EpisodeMetrics(1, 0.02, tilt_limit=15)
        for index, degrees in enumerate((0, 20, 80)):
            metrics.update(np.zeros((1, 2)), np.zeros(1), np.array([-np.cos(np.radians(degrees))]),
                           np.zeros((1, 12)), np.zeros((1, 3)), np.array([index == 1]), np.array([False]), 10)
        row = metrics.rows(3)[0]
        self.assertAlmostEqual(row["tilt_peak_deg"], 20)
        self.assertAlmostEqual(row["tilt_p95_deg"], 19)
        self.assertAlmostEqual(row["tilt_exceed_fraction"], 0.5)


class FixedStairReportTests(unittest.TestCase):
    def test_reports_keep_height_and_direction_results_separate(self):
        import json
        from dataclasses import asdict
        from src.utils.evaluation_metrics import COMMANDS
        from src.utils.evaluation_report import write_evaluation_report
        from src.utils.evaluation_scenarios import STAIR_COMMANDS, STAIR_GOAL, STAIR_PRESETS, expand_terrains

        options = EvalOptions(num_envs=6, seeds=(0,))
        scenarios = {t: STAIR_COMMANDS if t in STAIR_PRESETS else COMMANDS for t in expand_terrains(options.terrains)}
        rows = []
        for terrain, commands in scenarios.items():
            preset = STAIR_PRESETS.get(terrain)
            gain = preset["total_height_m"] * (-1 if preset["descending"] else 1) if preset else 0
            metrics = EpisodeMetrics(6, 0.02, commands, stair_goal=STAIR_GOAL if preset else None, stair_gain=gain)
            metrics.initial_positions = np.zeros((6, 3))
            position = np.tile([STAIR_GOAL + 0.1, 0, gain], (6, 1))
            # The 10/15 cm ascent fails while the other four stair scenes pass.
            if preset and not preset["descending"] and preset["height_cm"] >= 10:
                position[:] = 0
            metrics.update(np.zeros((6, 2)), np.zeros(6), -np.ones(6), np.zeros((6, 12)), np.zeros((6, 3)),
                           np.zeros(6, bool), np.zeros(6, bool), 10, position=position)
            for row in metrics.rows(1):
                row.update(iteration=500, terrain=terrain, seed=0,
                           failure_reason="" if row["task_success"] else "stairs_not_completed")
                rows.append(row)
        data = {"options": asdict(options), "rows": rows, "summary": summarize(rows),
                "selection": {"selected": 1, "available": 1, "all_tested": True},
                "models": [{"iteration": 500, "reasons": ["测试"], "sha256": "test"}],
                "scenarios": {t: [{"name": n, "label": label, "velocity": v} for n, label, v in commands] for t, commands in scenarios.items()},
                "stair_presets": STAIR_PRESETS, "stair_goal_m": STAIR_GOAL, "run": "/fixture", "steps": 1, "dt": 0.02,
                "parallel_models": 1, "total_parallel_envs": 6}
        result = {"convergence": {"status": "insufficient_data"}, "as_of_iteration": 500, "segment": 0}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_evaluation_report(result, data, output, plot=False)
            html = (output / "report.html").read_text()
            markdown = (output / "report.md").read_text()
            self.assertIn("实机优先模型", html)
            self.assertIn('id="hardware-readiness"', html)
            self.assertFalse(json.loads((output / "hardware_readiness.json").read_text())["hardware_validated"])
            for terrain, preset in STAIR_PRESETS.items():
                self.assertIn(f'<option value="{terrain}">{preset["label"]}</option>', html)
                self.assertIn(preset["label"], markdown)
            expected = "| model_500 | 6/6 (100.0%) | 6/6 (100.0%) | 0/6 (0.0%) | 6/6 (100.0%) | 0/6 (0.0%) | 6/6 (100.0%) |"
            self.assertIn(expected, markdown)
            embedded = html.split('<script id="case-data" type="application/json">')[1].split('</script>')[0]
            cases = json.loads(embedded)
            self.assertEqual(len(cases), 30)
            self.assertEqual({r["terrain"] for r in cases}, set(scenarios))

            # All 13 models are scored/exported, but only the best ten appear
            # in HTML tables, filters and embedded scene data.
            import csv
            data["rows"] = [{**row, "iteration": i} for i in range(500, 513) for row in rows]
            data["summary"] = summarize(data["rows"])
            data["models"] = [{"iteration": i, "reasons": ["全部评测"], "sha256": "test"} for i in range(500, 513)]
            data["selection"] = {"selected": 13, "available": 13, "all_tested": True}
            data["simulation_images"] = [{"label": "5 cm 上楼梯", "image": "simulation/scene_stairs_up_5cm.png"}]
            write_evaluation_report(result, data, output, plot=False)
            html = (output / "report.html").read_text()
            markdown = (output / "report.md").read_text()
            embedded = html.split('<script id="case-data" type="application/json">')[1].split('</script>')[0]
            self.assertEqual({r["iteration"] for r in json.loads(embedded)}, set(range(500, 510)))
            for i in range(510, 513):
                self.assertNotIn(f"model_{i}", html)
                self.assertNotIn(f"model_{i}", markdown)
            self.assertIn('id="simulation-section"', html)
            self.assertIn('simulation/scene_stairs_up_5cm.png', html)
            for filename, count in (("evaluation_ranking.csv", 13), ("evaluation_cases.csv", 13 * 30), ("evaluation.csv", 13 * 48)):
                with (output / filename).open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)

            # All runs can save model_500; ranking, terrain filters and CSVs
            # must still distinguish the checkpoint sources.
            data["rows"] = [{**row, "iteration": 500, "model_id": f"run{i}:500", "source_index": i,
                             "label": f"R{i+1} / model_500"} for i in range(13) for row in rows]
            data["summary"] = summarize(data["rows"])
            data["models"] = [{"iteration": 500, "model_id": f"run{i}:500", "label": f"R{i+1} / model_500",
                               "reasons": ["全部评测"], "sha256": "test"} for i in range(13)]
            write_evaluation_report(result, data, output, plot=False)
            html = (output / "report.html").read_text()
            embedded = html.split('<script id="case-data" type="application/json">')[1].split('</script>')[0]
            self.assertEqual({r["model_id"] for r in json.loads(embedded)}, {f"run{i}:500" for i in range(10)})
            self.assertIn('value="run0:500">R1 / model_500', html)
            self.assertNotIn('R11 / model_500', html)
            self.assertIn('| R2 / model_500 | 6/6', (output / "report.md").read_text())
            with (output / "evaluation_ranking.csv").open() as stream:
                self.assertEqual(len({r["model_id"] for r in csv.DictReader(stream)}), 13)


class SimulationFrameTests(unittest.TestCase):
    def test_same_slots_and_times_and_no_post_failure_frames(self):
        from types import SimpleNamespace
        from src.utils.evaluation_images import FrameRecorder
        from src.utils.evaluation_metrics import COMMANDS
        recorder = FrameRecorder([{"iteration": 500}, {"iteration": 600}], 6, COMMANDS, "flat", 0, 12, 1.0)
        active, done = np.ones(12, bool), np.zeros(12, bool)
        reasons, frames = [""] * 12, []
        for step in range(12):
            if step == 3:
                done[1], reasons[1] = True, "fell_over;"
            state = SimpleNamespace(qpos=torch.full((12, 3), float(step)), qvel=torch.zeros(12, 3))
            recorder.capture(step, state, active, done, reasons, torch.zeros(12, 3), frames)
            active[done] = False
            done[:] = False
        first = [f for f in frames if f["iteration"] == 500]
        second = [f for f in frames if f["iteration"] == 600]
        self.assertEqual([f["time_s"] for f in first], [2, 4])
        self.assertEqual([f["time_s"] for f in second], [2, 6, 12])
        self.assertTrue(first[-1]["terminal"])
        self.assertEqual(first[-1]["qpos"], [3, 3, 3])
        self.assertEqual({f["env_id"] for f in frames}, {1})


if __name__ == "__main__":
    unittest.main()
