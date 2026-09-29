"""Behavior tests; run with python -m unittest discover -s tests -v."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

from src.utils.best_model import (
    ANGULAR, LENGTH, LINEAR, REWARD, TERRAIN, Goal, Options, analyze,
)
from src.utils.training_logs import ConfigLoader, Point, Run, Segment, inspect_checkpoint, read_run, sha256, split_records
from src.utils.training_report import export_checkpoint, publish_best


def make_run(count=400, reward=None, schedule=None, first=0):
    series = {}
    for tag, value in [(REWARD, 10.0), (LENGTH, 1000.0), (LINEAR, 2.0), (ANGULAR, 1.5)]:
        series[tag] = {i: Point(i, reward(i) if reward and tag == REWARD else value, 1000+i)
                       for i in range(first, first+count)}
    segment = Segment(0, series, 1000+first, 999+first+count)
    env = {"episode_length_s": 20.0, "decimation": 4, "sim": {"mujoco": {"timestep": 0.005}},
           "curriculum": {}, "rewards": {"track_linear_velocity": {"weight": 3}, "track_angular_velocity": {"weight": 2}}}
    if schedule:
        env["curriculum"]["command_vel"] = {"params": {"velocity_stages": [{"step": step} for step in schedule]}}
    checkpoints = [{"path": f"/run/model_{i}.pt", "iteration": i, "mtime": 1000+i, "mtime_ns": 0,
                    "size": 1, "valid": True, "error": None, "env_steps": (i+1)*100, "segment": 0}
                   for i in range(first, first+count, 100)]
    return Run(Path("/run"), env, {"num_steps_per_env": 100, "resume": False}, [segment], checkpoints, [], {}, [])


class RankingTests(unittest.TestCase):
    def test_one_reward_spike_does_not_win(self):
        run = make_run(301, lambda i: 10000 if i == 150 else 10 if i <= 200 else 11)
        result = analyze(run, Options())
        self.assertEqual(result["best_current_stage"], 300)
        self.assertEqual(next(c for c in result["candidates"] if c["iteration"] == 200)["score"], 10)

    def test_reward_peak_without_checkpoint_is_not_returned(self):
        run = make_run(350, lambda i: 100 if i > 300 else 10)
        result = analyze(run, Options())
        self.assertEqual(result["best_current_stage"], 300)
        self.assertEqual(result["as_of_iteration"], 349)

    def test_curriculum_excludes_transition_and_earlier_easy_best(self):
        run = make_run(351, lambda i: 100 if i < 200 else 10, schedule=[0, 20000, 50000])
        result = analyze(run, Options())
        self.assertEqual(result["best_current_stage"], 300)
        self.assertEqual(result["top_by_stage"]["0"], [100])
        transition = next(c for c in result["candidates"] if c["iteration"] == 200)
        self.assertFalse(transition["eligible"])
        candidate = next(c for c in result["candidates"] if c["iteration"] == 300)
        self.assertEqual(candidate["window_start"], 220)
        self.assertEqual(candidate["samples"], 81)
        self.assertEqual(result["convergence"]["status"], "curriculum_in_progress")

    def test_no_current_stage_candidate_does_not_fall_back_to_old_best(self):
        run = make_run(220, schedule=[0, 20000])
        result = analyze(run, Options())
        self.assertIsNone(result["best_current_stage"])
        self.assertEqual(result["top_by_stage"]["0"], [100])

    def test_resumed_counter_changes_stage_boundaries(self):
        run = make_run(151, schedule=[0, 20000, 40000], first=500)
        run.agent["resume"] = True
        for c in run.checkpoints:
            c["env_steps"] = (c["iteration"]-500+1)*100
        result = analyze(run, Options())
        self.assertEqual(result["curriculum"]["offset"], -50000)
        self.assertEqual(result["current_stage"]["index"], 0)
        self.assertFalse(result["current_stage"]["final"])

    def test_resumed_counter_unknown_requires_override(self):
        run = make_run(151, schedule=[0, 20000], first=500)
        run.agent["resume"] = True
        for c in run.checkpoints:
            c["env_steps"] = None
        result = analyze(run, Options())
        self.assertIsNone(result["best_current_stage"])
        self.assertEqual(result["convergence"]["status"], "unknown_curriculum")
        self.assertTrue(analyze(run, Options(env_step_offset=-50000))["curriculum"]["known"])

    def test_bad_checkpoint_is_excluded(self):
        run = make_run()
        run.checkpoints[-1].update(valid=False, error="non-finite tensor")
        self.assertEqual(analyze(run, Options())["best_current_stage"], 200)

    def test_changing_tracking_weight_has_no_misleading_normalized_index(self):
        run = make_run()
        run.env["curriculum"]["tracking_weight"] = {"params": {
            "reward_name": "track_linear_velocity", "weight_stages": [{"step": 0, "weight": 3}]}}
        result = analyze(run, Options())
        self.assertNotIn("track_linear_velocity_index", result["candidates"][-1]["derived"])

    def test_nonfinite_reward_is_not_silently_dropped(self):
        run = make_run(301, lambda i: float("nan") if i == 250 else 10)
        result = analyze(run, Options())
        self.assertEqual(result["best_current_stage"], 200)
        self.assertEqual(result["candidates"][-1]["reason"], "non-finite reward samples")
        json.dumps(result, allow_nan=False)


class ConvergenceTests(unittest.TestCase):
    def test_flat_bad_reward_is_only_plateau(self):
        result = analyze(make_run(reward=lambda i: -20), Options())
        self.assertEqual(result["convergence"]["status"], "plateau")
        self.assertEqual(result["convergence"]["attainment"]["status"], "unspecified")

    def test_explicit_goals_and_stable_metrics(self):
        options = Options(goals=[Goal.parse("survival_ratio>=0.95"), Goal.parse(f"{LINEAR}>=1.8")])
        self.assertEqual(analyze(make_run(), options)["convergence"]["status"], "converged_candidate")
        failed = replace(options, goals=[Goal.parse(f"{LINEAR}>=2.8")])
        result = analyze(make_run(), failed)
        self.assertEqual(result["convergence"]["status"], "plateau")
        self.assertEqual(result["convergence"]["attainment"]["status"], "failed")

    def test_short_run_cannot_converge(self):
        self.assertEqual(analyze(make_run(200), Options())["convergence"]["status"], "insufficient_data")

    def test_gaps_are_not_filled_for_convergence(self):
        run = make_run()
        for i in range(100, 180):
            run.segments[0].series[REWARD].pop(i)
        self.assertEqual(analyze(run, Options())["convergence"]["status"], "insufficient_data")

    def test_negative_rewards_can_improve_or_degrade(self):
        improving = make_run(reward=lambda i: -50+i*0.05)
        degrading = make_run(reward=lambda i: -10-i*0.05)
        self.assertEqual(analyze(improving, Options())["convergence"]["status"], "improving")
        self.assertEqual(analyze(degrading, Options())["convergence"]["status"], "degrading")

    def test_changing_terrain_blocks_convergence(self):
        run = make_run()
        run.env["curriculum"]["terrain_levels"] = {}
        run.segments[0].series[TERRAIN] = {i: Point(i, i/100, 1000+i) for i in range(400)}
        self.assertEqual(analyze(run, Options())["convergence"]["status"], "curriculum_in_progress")

    def test_missing_terrain_log_is_insufficient(self):
        run = make_run()
        run.env["curriculum"]["terrain_levels"] = {}
        self.assertEqual(analyze(run, Options())["convergence"]["status"], "insufficient_data")


class InputTests(unittest.TestCase):
    def test_time_tags_do_not_create_restarts_and_duplicate_latest_wins(self):
        records = [(REWARD, Point(1, 1, 10)), (REWARD+"/time", Point(9999, 1, 11)),
                   (REWARD, Point(1, 2, 12)), (REWARD, Point(2, 3, 13))]
        segments, duplicates = split_records(records)
        self.assertEqual(len(segments), 1)
        self.assertEqual(duplicates, 1)
        self.assertEqual(segments[0].series[REWARD][1].value, 2)

    def test_rewind_does_not_mix_old_tail_with_resumed_branch(self):
        records = [(REWARD, Point(i, i, 100+i)) for i in range(10)]
        records += [(REWARD, Point(i, 100+i, 200+i)) for i in range(5, 8)]
        segments, _ = split_records(records)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[-1].last_step, 7)
        self.assertNotIn(9, segments[-1].series[REWARD])

    def test_yaml_python_tags_never_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"should_not_exist"
            value = yaml.load(f"!!python/object/apply:os.system ['touch {target}']", Loader=ConfigLoader)
            self.assertIsInstance(value, list)
            self.assertFalse(target.exists())
            self.assertEqual(yaml.load("x: !!python/tuple [1, 2]", Loader=ConfigLoader), {"x": [1, 2]})

    def test_checkpoint_integrity_and_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model_1.pt"
            torch.save({"iter": 1, "actor_state_dict": {"x": torch.tensor([float("nan")])}}, path)
            self.assertFalse(inspect_checkpoint(path, 1)["valid"])
            torch.save({"iter": 2, "actor_state_dict": {"x": torch.ones(1)}}, path)
            self.assertIn("filename", inspect_checkpoint(path, 1)["error"])

    def test_invalid_cli_options(self):
        with self.assertRaises(ValueError):
            Options(window=10, min_samples=50).validate()
        with self.assertRaises(ValueError):
            Goal.parse("survival_ratio>=nan")


class IntegrationTests(unittest.TestCase):
    def test_real_tensorboard_cli_copy_and_reports(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"params").mkdir()
            run = make_run(301)
            (root/"params/env.yaml").write_text(yaml.safe_dump(run.env))
            (root/"params/agent.yaml").write_text(yaml.safe_dump(run.agent))
            writer = EventFileWriter(str(root))
            for i in range(301):
                writer.add_event(Event(step=i, wall_time=1000+i, summary=Summary(value=[
                    Summary.Value(tag=tag, simple_value=points[i].value)
                    for tag, points in run.segments[0].series.items()])))
            writer.close()
            for i in (100, 200, 300):
                torch.save({"iter": i, "actor_state_dict": {"weight": torch.ones(2)},
                            "infos": {"env_state": {"common_step_counter": (i+1)*100}}}, root/f"model_{i}.pt")
            command = [sys.executable, "-B", "scripts/best_model.py", "--run", str(root), "--write-best", "--no-plot"]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((root/"best/selection.json").read_text())
            self.assertEqual(result["best_current_stage"], 300)
            self.assertEqual(sha256(root/"best/model_best.pt"), sha256(root/"model_300.pt"))
            self.assertTrue((root/"best/candidates.csv").exists())
            self.assertFalse((root/"model_best.pt").exists())
            # Re-reading must not include the best alias as a numeric candidate.
            loaded = read_run(root)
            self.assertEqual(len(loaded.checkpoints), 3)

    def test_tensor_scalars_and_incomplete_event_tail(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.util.tensor_util import make_tensor_proto
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = EventFileWriter(str(root))
            writer.add_event(Event(step=5, wall_time=1000, summary=Summary(value=[
                Summary.Value(tag=REWARD, tensor=make_tensor_proto(12.0))])))
            writer.close()
            path = next(root.glob("events.out.tfevents.*"))
            with path.open("ab") as stream:
                stream.write(b"partial")
            run = read_run(root)
            self.assertEqual(run.segments[0].series[REWARD][5].value, 12)
            self.assertTrue(any("incomplete or corrupt tail" in w for w in run.warnings))
            self.assertEqual(analyze(run, Options())["convergence"]["status"], "unknown_curriculum")

    def test_export_failure_keeps_previous_published_checkpoint(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"model_300.pt"
            source.write_bytes(b"new checkpoint")
            output = root/"best"
            output.mkdir()
            (output/"model_best.pt").write_bytes(b"old checkpoint")
            run = make_run()
            result = analyze(run, Options())
            candidate = result["candidates"][-1]
            candidate.update(path=str(source), size=source.stat().st_size, mtime_ns=source.stat().st_mtime_ns)
            with patch("src.utils.training_report.export_checkpoint", side_effect=ValueError("unsupported")):
                with self.assertRaises(ValueError):
                    publish_best(run, result, output, export_onnx=True)
            self.assertEqual((output/"model_best.pt").read_bytes(), b"old checkpoint")

    def test_ppo_onnx_matches_selected_weights(self):
        import onnxruntime as ort
        from tensordict import TensorDict
        from rsl_rl.models import MLPModel
        cfg = {"hidden_dims": [8, 4], "activation": "elu", "obs_normalization": True,
               "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5}}
        actor = MLPModel(TensorDict({"actor": torch.zeros(1, 5)}, batch_size=[1]), {"actor": ["actor"]}, "actor", 2, **cfg)
        actor.eval()
        run = make_run()
        run.agent["actor"] = {"class_name": "MLPModel", **cfg}
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory)/"model_300.pt", Path(directory)/"policy.onnx"
            torch.save({"actor_state_dict": actor.state_dict()}, path)
            export_checkpoint(run, path, output)
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            for batch in (1, 7):
                x = torch.randn(batch, 5)
                with torch.no_grad():
                    expected = actor.as_onnx(False)(x).numpy()
                actual = session.run(None, {"obs": x.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_him_onnx_preserves_history_normalization_and_clipping(self):
        import onnxruntime as ort
        from tensordict import TensorDict
        from rsl_rl.models import HIMActorModel
        cfg = {"hidden_dims": [8, 4], "activation": "elu", "obs_normalization": True,
               "num_one_step_obs": 6, "history_size": 2, "history_order": "frame_major_oldest_first",
               "action_clip": 2.0, "observation_clip": 10.0, "action_observation_slice": [4, 6],
               "estimator_cfg": {"enc_hidden_dims": [8, 3], "tar_hidden_dims": [8, 3]},
               "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5}}
        actor = HIMActorModel(TensorDict({"actor": torch.zeros(1, 12)}, batch_size=[1]),
                              {"actor": ["actor"]}, "actor", 2, **cfg).eval()
        actor.obs_normalizer._mean.fill_(0.25)
        actor.obs_normalizer._std.fill_(0.5)
        actor.obs_normalizer._var.fill_(0.25)
        contract = {"action_clip": 2.0, "observation_clip": 10.0, "history_size": 2,
                    "frame_size": 6, "action_observation_slice": [4, 6]}
        run = make_run()
        run.agent["actor"] = {"class_name": "HIMActorModel", **cfg}
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory)/"model_300.pt", Path(directory)/"policy.onnx"
            torch.save({"actor_state_dict": actor.state_dict(), "infos": {"him_numerics": contract}}, path)
            export_checkpoint(run, path, output)
            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            for batch, scale in [(1, 1), (5, 30)]:
                current_first = torch.randn(batch, 12) * scale
                training_order = current_first[:, torch.argsort(actor.history_permutation)]
                with torch.inference_mode():
                    expected = actor(TensorDict({"actor": training_order}, batch_size=[batch])).numpy()
                actual = session.run(None, {"obs_history": current_first.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertLessEqual(float(np.abs(actual).max()), 2.0)


if __name__ == "__main__":
    unittest.main()
