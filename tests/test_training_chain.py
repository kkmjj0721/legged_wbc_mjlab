"""Resume ancestry and checkpoint identity must survive overlapping iterations."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
import yaml

from src.utils.training_chain import analyze_chain, combine_runs, read_training_chain, write_lineage, resolve_run_directory
from src.utils.training_logs import model_key, source_run, sha256
from src.utils.policy_evaluation import EpisodeMetrics, ParallelActors, select_candidates, summarize
from src.utils.evaluation_repeatability import compare_evaluations
from src.utils.evaluation_report import publish_evaluation_best
from src.utils.best_model import Options
from test_best_model import make_run
from test_evaluation_repeatability import evaluation


class AncestryTests(unittest.TestCase):
    def directory(self, root, name, parent=None, checkpoint="model_500.pt"):
        path = root / name
        (path / "params").mkdir(parents=True)
        (path / "params/agent.yaml").write_text(yaml.safe_dump({"resume": parent is not None, "load_run": parent, "load_checkpoint": checkpoint}))
        (path / "model_500.pt").write_bytes(name.encode())
        return path

    def read(self, path):
        return replace(make_run(), path=path)

    def test_experiment_selects_newest_timestamp_and_traces_only_its_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "2026-09-29_16-14-31")
            self.directory(root, "2026-09-29_17-00-00_unrelated")
            b = self.directory(root, "2026-09-29_17-21-34", a.name)
            c = self.directory(root, "2026-09-29_21-37-06", b.name)
            (root / "best").mkdir()
            (root / "best/report.html").write_text("report")
            os.utime(a, (2_000_000_000, 2_000_000_000))
            os.utime(a / "params/agent.yaml", (2_000_000_000, 2_000_000_000))
            with patch("src.utils.training_chain.read_run", side_effect=self.read):
                self.assertEqual([r.path for r in read_training_chain(root)[0]], [a, b, c])
                self.assertEqual([r.path for r in read_training_chain(root, single=True)[0]], [c])

    def test_direct_run_stays_selected_and_custom_names_use_config_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "custom_a")
            b = self.directory(root, "custom_b")
            os.utime(a / "params/agent.yaml", (2000, 2000))
            os.utime(b / "params/agent.yaml", (1000, 1000))
            self.assertEqual(resolve_run_directory(root), a)
            self.directory(a, "2099-01-01_00-00-00")
            self.assertEqual(resolve_run_directory(a), a)

    def test_empty_experiment_is_reported_without_searching_nested_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.directory(root / "experiment", "2026-09-29_17-21-34")
            with self.assertRaisesRegex(ValueError, "immediate child"):
                resolve_run_directory(root)

    def test_latest_run_with_unresolved_ancestry_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.directory(root, "2026-09-29_16-14-31")
            self.directory(root, "2026-09-29_21-37-06", ".*", "model_.*.pt")
            with self.assertRaisesRegex(ValueError, "historical"):
                read_training_chain(root)

    def test_auto_traces_three_runs_in_source_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "a")
            b = self.directory(root, "b", "a")
            c = self.directory(root, "c", "b")
            with patch("src.utils.training_chain.read_run", side_effect=self.read):
                runs, links = read_training_chain(c)
                self.assertEqual([r.path for r in runs], [a, b, c])
                self.assertEqual(links[-1]["parent"], b)
                self.assertEqual(len(read_training_chain(c, single=True)[0]), 1)

    def test_regex_needs_explicit_order_instead_of_guessing_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "a")
            b = self.directory(root, "b", ".*", "model_.*.pt")
            with self.assertRaisesRegex(ValueError, "historical"):
                read_training_chain(b)
            with patch("src.utils.training_chain.read_run", side_effect=self.read):
                runs, links = read_training_chain(explicit=[a, b])
                self.assertEqual(len(runs), 2)
                self.assertIn("not verified", links[1]["evidence"])

    def test_cycle_and_duplicate_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "a", "b")
            b = self.directory(root, "b", "a")
            with self.assertRaisesRegex(ValueError, "cycle"):
                read_training_chain(b)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_training_chain(explicit=[a, a])

    def test_recorded_source_survives_move_but_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "a")
            b = self.directory(root, "b", ".*", "model_.*.pt")
            write_lineage(b, "test", a / "model_500.pt")
            file = b / "params/lineage.json"
            recorded = json.loads(file.read_text())
            recorded["parent_run"] = "/old/moved/experiment/a"
            file.write_text(json.dumps(recorded))
            with patch("src.utils.training_chain.read_run", side_effect=self.read):
                self.assertEqual(read_training_chain(b)[0][0].path, a)
                (a / "model_500.pt").write_bytes(b"overwritten")
                with self.assertRaisesRegex(ValueError, "has changed"):
                    read_training_chain(b)

    def test_explicit_order_must_match_known_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "a")
            b = self.directory(root, "b", "a")
            c = self.directory(root, "c")
            with self.assertRaisesRegex(ValueError, "preceding"):
                read_training_chain(explicit=[c, b])

    def test_resuming_published_best_tracks_owning_run_and_loaded_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.directory(root, "a")
            b = self.directory(root, "b", ".*")
            (a / "best").mkdir()
            best = a / "best/model_best_eval.pt"
            best.write_bytes(b"saved-model")
            write_lineage(b, "test", best, 500)
            with patch("src.utils.training_chain.read_run", side_effect=self.read):
                runs, links = read_training_chain(b)
                self.assertEqual([r.path for r in runs], [a, b])
                _, result = analyze_chain(runs, links, Options())
                self.assertEqual(result["chain"][-1]["resume_iteration"], 500)
                self.assertEqual(links[-1]["checkpoint"], "best/model_best_eval.pt")


class ChainModelsTests(unittest.TestCase):
    def runs(self):
        return [replace(make_run(), path=Path('/first')), replace(make_run(), path=Path('/resumed'))]

    def test_overlapping_models_are_all_selected_and_keep_original_iterations(self):
        runs = self.runs()
        combined = combine_runs(runs)
        selected = select_candidates(combined, {"top_current_stage": []})
        self.assertEqual(len(selected), sum(len(r.checkpoints) for r in runs))
        self.assertEqual(len({model_key(c) for c in selected}), len(selected))
        pair = select_candidates(combined, {"top_current_stage": []}, requested=[100])
        self.assertEqual(len(pair), 2)
        self.assertEqual([c["iteration"] for c in pair], [100, 100])
        self.assertIs(source_run(combined, pair[0]), runs[0])
        self.assertIs(source_run(combined, pair[1]), runs[1])

    def test_analysis_keeps_config_changes_and_each_runs_convergence(self):
        runs = self.runs()
        runs[1].env = copy.deepcopy(runs[1].env)
        runs[1].env["rewards"]["track_linear_velocity"]["weight"] = 100
        combined, result = analyze_chain(runs, [None, {"evidence": "test"}], Options())
        self.assertTrue(result["chain"][1]["config_changed"])
        self.assertEqual(len(result["chain"]), 2)
        self.assertEqual(result["chain"][0]["segments"][0]["max_episode_steps"], 1000)
        selected = select_candidates(combined, result, count=1)
        self.assertEqual(selected[0]["source_run"], "/resumed")

    def test_metrics_do_not_combine_two_model_500_files(self):
        metrics = EpisodeMetrics(2, .02)
        metrics.update(np.zeros((2, 2)), np.zeros(2), -np.ones(2), np.zeros((2, 12)), np.zeros((2, 3)),
                       np.array([True, False]), np.zeros(2, bool), 10)
        rows = metrics.rows(1)
        for i, row in enumerate(rows):
            row.update(iteration=500, model_id=f"run{i}:500", label=f"R{i+1} / model_500")
        summaries = summarize(rows)
        self.assertEqual(len(summaries), 2)
        self.assertEqual([r["episodes"] for r in summaries], [1, 1])
        self.assertEqual([r["task_success"] for r in summaries], [1, 0])

    def test_repeat_audit_detects_winner_switch_with_same_iteration(self):
        prior = evaluation()
        for kind in ("models", "rows", "summary"):
            for row in prior[kind]:
                row["model_id"] = str(row["iteration"])
                row["iteration"] = 500
        prior.update(best_iteration=500, best_model_id="10")
        current = copy.deepcopy(prior)
        current.update(best_model_id="20")
        current["summary"].reverse()
        audit = compare_evaluations(current, prior)
        self.assertEqual(audit["status"], "winner_changed")
        self.assertEqual(audit["episodes"], 8)

    def test_publication_uses_model_identity_not_duplicate_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sources = []
            for i in range(2):
                path = output / f"run{i}.pt"
                path.write_bytes(str(i).encode())
                sources.append({"iteration": 500, "model_id": str(i), "path": str(path), "sha256": sha256(path)})
            data = {"models": sources, "best_iteration": 500, "best_model_id": "1", "ranking_policy": "test"}
            publish_evaluation_best(self.runs()[0], data, output)
            self.assertEqual((output / "model_best_eval.pt").read_bytes(), b"1")

    def test_different_saved_actor_architectures_keep_own_forward_function(self):
        class Actor(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.net = torch.nn.Sequential(torch.nn.Linear(4, width), torch.nn.Tanh(), torch.nn.Linear(width, 2))
            def forward(self, td):
                return self.net(td["actor"])
        actors = [Actor(w).eval().requires_grad_(False) for w in (3, 7)]
        obs = torch.randn(2, 6, 4)
        expected = torch.stack([a(TensorDict({"actor": o}, batch_size=[6])) for a, o in zip(actors, obs)])
        torch.testing.assert_close(ParallelActors(actors)(obs), expected)

    def test_scalar_actor_configuration_is_not_replaced_by_first_actor(self):
        class Actor(torch.nn.Module):
            def __init__(self, limit):
                super().__init__()
                self.limit = limit
            def forward(self, td):
                return td["actor"].clamp(-self.limit, self.limit)
        actors = [Actor(1), Actor(2)]
        obs = torch.full((2, 6, 4), 3.)
        out = ParallelActors(actors, [{"limit": 1}, {"limit": 2}])(obs)
        torch.testing.assert_close(out[0], torch.ones(6, 4))
        torch.testing.assert_close(out[1], torch.full((6, 4), 2.))


if __name__ == "__main__":
    unittest.main()
