"""Present one report and distinguish simulated performance from log scores."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import best_model as cli
from src.utils.training_report import recommendation, log_candidate_label
from src.utils.training_chain_report import remove_legacy_training_page
from test_best_model import make_run


def simulated():
    return {"best_iteration": 100, "best_model_id": "parent:100", "best_label": "R1 / model_100",
            "summary": [{"iteration": 100, "model_id": "parent:100"}],
            "models": [{"iteration": 100, "model_id": "parent:100", "label": "R1 / model_100", "path": "/parent/model_100.pt"}]}


class RecommendationPresentationTests(unittest.TestCase):
    def test_simulation_recommendation_keeps_log_candidate_separate(self):
        result = {"best_current_stage": 300, "chain": [{"index": 1}, {"index": 2}]}
        selected = recommendation(result, simulated())
        self.assertEqual(selected["label"], "R1 / model_100")
        self.assertEqual(selected["artifact"], "model_best_eval.pt")
        self.assertEqual(selected["intended_use"], "simulation_screening")
        self.assertIsNone(selected["hardware_recommendation"])
        self.assertEqual(log_candidate_label(result), "R2 / model_300")
        self.assertEqual(result["best_current_stage"], 300)

    def test_inconsistent_saved_winner_is_rejected(self):
        data = simulated()
        data["best_model_id"] = "another:100"
        with self.assertRaisesRegex(ValueError, "first ranked"):
            recommendation({}, data)

    def test_cli_ends_with_simulation_recommendation_and_one_report_entry(self):
        run = make_run()
        captured = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(cli, "resolve_run_directory", return_value=run.path))
            stack.enter_context(patch.object(cli, "read_training_chain", return_value=([run], [None])))
            for name in ("write_reports", "write_chain_reports", "write_selection"):
                stack.enter_context(patch.object(cli, name))
            stack.enter_context(patch("src.utils.policy_evaluation.evaluate", return_value=simulated()))
            report = stack.enter_context(patch("src.utils.evaluation_report.write_evaluation_report"))
            stack.enter_context(patch("src.utils.evaluation_report.publish_evaluation_best"))
            stack.enter_context(patch("src.utils.evaluation_repeatability.repeatability_notes", return_value=[]))
            with contextlib.redirect_stdout(captured):
                code = cli.main(["--run", str(run.path), "--output", directory, "--evaluate", "--no-plot"])
            self.assertEqual(code, 0)
            text = captured.getvalue()
            self.assertIn("日志评分候选（训练曲线参考）: model_300", text)
            self.assertIn("本轮推荐（基于仿真）: R1 / model_100", text)
            self.assertIn("任务未全部通过不会", text)
            self.assertEqual(text.count("report.html"), 1)
            self.assertNotIn("training_history.html", text)
            self.assertNotIn("当前阶段 best", text)
            self.assertEqual(report.call_args.args[0]["recommendation"]["label"], "R1 / model_100")

    def test_only_known_generated_legacy_page_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            legacy = output / "training_history.html"
            legacy.write_text("user-authored page")
            remove_legacy_training_page(output)
            self.assertTrue(legacy.exists())
            legacy.write_text('<title>完整训练与续训</title>')
            remove_legacy_training_page(output)
            self.assertFalse(legacy.exists())


if __name__ == "__main__":
    unittest.main()
