"""Keep input changes separate from repeated-simulation variability."""

import copy
import json
import unittest

from src.utils.evaluation_repeatability import compare_evaluations, physics_fingerprint, repeatability_notes


def evaluation():
    return {"best_iteration": 10, "models": [{"iteration": i, "sha256": str(i)} for i in (10, 20)],
            "saved_config_sha256": {"env": "env", "agent": "agent"},
            "scenarios": {"flat": [{"name": "forward"}]},
            "options": {"num_envs": 6, "duration": 12, "seeds": [0, 1, 2], "tilt_limit": 20, "device": "cuda:0"},
            "parallel_models": 2, "dt": .02, "steps": 600, "ranking_policy": "test",
            "runtime": {"packages": {"mujoco": "test"}},
            "environments": {"flat": {"sha256": "cfg", "physics_sha256": "physics"}},
            "summary": [{"iteration": 10, "task_success": .75, "episodes": 4}, {"iteration": 20, "task_success": .5, "episodes": 4}],
            "rows": [{"iteration": iteration, "terrain": "flat", "seed": 0, "env_id": slot,
                      "initial_state_sha256": "shared", "task_success": slot < (3 if iteration == 10 else 2),
                      "success": True, "survival_s": 12., "linear_rmse": .1, "yaw_rmse": .1, "tilt_rms_deg": 1.}
                     for iteration in (10, 20) for slot in range(4)]}


class RepeatabilityTests(unittest.TestCase):
    def test_first_evaluation_does_not_claim_unique_best(self):
        audit = compare_evaluations(evaluation(), None)
        self.assertEqual(audit["status"], "not_checked")
        self.assertEqual(audit["lead_success_episodes"], 1)
        self.assertFalse(audit["unique_best_established"])

    def test_identical_measurements_and_reordered_rows(self):
        current, prior = evaluation(), evaluation()
        prior["rows"].reverse()
        audit = compare_evaluations(current, prior)
        self.assertEqual(audit["status"], "identical")
        self.assertEqual(audit["changed_metric_episodes"], 0)
        self.assertFalse(audit["unique_best_established"])

    def test_live_tuples_match_reloaded_json_lists(self):
        current = evaluation()
        current["options"]["seeds"] = (0, 1, 2)
        current["scenarios"]["flat"][0]["velocity"] = (.5, 0., 0.)
        prior = json.loads(json.dumps(current))
        self.assertEqual(compare_evaluations(current, prior)["status"], "identical")

    def test_changed_winner_is_reported_with_paired_episode_counts(self):
        prior, current = evaluation(), evaluation()
        current["rows"][2]["task_success"] = False
        current["rows"][6]["task_success"] = True
        current["best_iteration"] = 20
        current["summary"] = [{"iteration": 20, "task_success": .75, "episodes": 4}, {"iteration": 10, "task_success": .5, "episodes": 4}]
        audit = compare_evaluations(current, prior)
        self.assertEqual(audit["status"], "winner_changed")
        self.assertEqual(audit["changed_task_success_episodes"], 2)
        self.assertEqual(audit["max_model_success_delta_percentage_points"], 25)
        self.assertIn("排名不稳定", ''.join(repeatability_notes({"repeatability": audit})))

    def test_same_winner_does_not_hide_score_differences(self):
        prior, current = evaluation(), evaluation()
        current["rows"][0]["linear_rmse"] += .001
        audit = compare_evaluations(current, prior)
        self.assertEqual(audit["status"], "same_winner_scores_changed")
        self.assertEqual(audit["changed_metric_episodes"], 1)
        self.assertEqual(audit["changed_task_success_episodes"], 0)

    def test_old_incomplete_metadata_does_not_abort_new_evaluation(self):
        prior = evaluation()
        del prior["rows"][0]["initial_state_sha256"]
        audit = compare_evaluations(evaluation(), prior)
        self.assertEqual(audit["status"], "insufficient_metadata")
        self.assertFalse(audit["comparable"])
        self.assertIn("无法验证", ''.join(repeatability_notes({"repeatability": audit})))

    def test_changed_inputs_do_not_get_called_numerical_instability(self):
        for change in ("parallel", "weight", "seed", "physics", "runtime", "initial", "pool"):
            with self.subTest(change=change):
                prior, current = evaluation(), evaluation()
                if change == "parallel": current["parallel_models"] = 1
                if change == "weight": current["models"][0]["sha256"] = "changed"
                if change == "seed": current["options"]["seeds"] = [3, 4, 5]
                if change == "physics": current["environments"]["flat"]["physics_sha256"] = "changed"
                if change == "runtime": current["runtime"] = {"changed": True}
                if change == "initial": current["rows"][0]["initial_state_sha256"] = "changed"
                if change == "pool": current["models"].append({"iteration": 30, "sha256": "30"})
                audit = compare_evaluations(current, prior)
                self.assertEqual(audit["status"], "inputs_changed")
                self.assertFalse(audit["comparable"])

    def test_legacy_runtime_is_explicitly_unverified(self):
        prior, current = evaluation(), evaluation()
        del prior["runtime"]
        audit = compare_evaluations(current, prior)
        self.assertTrue(audit["comparable"])
        self.assertFalse(audit["runtime_recorded_both"])
        self.assertIn("旧报告", ''.join(repeatability_notes({"repeatability": audit})))

    def test_compiled_physics_ignores_names_but_detects_mass_change(self):
        import mujoco
        first = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="a"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        renamed = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="longer_random_name"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        self.assertEqual(physics_fingerprint(first), physics_fingerprint(renamed))
        changed = copy.copy(first)
        changed.body_mass[1] *= 2
        self.assertNotEqual(physics_fingerprint(first), physics_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
