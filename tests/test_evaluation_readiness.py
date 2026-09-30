"""Relative recommendations permit task shortfalls and retain deployment limits."""

import copy
import math
import unittest
from src.utils.evaluation_metrics import COMMANDS
from src.utils.evaluation_readiness import REQUIRED_TERRAINS, assess_hardware_readiness, apply_recommendation_ranking
from src.utils.evaluation_scenarios import STAIR_COMMANDS


def evaluation():
    rows = []
    for terrain in REQUIRED_TERRAINS:
        commands = STAIR_COMMANDS if terrain.startswith("stairs_") else COMMANDS
        for seed in (0, 1, 2):
            for index in range(6):
                command = commands[index % len(commands)]
                rows.append({"iteration": 500, "model_id": "parent:500", "label": "R1 / model_500",
                             "terrain": terrain, "command": command[0], "seed": seed, "env_id": index,
                             "success": True, "task_success": True, "failed": False, "failure_reason": "",
                             "linear_rmse": 0.0, "yaw_rmse": 0.0, "tilt_rms_deg": 0.0,
                             "action_rate_rms": 0.0, "action_accel_rms": 0.0})
    return {"summary": [{"iteration": 500, "model_id": "parent:500", "label": "R1 / model_500"}],
            "rows": rows, "options": {"seeds": [0, 1, 2], "duration": 12.0, "num_envs": 6}, "dt": 0.02}


def add_model(data, identity="resumed:600", iteration=600):
    original = copy.deepcopy(data["rows"])
    model = {"iteration": iteration, "model_id": identity, "label": f"R2 / model_{iteration}"}
    data["summary"].append(model)
    added = [{**r, **model} for r in original]
    data["rows"].extend(added)
    return added


class ReadinessTests(unittest.TestCase):
    def test_perfect_nominal_results_are_recommended_but_not_hardware_validated(self):
        audit = assess_hardware_readiness(evaluation())
        self.assertEqual(audit["hardware_recommendation"]["model_id"], "parent:500")
        self.assertEqual(audit["hardware_recommendation"]["score"], 100)
        self.assertFalse(audit["hardware_validated"])

    def test_failed_stairs_reduce_score_but_do_not_cancel_recommendation(self):
        data = evaluation()
        for row in data["rows"]:
            if row["terrain"] == "stairs_up_15cm":
                row.update(task_success=False, failure_reason="stairs_not_completed")
        before = copy.deepcopy(data)
        audit = assess_hardware_readiness(data)
        self.assertEqual(data, before)
        self.assertEqual(audit["status"], "relative_recommendation")
        self.assertLess(audit["hardware_recommendation"]["score"], 100)
        self.assertEqual(audit["models"][0]["terminated"], 0)

    def test_missing_invalid_or_incomparable_measurements_are_not_silently_ranked(self):
        missing = evaluation()
        del missing["rows"][0]["linear_rmse"]
        invalid = evaluation()
        invalid["rows"][0]["linear_rmse"] = float('nan')
        duplicated = evaluation()
        duplicated["rows"].append(duplicated["rows"][0].copy())
        unequal = evaluation()
        add_model(unequal)
        unequal["rows"].pop()
        for data in (missing, invalid, duplicated, unequal):
            self.assertIsNone(assess_hardware_readiness(data)["hardware_recommendation"])

    def test_standing_still_is_not_rewarded_over_following_commands(self):
        data = evaluation()
        commands = {c[0]: c[2] for c in COMMANDS}
        for r in data["rows"]:
            if r["terrain"].startswith('stairs_'):
                r['task_success'] = False
            else:
                cmd = commands[r['command']]
                r.update(linear_rmse=math.hypot(*cmd[:2]), yaw_rmse=abs(cmd[2]))
        moving = add_model(data)
        for r in moving:
            r.update(linear_rmse=.08, yaw_rmse=.08, tilt_rms_deg=5, action_rate_rms=8, action_accel_rms=280)
        self.assertEqual(assess_hardware_readiness(data)["hardware_recommendation"]["model_id"], "resumed:600")

    def test_scores_do_not_depend_on_candidate_count_and_ranking_is_idempotent(self):
        data = evaluation()
        before = assess_hardware_readiness(data)["hardware_recommendation"]["score"]
        worse = add_model(data)
        for r in worse:
            r.update(linear_rmse=.3, task_success=False)
        apply_recommendation_ranking(data)
        self.assertEqual(data['best_model_id'], 'parent:500')
        self.assertEqual(data['summary'][0]['deployment_score'], before)
        snapshot = copy.deepcopy(data)
        apply_recommendation_ranking(data)
        self.assertEqual(data, snapshot)

    def test_relative_winner_replaces_old_pass_rate_winner_consistently(self):
        data = evaluation()
        for r in data['rows']:
            r.update(linear_rmse=.25, yaw_rmse=.25)
        better = add_model(data)
        for r in better:
            r.update(linear_rmse=.05, yaw_rmse=.05)
            if r['terrain'] == 'stairs_up_15cm':
                r['task_success'] = False
        apply_recommendation_ranking(data)
        self.assertEqual(data['best_model_id'], 'resumed:600')
        self.assertEqual(data['summary'][0]['model_id'], data['best_model_id'])
        self.assertEqual(data['hardware_assessment']['hardware_recommendation']['model_id'], data['best_model_id'])


if __name__ == '__main__':
    unittest.main()
