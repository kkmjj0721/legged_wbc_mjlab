"""Reward curriculum, stair command coverage, and checkpoint regression tests."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from mjlab.managers.curriculum_manager import CurriculumManager, CurriculumTermCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from src.config.ri_4438.ri_4438_config import Ri4438Cfg
from src.tasks.locomotion.ri_4438_him.config.env_cfgs import (
    ri_4438_flat_env_cfg,
    ri_4438_rough_env_cfg,
)
from src.tasks.locomotion.ri_4438_him.mdp.curriculums import RewardBasedVelocityCurriculum
from src.tasks.locomotion.ri_4438_him.rl.runner import HIMOnPolicyRunner


def linear_tracking(env):
    return env.raw_linear


def angular_tracking(env):
    return env.raw_angular


def make_env(num_envs=2, scale_by_dt=True):
    ranges = SimpleNamespace(
        lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
    )
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges))
    env = SimpleNamespace(
        num_envs=num_envs, device="cpu", step_dt=0.02,
        max_episode_length=1000, max_episode_length_s=20.0,
        common_step_counter=0, episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        raw_linear=torch.zeros(num_envs), raw_angular=torch.zeros(num_envs),
    )
    env.reward_manager = RewardManager({
        "track_linear_velocity": RewardTermCfg(func=linear_tracking, weight=2.0),
        "track_angular_velocity": RewardTermCfg(func=angular_tracking, weight=1.0),
    }, env, scale_by_dt=scale_by_dt)
    env.curriculum_manager = CurriculumManager({
        "command_vel": CurriculumTermCfg(
            func=RewardBasedVelocityCurriculum, params={"command_name": "twist"}
        )
    }, env)
    return env


def curriculum(env):
    return env.curriculum_manager.get_term_cfg("command_vel").func


def finish_episode(env, counter, linear, angular, duration_steps=1000, ids=None):
    """Feed actual manager reward sums, then exercise the environment's reset order."""
    if ids is None:
        ids = torch.arange(env.num_envs)
    scale = env.step_dt if env.reward_manager._scale_by_dt else 1.0
    env.reward_manager._episode_sums["track_linear_velocity"][ids] = (
        linear * 2.0 * scale * duration_steps
    )
    env.reward_manager._episode_sums["track_angular_velocity"][ids] = (
        angular * scale * duration_steps
    )
    env.episode_length_buf[ids] = duration_steps
    env.common_step_counter = counter
    env.curriculum_manager.compute(ids)
    metrics = env.curriculum_manager.reset(ids)
    env.reward_manager.reset(ids)
    env.episode_length_buf[ids] = 0
    return metrics


class RewardCurriculumTests(unittest.TestCase):
    def test_initial_reset_keeps_small_ranges(self):
        env = make_env()
        env.curriculum_manager.compute(torch.arange(env.num_envs))
        self.assertEqual(curriculum(env).state_dict()["ranges"], {
            "lin_vel_x": (-0.5, 0.5), "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-0.5, 0.5),
        })
        self.assertEqual(curriculum(env).episode_count, 0)

    def test_planar_reward_expands_both_axes_without_expanding_yaw(self):
        env = make_env()
        metrics = finish_episode(env, 1000, 0.9, 0.5)
        ranges = curriculum(env).state_dict()["ranges"]
        self.assertEqual(ranges["lin_vel_x"], (-0.6, 0.6))
        self.assertEqual(ranges["lin_vel_y"], (-0.6, 0.6))
        self.assertEqual(ranges["ang_vel_z"], (-0.5, 0.5))
        self.assertAlmostEqual(metrics["Curriculum/command_vel/linear_score"], 0.9)

    def test_yaw_can_expand_without_planar_axes(self):
        env = make_env()
        finish_episode(env, 1000, 0.5, 0.9)
        ranges = curriculum(env).state_dict()["ranges"]
        self.assertEqual(ranges["lin_vel_x"], (-0.5, 0.5))
        self.assertEqual(ranges["lin_vel_y"], (-0.5, 0.5))
        self.assertEqual(ranges["ang_vel_z"], (-0.6, 0.6))

    def test_threshold_is_strict_and_all_axes_reach_the_limit(self):
        env = make_env()
        finish_episode(env, 1000, 0.8, 0.8)
        self.assertEqual(curriculum(env).state_dict()["ranges"]["lin_vel_x"], (-0.5, 0.5))
        for counter in range(2000, 9000, 1000):
            finish_episode(env, counter, 0.95, 0.95)
        self.assertTrue(all(bounds == (-1.0, 1.0)
                            for bounds in curriculum(env).state_dict()["ranges"].values()))

    def test_full_duration_normalization_penalizes_early_termination(self):
        env = make_env()
        env.raw_linear.fill_(1.0)
        env.raw_angular.fill_(1.0)
        for _ in range(500):
            env.reward_manager.compute(env.step_dt)
        env.common_step_counter = 1000
        env.episode_length_buf.fill_(500)
        env.curriculum_manager.compute(torch.arange(env.num_envs))
        self.assertTrue(torch.allclose(curriculum(env).scores, torch.tensor([0.5, 0.5]), atol=1e-5))
        self.assertTrue(all(bounds == (-0.5, 0.5)
                            for bounds in curriculum(env).state_dict()["ranges"].values()))

    def test_window_counts_episodes_instead_of_reset_batches(self):
        env = make_env(num_envs=3)
        finish_episode(env, 500, 1.0, 1.0, ids=torch.tensor([0, 1]))
        self.assertEqual(curriculum(env).episode_count, 2)
        self.assertEqual(curriculum(env).state_dict()["ranges"]["lin_vel_x"], (-0.5, 0.5))
        finish_episode(env, 1000, 0.5, 0.5, ids=torch.tensor([2]))
        self.assertTrue(torch.allclose(curriculum(env).scores, torch.tensor([5/6, 5/6])))
        self.assertTrue(all(bounds == (-0.6, 0.6)
                            for bounds in curriculum(env).state_dict()["ranges"].values()))

    def test_unscaled_rewards_have_the_same_curriculum_result(self):
        env = make_env(scale_by_dt=False)
        finish_episode(env, 1000, 0.9, 0.9)
        self.assertTrue(torch.allclose(curriculum(env).scores, torch.tensor([0.9, 0.9])))
        self.assertTrue(all(bounds == (-0.6, 0.6)
                            for bounds in curriculum(env).state_dict()["ranges"].values()))

    def test_checkpoint_preserves_unfinished_check_window(self):
        env = make_env()
        finish_episode(env, 1000, 0.9, 0.9)
        finish_episode(env, 1500, 0.6, 0.6)
        state = curriculum(env).state_dict()
        resumed = make_env()
        resumed.common_step_counter = 1500
        curriculum(resumed).load_state_dict(state)
        # Environment reset must not erase or recount restored global statistics.
        resumed.curriculum_manager.compute(torch.arange(resumed.num_envs))
        resumed.curriculum_manager.reset(torch.arange(resumed.num_envs))
        self.assertEqual(curriculum(resumed).episode_count, 2)
        finish_episode(env, 2000, 0.95, 0.95)
        finish_episode(resumed, 2000, 0.95, 0.95)
        self.assertEqual(curriculum(env).state_dict()["ranges"],
                         curriculum(resumed).state_dict()["ranges"])
        self.assertTrue(torch.equal(curriculum(env).scores, curriculum(resumed).scores))

    def test_legacy_schedule_uses_saved_step_counter(self):
        env = make_env()
        saved = {
            "commands": {"twist": {"ranges": {
                "lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_z": [-1, 1],
            }}},
            "curriculum": {"command_vel": {"params": {"velocity_stages": [
                {"step": 0, "lin_vel_x": [-0.5, 1], "lin_vel_y": [-0.5, 0.5],
                 "ang_vel_z": [-0.5, 0.5]},
                {"step": 200000, "lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1],
                 "ang_vel_z": [-1, 1]},
            ]}}},
        }
        env.common_step_counter = 200000
        self.assertTrue(curriculum(env).restore_legacy_ranges(saved))
        self.assertEqual(curriculum(env).state_dict()["ranges"]["ang_vel_z"], (-0.5, 0.5))
        env.common_step_counter += 1
        curriculum(env).restore_legacy_ranges(saved)
        self.assertTrue(all(bounds == (-1.0, 1.0)
                            for bounds in curriculum(env).state_dict()["ranges"].values()))


class CommandCoverageTests(unittest.TestCase):
    def test_initial_ranges_apply_to_rough_and_flat_without_changing_ppo_defaults(self):
        for factory in (ri_4438_rough_env_cfg, ri_4438_flat_env_cfg):
            cfg = factory()
            ranges = cfg.commands["twist"].ranges
            self.assertEqual((ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z),
                             ((-0.5, 0.5),) * 3)
            self.assertIs(cfg.curriculum["command_vel"].func, RewardBasedVelocityCurriculum)
        self.assertEqual(Ri4438Cfg.comaman.ranges.lin_vel_x, [-1.0, 1.0])
        self.assertEqual(ri_4438_rough_env_cfg(play=True).commands["twist"].ranges.lin_vel_x,
                         (-1.0, 1.0))

    def test_easy_and_hard_stairs_cover_the_active_velocity_range(self):
        cfg = ri_4438_rough_env_cfg()
        generator = cfg.scene.terrain.terrain_generator
        num_envs = 4096
        scene = type("Scene", (dict,), {})(robot=SimpleNamespace())
        scene.terrain = SimpleNamespace(
            cfg=SimpleNamespace(terrain_generator=generator),
            terrain_types=torch.full((num_envs,), list(generator.sub_terrains).index("pyramid_stairs")),
            terrain_levels=torch.zeros(num_envs, dtype=torch.long),
        )
        env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=scene)
        command_cfg = replace(cfg.commands["twist"], stair_forward_fraction=1.0, rel_standing_envs=0.0)
        command = command_cfg.build(env)
        ids = torch.arange(num_envs)
        for level, high in ((0, 0.5), (0, 1.0), (9, 1.0)):
            torch.manual_seed(4438)
            scene.terrain.terrain_levels.fill_(level)
            command.cfg.ranges.lin_vel_x = (-high, high)
            command._resample_command(ids)
            command._update_command()
            self.assertTrue((command.command[:, 0] >= 0.3).all())
            self.assertTrue((command.command[:, 0] <= high).all())
            self.assertGreater(command.command[:, 0].max().item(), high - 0.01)
            self.assertTrue((command.command[:, 1:] == 0).all())


class RunnerCheckpointTests(unittest.TestCase):
    def test_runner_saves_and_restores_all_velocity_curriculum_state(self):
        env = make_env()
        finish_episode(env, 1000, 0.9, 0.9)
        finish_episode(env, 1500, 0.7, 0.7)
        runner = HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
        runner.env = SimpleNamespace(unwrapped=env, reset=Mock())
        runner.cfg = {"upload_model": False}
        runner.logger = SimpleNamespace(writer=None)
        runner._validate_numerics = Mock()
        runner._numerical_contract = Mock(return_value={"version": 1})
        runner.export_policy_to_onnx = Mock()

        def save_base(instance, path, infos):
            torch.save({"infos": {**infos, "env_state": {
                "common_step_counter": instance.env.unwrapped.common_step_counter,
            }}}, path)

        def load_base(instance, path, *args):
            infos = torch.load(path, weights_only=True)["infos"]
            instance.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
            return infos

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.pt")
            with patch.object(MjlabOnPolicyRunner, "save", new=save_base):
                runner.save(path, {"extra_metadata": "preserved"})
            saved = torch.load(path, weights_only=True)["infos"]
            self.assertEqual(saved["extra_metadata"], "preserved")
            self.assertEqual(saved["him_velocity_curriculum"]["episode_count"], 2)
            resumed = make_env()
            runner.env = SimpleNamespace(unwrapped=resumed, reset=Mock())
            with patch.object(MjlabOnPolicyRunner, "load", new=load_base):
                runner.load(path)
            self.assertEqual(curriculum(env).state_dict()["ranges"],
                             curriculum(resumed).state_dict()["ranges"])
            self.assertTrue(torch.equal(curriculum(env).reward_sums, curriculum(resumed).reward_sums))
            runner.env.reset.assert_called_once()

    def test_legacy_python_tagged_config_is_read_as_data(self):
        env = make_env()
        env.common_step_counter = 200001
        with tempfile.TemporaryDirectory() as directory:
            params = Path(directory) / "params"
            params.mkdir()
            (params / "env.yaml").write_text("""commands:
  twist:
    ranges:
      lin_vel_x: !!python/tuple [-1, 1]
      lin_vel_y: !!python/tuple [-1, 1]
      ang_vel_z: !!python/tuple [-1, 1]
unused: !!python/name:unimportable.symbol ''
""")
            HIMOnPolicyRunner._restore_legacy_velocity_ranges(
                curriculum(env), Path(directory) / "model.pt"
            )
        self.assertTrue(all(bounds == (-1.0, 1.0)
                            for bounds in curriculum(env).state_dict()["ranges"].values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
