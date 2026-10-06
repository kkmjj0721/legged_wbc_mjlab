"""Focused CPU regressions for RI-4438's sensor and contact randomization.

Run with: PYTHONPATH=src .venv/bin/python -m unittest tools.test_ri4438_domain_randomization
"""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mjlab.envs import mdp
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg
from src.tasks.locomotion.ri_4438_him.config.env_cfgs import (
  ri_4438_flat_env_cfg,
  ri_4438_rough_env_cfg,
)
from src.tasks.locomotion.ri_4438_him.mdp.domain_randomization import (
  imu_ang_vel_with_errors,
  imu_projected_gravity_with_errors,
  randomize_foot_contact_softness,
  randomize_imu_errors,
)


class ImuErrorsTest(unittest.TestCase):
  def setUp(self):
    self.env = SimpleNamespace(num_envs=4, device="cpu")
    self.gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1)
    self.gyro = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)

  def test_nominal_before_startup_and_in_play(self):
    with patch.object(mdp, "builtin_sensor", return_value=self.gyro):
      torch.testing.assert_close(imu_ang_vel_with_errors(self.env, "gyro"), self.gyro)
    with patch.object(mdp, "projected_gravity", return_value=self.gravity):
      torch.testing.assert_close(imu_projected_gravity_with_errors(self.env), self.gravity)

  def test_common_frame_fixed_bias_and_unchanged_truth(self):
    # A known 90-degree pitch turns nominal forward gyro into +Z, gravity into +X.
    self.env._ri4438_imu_mount_quat = torch.tensor(
      [[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]]
    ).repeat(4, 1)
    self.env._ri4438_imu_gyro_bias = torch.full((4, 3), 0.02)
    with patch.object(mdp, "builtin_sensor", return_value=self.gyro):
      observed = imu_ang_vel_with_errors(self.env, "gyro")
      torch.testing.assert_close(observed, torch.tensor([[0.02, 0.02, 1.02]]).repeat(4, 1))
      torch.testing.assert_close(imu_ang_vel_with_errors(self.env, "gyro"), observed)
    with patch.object(mdp, "projected_gravity", return_value=self.gravity):
      observed = imu_projected_gravity_with_errors(self.env)
      torch.testing.assert_close(observed, torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1))
      torch.testing.assert_close(observed.norm(dim=-1), torch.ones(4))
    torch.testing.assert_close(self.gyro, torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1))
    torch.testing.assert_close(self.gravity, torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1))

  def test_sampling_and_partial_environment_selection(self):
    randomize_imu_errors(self.env, None, (-0.02, 0.02), (-0.01745, 0.01745))
    bias = self.env._ri4438_imu_gyro_bias.clone()
    quat = self.env._ri4438_imu_mount_quat.clone()
    self.assertTrue((bias.abs() <= 0.02).all())
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones(4))
    self.assertFalse(torch.equal(bias[0], bias[1]))
    randomize_imu_errors(self.env, torch.tensor([1, 3]), (0.0, 0.0), (0.0, 0.0))
    torch.testing.assert_close(self.env._ri4438_imu_gyro_bias[[0, 2]], bias[[0, 2]])
    torch.testing.assert_close(self.env._ri4438_imu_mount_quat[[0, 2]], quat[[0, 2]])
    torch.testing.assert_close(self.env._ri4438_imu_mount_quat[[1, 3]],
                               torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))


class ContactSoftnessTest(unittest.TestCase):
  def setUp(self):
    self.defaults = torch.tensor([[0.02, 1.0]]).repeat(7, 1)
    self.field = self.defaults.unsqueeze(0).repeat(4, 1, 1)
    self.env = SimpleNamespace(
      num_envs=4, device="cpu", physics_dt=0.005,
      scene={"robot": SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([2, 4, 5, 6])))},
      sim=SimpleNamespace(model=SimpleNamespace(geom_solref=self.field),
                          get_default_field=lambda name: self.defaults),
    )
    self.asset_cfg = SceneEntityCfg("robot", geom_ids=[0, 2])

  def test_only_selected_feet_and_worlds_change(self):
    randomize_foot_contact_softness(self.env, torch.tensor([1, 3]), (0.025, 0.025), self.asset_cfg)
    expected = self.defaults.unsqueeze(0).repeat(4, 1, 1)
    expected[torch.tensor([1, 3])[:, None], torch.tensor([2, 5]), 0] = 0.025
    torch.testing.assert_close(self.field, expected)
    # Absolute sampling does not accumulate on a second invocation.
    randomize_foot_contact_softness(self.env, torch.tensor([1, 3]), (0.025, 0.025), self.asset_cfg)
    torch.testing.assert_close(self.field, expected)

  def test_shared_per_world_and_distinct_across_worlds(self):
    randomize_foot_contact_softness(self.env, None, (0.015, 0.03), self.asset_cfg)
    torch.testing.assert_close(self.field[:, 2, 0], self.field[:, 5, 0])
    self.assertTrue((self.field[:, 2, 0] >= 0.015).all())
    self.assertTrue((self.field[:, 2, 0] <= 0.03).all())
    self.assertGreater(self.field[:, 2, 0].std().item(), 0.0)

  def test_reject_unsafe_timestep_and_direct_solref(self):
    with self.assertRaises(ValueError):
      randomize_foot_contact_softness(self.env, None, (0.005, 0.03), self.asset_cfg)
    self.defaults[2, 0] = -100.0
    with self.assertRaises(ValueError):
      randomize_foot_contact_softness(self.env, None, (0.015, 0.03), self.asset_cfg)


class FiniteForceTest(unittest.TestCase):
  def test_force_expires_and_partial_reset_clears_only_reset_world(self):
    wrench = torch.zeros(2, 2, 6)

    def write(forces, torques, *, env_ids, body_ids):
      wrench[env_ids[:, None], body_ids] = torch.cat((forces, torques), dim=-1)

    asset = SimpleNamespace(num_bodies=2, write_external_wrench_to_sim=write)
    env = SimpleNamespace(scene={"robot": asset}, num_envs=2, device="cpu", step_dt=0.02)
    cfg = EventTermCfg(func=mdp.apply_body_impulse, mode="step", params={
      "asset_cfg": SceneEntityCfg("robot", body_ids=[0]),
      "force_range": (3.0, 3.0), "torque_range": (0.0, 0.0),
      "duration_s": (0.06, 0.06), "cooldown_s": (0.04, 0.04),
    })
    event = cfg.func(cfg, env)
    event(env, None, **cfg.params)
    self.assertEqual(wrench.count_nonzero(), 0)
    event(env, None, **cfg.params)
    torch.testing.assert_close(wrench[:, 0, :3], torch.full((2, 3), 3.0))
    self.assertEqual(wrench[:, 1].count_nonzero(), 0)
    event.reset(torch.tensor([0]))
    self.assertEqual(wrench[0].count_nonzero(), 0)
    self.assertEqual(wrench[1, 0, :3].count_nonzero(), 3)
    # Keep the reset world's next trigger out of the expiration observation.
    event._interval_time_left[0] = 10.0
    for _ in range(3):
      event(env, None, **cfg.params)
    self.assertEqual(wrench.count_nonzero(), 0)
    self.assertFalse(event._active.any())


class ObservationConfigurationTest(unittest.TestCase):
  def test_training_truth_play_and_existing_reset_contract(self):
    for factory in (ri_4438_rough_env_cfg, ri_4438_flat_env_cfg):
      cfg = factory()
      for name, nominal, corrupted in (
        ("base_ang_vel", mdp.builtin_sensor, imu_ang_vel_with_errors),
        ("projected_gravity", mdp.projected_gravity, imu_projected_gravity_with_errors),
      ):
        self.assertIs(cfg.observations["actor"].terms[name].params["source_func"], corrupted)
        for group in ("critic", "estimator"):
          self.assertIs(cfg.observations[group].terms[name].params["source_func"], nominal)
          self.assertEqual(cfg.observations[group].terms[name].delay_max_lag, 0)
      self.assertEqual(cfg.observations["actor"].history_length, 6)
      self.assertEqual(cfg.events["reset_robot_joints"].params["position_range"], (0.0, 0.0))
      self.assertEqual(cfg.events["reset_robot_joints"].params["velocity_range"], (0.0, 0.0))
      self.assertEqual(len(cfg.events["foot_contact_softness"].params["asset_cfg"].geom_names), 4)
      play = factory(play=True)
      for name in ("joint_friction", "joint_armature", "joint_damping", "imu_errors",
                   "external_force", "foot_contact_softness"):
        self.assertIn(name, cfg.events)
        self.assertNotIn(name, play.events)
      self.assertEqual(cfg.events["external_force"].mode, "step")


if __name__ == "__main__":
  unittest.main()
