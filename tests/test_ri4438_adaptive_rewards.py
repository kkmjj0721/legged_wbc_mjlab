"""Behavioral checks for terrain relaxation and complete foot-contact sequences.

Run with: .venv/bin/python -m unittest discover -s tests -p 'test_ri4438_adaptive_rewards.py'
"""

import math
from types import SimpleNamespace as NS
import unittest

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.locomotion.ri_4438_him.mdp import rewards as r
from src.tasks.locomotion.ri_4438_him.mdp.terrain import (
  TerrainDifficultyCfg, reset_terrain_difficulty, terrain_difficulty,
)


def make_env(n=2, dt=0.02):
  axis = torch.arange(-3, 4, dtype=torch.float32) * 0.1
  x, y = torch.meshgrid(axis, axis, indexing="ij")
  hits = torch.stack((x.flatten(), y.flatten(), torch.zeros(x.numel())), dim=-1)
  hits = hits.unsqueeze(0).repeat(n, 1, 1)
  scan = NS(
    hit_pos_w=hits,
    distances=torch.ones(n, x.numel()),
    normals_w=torch.tensor([0., 0., 1.]).expand(n, x.numel(), 3).clone(),
    frame_pos_w=torch.tensor([0., 0., 0.3]).expand(n, 1, 3).clone(),
  )
  feet = torch.zeros(n, 4, 3)
  feet[..., 2] = 0.01573
  data = NS(
    site_pos_w=feet,
    site_lin_vel_w=torch.zeros_like(feet),
    root_link_lin_vel_b=torch.zeros(n, 3),
    joint_pos=torch.full((n, 3), 0.1),
    default_joint_pos=torch.zeros(n, 3),
    projected_gravity_b=torch.tensor([0.2, 0.1, -0.9747]).expand(n, 3).clone(),
  )
  contact = NS(
    found=torch.ones(n, 4),
    force=torch.tensor([0., 0., 10.]).expand(n, 4, 3).clone(),
    current_contact_time=torch.ones(n, 4),
  )
  command = torch.tensor([0.5, 0., 0.]).expand(n, 3).clone()
  robot = NS(data=data, find_joints=lambda names: ([0, 1, 2], ["hip", "thigh", "calf"]))
  return NS(
    num_envs=n, device="cpu", step_dt=dt, common_step_counter=0,
    episode_length_buf=torch.zeros(n, dtype=torch.long),
    scene={"robot": robot, "terrain_scan": NS(data=scan), "clearance_scan": NS(data=scan),
           "feet_ground_contact": NS(data=contact)},
    command_manager=NS(get_command=lambda name: command),
    extras={"log": {}}, command=command,
  )


def roughen(env, ids=slice(None)):
  hits = env.scene["terrain_scan"].data.hit_pos_w
  hits[ids, :, 2] = (hits[ids, :, 0] > 0).float() * 0.15


def next_step(env):
  env.common_step_counter += 1
  env.episode_length_buf += 1


FEET = SceneEntityCfg("robot", site_ids=[0, 1, 2, 3])


class TerrainTests(unittest.TestCase):
  def test_flat_rough_and_world_translation(self):
    env = make_env(3)
    roughen(env, 1)
    scan = env.scene["terrain_scan"].data
    scan.hit_pos_w[2] += torch.tensor([100., -100., 12.])
    scan.frame_pos_w[2] += torch.tensor([100., -100., 12.])
    q = terrain_difficulty(env, TerrainDifficultyCfg())
    torch.testing.assert_close(q, torch.tensor([0., 1., 0.]))

  def test_smoothing_once_per_step_and_partial_reset(self):
    env = make_env()
    cfg = TerrainDifficultyCfg()
    terrain_difficulty(env, cfg)
    roughen(env)
    next_step(env)
    first = terrain_difficulty(env, cfg).clone()
    torch.testing.assert_close(first, torch.full((2,), 1 - math.exp(-0.02 / 0.06)))
    for _ in range(5):
      torch.testing.assert_close(terrain_difficulty(env, cfg), first)
    reset_terrain_difficulty(env, torch.tensor([0]))
    env.scene["terrain_scan"].data.hit_pos_w[0, :, 2] = 0
    q = terrain_difficulty(env, cfg)
    self.assertEqual(q[0].item(), 0.)
    self.assertEqual(q[1].item(), first[1].item())

  def test_invalid_scan_preserves_difficulty_and_stays_finite(self):
    env = make_env()
    cfg = TerrainDifficultyCfg()
    roughen(env)
    terrain_difficulty(env, cfg)
    scan = env.scene["terrain_scan"].data
    scan.distances.fill_(-1)
    scan.hit_pos_w.fill_(float("nan"))
    next_step(env)
    torch.testing.assert_close(terrain_difficulty(env, cfg), torch.ones(2))
    reset_terrain_difficulty(env)
    torch.testing.assert_close(terrain_difficulty(env, cfg), torch.zeros(2))


class RelaxationTests(unittest.TestCase):
  def setUp(self):
    self.env = make_env()
    roughen(self.env, 1)
    self.cfg = TerrainDifficultyCfg()

  def test_gait_flat_parity_rough_freedom_and_standing(self):
    env = self.env
    params = dict(period=0.6, offset=[0., 0.5, 0.5, 0.], threshold=0.56,
                  command_threshold=0.1, command_name="twist", sensor_name="feet_ground_contact")
    env.episode_length_buf.fill_(5)
    contact = env.scene["feet_ground_contact"].data.current_contact_time
    contact[:] = torch.tensor([0., 1., 1., 0.])  # Opposite of the desired contacts.
    reward = r.feet_gait(env, **params, terrain_cfg=self.cfg)
    torch.testing.assert_close(reward, torch.tensor([0., 0.85]))
    contact[:] = 1 - contact
    torch.testing.assert_close(r.feet_gait(env, **params, terrain_cfg=self.cfg), torch.ones(2))
    env.command.zero_()
    torch.testing.assert_close(r.feet_gait(env, **params, terrain_cfg=self.cfg), torch.zeros(2))

  def test_z_hip_orientation_and_pose_relaxation(self):
    env = self.env
    env.scene["robot"].data.root_link_lin_vel_b[:] = torch.tensor([0.5, 0., 0.3])
    reward = r.track_linear_velocity(env, 0.5, "twist", terrain_cfg=self.cfg)
    torch.testing.assert_close(reward, torch.tensor([math.exp(-0.72), math.exp(-0.072)]))
    hip = r.hip_joint_deviation_penalty(env, "twist", terrain_cfg=self.cfg)
    self.assertAlmostEqual((hip[1] / hip[0]).item(), 0.3, places=6)
    orientation = r.body_orientation_l2(env, SceneEntityCfg("robot", body_ids=[]), self.cfg)
    self.assertAlmostEqual((orientation[1] / orientation[0]).item(), 0.5, places=6)
    std = {"hip": 0.1, "thigh": 0.2, "calf": 0.3}
    params = dict(std_standing=std, std_walking=std, std_running=std,
                  command_name="twist", asset_cfg=SceneEntityCfg("robot", joint_names=".*"))
    pose = r.variable_posture(RewardTermCfg(func=r.variable_posture, weight=0.7, params=params), env)
    flat = pose(env, **params)
    adaptive = pose(env, **params, terrain_cfg=self.cfg)
    self.assertEqual(flat[0].item(), adaptive[0].item())
    self.assertGreater(adaptive[1].item(), adaptive[0].item())


class ClearanceTests(unittest.TestCase):
  def setUp(self):
    self.env = make_env(1)
    env = self.env
    env.scene["robot"].data.site_pos_w[..., 2] = 0.04
    env.scene["robot"].data.site_lin_vel_w[..., 0] = 0.5
    env.scene["feet_ground_contact"].data.found.zero_()
    env.scene["feet_ground_contact"].data.force.zero_()
    env.scene["clearance_scan"].data = NS(
      hit_pos_w=torch.tensor([[[0.05, 0., 0.10], [0.5, 0.5, 2.0]]]),
      distances=torch.ones(1, 2), normals_w=torch.tensor([[[0., 0., 1.], [0., 0., 1.]]]),
    )

  def cost(self):
    return r.feet_clearance_adaptive(self.env, "feet_ground_contact", "clearance_scan", "twist", FEET)

  def test_obstacle_above_foot_and_no_high_lift_penalty(self):
    self.assertGreater(self.cost().item(), 0)
    self.env.scene["robot"].data.site_pos_w[..., 2] = 0.3
    self.assertEqual(self.cost().item(), 0)

  def test_path_follows_motion_and_ignores_unrelated_obstacle(self):
    self.env.scene["robot"].data.site_lin_vel_w[..., 0] = -0.5
    self.assertEqual(self.cost().item(), 0)

  def test_stance_standing_missing_and_side_collision(self):
    contact = self.env.scene["feet_ground_contact"].data
    contact.found.fill_(1)
    contact.force[..., 2] = 10
    self.assertEqual(self.cost().item(), 0)
    contact.force[:] = torch.tensor([10., 0., 0.])
    self.assertGreater(self.cost().item(), 0)
    self.env.command.zero_()
    self.assertEqual(self.cost().item(), 0)
    self.env.command[:, 0] = 0.5
    self.env.scene["clearance_scan"].data.distances.fill_(-1)
    self.env.scene["clearance_scan"].data.hit_pos_w.fill_(float("nan"))
    self.assertEqual(self.cost().item(), 0)


class SwingTests(unittest.TestCase):
  def setUp(self):
    self.env = make_env()
    self.params = dict(sensor_name="feet_ground_contact", command_name="twist", asset_cfg=FEET,
                       terrain_cfg=TerrainDifficultyCfg())
    self.term = r.feet_swing_peak(RewardTermCfg(func=r.feet_swing_peak, weight=-0.02, params=self.params), self.env)

  def step(self, z, support=True, side_hit=False):
    env = self.env
    next_step(env)
    env.scene["robot"].data.site_pos_w[..., 2] = z
    contact = env.scene["feet_ground_contact"].data
    contact.found.fill_(float(support or side_hit))
    contact.force[:] = torch.tensor([10., 0., 0.] if side_hit else [0., 0., 10. if support else 0.])
    return self.term(env, **self.params)

  def establish_support(self):
    self.step(1.01573)
    self.step(1.01573)

  def test_once_per_landing_and_no_reward_from_stepping_down(self):
    self.establish_support()
    self.step(1.04573, support=False)
    self.step(1.04573, support=False)
    self.assertTrue((self.step(0.51573) == 0).all())
    cost = self.step(0.51573) * self.env.step_dt
    # Actual lift was 3 cm; a 50 cm drop below the foot must not satisfy the band.
    torch.testing.assert_close(cost, torch.full((2,), 4 * (0.03 / 0.04)**2))
    self.assertTrue((self.step(0.51573) == 0).all())

  def test_good_swing_and_rough_terrain_have_no_band_penalty(self):
    roughen(self.env, 1)
    self.establish_support()
    self.step(1.09573, support=False)
    self.step(1.09573, support=False)
    self.step(1.01573)
    self.assertTrue((self.step(1.01573) == 0).all())
    self.step(1.19573, support=False)
    self.step(1.19573, support=False)
    self.env.scene["terrain_scan"].data.hit_pos_w[..., 2] = 0
    self.step(1.01573)
    cost = self.step(1.01573)
    self.assertGreater(cost[0].item(), 0)
    self.assertEqual(cost[1].item(), 0)  # Retain roughness seen during the swing.

  def test_partial_reset_does_not_bill_old_swing(self):
    self.establish_support()
    self.step(1.04573, support=False)
    self.step(1.04573, support=False)
    self.term.reset(torch.tensor([0]))
    reset_terrain_difficulty(self.env, torch.tensor([0]))
    self.step(1.01573)
    cost = self.step(1.01573)
    self.assertEqual(cost[0].item(), 0)
    self.assertGreater(cost[1].item(), 0)

  def test_event_cost_is_independent_of_control_dt(self):
    costs = []
    for dt in (0.01, 0.02):
      self.env = make_env(dt=dt)
      self.term = r.feet_swing_peak(
        RewardTermCfg(func=r.feet_swing_peak, weight=-0.02, params=self.params), self.env,
      )
      self.establish_support()
      for _ in range(round(0.04 / dt)):
        self.step(1.04573, support=False)
      self.step(1.01573)
      # Match RewardManager's configured weight and integration over one step.
      costs.append(self.step(1.01573) * -0.02 * dt)
    torch.testing.assert_close(costs[0], costs[1])
    self.assertTrue((costs[0] < 0).all())

  def test_spawn_contact_chatter_side_hits_and_zero_command(self):
    self.step(1.04, support=False)
    self.step(1.04, support=False)
    self.establish_support()  # Spawn landing is not a completed swing.
    self.step(1.02, support=False)  # One-frame loss of support is ignored.
    self.step(1.01573)
    self.assertTrue((self.step(1.01573) == 0).all())
    self.step(1.04573, support=False)
    self.step(1.04573, support=False)
    for _ in range(3):
      self.assertTrue((self.step(1.04573, support=False, side_hit=True) == 0).all())
    self.step(1.01573)
    self.assertTrue((self.step(1.01573) > 0).all())
    self.env.command.zero_()
    self.step(1.04573, support=False)
    self.step(1.04573, support=False)
    self.step(1.01573)
    self.assertTrue((self.step(1.01573) == 0).all())


if __name__ == "__main__":
  unittest.main()
