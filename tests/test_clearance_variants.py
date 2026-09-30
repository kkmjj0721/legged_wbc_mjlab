"""Unit tests for the experimental foot-clearance reward variants.

Pure-CPU, no simulator.  They pin down the maths of

* :mod:`tests.exp.clearance_variants` kernels and reward terms, and
* the equivalence of the legacy re-implementations with the repository code.

Run with::

    uv run python -m unittest tests.test_clearance_variants -v
"""

from __future__ import annotations

import math
import sys
import types
import unittest

import torch

# --------------------------------------------------------------------------- #
# Minimal fakes so the module can be imported and exercised without mjlab envs.
# --------------------------------------------------------------------------- #


class _FakeEntityData:
  def __init__(self, num_envs: int, num_feet: int):
    self.site_pos_w = torch.zeros(num_envs, num_feet, 3)
    self.site_lin_vel_w = torch.zeros(num_envs, num_feet, 3)


class _FakeEntity:
  def __init__(self, num_envs: int, num_feet: int, site_names):
    self.data = _FakeEntityData(num_envs, num_feet)
    self.site_names = list(site_names)

  def find_geoms(self, patterns):  # pragma: no cover - not used by variants
    return None


class _FakeContactData:
  def __init__(self, num_envs: int, num_feet: int):
    self.current_contact_time = torch.zeros(num_envs, num_feet)


class _FakeContactSensor:
  def __init__(self, num_envs: int, num_feet: int, primary_names):
    self.data = _FakeContactData(num_envs, num_feet)
    self.primary_names = list(primary_names)


class _FakeHeightSensor:
  def __init__(self, num_envs: int, num_feet: int):
    self.data = types.SimpleNamespace(heights=torch.zeros(num_envs, num_feet))


class _FakeSensors(dict):
  def get(self, key, default=None):
    return dict.get(self, key, default)


class _FakeScene:
  def __init__(self, entities, sensors):
    self._entities = entities
    self.sensors = _FakeSensors(sensors)

  def __getitem__(self, key):
    if key in self._entities:
      return self._entities[key]
    if key in self.sensors:
      return self.sensors[key]
    raise KeyError(key)


class _FakeCommandManager:
  def __init__(self, command: torch.Tensor):
    self._command = command

  def get_command(self, name):
    assert name == "twist"
    return self._command


class FakeEnv:
  """Just enough of ManagerBasedRlEnv for the reward terms under test."""

  def __init__(self, num_envs=4, num_feet=4, command_value=1.0, reverse_contact=False):
    self.num_envs = num_envs
    self.device = "cpu"
    self.step_dt = 0.02
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self.extras: dict = {"log": {}}
    site_names = ["FL", "FR", "RL", "RR"][:num_feet]
    self.robot = _FakeEntity(num_envs, num_feet, site_names)
    contact_names = [f"{n}_foot_collision" for n in site_names]
    if reverse_contact:
      contact_names = list(reversed(contact_names))
    self.contact = _FakeContactSensor(num_envs, num_feet, contact_names)
    self.heights = _FakeHeightSensor(num_envs, num_feet)
    self.scene = _FakeScene(
      {"robot": self.robot},
      {"feet_ground_contact": self.contact, "feet_terrain_height": self.heights},
    )
    command = torch.zeros(num_envs, 3)
    command[:, 0] = command_value
    self.command_manager = _FakeCommandManager(command)


def make_cfg(params: dict, weight: float = -1.0):
  from mjlab.managers.reward_manager import RewardTermCfg

  return RewardTermCfg(func=None, weight=weight, params=params)


# Make the modules importable when unittest is discovered from the repo root.
sys.path.insert(0, ".")  # noqa: E402

from tests.exp import clearance_variants as cv  # noqa: E402


class BandCostTest(unittest.TestCase):
  def test_zero_inside_band_and_boundaries(self):
    h = torch.tensor([0.06, 0.08, 0.10, 0.12, 0.14])
    cost = cv.band_cost(h, 0.06, 0.14)
    self.assertTrue(torch.allclose(cost, torch.zeros(5)))

  def test_asymmetry(self):
    h = torch.tensor([0.02, 0.18])
    cost = cv.band_cost(h, 0.06, 0.14, gain_below=1.0, gain_above=0.25)
    # Both violations are 0.04 m, normalised by the 0.08 m band width -> 0.5.
    self.assertAlmostEqual(cost[0].item(), 0.25, places=5)
    self.assertAlmostEqual(cost[1].item(), 0.25 * 0.25, places=5)

  def test_scale_invariance(self):
    h = torch.tensor([0.0, 0.4])
    narrow = cv.band_cost(h, 0.1, 0.2)
    wide = cv.band_cost(h, 0.1, 0.6)
    # Below-side violation relative to the band width is identical for the
    # first sample (0.1/0.1 == 0.1/0.5 * 5); check the *normalisation* instead.
    self.assertGreater(narrow[0].item(), wide[0].item())
    explicit = cv.band_cost(h, 0.1, 0.2, scale=0.25)
    self.assertAlmostEqual(explicit[0].item(), (0.1 / 0.25) ** 2, places=6)

  def test_l2_smooth_at_boundary(self):
    eps = 1e-5
    inside = cv.band_cost(torch.tensor([0.06 - eps]), 0.06, 0.14)
    outside = cv.band_cost(torch.tensor([0.06 + eps]), 0.06, 0.14)
    self.assertLess(inside.item(), 1e-6)
    self.assertEqual(outside.item(), 0.0)
    below = cv.band_cost(torch.tensor([0.06 - 0.01]), 0.06, 0.14)
    # quadratic growth: halving the violation quarters the cost
    below_half = cv.band_cost(torch.tensor([0.06 - 0.005]), 0.06, 0.14)
    self.assertAlmostEqual(below.item(), 4.0 * below_half.item(), places=6)


class TargetKernelTest(unittest.TestCase):
  def test_exp_reward_peaks_at_target(self):
    h = torch.tensor([0.09, 0.09 + 0.04, 0.09 - 0.08])
    r = cv.exp_reward(h, 0.09, 0.04)
    self.assertAlmostEqual(r[0].item(), 1.0, places=6)
    self.assertAlmostEqual(r[1].item(), math.exp(-1.0), places=6)
    self.assertAlmostEqual(r[2].item(), math.exp(-4.0), places=6)

  def test_target_cost_l1_l2(self):
    h = torch.tensor([0.13])
    self.assertAlmostEqual(cv.target_cost(h, 0.09, 0.04).item(), 1.0, places=6)
    self.assertAlmostEqual(cv.target_cost(h, 0.09, 0.04, mode="l1").item(), 1.0, places=6)


class LegacyIntervalTest(unittest.TestCase):
  def test_matches_reference_formula(self):
    env = FakeEnv(num_envs=2, num_feet=4, command_value=1.0)
    env.robot.data.site_pos_w = torch.tensor(
      [
        [[0, 0, 0.02], [0, 0, 0.13], [0, 0, 0.20], [0, 0, 0.10]],
        [[0, 0, 0.10], [0, 0, 0.16], [0, 0, 0.00], [0, 0, 0.30]],
      ]
    )
    env.robot.data.site_lin_vel_w = torch.tensor(
      [
        [[1, 0, 0], [1, 0, 0], [0.5, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [2, 0, 0], [1, 0, 0], [1, 0, 0]],
      ]
    )
    asset_cfg = types.SimpleNamespace(name="robot", site_ids=slice(None))
    got = cv.feet_clearance_interval(
      env, height_range=(0.10, 0.16), command_name="twist", asset_cfg=asset_cfg
    )
    # Independent reference computation.
    z = env.robot.data.site_pos_w[:, :, 2]
    v = torch.norm(env.robot.data.site_lin_vel_w[:, :, :2], dim=-1)
    delta = (0.10 - z).clamp_min(0.0) + (z - 0.16).clamp_min(0.0)
    want = (delta * v).sum(dim=1)
    self.assertTrue(torch.allclose(got, want, atol=1e-6))

  def test_command_gate(self):
    env = FakeEnv(num_envs=2, num_feet=4, command_value=0.0)
    env.robot.data.site_pos_w[:, :, 2] = 0.5
    env.robot.data.site_lin_vel_w[:, :, 0] = 1.0
    asset_cfg = types.SimpleNamespace(name="robot", site_ids=slice(None))
    got = cv.feet_clearance_interval(
      env, height_range=(0.10, 0.16), command_name="twist", asset_cfg=asset_cfg
    )
    self.assertTrue(torch.allclose(got, torch.zeros(2)))


class ClearanceBandTest(unittest.TestCase):
  def _params(self, **overrides):
    params = dict(
      height_range=(0.06, 0.14),
      ref="terrain",
      gate="swing",
      gain_below=1.0,
      gain_above=0.25,
      mode="l2",
      vel_weight=0.0,
      vel_scale=0.2,
      combine="sum",
      period=0.6,
      offset=[0.0, 0.5, 0.5, 0.0],
      threshold=0.56,
      sensor_name="feet_ground_contact",
      height_sensor_name="feet_terrain_height",
      touchdown_tau=0.1,
      command_name="twist",
      command_threshold=0.1,
    )
    params.update(overrides)
    site_ids = torch.arange(len(params["offset"]))
    params["asset_cfg"] = types.SimpleNamespace(
      name="robot", site_ids=site_ids, site_names=("FL", "FR", "RL", "RR")
    )
    return params

  def test_terrain_reference_and_swing_gate(self):
    env = FakeEnv(num_envs=1, num_feet=4, command_value=1.0)
    term = cv.ClearanceBand(make_cfg(self._params()), env)
    env.contact.data.current_contact_time = torch.tensor(
      [[0.0, 0.5, 0.0, 0.5]]  # FL swing, FR stance, RL swing, RR stance
    )
    env.heights.data.heights = torch.tensor([[0.10, 0.02, 0.20, 0.10]])
    value = term(
      env,
      **{k: v for k, v in self._params().items() if k != "asset_cfg"},
      asset_cfg=self._params()["asset_cfg"],
    )
    # FL inside band -> 0; FR is in stance -> skipped; RL is above the band by
    # 0.06 m (0.06 / 0.08 = 0.75, times gain_above 0.25); RR in stance.
    self.assertAlmostEqual(value.item(), 0.25 * 0.75**2, places=6)

  def test_contact_permutation_is_resolved_by_name(self):
    env = FakeEnv(num_envs=1, num_feet=4, command_value=1.0, reverse_contact=True)
    params = self._params()
    term = cv.ClearanceBand(make_cfg(params), env)
    self.assertTrue(torch.equal(term._contact_perm, torch.tensor([3, 2, 1, 0])))
    # Primary order is RR, RL, FR, FL; make FR swing.  In site order that is
    # contact = [0.5, 0.0, 0.5, 0.5].
    env.contact.data.current_contact_time = torch.tensor([[0.5, 0.5, 0.0, 0.5]])
    env.heights.data.heights = torch.tensor([[0.10, 0.02, 0.10, 0.10]])
    value = term(
      env,
      **{k: v for k, v in params.items() if k != "asset_cfg"},
      asset_cfg=params["asset_cfg"],
    )
    # Only the swinging FR foot contributes: 0.04 m below the band -> 0.25.
    self.assertAlmostEqual(value.item(), (0.04 / 0.08) ** 2, places=6)

  def test_command_gate_zeroes_cost(self):
    env = FakeEnv(num_envs=1, num_feet=4, command_value=0.0)
    term = cv.ClearanceBand(make_cfg(self._params()), env)
    env.heights.data.heights = torch.zeros(1, 4)
    value = term(
      env,
      **{k: v for k, v in self._params().items() if k != "asset_cfg"},
      asset_cfg=self._params()["asset_cfg"],
    )
    self.assertEqual(value.item(), 0.0)

  def test_touchdown_reference_tracks_contact_height(self):
    params = self._params(ref="touchdown")
    env = FakeEnv(num_envs=1, num_feet=4, command_value=1.0)
    term = cv.ClearanceBand(make_cfg(params), env)
    kwargs = {k: v for k, v in params.items() if k != "asset_cfg"}
    # Start with all feet in contact on a 0.15 m step -> reference latches 0.15.
    env.contact.data.current_contact_time = torch.full((1, 4), 0.5)
    env.robot.data.site_pos_w[:, :, 2] = 0.15
    term(env, **kwargs, asset_cfg=params["asset_cfg"])
    # Lift one foot 0.10 m above the reference while swinging: clearance 0.10.
    env.contact.data.current_contact_time = torch.tensor([[0.5, 0.5, 0.5, 0.0]])
    env.robot.data.site_pos_w[0, 3, 2] = 0.25
    value = term(env, **kwargs, asset_cfg=params["asset_cfg"])
    self.assertAlmostEqual(value.item(), 0.0, places=6)
    # Same foot at reference height + 0.02 -> below band (0.06) by 0.04.
    env.robot.data.site_pos_w[0, 3, 2] = 0.17
    value = term(env, **kwargs, asset_cfg=params["asset_cfg"])
    self.assertAlmostEqual(value.item(), (0.04 / 0.08) ** 2, places=6)

  def test_reset_clears_state(self):
    params = self._params(ref="touchdown")
    env = FakeEnv(num_envs=2, num_feet=4, command_value=1.0)
    term = cv.ClearanceBand(make_cfg(params), env)
    kwargs = {k: v for k, v in params.items() if k != "asset_cfg"}
    env.contact.data.current_contact_time = torch.full((2, 4), 0.5)
    env.robot.data.site_pos_w[:, :, 2] = 0.3
    term(env, **kwargs, asset_cfg=params["asset_cfg"])
    term.reset(torch.tensor([0]))
    self.assertTrue(torch.all(term._ref_init[0].logical_not()))
    self.assertTrue(torch.all(term._ref_init[1]))


class ClearanceRewardTest(unittest.TestCase):
  def _params(self, **overrides):
    params = dict(
      ref="terrain",
      gate="swing",
      target_height=0.09,
      sigma=0.04,
      sensor_name="feet_ground_contact",
      height_sensor_name="feet_terrain_height",
      touchdown_tau=0.1,
      period=0.6,
      offset=[0.0, 0.5, 0.5, 0.0],
      threshold=0.56,
      command_name="twist",
      command_threshold=0.1,
    )
    params.update(overrides)
    params["asset_cfg"] = types.SimpleNamespace(
      name="robot",
      site_ids=torch.arange(4),
      site_names=("FL", "FR", "RL", "RR"),
    )
    return params

  def test_reward_only_counts_swing_feet_at_target(self):
    params = self._params()
    env = FakeEnv(num_envs=1, num_feet=4, command_value=1.0)
    term = cv.ClearanceReward(make_cfg(params, weight=1.0), env)
    kwargs = {k: v for k, v in params.items() if k != "asset_cfg"}
    env.contact.data.current_contact_time = torch.tensor([[0.0, 0.5, 0.0, 0.5]])
    # Two swing feet: FL exactly at target, RL far from target; stance feet at 0.
    env.heights.data.heights = torch.tensor([[0.09, 0.0, 0.20, 0.0]])
    value = term(env, **kwargs, asset_cfg=params["asset_cfg"])
    want = (1.0 + math.exp(-((0.20 - 0.09) / 0.04) ** 2)) / 2.0
    self.assertAlmostEqual(value.item(), want, places=5)

  def test_no_active_feet_is_finite(self):
    params = self._params()
    env = FakeEnv(num_envs=1, num_feet=4, command_value=1.0)
    term = cv.ClearanceReward(make_cfg(params, weight=1.0), env)
    kwargs = {k: v for k, v in params.items() if k != "asset_cfg"}
    env.contact.data.current_contact_time = torch.full((1, 4), 0.5)
    value = term(env, **kwargs, asset_cfg=params["asset_cfg"])
    self.assertEqual(value.item(), 0.0)
    self.assertTrue(torch.isfinite(value).all())


class PhaseGateTest(unittest.TestCase):
  def test_trot_phase_alternates(self):
    env = FakeEnv(num_envs=1, num_feet=4, command_value=1.0)
    offs = [0.0, 0.5, 0.5, 0.0]
    period_steps = int(round(0.6 / env.step_dt))  # 30
    stance_steps = math.ceil(0.56 * period_steps)  # 17
    swings = []
    for step in range(period_steps):
      env.episode_length_buf[0] = step
      swings.append(cv.phase_swing(env, 0.6, offs, 0.56)[0].clone())
    swings = torch.stack(swings)
    # Each leg swings for exactly period - stance steps.
    counts = swings.sum(dim=0)
    self.assertTrue(torch.all(counts == period_steps - stance_steps))
    # Diagonal legs are in phase, left/right legs are anti-phase.
    self.assertTrue(torch.equal(swings[:, 0], swings[:, 3]))
    self.assertTrue(torch.equal(swings[:, 1], swings[:, 2]))
    self.assertFalse(torch.any(swings[:, 0] & swings[:, 1]))


if __name__ == "__main__":
  unittest.main()
