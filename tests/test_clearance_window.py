"""Boundary tests for the design-doc foot-clearance window reward.

Exercises :class:`tests.exp.clearance_window.FeetClearanceWindow` with a hand
driven fake env (no simulator) and pins every row of the design doc's
section 5.1 "内存边界测试" table plus the parameter validation rules of
section 4.

Run with::

    .venv/bin/python -m unittest tests.test_clearance_window -v
"""

from __future__ import annotations

import math
import sys
import types
import unittest

import torch

sys.path.insert(0, ".")

from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402

from tests.exp.clearance_window import FeetClearanceWindow  # noqa: E402

FEET = ("FL", "FR", "RL", "RR")
REST = 0.01573
TARGET = 0.10
MAXH = 0.20
NORM = TARGET - REST


# --------------------------------------------------------------------------- #
# Fake env
# --------------------------------------------------------------------------- #
class _Sensors(dict):
  def get(self, key, default=None):
    return dict.get(self, key, default)


class _Scene:
  def __init__(self, entities, sensors):
    self._entities = entities
    self.sensors = _Sensors(sensors)

  def __getitem__(self, key):
    if key in self._entities:
      return self._entities[key]
    return self.sensors[key]


class Rig:
  """Scriptable four-foot contact/height source for the reward term."""

  def __init__(self, num_envs: int = 1, command: float = 1.0):
    self.B = num_envs
    self.found = torch.zeros(num_envs, 4)
    self.force = torch.zeros(num_envs, 4, 3)
    self.heights = torch.zeros(num_envs, 4)
    self.contact = types.SimpleNamespace(
      data=types.SimpleNamespace(
        found=self.found, force=self.force, current_contact_time=None
      ),
      primary_names=[f"{n}_foot_collision" for n in FEET],
    )
    self.height_sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(heights=self.heights)
    )
    robot = types.SimpleNamespace(
      data=types.SimpleNamespace(site_pos_w=torch.zeros(num_envs, 4, 3))
    )
    cmd = torch.zeros(num_envs, 3)
    cmd[:, 0] = command
    self.env = types.SimpleNamespace(
      num_envs=num_envs,
      device="cpu",
      step_dt=0.02,
      command_manager=types.SimpleNamespace(get_command=lambda name: cmd),
      extras={"log": {}},
      scene=_Scene(
        {"robot": robot},
        {
          "feet_ground_contact": self.contact,
          "feet_terrain_height": self.height_sensor,
        },
      ),
    )

  # ---------------------------------------------------------------- scripting
  def set_foot(
    self,
    foot: int,
    *,
    contact: bool,
    height: float = REST,
    fz: float = 30.0,
    fx: float = 0.0,
    fy: float = 0.0,
    env: int | None = None,
  ) -> None:
    rows = range(self.B) if env is None else (env,)
    for e in rows:
      self.found[e, foot] = 1.0 if contact else 0.0
      self.force[e, foot, 0] = fx if contact else 0.0
      self.force[e, foot, 1] = fy if contact else 0.0
      self.force[e, foot, 2] = fz if contact else 0.0
      self.heights[e, foot] = height

  def stand(self, foot: int, height: float = REST, fz: float = 30.0, **kw) -> None:
    self.set_foot(foot, contact=True, height=height, fz=fz, **kw)

  def stand_all(self, height: float = REST, fz: float = 30.0) -> None:
    for f in range(4):
      self.stand(f, height, fz)

  def air(self, foot: int, height: float, **kw) -> None:
    self.set_foot(foot, contact=False, height=height, **kw)

  def air_all(self, height: float, **kw) -> None:
    for f in range(4):
      self.air(f, height, **kw)

  def wall(self, foot: int, height: float = 0.08, fz: float = 30.0) -> None:
    """A riser/edge hit: large tangential force, too high to be support."""
    self.set_foot(foot, contact=True, height=height, fz=fz, fx=8.0 * fz)


def make_term(rig: Rig, **overrides):
  params = dict(
    height_range=(TARGET, MAXH),
    rest_height=REST,
    step_window=1.0,
    late_ramp=0.4,
    min_air_time=0.04,
    contact_confirm=0.04,
    support_height=0.04,
    min_support_force=1.0,
    wall_force_ratio=5.0,
    command_name="twist",
    command_threshold=0.1,
    sensor_name="feet_ground_contact",
    height_sensor_name="feet_terrain_height",
    asset_cfg=types.SimpleNamespace(
      name="robot", site_ids=torch.arange(4), site_names=FEET
    ),
  )
  params.update(overrides)
  cfg = RewardTermCfg(func=None, weight=-0.25, params=params)
  term = FeetClearanceWindow(cfg, rig.env)
  return term


def call(term: FeetClearanceWindow, rig: Rig) -> torch.Tensor:
  return term(rig.env, **term.p)


def swing_heights(peak: float, frames: int) -> list[float]:
  return [peak * math.sin(math.pi * (i + 0.5) / frames) for i in range(frames)]


def run_cycles(
  rig: Rig,
  term: FeetClearanceWindow,
  cycles: int = 3,
  stance: int = 18,
  peak: float = 0.12,
  swing: int = 12,
  delay: int = 0,
  pattern: list[int] | None = None,
) -> list[float]:
  """Run ``cycles`` of ``stance`` support frames + ``swing`` airborne frames.

  ``delay`` shifts each foot's cycle by that many frames (a clock offset of
  ``0.02 * delay`` seconds).  ``pattern`` optionally overrides the per-foot
  cycle offset in frames.
  """
  heights = swing_heights(peak, swing)
  period = stance + swing
  total = delay + cycles * period
  costs = []
  for step in range(total):
    for f in range(4):
      offset = (pattern[f] if pattern else 0) + delay
      local = step - offset
      if local < 0:
        rig.stand(f)
        continue
      phase = local % period
      if phase < stance:
        rig.stand(f)
      else:
        rig.air(f, max(heights[phase - stance], REST))
    costs.append(float(call(term, rig)))
  # Let the last landing settle (contact_confirm frames of support).
  for _ in range(4):
    rig.stand_all()
    costs.append(float(call(term, rig)))
  return costs


def steady(costs: list[float], tail: int = 20) -> float:
  return sum(costs[-tail:]) / min(len(costs), tail)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class QualifiedTrajectoryTest(unittest.TestCase):
  def test_qualified_gait_costs_nothing(self):
    rig, term = Rig(), None
    term = make_term(rig)
    costs = run_cycles(rig, term, cycles=3)
    self.assertAlmostEqual(steady(costs), 0.0, places=6)

  def test_clock_shift_does_not_add_penalty(self):
    """A qualified trajectory shifted by 0.1 s scores the same."""
    base_rig, base_term = Rig(), None
    base_term = make_term(base_rig)
    base = run_cycles(base_rig, base_term, cycles=3, delay=0)

    shifted_rig, shifted_term = Rig(), None
    shifted_term = make_term(shifted_rig)
    shifted = run_cycles(shifted_rig, shifted_term, cycles=3, delay=5)

    self.assertAlmostEqual(steady(base), steady(shifted), places=6)
    self.assertAlmostEqual(steady(shifted), 0.0, places=6)

  def test_longer_step_period_still_free(self):
    """0.6 -> 0.8 / 0.9 s cycles stay inside the 1.0 s window."""
    for stance, label in ((28, "0.80 s"), (33, "0.90 s")):
      rig = Rig()
      term = make_term(rig)
      costs = run_cycles(rig, term, cycles=2, stance=stance)
      self.assertLess(max(costs), 1e-6, msg=label)
    # Past step_window + late_ramp the late penalty is expected, and bounded.
    rig = Rig()
    term = make_term(rig)
    costs = run_cycles(rig, term, cycles=2, stance=54)  # 1.32 s cycle
    self.assertGreater(max(costs), 0.0)
    self.assertLessEqual(max(costs), 2.0 + 1e-6)


class TimeoutTest(unittest.TestCase):
  def test_all_feet_grounded_reaches_full_penalty(self):
    rig = Rig()
    term = make_term(rig)
    costs = []
    for _ in range(120):  # 2.4 s
      rig.stand_all()
      costs.append(float(call(term, rig)))
    self.assertAlmostEqual(costs[49], 0.0, places=6)  # t = 1.0 s: still on time
    self.assertAlmostEqual(costs[-1], 2.0, places=5)  # t >= 1.4 s: saturated

  def test_single_foot_airborne_contributes_an_eighth(self):
    """One foot never lands: after the grace it contributes -0.125 weighted."""
    rig = Rig()
    term = make_term(rig)
    heights = swing_heights(0.12, 12)
    for step in range(4 * 30 + 10):
      for f in (0, 1, 2):
        phase = step % 30
        if phase < 18:
          rig.stand(f)
        else:
          rig.air(f, max(heights[phase - 18], REST))
      rig.air(3, 0.12)
      cost = float(call(term, rig))
    # 0.75 * 0 + 0.25 * 1 = 0.25 -> raw 0.5 -> weighted -0.125.
    self.assertAlmostEqual(cost, 0.5, places=5)
    self.assertAlmostEqual(-0.25 * cost, -0.125, places=5)

  def test_wall_contact_cannot_settle_a_step(self):
    rig = Rig()
    term = make_term(rig)
    heights = swing_heights(0.12, 12)
    for step in range(4 * 30 + 10):
      for f in (1, 2, 3):
        phase = step % 30
        if phase < 18:
          rig.stand(f)
        else:
          rig.air(f, max(heights[phase - 18], REST))
      # The blocked foot keeps slamming into a riser instead of landing.
      if step % 2 == 0:
        rig.air(0, 0.05)
      else:
        rig.wall(0, 0.08)
      cost = float(call(term, rig))
    self.assertEqual(float(term.last_deficit[0, 0]), 0.0)
    self.assertGreater(float(term.settled_steps), 0.0)  # the other feet settle
    # Only the blocked foot is late: 0.25 * 1 -> raw 0.5.
    self.assertAlmostEqual(cost, 0.5, places=5)


class StepDetectionTest(unittest.TestCase):
  def test_single_frame_contact_break_is_not_a_step(self):
    rig = Rig()
    term = make_term(rig)
    heights = swing_heights(0.12, 12)
    # Confirm support, then take off.
    for _ in range(4):
      rig.stand_all()
      call(term, rig)
    settled_before = float(term.settled_steps)
    # Flight with a one-frame contact break in the middle.  The break is far
    # too short to be a fresh support (contact_confirm = 0.04 s), so it must
    # neither settle the current step nor start a new one.
    profile = [("air", heights[i]) for i in range(5)]
    profile.append(("stand", REST))
    profile += [("air", heights[i]) for i in range(5, 12)]
    for kind, h in profile:
      for f in range(4):
        if kind == "stand":
          rig.stand(f)
        else:
          rig.air(f, max(h, REST))
      call(term, rig)
    self.assertEqual(float(term.settled_steps), settled_before)
    self.assertTrue(bool((term.in_flight).all()))
    self.assertGreater(float(term.peak.mean()), 0.05)  # record preserved
    # Now land and stay: exactly one step per foot settles, with the peak
    # collected across both airborne segments (a qualified 0.12 profile).
    for _ in range(6):
      rig.stand_all()
      call(term, rig)
    self.assertEqual(float(term.settled_steps), settled_before + 4.0)
    self.assertLess(float(term.last_deficit.mean()), 1e-6)

  def test_born_airborne_then_landing_gets_no_credit(self):
    rig = Rig()
    term = make_term(rig)
    for _ in range(15):
      rig.air_all(0.12)
      call(term, rig)
    for _ in range(10):
      rig.stand_all()
      call(term, rig)
    self.assertEqual(float(term.settled_steps), 0.0)
    self.assertLess(float(term.last_deficit.mean()), 1e-6)

  def test_short_air_time_is_not_a_step(self):
    """A 0.02 s hop (one frame) is below min_air_time and cannot settle."""
    rig = Rig()
    term = make_term(rig)
    for _ in range(4):
      rig.stand_all()
      call(term, rig)
    rig.air_all(0.12)
    call(term, rig)
    for _ in range(4):
      rig.stand_all()
      call(term, rig)
    self.assertEqual(float(term.settled_steps), 0.0)


class DeficitTest(unittest.TestCase):
  @staticmethod
  def _expected_deficit(peak: float, frames: int = 12, spike_at: int | None = None):
    """Deficit implied by the sustained (two-frame min) peak of the profile."""
    heights = [max(h, REST) for h in swing_heights(peak, frames)]
    if spike_at is not None:
      heights[spike_at] = 0.15
    sustained = max(
      min(heights[i], heights[i + 1]) for i in range(len(heights) - 1)
    )
    return ((TARGET - sustained) / NORM) ** 2

  def test_low_amplitude_steps_stay_penalised(self):
    """~2 cm net lift every 0.6 s is not laundered by the timer refresh."""
    rig = Rig()
    term = make_term(rig)
    peak = REST + 0.02
    costs = run_cycles(rig, term, cycles=3, stance=18, swing=12, peak=peak)
    expected = self._expected_deficit(peak)
    self.assertAlmostEqual(steady(costs), 2.0 * expected, places=3)
    self.assertAlmostEqual(float(term.last_deficit.mean()), expected, places=4)
    self.assertGreater(steady(costs), 1.0)
    self.assertLess(steady(costs), 1.2)

  def test_single_frame_spike_is_filtered(self):
    """One high frame inside a low trajectory does not reset the score."""
    rig = Rig()
    term = make_term(rig)
    heights = swing_heights(REST + 0.02, 12)
    heights[5] = 0.15  # isolated spike
    for cycle in range(3):
      for i in range(18):
        rig.stand_all()
        call(term, rig)
      for h in heights:
        rig.air_all(max(h, REST))
        call(term, rig)
      for _ in range(2):
        rig.stand_all()
        call(term, rig)
    expected = self._expected_deficit(REST + 0.02, spike_at=5)
    self.assertAlmostEqual(float(term.last_deficit.mean()), expected, places=4)

  def test_low_quality_step_replaces_previous_score(self):
    rig = Rig()
    term = make_term(rig)
    # Qualified step.
    run_cycles(rig, term, cycles=1, peak=0.12)
    self.assertLess(float(term.last_deficit.mean()), 1e-6)
    # Then a low step with the same timer: the new score replaces the old one.
    run_cycles(rig, term, cycles=1, peak=REST + 0.02)
    self.assertGreater(float(term.last_deficit.mean()), 0.5)

  def test_over_height_is_penalised(self):
    rig = Rig()
    term = make_term(rig)
    for _ in range(3):
      rig.air_all(0.30)
      cost = float(call(term, rig))
    self.assertGreater(cost, 0.0)
    self.assertLessEqual(cost, 2.0)


class GatingAndResetTest(unittest.TestCase):
  def test_stop_command_returns_zero_and_clears(self):
    rig = Rig()
    term = make_term(rig)
    for _ in range(30):
      rig.stand_all()
      call(term, rig)
    self.assertGreater(float(term.age.mean()), 0.0)
    for _ in range(10):
      rig.air_all(0.12)
      call(term, rig)
    self.assertTrue(bool(term.in_flight.any()))
    rig.env.command_manager.get_command = lambda name: torch.zeros(1, 3)
    for _ in range(3):
      cost = float(call(term, rig))
    self.assertEqual(cost, 0.0)
    self.assertFalse(bool(term.in_flight.any()))
    self.assertLess(float(term.age.mean()), 1e-6)
    self.assertLess(float(term.peak.mean()), 1e-6)

  def test_local_reset_only_touches_given_envs(self):
    rig = Rig(num_envs=2)
    term = make_term(rig)
    for _ in range(60):
      rig.stand_all()
      call(term, rig)
    self.assertGreater(float(term.age[1].mean()), 0.0)
    term.reset(torch.tensor([0]))
    self.assertLess(float(term.age[0].mean()), 1e-6)
    self.assertGreater(float(term.age[1].mean()), 0.5)

  def test_full_reset_clears_everything(self):
    rig = Rig(num_envs=3)
    term = make_term(rig)
    for _ in range(60):
      rig.stand_all()
      call(term, rig)
    term.reset()
    self.assertLess(float(term.age.abs().max()), 1e-6)
    self.assertLess(float(term.last_deficit.abs().max()), 1e-6)
    self.assertFalse(bool(term.in_flight.any()))


class RobustnessTest(unittest.TestCase):
  def test_invalid_height_keeps_output_finite(self):
    rig = Rig()
    term = make_term(rig)
    for _ in range(4):
      rig.stand_all()
      call(term, rig)
    heights = [float("nan"), 0.12, -0.5, 0.12, float("inf"), 0.12]
    for h in heights:
      rig.air_all(h)
      cost = float(call(term, rig))
      self.assertTrue(math.isfinite(cost), msg=f"h={h}")
    for _ in range(4):
      rig.stand_all()
      call(term, rig)
    # NaN/negative/inf samples never refreshed the qualified peak.
    self.assertGreater(float(term.last_deficit.mean()), 0.0)

  def test_wall_force_below_ratio_counts_as_support(self):
    """|F_xy| <= 5 |F_z| at ground height is support, not a wall hit."""
    rig = Rig()
    term = make_term(rig)
    for _ in range(4):
      for f in range(4):
        rig.set_foot(f, contact=True, height=REST, fz=30.0, fx=2.0 * 30.0)
      call(term, rig)
    self.assertEqual(float(term.settled_steps), 0.0)
    self.assertAlmostEqual(float(term.support_timer.mean()), 0.08, places=6)

  def test_height_above_support_window_is_not_support(self):
    """A foot at 8 cm cannot be 'standing', even with vertical force."""
    rig = Rig()
    term = make_term(rig)
    for _ in range(4):
      rig.stand_all(height=0.08)
      call(term, rig)
    self.assertLess(float(term.support_timer.mean()), 1e-6)

  def test_parameter_validation(self):
    rig = Rig()
    with self.assertRaises(ValueError):
      make_term(rig, height_range=(0.20, 0.10))
    with self.assertRaises(ValueError):
      make_term(rig, rest_height=0.2)
    with self.assertRaises(ValueError):
      make_term(rig, late_ramp=0.0)
    with self.assertRaises(ValueError):
      make_term(rig, contact_confirm=0.0)


if __name__ == "__main__":
  unittest.main()
