"""Phase-agnostic foot-clearance reward ("window") from the RI4438 design doc.

Implements the design described in
``docs/RI4438-保留相位的足端间隙奖励设计方案.md``:

    confirm support -> actually leave the ground -> record the clearance peak
    -> confirm touchdown -> update the score and the timer.

The reward does **not** read the gait clock.  Every foot owns

* ``last_deficit``  height shortfall of its last *completed* step in ``[0, 1]``
* ``age``           seconds since its last completed step
* ``peak``          clearance peak of the step currently in progress

and pays::

    late      = clamp((age - step_window) / late_ramp, 0, 1)
    cost      = last_deficit + (1 - last_deficit) * late
    high_cost = clamp((h - max_height) / (target - rest), 0, 1) * I(airborne & valid)
    raw       = 2.0 * max(cost, high_cost).mean(dim=1)

which keeps the weighted term in ``[-|w|, 0]`` for four feet (``2 * mean`` is
``0.5 * sum``), matching the old term's range at ``weight=-0.25``.

This module is experiment sandbox code: it never modifies anything under
``src/``.  :mod:`tests.exp.window_lab` drops :class:`FeetClearanceWindow` into
an in-memory env config and resumes training from a checkpoint.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from tests.exp.clearance_variants import resolve_contact_perm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_EPS = 1e-9


def _log(env: ManagerBasedRlEnv, key: str, value: torch.Tensor) -> None:
  env.extras.setdefault("log", {})[key] = torch.nan_to_num(value.detach().mean())


class FeetClearanceWindow:
  """Per-foot "did the last completed step lift enough, and how long ago" term.

  Parameters (passed through ``RewardTermCfg.params``)
  ---------------------------------------------------
  height_range : (float, float)
      ``(target_height, max_height)`` in metres above the terrain under the
      foot site.  ``target_height`` is the fully-satisfied lift, ``max_height``
      the point where over-lifting starts to be penalised.
  rest_height : float
      Clearance of the foot site while standing on flat ground (``0.01573``
      for the current RI-4438 foot geometry).  Normalises the shortfall.
  step_window : float
      Seconds after a completed step before the late penalty starts (``1.0``).
  late_ramp : float
      Seconds over which the late penalty ramps to 1 (``0.4``).
  min_air_time : float
      Airborne time required for a takeoff to count as a step (``0.04``).
  contact_confirm : float
      Continuous support time required both to arm a step and to settle it
      (``0.04``).
  support_height : float
      Maximum terrain-relative site height (m) that can still count as support.
  min_support_force : float
      Minimum ``|F_z|`` (N) for a contact to count as support.
  wall_force_ratio : float
      Contacts with ``|F_xy| > ratio * |F_z|`` are treated as wall/edge hits,
      not support.
  command_name, command_threshold : moving gate; the term returns 0 and clears
      its per-foot records while the command is below the threshold.
  sensor_name, height_sensor_name : contact and terrain-height sensors.
  asset_cfg : robot foot sites.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.p = p
    self.asset_cfg: SceneEntityCfg = p["asset_cfg"]

    height_range = tuple(float(v) for v in p["height_range"])
    if len(height_range) != 2 or not all(math.isfinite(v) for v in height_range):
      raise ValueError("height_range must be a finite (target, max) pair")
    self.target_height, self.max_height = height_range
    self.rest_height = float(p.get("rest_height", 0.01573))
    if not (self.max_height >= self.target_height > self.rest_height >= 0.0):
      raise ValueError(
        "require max_height >= target_height > rest_height >= 0, got "
        f"{self.max_height} / {self.target_height} / {self.rest_height}"
      )

    self.step_window = float(p.get("step_window", 1.0))
    self.late_ramp = float(p.get("late_ramp", 0.4))
    self.min_air_time = float(p.get("min_air_time", 0.04))
    self.contact_confirm = float(p.get("contact_confirm", 0.04))
    if self.step_window <= 0.0 or self.late_ramp <= 0.0:
      raise ValueError("step_window and late_ramp must be > 0")
    if self.min_air_time < 0.0 or self.contact_confirm <= 0.0:
      raise ValueError("min_air_time >= 0 and contact_confirm > 0 required")

    self.support_height = float(p.get("support_height", 0.04))
    self.min_support_force = float(p.get("min_support_force", 1.0))
    self.wall_force_ratio = float(p.get("wall_force_ratio", 5.0))
    if self.support_height < 0.0 or self.min_support_force < 0.0:
      raise ValueError("support_height >= 0 and min_support_force >= 0 required")
    if self.wall_force_ratio <= 0.0:
      raise ValueError("wall_force_ratio must be > 0")

    self.sensor_name = p.get("sensor_name", "feet_ground_contact")
    self.height_sensor_name = p.get("height_sensor_name", "feet_terrain_height")
    self.command_name = p.get("command_name", "twist")
    self.command_threshold = float(p.get("command_threshold", 0.1))
    self.reward_scale = float(p.get("reward_scale", 2.0))
    self.normaliser = self.target_height - self.rest_height

    asset: Entity = env.scene[self.asset_cfg.name]
    self.n_feet = len(self.asset_cfg.site_ids)
    self._dtype = asset.data.site_pos_w.dtype
    self._device = env.device
    self._contact_perm = resolve_contact_perm(
      env, self.sensor_name, self.asset_cfg.site_names
    )

    shape = (env.num_envs, self.n_feet)
    self.last_deficit = torch.zeros(shape, dtype=self._dtype, device=env.device)
    self.age = torch.zeros(shape, dtype=self._dtype, device=env.device)
    self.peak = torch.zeros(shape, dtype=self._dtype, device=env.device)
    self.in_flight = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.air_ok = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.support_timer = torch.zeros(shape, dtype=self._dtype, device=env.device)
    self.air_timer = torch.zeros(shape, dtype=self._dtype, device=env.device)
    self.prev_valid = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.prev_height = torch.zeros(shape, dtype=self._dtype, device=env.device)
    # Running tally of settled steps, exposed for diagnostics.
    self.settled_steps = torch.zeros((), dtype=self._dtype, device=env.device)

  # ------------------------------------------------------------------ reset
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear only the given environments; other envs keep their records."""
    if env_ids is None:
      env_ids = slice(None)
    self.last_deficit[env_ids] = 0.0
    self.age[env_ids] = 0.0
    self.peak[env_ids] = 0.0
    self.in_flight[env_ids] = False
    self.air_ok[env_ids] = False
    self.support_timer[env_ids] = 0.0
    self.air_timer[env_ids] = 0.0
    self.prev_valid[env_ids] = False
    self.prev_height[env_ids] = 0.0

  # ------------------------------------------------------------------ steps
  def _command_gate(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """1 where a non-trivial velocity command is active, else 0. Shape [B]."""
    if self.command_name is None:
      return torch.ones(env.num_envs, dtype=self._dtype, device=env.device)
    command = env.command_manager.get_command(self.command_name)
    if command is None:
      return torch.ones(env.num_envs, dtype=self._dtype, device=env.device)
    command = torch.nan_to_num(command)
    moving = torch.linalg.vector_norm(command[:, :2], dim=1) > self.command_threshold
    moving = moving | (torch.abs(command[:, 2]) > self.command_threshold)
    return moving.to(self._dtype)

  def _contact(self, env: ManagerBasedRlEnv):
    """Return ``(found, force, height)`` aligned with the site order, or None."""
    sensor = env.scene.sensors.get(self.sensor_name)
    if sensor is None:
      raise RuntimeError(f"contact sensor {self.sensor_name!r} not found")
    data = sensor.data
    if data.found is None or data.force is None:
      raise RuntimeError(
        f"contact sensor {self.sensor_name!r} needs the 'found' and 'force' fields"
      )
    found = data.found
    force = data.force
    if found.shape[1] != self.n_feet:
      found = found[:, self._contact_perm]
      force = force[:, self._contact_perm]
    height = env.scene.sensors[self.height_sensor_name].data.heights
    return found, force, height

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    height_range: tuple[float, float] = (0.10, 0.20),
    rest_height: float = 0.01573,
    step_window: float = 1.0,
    late_ramp: float = 0.4,
    min_air_time: float = 0.04,
    contact_confirm: float = 0.04,
    support_height: float = 0.04,
    min_support_force: float = 1.0,
    wall_force_ratio: float = 5.0,
    command_name: str | None = "twist",
    command_threshold: float = 0.1,
    sensor_name: str = "feet_ground_contact",
    height_sensor_name: str = "feet_terrain_height",
    reward_scale: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    # All configuration is captured in ``__init__``; the keyword list mirrors
    # ``RewardTermCfg.params`` because the manager splats params into the call.
    del (
      height_range,
      rest_height,
      step_window,
      late_ramp,
      min_air_time,
      contact_confirm,
      support_height,
      min_support_force,
      wall_force_ratio,
      command_name,
      command_threshold,
      sensor_name,
      height_sensor_name,
      reward_scale,
      asset_cfg,
    )
    dt = float(env.step_dt)
    found, force, height = self._contact(env)
    active = self._command_gate(env)
    moving = active > 0.0

    raw_contact = found > 0
    force_z = force[..., 2].abs()
    force_xy = torch.linalg.vector_norm(force[..., :2], dim=-1)
    valid_height = torch.isfinite(height) & (height >= 0.0)
    height = torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)

    support = (
      raw_contact
      & (force_z > self.min_support_force)
      & (force_xy <= self.wall_force_ratio * force_z)
      & valid_height
      & (height <= self.support_height)
    )

    # ---------------------------------------------------------------- state
    # ``ready_prev`` is the previous step's confirmed-support state: a takeoff
    # is only armed if the foot was already standing on confirmed support.
    ready_prev = self.support_timer >= self.contact_confirm - _EPS
    self.support_timer = torch.where(
      support, self.support_timer + dt, torch.zeros_like(self.support_timer)
    )
    self.air_timer = torch.where(
      ~raw_contact, self.air_timer + dt, torch.zeros_like(self.air_timer)
    )
    armed = ready_prev & ~raw_contact & moving.unsqueeze(1)  # stable support lost again

    # Start a fresh step record on the first frame after a confirmed support.
    self.in_flight = self.in_flight | armed
    self.air_ok = torch.where(armed, torch.zeros_like(self.air_ok), self.air_ok)
    self.peak = torch.where(armed, torch.zeros_like(self.peak), self.peak)
    self.prev_valid = torch.where(armed, torch.zeros_like(self.prev_valid), self.prev_valid)

    # A step only counts once the foot has really been in the air for
    # ``min_air_time`` seconds; single-frame contact breaks cannot latch it.
    self.air_ok = self.air_ok | (
      self.in_flight & (self.air_timer >= self.min_air_time - _EPS)
    )

    # Peak of the sustained (two adjacent airborne samples) clearance.
    usable = self.in_flight & valid_height & ~raw_contact
    paired = usable & self.prev_valid
    sustained = torch.minimum(height, self.prev_height)
    self.peak = torch.where(paired & (sustained > self.peak), sustained, self.peak)
    self.prev_height = torch.where(usable, height, self.prev_height)
    self.prev_valid = usable

    # Only a fresh, confirmed support (>= contact_confirm s) after a real
    # airborne phase settles a step.
    settle = (
      self.in_flight
      & self.air_ok
      & support
      & (self.support_timer >= self.contact_confirm - _EPS)
    )

    # --------------------------------------------------------------- timing
    self.age = self.age + dt
    deficit = (
      ((self.target_height - self.peak) / self.normaliser).clamp(0.0, 1.0) ** 2
    )
    self.last_deficit = torch.where(settle, deficit, self.last_deficit)
    self.age = torch.where(settle, torch.zeros_like(self.age), self.age)
    self.peak = torch.where(settle, torch.zeros_like(self.peak), self.peak)
    self.in_flight = self.in_flight & ~settle
    self.air_ok = self.air_ok & ~settle
    self.settled_steps = self.settled_steps + settle.sum()

    # ---------------------------------------------------------------- costs
    late = ((self.age - self.step_window) / self.late_ramp).clamp(0.0, 1.0)
    cost = self.last_deficit + (1.0 - self.last_deficit) * late
    high_cost = (
      ((height - self.max_height) / self.normaliser).clamp(0.0, 1.0)
      * (~raw_contact & valid_height).to(self._dtype)
    )
    per_foot = torch.maximum(cost, high_cost)
    raw = self.reward_scale * per_foot.mean(dim=1) * active

    # ------------------------------------------------- inactive: clear all
    idle = ~moving
    if bool(idle.any()):
      idle2 = idle.unsqueeze(1)
      for buf in (
        self.last_deficit,
        self.age,
        self.peak,
        self.support_timer,
        self.air_timer,
        self.prev_height,
      ):
        buf[idle2.expand_as(buf)] = 0.0
      for flag in (self.in_flight, self.air_ok, self.prev_valid):
        flag[idle2.expand_as(flag)] = False

    # ---------------------------------------------------------- diagnostics
    _log(env, "Diag/Window/raw_cost", raw)
    _log(env, "Diag/Window/last_deficit", self.last_deficit)
    _log(env, "Diag/Window/age", self.age)
    _log(env, "Diag/Window/cost", per_foot)
    _log(env, "Diag/Window/late_frac", (late > 0.0).to(self._dtype))
    _log(env, "Diag/Window/full_cost_frac", (per_foot >= 1.0 - 1e-6).to(self._dtype))
    _log(env, "Diag/Window/support_frac", support.to(self._dtype))
    _log(env, "Diag/Window/flight_frac", self.in_flight.to(self._dtype))
    _log(env, "Diag/Window/air_ok_frac", self.air_ok.to(self._dtype))
    _log(env, "Diag/Window/peak_mean", self.peak)
    _log(env, "Diag/Window/steps_settled", settle.to(self._dtype))
    _log(env, "Diag/Window/height_mean", height)
    _log(env, "Diag/Window/raw_contact_frac", raw_contact.to(self._dtype))
    env.extras.setdefault("log", {})["Diag/Window/total_settled"] = self.settled_steps
    return raw
