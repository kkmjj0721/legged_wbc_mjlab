"""Foot-clearance reward variants for blind quadruped locomotion (experiment sandbox).

This module deliberately does **not** modify anything under ``src/``.  It only
provides alternative implementations of the ``foot_clearance`` reward term that
an in-memory training / diagnostics harness can drop into an env config.

Contents
--------
* Pure, unit-testable cost kernels:
    - :func:`band_cost`      asymmetric, scale-normalised band cost
    - :func:`target_cost`    distance to a target clearance
    - :func:`exp_reward`     Gaussian "clearance achieved" reward
* Stateless legacy terms reproducing the repository's two historical
  implementations:
    - :func:`feet_clearance_interval`  world-z interval x |v_xy|  (current code)
    - :func:`feet_clearance_target`    |world-z - h*| x |v_xy|    (PPO task)
* A configurable stateful term :class:`ClearanceBand` covering the proposed
  designs (terrain-relative or proprioceptive reference, swing gating,
  asymmetric cost, optional velocity weighting).
* A positive term :class:`ClearanceReward` that rewards *achieving* clearance
  during swing instead of penalising deviation.

Every term writes scalar diagnostics to ``env.extras["log"]`` under
``Diag/...`` so a training run exposes the reward's internal behaviour.

Reference frames for the clearance ``h`` of foot ``i``
-----------------------------------------------------
``world``      h = site world z.  This is what the current repository code uses,
               and it silently couples the reward to the absolute terrain level:
               on a 15 cm step the *stance* foot already sits at z = 0.15.
``terrain``    h = site world z - terrain height under the site (ray cast).
               Privileged but legal for a blind policy: only the *reward* sees
               it, the actor observation stays proprioceptive.
``touchdown``  h = site world z - reference z of the same foot, where the
               reference is an EMA of the foot height while it is in contact.
               Fully proprioceptive proxy for the local ground level; no extra
               sensor and no privileged information.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

Gate = Literal["none", "swing", "phase"]
Reference = Literal["world", "terrain", "touchdown"]


# --------------------------------------------------------------------------- #
# Pure kernels (no env, CPU-testable)
# --------------------------------------------------------------------------- #
def band_cost(
  h: torch.Tensor,
  h_min: float | torch.Tensor,
  h_max: float | torch.Tensor,
  *,
  gain_below: float = 1.0,
  gain_above: float = 0.25,
  mode: Literal["l1", "l2"] = "l2",
  scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
  """Asymmetric, bounded cost for a clearance ``h`` outside ``[h_min, h_max]``.

  ``below`` and ``above`` are normalised by the band width so the cost is
  invariant to the chosen band width and stays O(1).  No cost is charged inside
  the band (including its boundary).

  Args:
    h: clearance tensor (any shape), metres.
    h_min, h_max: band bounds; floats or broadcastable tensors.
    gain_below: weight of the under-clearance (stubbing) side.
    gain_above: weight of the over-clearance (wasted lift) side.
    mode: ``"l2"`` (smooth, zero slope at the boundary) or ``"l1"``.
    scale: normaliser; defaults to ``h_max - h_min`` (floored at 1 mm).
  """
  h_min_t = torch.as_tensor(h_min, dtype=h.dtype, device=h.device)
  h_max_t = torch.as_tensor(h_max, dtype=h.dtype, device=h.device)
  if scale is None:
    scale_t = (h_max_t - h_min_t).clamp_min(1e-3)
  else:
    scale_t = torch.as_tensor(scale, dtype=h.dtype, device=h.device).clamp_min(1e-3)
  below = ((h_min_t - h) / scale_t).clamp_min(0.0)
  above = ((h - h_max_t) / scale_t).clamp_min(0.0)
  if mode == "l2":
    return gain_below * below.square() + gain_above * above.square()
  if mode == "l1":
    return gain_below * below + gain_above * above
  raise ValueError(f"unsupported mode {mode!r}")


def target_cost(
  h: torch.Tensor,
  target: float | torch.Tensor,
  sigma: float | torch.Tensor,
  *,
  mode: Literal["l1", "l2"] = "l2",
) -> torch.Tensor:
  """Normalised distance to a single target clearance."""
  e = (h - torch.as_tensor(target, dtype=h.dtype, device=h.device)) / torch.as_tensor(
    sigma, dtype=h.dtype, device=h.device
  ).clamp_min(1e-6)
  return e.square() if mode == "l2" else e.abs()


def exp_reward(
  h: torch.Tensor,
  target: float | torch.Tensor,
  sigma: float | torch.Tensor,
) -> torch.Tensor:
  """Gaussian clearance reward in ``(0, 1]``; 1 when ``h == target``."""
  return torch.exp(-target_cost(h, target, sigma, mode="l2"))


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _log(env: ManagerBasedRlEnv, key: str, value: torch.Tensor) -> None:
  env.extras.setdefault("log", {})[key] = torch.nan_to_num(value.detach().mean())


def _command_gate(
  env: ManagerBasedRlEnv,
  command_name: str | None,
  command_threshold: float,
  dtype: torch.dtype,
) -> torch.Tensor:
  """1 where a non-trivial velocity command is active, else 0. Shape [B]."""
  if command_name is None:
    return torch.ones(env.num_envs, dtype=dtype, device=env.device)
  command = env.command_manager.get_command(command_name)
  if command is None:
    return torch.ones(env.num_envs, dtype=dtype, device=env.device)
  command = torch.nan_to_num(command)
  moving = torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold
  moving = moving | (torch.abs(command[:, 2]) > command_threshold)
  return moving.to(dtype)


def resolve_contact_perm(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: tuple[str, ...] | list[str] | None,
) -> torch.Tensor:
  """Permutation mapping site order -> contact-sensor primary order.

  The contact sensor resolves geom patterns independently from the site order in
  ``asset_cfg``; matching them by name prefix keeps every per-foot gate aligned.
  Falls back to identity when the mapping cannot be established.
  """
  sensor = env.scene.sensors.get(sensor_name)
  if sensor is None or not site_names or not hasattr(sensor, "primary_names"):
    return torch.arange(len(site_names or ()), device=env.device)
  primaries = list(sensor.primary_names)
  perm: list[int] = []
  for site in site_names:
    hit = next((i for i, n in enumerate(primaries) if n.startswith(site)), None)
    perm.append(hit if hit is not None else -1)
  if any(i < 0 for i in perm):
    return torch.arange(len(site_names), device=env.device)
  return torch.tensor(perm, device=env.device, dtype=torch.long)


def contact_state(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  perm: torch.Tensor,
) -> torch.Tensor | None:
  """Per-foot in-contact mask aligned with the site order, or None."""
  sensor = env.scene.sensors.get(sensor_name)
  if sensor is None:
    return None
  contact_time = sensor.data.current_contact_time
  if contact_time is None:
    return None
  if contact_time.shape[1] != perm.numel():
    return contact_time > 0.0
  return contact_time[:, perm] > 0.0


def phase_swing(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
) -> torch.Tensor:
  """Per-foot expected-swing mask from the trot clock. Shape [B, F]."""
  period_steps = int(round(period / env.step_dt))
  if period_steps < 3 or not math.isclose(period_steps * env.step_dt, period, abs_tol=1e-6):
    raise ValueError("period must be an integer multiple of env.step_dt")
  if not 0.0 < threshold < 1.0:
    raise ValueError("threshold must be in (0, 1)")
  stance_steps = math.ceil(threshold * period_steps)
  offset_steps = torch.tensor(
    [int(round(v * period_steps)) for v in offset], device=env.device, dtype=torch.long
  )
  leg_step = torch.remainder(
    env.episode_length_buf.to(torch.long).unsqueeze(1) + offset_steps.unsqueeze(0),
    period_steps,
  )
  return leg_step >= stance_steps


# --------------------------------------------------------------------------- #
# Stateless legacy terms
# --------------------------------------------------------------------------- #
def feet_clearance_interval(
  env: ManagerBasedRlEnv,
  height_range: tuple[float, float],
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reproduce the repository's current ``feet_clearance`` (world-z interval).

  ``c = I(cmd) * sum_i (relu(h_min - z_i) + relu(z_i - h_max)) * ||v_i,xy||``
  """
  min_height, max_height = height_range
  if not all(math.isfinite(v) for v in height_range) or min_height > max_height:
    raise ValueError("height_range must be finite with min_height <= max_height")
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  delta = (min_height - foot_z).clamp_min(0.0) + (foot_z - max_height).clamp_min(0.0)
  cost = torch.sum(delta * vel_norm, dim=1)
  cost = cost * _command_gate(env, command_name, command_threshold, cost.dtype)
  _log(env, "Diag/Clearance/raw_cost", cost)
  _log(env, "Diag/Clearance/world_z_mean", foot_z)
  return cost


def feet_clearance_target(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reproduce the PPO-task ``feet_clearance`` (|world-z - h*| x |v_xy|)."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  delta = torch.abs(foot_z - target_height)
  cost = torch.sum(delta * vel_norm, dim=1)
  cost = cost * _command_gate(env, command_name, command_threshold, cost.dtype)
  _log(env, "Diag/Clearance/raw_cost", cost)
  return cost


# --------------------------------------------------------------------------- #
# Configurable stateful term
# --------------------------------------------------------------------------- #
class ClearanceBand:
  """Configurable swing-foot clearance cost (stateful reward term).

  Parameters (all passed through ``RewardTermCfg.params``)
  -------------------------------------------------------
  height_range : (float, float)
      Target clearance band ``[h_min, h_max]`` in metres, measured in ``ref``.
  ref : {"world", "terrain", "touchdown"}
      Reference frame; see the module docstring.
  gate : {"none", "swing", "phase"}
      Which feet are evaluated.  ``swing`` = not in contact (contact sensor),
      ``phase`` = expected swing window of the trot clock.
  gain_below, gain_above : float
      Asymmetric gains; ``gain_below > gain_above`` biases towards stubbing.
  mode : {"l1", "l2"}
      Cost shape.  ``l2`` is smooth (no gradient jump at the band edge).
  vel_weight : float
      If > 0 the per-foot cost is scaled by ``min(||v_foot,xy|| / vel_scale, 1)``
      (legacy behaviour); 0 disables velocity weighting so the term does not
      scale with the commanded speed.
  vel_scale : float
      Speed (m/s) at which the velocity weight saturates.
  combine : {"sum", "mean"}
      Per-foot reduction.
  period, offset, threshold : gait-clock parameters for ``gate="phase"``.
  sensor_name : contact sensor used by the swing/phase gate and the
      ``touchdown`` reference.
  height_sensor_name : terrain-height sensor used when ``ref="terrain"``.
  touchdown_tau : EMA rate for ``ref="touchdown"`` while the foot is in contact.
  command_name, command_threshold : velocity-command gate.
  asset_cfg : robot sites (one per foot).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.p = p
    self.asset_cfg: SceneEntityCfg = p["asset_cfg"]
    self.ref: Reference = p.get("ref", "world")
    self.gate: Gate = p.get("gate", "none")
    self.gain_below = float(p.get("gain_below", 1.0))
    self.gain_above = float(p.get("gain_above", 0.25))
    self.mode = p.get("mode", "l2")
    self.vel_weight = float(p.get("vel_weight", 0.0))
    self.vel_scale = float(p.get("vel_scale", 0.2))
    self.combine = p.get("combine", "sum")
    self.sensor_name = p.get("sensor_name", "feet_ground_contact")
    self.height_sensor_name = p.get("height_sensor_name", "feet_terrain_height")
    self.command_name = p.get("command_name", "twist")
    self.command_threshold = float(p.get("command_threshold", 0.1))
    self.touchdown_tau = float(p.get("touchdown_tau", 0.1))

    asset: Entity = env.scene[self.asset_cfg.name]
    self.n_feet = len(self.asset_cfg.site_ids)
    self._dtype = asset.data.site_pos_w.dtype
    self._contact_perm = resolve_contact_perm(
      env, self.sensor_name, self.asset_cfg.site_names
    )
    self._ref_z = torch.zeros(env.num_envs, self.n_feet, dtype=self._dtype, device=env.device)
    self._ref_init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._ref_z[env_ids] = 0.0
    self._ref_init[env_ids] = False

  def _clearance(self, env: ManagerBasedRlEnv, foot_z: torch.Tensor) -> torch.Tensor:
    if self.ref == "terrain":
      return torch.nan_to_num(env.scene.sensors[self.height_sensor_name].data.heights)
    if self.ref == "touchdown":
      in_contact = self._active_contact(env)
      if in_contact is None:
        raise RuntimeError("ref='touchdown' requires a contact sensor")
      fresh = ~self._ref_init
      self._ref_z = torch.where(fresh.unsqueeze(1), foot_z, self._ref_z)
      self._ref_init = self._ref_init | fresh
      updated = (1.0 - self.touchdown_tau) * self._ref_z + self.touchdown_tau * foot_z
      self._ref_z = torch.where(in_contact, updated, self._ref_z)
      return foot_z - self._ref_z
    return foot_z

  def _active_contact(self, env: ManagerBasedRlEnv) -> torch.Tensor | None:
    return contact_state(env, self.sensor_name, self._contact_perm)

  def _mask(self, env: ManagerBasedRlEnv, shape: torch.Size) -> torch.Tensor:
    if self.gate == "swing":
      in_contact = self._active_contact(env)
      if in_contact is None:
        return torch.ones(shape, dtype=torch.bool, device=env.device)
      return ~in_contact
    if self.gate == "phase":
      return phase_swing(env, self.p["period"], list(self.p["offset"]), self.p["threshold"])
    return torch.ones(shape, dtype=torch.bool, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    height_range: tuple[float, float],
    ref: Reference = "world",
    gate: Gate = "none",
    gain_below: float = 1.0,
    gain_above: float = 0.25,
    mode: Literal["l1", "l2"] = "l2",
    vel_weight: float = 0.0,
    vel_scale: float = 0.2,
    combine: Literal["sum", "mean"] = "sum",
    period: float = 0.6,
    offset: list[float] | None = None,
    threshold: float = 0.56,
    sensor_name: str = "feet_ground_contact",
    height_sensor_name: str = "feet_terrain_height",
    touchdown_tau: float = 0.1,
    command_name: str | None = "twist",
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    # All configuration is captured in __init__; the keyword list mirrors
    # RewardTermCfg.params because the manager splats params into the call.
    del (
      ref,
      gate,
      gain_below,
      gain_above,
      mode,
      vel_weight,
      vel_scale,
      combine,
      period,
      offset,
      threshold,
      sensor_name,
      height_sensor_name,
      touchdown_tau,
      asset_cfg,
    )
    asset: Entity = env.scene[self.asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, self.asset_cfg.site_ids, 2]
    clearance = self._clearance(env, foot_z)
    h_min, h_max = height_range
    cost = band_cost(
      clearance,
      h_min,
      h_max,
      gain_below=self.gain_below,
      gain_above=self.gain_above,
      mode=self.mode,
    )

    mask = self._mask(env, cost.shape)
    if self.vel_weight > 0.0:
      vel = torch.norm(asset.data.site_lin_vel_w[:, self.asset_cfg.site_ids, :2], dim=-1)
      weight = mask.to(cost.dtype) * (vel / self.vel_scale).clamp(max=1.0)
    else:
      weight = mask.to(cost.dtype)

    gated = cost * weight
    value = gated.sum(dim=1) if self.combine == "sum" else gated.mean(dim=1)
    value = value * _command_gate(env, self.command_name, self.command_threshold, value.dtype)

    active = weight > 0
    denom = active.sum().clamp_min(1)
    _log(env, "Diag/Clearance/raw_cost", value)
    _log(env, "Diag/Clearance/h_mean", clearance)
    _log(env, "Diag/Clearance/h_active_mean", (clearance * active).sum() / denom)
    _log(env, "Diag/Clearance/active_fraction", active.to(cost.dtype).mean())
    _log(env, "Diag/Clearance/below_frac", ((clearance < h_min) & active).sum() / denom)
    _log(env, "Diag/Clearance/above_frac", ((clearance > h_max) & active).sum() / denom)
    return value


class ClearanceReward:
  """Positive reward for *achieving* a target swing clearance.

  ``r = I(cmd) * mean_i I(active_i) * exp(-((h_i - h*)/sigma)^2)``

  Only feet inside the active set (swing / swing phase) contribute, and feet in
  stance are dropped from the mean rather than rewarded for sitting on the
  ground.  This turns clearance shaping into an incentive instead of a penalty,
  which composes better with the velocity-tracking reward.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.p = p
    self.asset_cfg: SceneEntityCfg = p["asset_cfg"]
    self.ref: Reference = p.get("ref", "terrain")
    self.gate: Gate = p.get("gate", "swing")
    self.target = float(p.get("target_height", 0.09))
    self.sigma = float(p.get("sigma", 0.04))
    self.sensor_name = p.get("sensor_name", "feet_ground_contact")
    self.height_sensor_name = p.get("height_sensor_name", "feet_terrain_height")
    self.command_name = p.get("command_name", "twist")
    self.command_threshold = float(p.get("command_threshold", 0.1))
    self.touchdown_tau = float(p.get("touchdown_tau", 0.1))

    asset: Entity = env.scene[self.asset_cfg.name]
    self.n_feet = len(self.asset_cfg.site_ids)
    self._dtype = asset.data.site_pos_w.dtype
    self._contact_perm = resolve_contact_perm(
      env, self.sensor_name, self.asset_cfg.site_names
    )
    self._ref_z = torch.zeros(env.num_envs, self.n_feet, dtype=self._dtype, device=env.device)
    self._ref_init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._ref_z[env_ids] = 0.0
    self._ref_init[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ref: Reference = "terrain",
    gate: Gate = "swing",
    target_height: float = 0.09,
    sigma: float = 0.04,
    sensor_name: str = "feet_ground_contact",
    height_sensor_name: str = "feet_terrain_height",
    touchdown_tau: float = 0.1,
    period: float = 0.6,
    offset: list[float] | None = None,
    threshold: float = 0.56,
    command_name: str | None = "twist",
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del (
      ref,
      gate,
      target_height,
      sigma,
      sensor_name,
      height_sensor_name,
      touchdown_tau,
      period,
      offset,
      threshold,
      asset_cfg,
    )
    asset: Entity = env.scene[self.asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, self.asset_cfg.site_ids, 2]

    if self.ref == "terrain":
      clearance = torch.nan_to_num(
        env.scene.sensors[self.height_sensor_name].data.heights
      )
    elif self.ref == "touchdown":
      in_contact = contact_state(env, self.sensor_name, self._contact_perm)
      if in_contact is None:
        raise RuntimeError("ref='touchdown' requires a contact sensor")
      fresh = ~self._ref_init
      self._ref_z = torch.where(fresh.unsqueeze(1), foot_z, self._ref_z)
      self._ref_init = self._ref_init | fresh
      self._ref_z = torch.where(
        in_contact,
        (1.0 - self.touchdown_tau) * self._ref_z + self.touchdown_tau * foot_z,
        self._ref_z,
      )
      clearance = foot_z - self._ref_z
    else:
      clearance = foot_z

    reward = exp_reward(clearance, self.target, self.sigma)

    if self.gate == "swing":
      in_contact = contact_state(env, self.sensor_name, self._contact_perm)
      active = (~in_contact) if in_contact is not None else torch.ones_like(reward, dtype=torch.bool)
    elif self.gate == "phase":
      active = phase_swing(env, self.p["period"], list(self.p["offset"]), self.p["threshold"])
    else:
      active = torch.ones_like(reward, dtype=torch.bool)

    active_f = active.to(reward.dtype)
    value = (reward * active_f).sum(dim=1) / active_f.sum(dim=1).clamp_min(1.0)
    value = value * _command_gate(env, self.command_name, self.command_threshold, value.dtype)

    _log(env, "Diag/ClearanceReward/value", value)
    _log(
      env,
      "Diag/ClearanceReward/h_active_mean",
      (clearance * active_f).sum() / active_f.sum().clamp_min(1.0),
    )
    _log(env, "Diag/ClearanceReward/active_fraction", active_f.mean())
    return value
