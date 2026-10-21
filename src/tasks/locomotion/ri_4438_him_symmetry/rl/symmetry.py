"""Name-resolved left/right reflection for RI-4438 HIM observation histories.

Only tensor samples are reflected. Physics, rollout clocks, and estimator
supervision pairs are not modified by this adapter.
"""

from __future__ import annotations

import re

import torch
from tensordict import TensorDict

from .layouts import HimLayout, PHASE47, validate_env_cfg

MAPPING_VERSION = "ri4438_lr_v1"
_OTHER = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


def joint_rule(names) -> tuple[list[int], list[float]]:
    names = [name.split("/")[-1] for name in names]
    expected = {f"{leg}_{joint}_joint" for leg in _OTHER for joint in ("hip", "thigh", "calf")}
    if len(names) != 12 or set(names) != expected:
        raise ValueError(f"Expected all 12 unique RI-4438 joints, got {names}")
    indices, signs = [], []
    for name in names:
        leg, joint = re.fullmatch(r"(FL|FR|RL|RR)_(hip|thigh|calf)_joint", name).groups()
        indices.append(names.index(f"{_OTHER[leg]}_{joint}_joint"))
        signs.append(-1.0 if joint == "hip" else 1.0)
    return indices, signs


def foot_rule(names) -> list[int]:
    names = [name.split("/")[-1] for name in names]
    expected = {f"{leg}_foot_collision" for leg in _OTHER}
    if len(names) != 4 or set(names) != expected:
        raise ValueError(f"Expected four unique RI-4438 foot collision geoms, got {names}")
    return [names.index(f"{_OTHER[name[:2]]}_foot_collision") for name in names]


def ray_rule(positions: torch.Tensor, directions: torch.Tensor | None = None) -> torch.Tensor:
    if positions.shape != (187, 3) or not torch.isfinite(positions).all():
        raise ValueError("Expected 187 finite local ray origins with shape [187,3]")
    reflection = positions.new_tensor([1, -1, 1])
    distances = (positions[:, None, :] * reflection - positions[None, :, :]).abs().amax(-1)
    matches = distances < 1e-5
    if not torch.all(matches.sum(0) == 1) or not torch.all(matches.sum(1) == 1):
        raise ValueError("Terrain scan has no bijective left/right reflection")
    indices = matches.to(torch.int64).argmax(1)
    if directions is not None:
        if directions.shape != positions.shape or not torch.allclose(
            directions * reflection, directions[indices], rtol=0, atol=1e-6
        ):
            raise ValueError("Terrain ray directions do not respect left/right reflection")
    return indices


class Ri4438Mirror:
    def __init__(self, layout: HimLayout, joint_names: dict, action_names, foot_names,
                 ray_positions: torch.Tensor, ray_directions: torch.Tensor | None = None):
        self.layout = layout
        self.device = ray_positions.device
        self.action_indices, self.action_signs = self._tensors(*joint_rule(action_names))
        feet = foot_rule(foot_names)
        rays = ray_rule(ray_positions, ray_directions).tolist()
        self.rules = {}
        for group in ("actor", "critic", "estimator"):
            indices, signs, offset = [], [], 0
            for name, width in zip(layout.terms(group), layout.widths(group), strict=True):
                index, sign = list(range(width)), [1.0] * width
                if name == "base_ang_vel":
                    sign = [-1.0, 1.0, -1.0]
                elif name in ("projected_gravity", "base_lin_vel", "base_com"):
                    sign = [1.0, -1.0, 1.0]
                elif name == "command":
                    sign = [1.0, -1.0, -1.0]
                elif name == "phase":
                    sign = [-1.0, -1.0]
                elif name in ("joint_pos", "joint_vel"):
                    index, sign = joint_rule(joint_names[group, name])
                elif name == "actions":
                    index, sign = joint_rule(action_names)
                elif name == "foot_contact":
                    index = feet
                elif name == "height_scan":
                    index = rays
                indices.extend(offset + i for i in index)
                signs.extend(sign)
                offset += width
            self.rules[group] = self._tensors(indices, signs)

    def _tensors(self, indices, signs):
        return (torch.tensor(indices, dtype=torch.long, device=self.device),
                torch.tensor(signs, dtype=torch.float32, device=self.device))

    @staticmethod
    def _apply(value, rule):
        indices, signs = rule
        return value.index_select(-1, indices.to(value.device)) * signs.to(device=value.device, dtype=value.dtype)

    def mirror_group(self, group: str, value: torch.Tensor) -> torch.Tensor:
        if group not in self.rules:
            raise ValueError(f"Unsupported symmetry observation group {group!r}")
        expected = ((self.layout.history_size, self.layout.frame_size) if group == "actor"
                    else (sum(self.layout.widths(group)),))
        if tuple(value.shape[1:]) != expected:
            raise ValueError(f"{self.layout.name}/{group}: expected [B,{expected}], got {tuple(value.shape)}")
        return self._apply(value, self.rules[group])

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != 12:
            raise ValueError("RI-4438 actions must have shape [B,12]")
        return self._apply(actions, (self.action_indices, self.action_signs))

    def augment(self, obs: TensorDict | None, actions: torch.Tensor | None):
        if obs is not None and len(obs.batch_size) != 1:
            raise ValueError("Symmetry expects a flattened mini-batch TensorDict")
        if obs is not None and actions is not None and obs.batch_size[0] != actions.shape[0]:
            raise ValueError("Observation and action batch sizes differ")
        augmented_obs = None
        if obs is not None:
            augmented_obs = TensorDict({
                name: torch.cat((value, self.mirror_group(name, value)), dim=0)
                for name, value in obs.items()
            }, batch_size=[2 * obs.batch_size[0]], device=obs.device)
        augmented_actions = None if actions is None else torch.cat((actions, self.mirror_actions(actions)), dim=0)
        return augmented_obs, augmented_actions

    @classmethod
    def from_env(cls, env, layout: HimLayout):
        validate_env_cfg(env.cfg, layout)
        manager, robot = env.observation_manager, env.scene["robot"]
        joint_names = {}
        for group in ("actor", "critic", "estimator"):
            if tuple(manager.active_terms[group]) != layout.terms(group):
                raise ValueError(f"Runtime {group} term order differs from {layout.name}")
            expected_shape = ((layout.history_size, layout.frame_size) if group == "actor" else (sum(layout.widths(group)),))
            if tuple(manager.group_obs_dim[group]) != expected_shape:
                raise ValueError(f"Runtime {group} shape differs from {layout.name}: {manager.group_obs_dim[group]}")
            expected_terms = [((layout.history_size, width) if group == "actor" else (width,))
                              for width in layout.widths(group)]
            if list(manager.group_obs_term_dim[group]) != expected_terms:
                raise ValueError(f"Runtime {group} term dimensions differ from {layout.name}")
            for name in ("joint_pos", "joint_vel"):
                params = manager.get_term_cfg(group, name).params
                asset = params.get("asset_cfg")
                if asset is not None and asset.name != "robot":
                    raise ValueError("Symmetry joint observations must refer to robot")
                ids = slice(None) if asset is None else asset.joint_ids
                selected = torch.arange(len(robot.joint_names))[ids].tolist()
                names = [robot.joint_names[i] for i in selected]
                joint_names[group, name] = names
                index, sign = joint_rule(names)
                defaults = (robot.data.default_joint_pos if name == "joint_pos" else robot.data.default_joint_vel)
                values = defaults[:, selected]
                reflected = values[:, index] * values.new_tensor(sign)
                if not torch.allclose(values, reflected, atol=1e-6, rtol=0):
                    raise ValueError(f"{group}/{name} defaults are not mirror-compatible")
        if tuple(env.action_manager.active_terms) != ("joint_pos",):
            raise ValueError("Symmetry requires exactly one joint_pos action term")
        action = env.action_manager.get_term("joint_pos")
        if action.cfg.entity_name != "robot":
            raise ValueError("Symmetry actions must refer to robot")
        index, sign = joint_rule(action.target_names)
        for name, signed in (("scale", False), ("offset", True)):
            values = torch.as_tensor(getattr(action, name), device=env.device).expand(env.num_envs, 12)
            reflected = values[:, index] * (values.new_tensor(sign) if signed else 1)
            if not torch.isfinite(values).all() or not torch.allclose(values, reflected, atol=1e-6, rtol=0):
                raise ValueError(f"Action {name} is not mirror-compatible")
            if name == "scale" and (values == 0).any():
                raise ValueError("Action scales must be nonzero")
        # Validate the actual model axes, rather than accepting any 12-joint asset.
        for name in robot.joint_names:
            expected = [1.0, 0.0, 0.0] if "_hip_" in name else [0.0, 1.0, 0.0]
            axis = env.sim.mj_model.joint(f"robot/{name}").axis
            if not torch.allclose(torch.as_tensor(axis), torch.tensor(expected, dtype=torch.float64)):
                raise ValueError(f"Unexpected joint axis for {name}: {axis}")
        foot_sensor = env.scene[manager.get_term_cfg("critic", "foot_contact").params["sensor_name"]]
        scan = env.scene[manager.get_term_cfg("critic", "height_scan").params["sensor_name"]]
        if scan.cfg.ray_alignment != "yaw" or scan._num_frames != 1:
            raise ValueError("Symmetry requires one yaw-aligned local terrain scan")
        mirror = cls(layout, joint_names, action.target_names, foot_sensor.primary_names,
                     scan._local_offsets, scan._local_directions)
        if layout.with_phase:
            gait = env.cfg.rewards["foot_gait"].params
            if gait["sensor_name"] != foot_sensor.cfg.name or gait["period"] <= 0:
                raise ValueError("Gait must use the mirrored foot sensor and a positive period")
            offsets = torch.tensor(gait["offset"], dtype=torch.float64)
            if offsets.shape != (4,) or not torch.allclose(
                (offsets[foot_rule(foot_sensor.primary_names)] - offsets).remainder(1),
                torch.full((4,), 0.5, dtype=torch.float64), atol=1e-8, rtol=0,
            ):
                raise ValueError("phase47_v1 requires a half-cycle left/right gait shift")
        return mirror


def get_adapter(env, layout: HimLayout) -> Ri4438Mirror:
    env = getattr(env, "unwrapped", env)
    cache = getattr(env, "_ri4438_symmetry_adapters", None)
    if cache is None:
        cache = {}
        env._ri4438_symmetry_adapters = cache
    if layout.name not in cache:
        cache[layout.name] = Ri4438Mirror.from_env(env, layout)
    return cache[layout.name]


def augment_phase47(*, env, obs=None, actions=None):
    return get_adapter(env, PHASE47).augment(obs, actions)
