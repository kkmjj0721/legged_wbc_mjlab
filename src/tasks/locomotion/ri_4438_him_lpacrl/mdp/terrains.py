"""Fixed family/level columns and independently seeded replica rows."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import torch
from mjlab.terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    TerrainEntityCfg,
    TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg

from src.config.ri_4438.ri_4438_him_lpacrl_config import Ri4438HimLpacrlCfg

from .task_catalog import stable_seed

FAMILIES = {
    "flat",
    "random_rough",
    "stairs_outward_up",
    "stairs_outward_down",
    "slope_outward_up",
    "slope_outward_down",
}


def geometry_parameters(family, values, size):
    if family not in FAMILIES:
        raise ValueError(f"Unsupported terrain family: {family}")
    category = family.split("_outward_")[0]
    defaults = deepcopy(Ri4438HimLpacrlCfg.terrain.defaults[category])
    if set(values) - set(defaults):
        raise ValueError(
            f"{family}: unknown geometry parameters {set(values) - set(defaults)}"
        )
    params = {**defaults, **values}
    for name, value in params.items():
        if value is not None and not isinstance(value, bool):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{family}/{name} must be finite and nonnegative")
            if name not in {"gradient", "border_width"} and value == 0:
                raise ValueError(f"{family}/{name} must be positive")
    if 2 * params.get("border_width", 0) + params.get("platform_width", 0) >= min(size):
        raise ValueError(f"{family}: platform/border do not fit the patch")
    if "horizontal_scale" in params:
        hs = params["horizontal_scale"]
        border = params["border_width"]
        if border and border < hs:
            raise ValueError("Heightfield border must be at least horizontal_scale")
        if (
            params.get("downsampled_scale") is not None
            and params["downsampled_scale"] < hs
        ):
            raise ValueError("downsampled_scale must be >= horizontal_scale")
        ds = params.get("downsampled_scale") or hs
        if min(size) - 2 * border < 4 * ds:
            raise ValueError(
                "Heightfield must have at least four interpolation samples per axis"
            )
    if (
        family == "random_rough"
        and int(params["noise_step"] / params["vertical_scale"]) < 1
    ):
        raise ValueError("noise_step is below the vertical quantization unit")
    return params


def _base(family, params, size):
    p = dict(params)
    if family == "flat":
        return BoxFlatTerrainCfg(size=size)
    if family == "random_rough":
        amplitude = p.pop("height_max_m")
        return HfRandomUniformTerrainCfg(
            size=size, noise_range=(0.0, amplitude), scale_with_difficulty=False, **p
        )
    if family.startswith("stairs"):
        height = p.pop("step_height_m")
        cls = (
            BoxInvertedPyramidStairsTerrainCfg
            if family.endswith("up")
            else BoxPyramidStairsTerrainCfg
        )
        return cls(size=size, step_height_range=(height, height), **p)
    gradient = p.pop("gradient")
    if gradient == 0:
        return BoxFlatTerrainCfg(size=size)
    return HfPyramidSlopedTerrainCfg(
        size=size, slope_range=(gradient, gradient), inverted=family.endswith("up"), **p
    )


@dataclass(kw_only=True)
class FixedLevelTerrainCfg(SubTerrainCfg):
    """One fixed family/level column; generator difficulty encodes replica rows.

    Actual height/slope comes from parameters. MjLab's row difficulty must not
    scale it again, otherwise the saved task ID would no longer identify one
    fixed geometry level.
    """

    family: str
    level: str
    parameters: dict = field(default_factory=dict)
    replicas: int = 1
    geometry_seed: int = 0

    def function(self, difficulty, spec, rng):
        del rng
        row = round(difficulty * (self.replicas - 1)) if self.replicas > 1 else 0
        encoded = row / max(self.replicas - 1, 1)
        if not 0 <= row < self.replicas or not math.isclose(
            difficulty, encoded, abs_tol=1e-7
        ):
            raise ValueError("Unexpected replica/difficulty encoding")
        local_rng = np.random.default_rng(
            stable_seed(self.geometry_seed, f"{self.family}/{self.level}/{row}")
        )
        return _base(self.family, self.parameters, self.size).function(
            0.0, spec, local_rng
        )


class TerrainMap:
    def __init__(self, catalog, cfg):
        used = {(t.terrain, t.level) for t in catalog.records}
        self.columns = [
            (family, level)
            for family, spec in cfg.task_space.terrains.items()
            for level in spec.levels
            if (family, level) in used
        ]
        self.replicas = cfg.geometry.replicas
        self.size = tuple(cfg.geometry.patch_size_m)
        self.plane = all(family == "flat" for family, _ in self.columns)
        if self.plane and len(self.columns) != 1:
            raise ValueError("Pure plane has one semantic flat level")
        if self.plane and self.replicas != 1:
            raise ValueError(
                "Pure plane uses per-environment grid origins; geometry.replicas must be 1"
            )
        self.parameters = [
            geometry_parameters(
                family, cfg.task_space.terrains[family].levels[level], self.size
            )
            for family, level in self.columns
        ]
        self.task_columns = np.array(
            [self.columns.index((t.terrain, t.level)) for t in catalog.records],
            dtype=np.int64,
        )
        self.origins = None

    def entity_cfg(self, cfg):
        if self.plane:
            return TerrainEntityCfg(terrain_type="plane")
        sub_terrains = {
            f"g{i}_{family}_{level}": FixedLevelTerrainCfg(
                family=family,
                level=level,
                size=self.size,
                parameters=self.parameters[i],
                replicas=self.replicas,
                geometry_seed=cfg.geometry.seed,
            )
            for i, (family, level) in enumerate(self.columns)
        }
        return TerrainEntityCfg(
            terrain_type="generator",
            max_init_terrain_level=self.replicas - 1,
            terrain_generator=TerrainGeneratorCfg(
                seed=cfg.geometry.seed,
                curriculum=True,
                size=self.size,
                num_rows=self.replicas,
                num_cols=len(sub_terrains),
                difficulty_range=(0.0, 1.0),
                sub_terrains=sub_terrains,
            ),
        )

    def bind(self, origins):
        if self.plane:
            if origins is not None:
                raise ValueError("Plane unexpectedly has tile origins")
        else:
            if origins is None or tuple(origins.shape) != (
                self.replicas,
                len(self.columns),
                3,
            ):
                raise ValueError(
                    "Actual terrain origins differ from the compiled mapping"
                )
            if not torch.isfinite(origins).all():
                raise ValueError("Nonfinite terrain origins")
            self.origins = origins


class TraversalMetrics:
    """评估实际运动和台阶接触；出界重置的瞬移不计入行走距离。"""

    def __init__(self, env, controller):
        self.env = env
        self.robot = env.scene["robot"]
        self.contact = env.scene["feet_ground_contact"]
        names = [
            name.rsplit("/", 1)[-1].removesuffix("_foot_collision")
            for name in self.contact.primary_names
        ]
        self.foot_ids, _ = self.robot.find_sites(names, preserve_order=True)
        if len(self.foot_ids) != 4:
            raise ValueError("Traversal metrics require four ordered foot contacts")
        n, device = env.num_envs, env.device
        self.distance = torch.zeros(n, device=device)
        self.stair_contact_seconds = torch.zeros(n, device=device)
        self.stalled_seconds = torch.zeros(n, device=device)
        self.previous_xy = self.robot.data.root_link_pos_w[:, :2].clone()
        # 使用生成器离散踏步后实际形成的平台宽度，而非仅用请求值。
        half_platforms = []
        for task in controller.catalog.records:
            if not task.terrain.startswith("stairs"):
                half_platforms.append((math.inf, math.inf))
                continue
            column = controller.terrain_map.columns.index((task.terrain, task.level))
            params = controller.terrain_map.parameters[column]
            inner = [
                s - 2 * params["border_width"] for s in controller.terrain_map.size
            ]
            count = int(min(
                (s - params["platform_width"]) / (2 * params["step_width"])
                for s in inner
            ))
            half_platforms.append(
                tuple(s / 2 - count * params["step_width"] for s in inner)
            )
        self.half_platform = torch.tensor(half_platforms, device=device)[
            controller.episode_task_id
        ]

    def observe(self, active, commands):
        robot, dt = self.robot, self.env.step_dt
        xy = robot.data.root_link_pos_w[:, :2]
        self.distance[active] += (xy - self.previous_xy).norm(dim=-1)[active]
        self.previous_xy.copy_(xy)
        feet = robot.data.site_pos_w[:, self.foot_ids, :2]
        outside = (
            (feet - self.env.scene.env_origins[:, None, :2]).abs()
            > self.half_platform[:, None, :]
        ).any(dim=-1)
        forces = self.contact.data.force.norm(dim=-1).reshape(
            self.env.num_envs, 4, -1
        )
        touching = forces.amax(dim=-1) > 1.0
        self.stair_contact_seconds[active] += (
            (outside & touching).any(dim=-1)[active] * dt
        )
        stalled = (
            (commands[:, :2].norm(dim=-1) > 0.2)
            & (robot.data.root_link_lin_vel_b[:, :2].norm(dim=-1) < 0.05)
        )
        self.stalled_seconds[active] += stalled[active] * dt

    def reset_segments(self, env_ids):
        self.previous_xy[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]

    def trial(self, slot):
        return {
            "distance_m": float(self.distance[slot]),
            "stair_contact_seconds": float(self.stair_contact_seconds[slot]),
            "stalled_seconds": float(self.stalled_seconds[slot]),
        }


class GeometryMonitor:
    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.scan_crossings = self.foot_crossings = self.checked_states = 0
        self.max_relative_extent = 0.0
        self._site_ids = None

    def observe(self, terminal_mask):
        if self.controller.terrain_map.plane:
            return
        env = self.env
        valid = ~terminal_mask
        if not valid.any():
            return
        robot = env.scene["robot"]
        if self._site_ids is None:
            self._site_ids, _ = robot.find_sites(("FL", "FR", "RL", "RR"))
            if len(self._site_ids) != 4:
                raise ValueError(
                    "Expected four named foot sites for geometry validation"
                )
        feet = robot.data.site_pos_w[:, self._site_ids, :2]
        scan = env.scene["terrain_scan"].data.hit_pos_w[..., :2]
        origins = env.scene.env_origins[:, None, :2]
        root = robot.data.root_link_pos_w[:, None, :2]
        half = (
            torch.tensor(self.controller.cfg.geometry.patch_size_m, device=env.device)
            / 2
        )
        scan_cross = ((scan - origins).abs() > half).any(dim=-1).any(dim=-1)
        foot_cross = ((feet - origins).abs() > half).any(dim=-1).any(dim=-1)
        self.scan_crossings += int((scan_cross & valid).sum())
        self.foot_crossings += int((foot_cross & valid).sum())
        self.checked_states += int(valid.sum())
        extent = max(
            float((scan[valid] - root[valid]).abs().max()),
            float((feet[valid] - root[valid]).abs().max()),
        )
        self.max_relative_extent = max(self.max_relative_extent, extent)
        if ((scan_cross | foot_cross) & valid).any():
            raise RuntimeError(
                f"Nonterminal feet/scan crossed assigned patch: {self.summary()}. "
                f"Increase boundary_margin_m to at least {self.max_relative_extent + 0.1:.3f} and revalidate with a new protocol."
            )

    def summary(self):
        return {
            "checked_states": self.checked_states,
            "scan_crossings": self.scan_crossings,
            "foot_crossings": self.foot_crossings,
            "max_relative_extent_m": self.max_relative_extent,
        }
