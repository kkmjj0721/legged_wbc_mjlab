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
} | {
    f"stairs_{depth}_outward_{direction}"
    for depth in ("018", "021", "025", "028")
    for direction in ("up", "down")
}


def geometry_parameters(family, values, size):
    if family not in FAMILIES:
        raise ValueError(f"Unsupported terrain family: {family}")
    category = "stairs" if family.startswith("stairs_") else family.split("_outward_")[0]
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


@dataclass(kw_only=True)
class ConnectedColumnCfg(SubTerrainCfg):
    """Encode an explicit row table through mjlab's deterministic column API.

    Difficulty is only a row address. It never scales physical terrain levels.
    """

    tiles: tuple = ()
    geometry_seed: int = 0
    column: int = 0

    def function(self, difficulty, spec, rng):
        del rng
        row = round(difficulty * (len(self.tiles) - 1))
        if not math.isclose(difficulty, row / (len(self.tiles) - 1), abs_tol=1e-7):
            raise ValueError("Invalid connected map row address")
        family, level, params = self.tiles[row]
        seed = stable_seed(self.geometry_seed, f"tile/{row}/{self.column}/{family}/{level}")
        output = _base(family, params, self.size).function(0.0, spec, np.random.default_rng(seed))
        for geometry in output.geometries:
            if geometry.hfield is not None:
                field = geometry.hfield
                heights = np.asarray(field.userdata).reshape(field.nrow, field.ncol)
                y, x = (field.nrow-1)/2, (field.ncol-1)/2
                r, col = int(y), int(x)
                r1, c1 = min(r+1, field.nrow-1), min(col+1, field.ncol-1)
                fy, fx = y-r, x-col
                # MuJoCo triangulates a cell along its lower-left/upper-right diagonal.
                h00, h01, h10, h11 = heights[r,col], heights[r,c1], heights[r1,col], heights[r1,c1]
                z = (h00 + fx*(h01-h00) + fy*(h11-h01) if fx >= fy else
                     h00 + fx*(h11-h10) + fy*(h10-h00))
                output.origin[2] = geometry.geom.pos[2] + field.size[2] * z
        return output


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
        self.connected = cfg.geometry.map_protocol == "connected_v7" and not self.plane
        self.tiles = []
        self.spawn_tiles = [[] for _ in self.columns]
        if self.connected:
            families = list(cfg.geometry.map_levels)
            if len(families) != 11 or sum(len(v) for v in cfg.geometry.map_levels.values()) != 52:
                raise ValueError("Connected V7 requires the complete 11-family, 52-level map")
            rings = cfg.geometry.surrounding_rings
            self.rows, self.cols = cfg.geometry.core_rows + 2 * rings, len(families) + 2 * rings
            for row in range(self.rows):
                for col in range(self.cols):
                    family = families[(col - rings) % len(families)]
                    levels = list(cfg.geometry.map_levels[family])
                    level = levels[min(max(row - rings, 0), len(levels) - 1)]
                    params = geometry_parameters(family, cfg.geometry.map_levels[family][level], self.size)
                    tile_id = row * self.cols + col
                    core = rings <= row < self.rows - rings and rings <= col < self.cols - rings
                    self.tiles.append({"id": tile_id, "row": row, "col": col,
                                       "terrain": family, "level": level, "parameters": params,
                                       "core": core,
                                       "seed": stable_seed(cfg.geometry.seed, f"tile/{row}/{col}/{family}/{level}"),
                                       "exit_half_extent": [s / 2 - (params["border_width"] if family.startswith("stairs") else 0) for s in self.size],
                                       "neighbors": [r * self.cols + c for r, c in ((row-1,col),(row+1,col),(row,col-1),(row,col+1))
                                                     if 0 <= r < self.rows and 0 <= c < self.cols]})
                    if core and (family, level) in self.columns:
                        index = self.columns.index((family, level))
                        if params != self.parameters[index]:
                            raise ValueError("Task terrain differs from the frozen connected map geometry")
                        self.spawn_tiles[index].append(tile_id)
            if not all(self.spawn_tiles):
                raise ValueError("Every birth task must have a core spawn tile")
            self.support_half_extent = np.array([self.rows, self.cols]) * self.size / 2 + cfg.geometry.outer_flat_width_m
        else:
            self.rows, self.cols = self.replicas, len(self.columns)
            self.support_half_extent = np.array([self.rows, self.cols]) * self.size / 2
        self.manifest = {"protocol": cfg.geometry.map_protocol, "tiles": self.tiles,
                         "origin_height": "actual_center_surface_v7" if self.connected else "generator_origin_v6",
                         "spawn_tiles": self.spawn_tiles, "spawn_probability": "uniform_given_task",
                         "support_half_extent": self.support_half_extent.tolist(),
                         "seed": cfg.geometry.seed}

    def choose_spawns(self, tasks, rng=None, choices=None):
        """Return physical row, column, ID; task probability never depends on copies."""
        columns = self.task_columns[tasks]
        if not self.connected:
            rows = (rng.integers(self.replicas, size=len(tasks)) if choices is None
                    else np.asarray(choices, dtype=np.int64) % self.replicas)
            return rows, columns, columns * self.replicas + rows
        fractions = rng.random(len(tasks)) if choices is None else np.asarray(choices)
        ids = np.array([self.spawn_tiles[c][min(int(u * len(self.spawn_tiles[c])), len(self.spawn_tiles[c])-1)]
                        for c, u in zip(columns, fractions)])
        return ids // self.cols, ids % self.cols, ids

    def locate(self, xy):
        """World XY -> physical tile; -1 outer flat frame, -2 outside support."""
        half = xy.new_tensor([self.rows * self.size[0], self.cols * self.size[1]]) / 2
        rc = torch.floor((xy + half) / xy.new_tensor(self.size)).long()
        inside = ((rc >= 0) & (rc < rc.new_tensor([self.rows, self.cols]))).all(-1)
        supported = (xy.abs() < xy.new_tensor(self.support_half_extent)).all(-1)
        return torch.where(inside, rc[..., 0] * self.cols + rc[..., 1],
                           torch.where(supported, -1, -2))

    def entity_cfg(self, cfg):
        if self.plane:
            return TerrainEntityCfg(terrain_type="plane")
        if self.connected:
            columns = {
                f"connected_column_{col}": ConnectedColumnCfg(
                    size=self.size, geometry_seed=cfg.geometry.seed, column=col,
                    tiles=tuple((t["terrain"], t["level"], t["parameters"])
                                for t in self.tiles if t["col"] == col))
                for col in range(self.cols)
            }
            return TerrainEntityCfg(
                terrain_type="generator", max_init_terrain_level=self.rows - 1,
                terrain_generator=TerrainGeneratorCfg(
                    seed=cfg.geometry.seed, curriculum=True, size=self.size,
                    num_rows=self.rows, num_cols=self.cols, difficulty_range=(0.0, 1.0),
                    border_width=cfg.geometry.outer_flat_width_m, border_height=1.0,
                    sub_terrains=columns))
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
                self.rows,
                self.cols,
                3,
            ):
                raise ValueError(
                    "Actual terrain origins differ from the compiled mapping"
                )
            if not torch.isfinite(origins).all():
                raise ValueError("Nonfinite terrain origins")
            self.origins = origins
            if self.connected:
                expected = origins.new_tensor([[(r + .5 - self.rows/2)*self.size[0],
                                                (c + .5 - self.cols/2)*self.size[1]]
                                               for r in range(self.rows) for c in range(self.cols)])
                if not torch.allclose(origins[..., :2].reshape(-1, 2), expected, atol=1e-5):
                    raise ValueError("Connected map centres differ from the physical grid")


class TraversalMetrics:
    """评估实际运动和台阶接触；出界重置的瞬移不计入行走距离。"""

    def __init__(self, env, controller):
        self.env = env
        self.connected = None
        if controller.terrain_map.connected:
            from .episode_diagnostics import ConnectedTraversal

            self.connected = ConnectedTraversal(env, controller)
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
        half_platforms, widths, heights, step_counts = [], [], [], []
        for task in controller.catalog.records:
            if not task.terrain.startswith("stairs"):
                half_platforms.append((math.inf, math.inf))
                widths.append(1.0); heights.append(0.0); step_counts.append(0)
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
            widths.append(params["step_width"])
            heights.append(params["step_height_m"] * (1 if task.terrain.endswith("up") else -1))
            step_counts.append(count)
        self.half_platform = torch.tensor(half_platforms, device=device)[
            controller.episode_task_id
        ]
        task_ids = controller.episode_task_id
        self.width = torch.tensor(widths, device=device)[task_ids]
        self.height = torch.tensor(heights, device=device)[task_ids]
        self.step_count = torch.tensor(step_counts, device=device)[task_ids]
        self.levels = torch.arange(1, max(step_counts, default=0) + 1, device=device)
        self.support_seconds = torch.zeros(n, len(self.levels), device=device)
        self.crossed_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.segment_start_z = self.robot.data.root_link_pos_w[:, 2].clone()
        self.displacement_xy = torch.zeros(n, 2, device=device)
        self.yaw_error = torch.zeros(n, device=device)
        self.action_delta_sum = torch.zeros(n, device=device)
        self.action_delta_count = torch.zeros(n, device=device)
        self.previous_actions = None
        self.action_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.foot_slip_sum = torch.zeros(n, device=device)
        self.foot_contact_count = torch.zeros(n, device=device)

    def observe(self, active, commands, actions=None):
        robot, dt = self.robot, self.env.step_dt
        if self.connected is not None:
            self.connected.observe(active, self.env.termination_manager.terminated)
        xy = robot.data.root_link_pos_w[:, :2]
        delta = xy - self.previous_xy
        self.distance[active] += delta.norm(dim=-1)[active]
        self.displacement_xy[active] += delta[active]
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
        self.yaw_error[active] += (
            robot.data.root_link_ang_vel_b[:, 2] - commands[:, 2]
        )[active] * dt
        foot_speed = robot.data.site_lin_vel_w[:, self.foot_ids, :2].norm(dim=-1)
        self.foot_slip_sum[active] += (foot_speed * touching).sum(dim=-1)[active]
        self.foot_contact_count[active] += touching.sum(dim=-1)[active]
        if actions is not None:
            if self.previous_actions is not None:
                valid = active & self.action_valid
                self.action_delta_sum[valid] += (
                    actions - self.previous_actions
                ).square().mean(dim=-1)[valid]
                self.action_delta_count[valid] += 1
            self.previous_actions = actions.clone()
            self.action_valid.copy_(active)

        # Conservative geometric proxy: root outside the platform and moving
        # vertically in the staircase direction, with >=2 feet supported on
        # actual treads at this level or farther, continuously for >=0.1 s.
        # Foot sites are sphere centres (RI-4438 radius 15.73 mm). The height
        # tolerance admits contact compliance, but excludes neighbouring treads.
        origins = self.env.scene.env_origins
        excess = ((feet - origins[:, None, :2]).abs() - self.half_platform[:, None, :]).amax(dim=-1)
        foot_level = (torch.floor(excess / self.width[:, None]) + 1).clamp_min(0)
        on_stairs = (foot_level > 0) & (foot_level <= self.step_count[:, None])
        surface_z = origins[:, None, 2] + foot_level * self.height[:, None]
        clearance = robot.data.site_pos_w[:, self.foot_ids, 2] - surface_z
        supported = touching & on_stairs & (clearance >= -0.005) & (clearance <= 0.035)
        root_excess = ((xy - origins[:, :2]).abs() - self.half_platform).amax(dim=-1)
        root_level = (torch.floor(root_excess / self.width) + 1).clamp_min(0)
        vertical = (robot.data.root_link_pos_w[:, 2] - self.segment_start_z) * self.height.sign()
        levels = self.levels[None, :]
        evidence = (
            (root_level[:, None] >= levels)
            & (levels <= self.step_count[:, None])
            & (vertical[:, None] >= self.height.abs()[:, None] * (levels - 0.5))
            & ((supported[:, :, None] & (foot_level[:, :, None] >= levels[:, None, :])).sum(dim=1) >= 2)
        )
        self.support_seconds[active] = torch.where(
            evidence[active], self.support_seconds[active] + dt, 0.0,
        )
        if len(self.levels):
            crossed = torch.where(self.support_seconds >= 0.1 - 1e-6, levels, 0).amax(dim=1)
            self.crossed_steps[active] = torch.maximum(self.crossed_steps[active], crossed[active])

    def reset_segments(self, env_ids):
        self.previous_xy[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]
        self.segment_start_z[env_ids] = self.robot.data.root_link_pos_w[env_ids, 2]
        self.support_seconds[env_ids] = 0
        self.action_valid[env_ids] = False

    def trial(self, slot):
        return {
            **(self.connected.trial(slot) if self.connected is not None else {}),
            "distance_m": float(self.distance[slot]),
            "stair_contact_seconds": float(self.stair_contact_seconds[slot]),
            "stalled_seconds": float(self.stalled_seconds[slot]),
            "net_displacement_m": float(self.displacement_xy[slot].norm()),
            "integrated_yaw_error_rad": float(self.yaw_error[slot]),
            "action_delta_rms": float((self.action_delta_sum[slot] / self.action_delta_count[slot].clamp_min(1)).sqrt()),
            "foot_slip_m_s": float(self.foot_slip_sum[slot] / self.foot_contact_count[slot].clamp_min(1)),
            "stair_steps_crossed": int(self.crossed_steps[slot]),
            "stair_step_count": int(self.step_count[slot]),
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
        if self.controller.terrain_map.connected:
            half = feet.new_tensor(self.controller.terrain_map.support_half_extent)
            origins = torch.zeros_like(origins)
        scan_cross = ((scan - origins).abs() > half).any(dim=-1).any(dim=-1)
        if self.controller.terrain_map.connected:
            scan_cross |= (~torch.isfinite(scan).all(-1) | (env.scene["terrain_scan"].data.distances < 0)).any(-1)
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
                f"Nonterminal feet/scan outside supported region: {self.summary()}. "
                f"Increase boundary_margin_m to at least {self.max_relative_extent + 0.1:.3f} and revalidate with a new protocol."
            )

    def summary(self):
        return {
            "checked_states": self.checked_states,
            "scan_crossings": self.scan_crossings,
            "foot_crossings": self.foot_crossings,
            "max_relative_extent_m": self.max_relative_extent,
        }
