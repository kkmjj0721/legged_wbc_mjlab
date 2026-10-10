"""Fixed family/level columns and independently seeded replica rows."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
import torch
from mjlab.terrains import (
    BoxFlatTerrainCfg, BoxPyramidStairsTerrainCfg, BoxInvertedPyramidStairsTerrainCfg,
    HfPyramidSlopedTerrainCfg, HfRandomUniformTerrainCfg, TerrainEntityCfg, TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg
from .task_catalog import stable_seed

FAMILIES = {"flat", "random_rough", "stairs_outward_up", "stairs_outward_down", "slope_outward_up", "slope_outward_down"}


def geometry_parameters(family, values, size):
    if family not in FAMILIES:
        raise ValueError(f"Unsupported terrain family: {family}")
    if family == "flat":
        defaults = {}
    elif family == "random_rough":
        defaults = dict(height_max_m=0.05, noise_step=0.005, horizontal_scale=0.1,
                        vertical_scale=0.005, downsampled_scale=None, border_width=0.25)
    elif family.startswith("stairs"):
        defaults = dict(step_height_m=0.1, step_width=0.30, platform_width=3.0, border_width=1.0, holes=False)
    else:
        defaults = dict(gradient=0.2, platform_width=2.0, border_width=0.25,
                        horizontal_scale=0.1, vertical_scale=0.005)
    if set(values) - set(defaults):
        raise ValueError(f"{family}: unknown geometry parameters {set(values) - set(defaults)}")
    params = {**defaults, **values}
    for name, value in params.items():
        if value is not None and not isinstance(value, bool):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
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
        if params.get("downsampled_scale") is not None and params["downsampled_scale"] < hs:
            raise ValueError("downsampled_scale must be >= horizontal_scale")
        ds = params.get("downsampled_scale") or hs
        if min(size) - 2 * border < 4 * ds:
            raise ValueError("Heightfield must have at least four interpolation samples per axis")
    if family == "random_rough" and int(params["noise_step"] / params["vertical_scale"]) < 1:
        raise ValueError("noise_step is below the vertical quantization unit")
    return params


def _base(family, params, size):
    p = dict(params)
    if family == "flat":
        return BoxFlatTerrainCfg(size=size)
    if family == "random_rough":
        amplitude = p.pop("height_max_m")
        return HfRandomUniformTerrainCfg(size=size, noise_range=(0.0, amplitude), scale_with_difficulty=False, **p)
    if family.startswith("stairs"):
        height = p.pop("step_height_m")
        cls = BoxInvertedPyramidStairsTerrainCfg if family.endswith("up") else BoxPyramidStairsTerrainCfg
        return cls(size=size, step_height_range=(height, height), **p)
    gradient = p.pop("gradient")
    if gradient == 0:
        return BoxFlatTerrainCfg(size=size)
    return HfPyramidSlopedTerrainCfg(size=size, slope_range=(gradient, gradient),
                                    inverted=family.endswith("up"), **p)


@dataclass(kw_only=True)
class FixedLevelTerrainCfg(SubTerrainCfg):
    family: str
    level: str
    parameters: dict = field(default_factory=dict)
    replicas: int = 1
    geometry_seed: int = 0

    def function(self, difficulty, spec, rng):
        del rng
        row = round(difficulty * (self.replicas - 1)) if self.replicas > 1 else 0
        encoded = row / max(self.replicas - 1, 1)
        if not 0 <= row < self.replicas or not math.isclose(difficulty, encoded, abs_tol=1e-7):
            raise ValueError("Unexpected replica/difficulty encoding")
        local_rng = np.random.default_rng(stable_seed(self.geometry_seed, f"{self.family}/{self.level}/{row}"))
        return _base(self.family, self.parameters, self.size).function(0.0, spec, local_rng)


class TerrainMap:
    def __init__(self, catalog, cfg):
        used = {(t.terrain, t.level) for t in catalog.records}
        self.columns = [(f, l) for f, spec in cfg.task_space.terrains.items() for l in spec.levels if (f, l) in used]
        self.replicas = cfg.geometry.replicas
        self.size = tuple(cfg.geometry.patch_size_m)
        self.plane = all(f == "flat" for f, _ in self.columns)
        if self.plane and len(self.columns) != 1:
            raise ValueError("Pure plane has one semantic flat level")
        if self.plane and self.replicas != 1:
            raise ValueError("Pure plane uses per-environment grid origins; geometry.replicas must be 1")
        self.parameters = [geometry_parameters(f, cfg.task_space.terrains[f].levels[l], self.size) for f, l in self.columns]
        self.task_columns = np.array([self.columns.index((t.terrain, t.level)) for t in catalog.records], dtype=np.int64)
        self.origins = None

    def entity_cfg(self, cfg):
        if self.plane:
            return TerrainEntityCfg(terrain_type="plane")
        sub = {f"g{i}_{f}_{l}": FixedLevelTerrainCfg(family=f, level=l, size=self.size,
                 parameters=self.parameters[i], replicas=self.replicas, geometry_seed=cfg.geometry.seed)
               for i, (f, l) in enumerate(self.columns)}
        return TerrainEntityCfg(terrain_type="generator", max_init_terrain_level=self.replicas-1,
                                terrain_generator=TerrainGeneratorCfg(seed=cfg.geometry.seed, curriculum=True,
                                    size=self.size, num_rows=self.replicas, num_cols=len(sub),
                                    difficulty_range=(0.0, 1.0), sub_terrains=sub))

    def bind(self, origins):
        if self.plane:
            if origins is not None:
                raise ValueError("Plane unexpectedly has tile origins")
        else:
            if origins is None or tuple(origins.shape) != (self.replicas, len(self.columns), 3):
                raise ValueError("Actual terrain origins differ from the compiled mapping")
            if not torch.isfinite(origins).all():
                raise ValueError("Nonfinite terrain origins")
            self.origins = origins

    def summary(self):
        if self.plane:
            return [dict(family="flat", level=self.columns[0][1], replica=None, row=None, column=None,
                         parameters={}, origin="per-environment plane grid (unbound)")]
        result = []
        for col, ((family, level), params) in enumerate(zip(self.columns, self.parameters)):
            for row in range(self.replicas):
                entry = dict(family=family, level=level, replica=row, row=row, column=col, parameters=params,
                             origin="unbound", deterministic_duplicates=family != "random_rough" and self.replicas > 1)
                if family == "random_rough":
                    unit = params["vertical_scale"]
                    step = int(params["noise_step"] / unit)
                    hi = int(params["height_max_m"] / unit)
                    entry["quantized_noise_levels_m"] = (np.arange(0, hi + step, step) * unit).tolist()
                result.append(entry)
        return result
