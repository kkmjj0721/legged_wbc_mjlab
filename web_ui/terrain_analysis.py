"""CPU terrain layout and actual MuJoCo geometry counts for the workbench."""

from collections import Counter
from copy import deepcopy
import importlib
import math
import time
from types import SimpleNamespace

from .terrain_parameters import sub_terrain_label


# Guard the optional synchronous CPU build without altering the user's config.
MAX_PATCHES = 1024
MAX_GEOMS = 50000
MAX_HEIGHT_SAMPLES = 4000000


def _lp_terrain(parameters, terrain):
    """Build the effective LP map through its existing CPU configuration path."""
    from src.tasks.locomotion.ri_4438_him_lpacrl.config.curriculum_cfg import (
        build_curriculum, make_lpacrl_cfg,
    )

    cfg = make_lpacrl_cfg()
    values = parameters["terrain"]
    defaults = values["defaults"]
    levels = {}
    for family, entries in values["levels"].items():
        category = "stairs" if family.startswith("stairs_") else family.split("_outward_")[0]
        levels[family] = {name: {**deepcopy(defaults[category]), **deepcopy(data)}
                          for name, data in entries.items()}
    cfg.task_space.terrains = {
        family: SimpleNamespace(levels=entries) for family, entries in levels.items()
    }
    cfg.geometry.map_levels = deepcopy(levels)
    for name in vars(cfg.geometry):
        if name in values:
            setattr(cfg.geometry, name, deepcopy(values[name]))
    if terrain == "Flat":
        cfg.task_space.terrains = {"flat": SimpleNamespace(levels={"base": {}})}
    # The map may reference reset and command settings when checking its borders.
    for group, target in (("reset", cfg.robot.reset), ("curriculum", cfg.curriculum)):
        for name, value in parameters.get(group, {}).items():
            if hasattr(target, name):
                setattr(target, name, deepcopy(value))
    setup = build_curriculum(cfg)
    return setup.terrain_map.entity_cfg(setup.cfg)


def _generator_cfg(algorithm, parameters, terrain):
    from .template_runtime import apply_terrain_parameters
    from .templates_catalog import TEMPLATES

    if algorithm == "him_lpacrl":
        return _lp_terrain(parameters, terrain).terrain_generator
    if terrain == "Flat":
        return None
    package = TEMPLATES[algorithm][1]
    module = importlib.import_module(f"src.tasks.locomotion.{package}.config.env_cfgs")
    cfg = module.ri_4438_rough_env_cfg().scene.terrain.terrain_generator
    apply_terrain_parameters(cfg, parameters.get("terrain", {}))
    if not parameters.get("curriculum", {}).get("enabled", True):
        cfg.curriculum = False
    return cfg


def _complexity(sub, size):
    """Conservative allocation estimates, never presented as actual geom counts."""
    kind = type(sub).__name__
    if hasattr(sub, "tiles"):
        estimates = [_complexity(_lp_base(family, params, size), size)
                     for family, _level, params in sub.tiles]
        return (max((item[0] for item in estimates), default=1),
                max((item[1] for item in estimates), default=0))
    if hasattr(sub, "parameters") and hasattr(sub, "family"):
        return _complexity(_lp_base(sub.family, sub.parameters, size), size)
    if "Stairs" in kind:
        steps = _bounded_ratio(max(0, min(size) - 2 * sub.border_width - sub.platform_width),
                               2 * sub.step_width, MAX_GEOMS)
        return 4 * steps + 5, 0
    if hasattr(sub, "grid_width"):
        return (_bounded_ratio(size[0], sub.grid_width, MAX_GEOMS, ceil=True)
                * _bounded_ratio(size[1], sub.grid_width, MAX_GEOMS, ceil=True) + 5), 0
    if hasattr(sub, "horizontal_scale"):
        points = ((_bounded_ratio(size[0], sub.horizontal_scale, MAX_HEIGHT_SAMPLES, ceil=True) + 1)
                  * (_bounded_ratio(size[1], sub.horizontal_scale, MAX_HEIGHT_SAMPLES, ceil=True) + 1))
        return 5, points
    return 1, 0


def _bounded_ratio(numerator, denominator, limit, *, ceil=False):
    ratio = numerator / denominator
    if not math.isfinite(ratio) or ratio > limit:
        return limit + 1
    return math.ceil(ratio) if ceil else int(ratio)


def _lp_base(family, params, size):
    from src.tasks.locomotion.ri_4438_him_lpacrl.mdp.terrains import _base
    return _base(family, params, size)


def _model_statistics(model):
    import mujoco

    types = Counter(mujoco.mjtGeom(int(kind)).name.removeprefix("mjGEOM_").lower()
                    for kind in model.geom_type)
    collision = sum(int(t != 0 or a != 0)
                    for t, a in zip(model.geom_contype, model.geom_conaffinity))
    return {
        "geoms": int(model.ngeom), "collision_geoms": collision,
        "visual_only_geoms": int(model.ngeom) - collision,
        "geom_types": dict(types), "meshes": int(model.nmesh),
        "mesh_vertices": int(model.nmeshvert), "mesh_faces": int(model.nmeshface),
        "heightfields": int(model.nhfield), "heightfield_samples": int(model.nhfielddata),
    }


def analyze_terrain(spec, scope="common", terrain="Rough", *, seed=42, build=False):
    """Return effective map metadata; optionally build its full CPU geometry.

    ``global_geoms`` describes one MuJoCo model: the shared map plus one robot.
    MjLab batches that model's states across ``num_envs``; it does not multiply
    ``MjModel.ngeom`` by the environment count.
    """
    import mujoco
    from .generator import _model_details, _parameters

    started = time.monotonic()
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("地形分析种子必须为 [0, 2³²) 的整数")
    if type(build) is not bool:
        raise ValueError("build 必须为布尔值")
    if terrain not in ("Flat", "Rough"):
        raise ValueError("地形类型必须为 Flat 或 Rough")
    tasks = spec.get("tasks", {})
    algorithm = next(iter(tasks), None) if scope == "common" else scope
    if algorithm not in tasks:
        raise ValueError("请选择一个已启用的训练方案后分析地形")
    _root, _entry, bindings, limits, robot_model = _model_details(spec)
    parameters = _parameters(spec, bindings, limits, allow_unconfigured_pd=True)[algorithm]
    cfg = _generator_cfg(algorithm, parameters, terrain)
    robot = _model_statistics(robot_model)
    result = {
        "version": 1, "scope": scope, "algorithm": algorithm, "terrain": terrain,
        "seed": seed, "num_envs": parameters["environment"]["num_envs"],
        "note": "全局 geom = 整张共享地图 geom + 一个机器人模型的全部 geom（含纯视觉）。"
                "这是单个 MuJoCo 模型的 ngeom；MjLab 批量环境共享模型结构，不乘 num_envs。"
                "geom 数不是 mesh 三角面数，也不是接触点数。分析种子仅用于此次计算，不改写训练配置。",
        "robot": robot,
    }
    if cfg is None:
        summary = {
            "mode": "plane", "rows": None, "columns": None, "patch_count": 0,
            "patch_size_m": None, "grid_size_m": None, "total_size_m": None,
            "bounds_xy_m": None, "difficulty_range": None, "sub_terrains": [],
            "description": "Flat 使用无限平面；生成器的网格和子地形设置不生效。",
        }
        estimate_geoms, estimate_samples = 1, 0
    else:
        cfg = deepcopy(cfg)
        # A fixed analysis seed does not overwrite the task's seed setting.
        cfg.seed = seed
        for sub in cfg.sub_terrains.values():
            if hasattr(sub, "geometry_seed"):
                sub.geometry_seed = seed
        rows = cfg.num_rows
        columns = len(cfg.sub_terrains) if cfg.curriculum else cfg.num_cols
        grid_size = [rows * cfg.size[0], columns * cfg.size[1]]
        total_size = [side + 2 * cfg.border_width for side in grid_size]
        if any(not math.isfinite(side) for side in total_size):
            raise ValueError("整图尺寸超出有限数值范围，请减少地块尺寸、行列数或外围边宽")
        weights = sum(sub.proportion for sub in cfg.sub_terrains.values())
        if weights <= 0:
            raise ValueError("子地形权重总和必须大于 0")
        types = [{"name": name, "label": sub_terrain_label(name),
                  "class_name": type(sub).__name__, "weight": sub.proportion,
                  "normalized_weight": sub.proportion / weights,
                  "layout_share": 1 / columns if cfg.curriculum else None,
                  "patch_count": rows if cfg.curriculum else None}
                 for name, sub in cfg.sub_terrains.items()]
        complexity = [_complexity(sub, cfg.size) for sub in cfg.sub_terrains.values()]
        if cfg.curriculum:
            estimate_geoms = rows * sum(item[0] for item in complexity) + (4 if cfg.border_width else 0)
            estimate_samples = rows * sum(item[1] for item in complexity)
        else:
            estimate_geoms = rows * columns * max(item[0] for item in complexity) + (4 if cfg.border_width else 0)
            estimate_samples = rows * columns * max(item[1] for item in complexity)
        summary = {
            "mode": "curriculum" if cfg.curriculum else "random",
            "rows": rows, "columns": columns, "requested_columns": cfg.num_cols,
            "patch_count": rows * columns, "patch_size_m": list(cfg.size),
            "grid_size_m": grid_size, "total_size_m": total_size,
            "bounds_xy_m": [[-side / 2, side / 2] for side in total_size],
            "border_width_m": cfg.border_width, "difficulty_range": list(cfg.difficulty_range),
            "sub_terrains": types,
        }
    result["summary"] = summary
    result["geometry"] = {
        "status": "not_built", "robot_geoms": robot["geoms"], "terrain_geoms": None,
        "global_geoms": None, "estimate_geom_upper_bound": estimate_geoms,
        "estimate_heightfield_sample_upper_bound": estimate_samples,
        "estimate_note": "预算估算上限，仅用于限制 CPU 构建；实际数量以完成构建后的统计为准。",
    }
    if build:
        over_budget = (summary["patch_count"] > MAX_PATCHES
                       or estimate_geoms > MAX_GEOMS or estimate_samples > MAX_HEIGHT_SAMPLES)
        if over_budget:
            result["geometry"].update(status="budget_exceeded", message=(
                f"配置超过同步 CPU 分析预算（{MAX_PATCHES} 地块 / {MAX_GEOMS} 估算 geom / "
                f"{MAX_HEIGHT_SAMPLES} 高度场采样）。未缩小或改写地图；请先减少规模后计算。"))
        else:
            terrain_spec = mujoco.MjSpec()
            if cfg is None:
                terrain_spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.1])
            else:
                from mjlab.terrains.terrain_generator import TerrainGenerator

                actual_counts = Counter()
                identities = {id(sub): name for name, sub in cfg.sub_terrains.items()}

                class CountingGenerator(TerrainGenerator):
                    def _create_terrain_geom(self, spec, world_position, difficulty, sub, sub_row, sub_col):
                        actual_counts[identities[id(sub)]] += 1
                        return super()._create_terrain_geom(spec, world_position, difficulty, sub, sub_row, sub_col)

                CountingGenerator(cfg, device="cpu").compile(terrain_spec)
                for item in summary["sub_terrains"]:
                    count = actual_counts[item["name"]]
                    item["patch_count"] = count
                    item["layout_share"] = count / summary["patch_count"]
            terrain_model = terrain_spec.compile()
            actual = _model_statistics(terrain_model)
            result["terrain_geometry"] = actual
            result["geometry"].update(
                status="actual", terrain_geoms=actual["geoms"],
                global_geoms=robot["geoms"] + actual["geoms"],
                geom_types=dict(Counter(robot["geom_types"]) + Counter(actual["geom_types"])),
                meshes=robot["meshes"] + actual["meshes"],
                mesh_vertices=robot["mesh_vertices"] + actual["mesh_vertices"],
                mesh_faces=robot["mesh_faces"] + actual["mesh_faces"],
                heightfields=robot["heightfields"] + actual["heightfields"],
                heightfield_samples=robot["heightfield_samples"] + actual["heightfield_samples"],
            )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result
