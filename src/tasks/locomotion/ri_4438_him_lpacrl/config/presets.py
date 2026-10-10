"""Paper task axes with explicitly chosen RI-4438 geometry."""
from src.config.ri_4438.ri_4438_him_lpacrl_config import (
    CommandAxisCfg, ExperimentCfg, TerrainFamilyCfg,
)

PRESETS = ("ri_flat_smoke", "ri_joint_smoke", "paper_flat_velocity", "paper_terrain_types", "paper_joint_space", "custom")


def preset(name: str) -> ExperimentCfg:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose {PRESETS}")
    cfg = ExperimentCfg()
    if name == "custom":
        cfg.task_space.terrains = {}
    if name in {"ri_joint_smoke", "paper_joint_space", "paper_terrain_types"}:
        count = 4 if name == "paper_joint_space" else 2
        heights = [0.05 * (i + 1) for i in range(count)]
        rough = [0.02, 0.05] if count == 2 else [0.02 + 0.08 * i / 3 for i in range(4)]
        families = {
            "random_rough": ("height_max_m", rough),
            "stairs_outward_up": ("step_height_m", heights),
            "stairs_outward_down": ("step_height_m", heights),
        }
        if name != "ri_joint_smoke":
            families.update({k: ("gradient", [0.4 * i / 3 for i in range(4)])
                             for k in ("slope_outward_up", "slope_outward_down")})
        cfg.task_space.terrains = {k: TerrainFamilyCfg({f"l{i}": {param: v} for i, v in enumerate(vals)})
                                  for k, (param, vals) in families.items()}
        if name == "paper_terrain_types":
            cfg.task_space.terrains = {"flat": TerrainFamilyCfg(), **{
                k: TerrainFamilyCfg({"base": {param: {"height_max_m": 0.05, "step_height_m": 0.10, "gradient": 0.20}[param]}})
                for k, (param, _) in families.items()}}
            cfg.task_space.command_axes = {a: CommandAxisCfg(mode="continuous", range=(-1.0, 1.0)) for a in ("vx", "vy", "wz")}
    if name == "paper_flat_velocity":
        cfg.task_space.command_axes["vx"].edges = [i * 0.5 for i in range(9)]
    if name == "paper_joint_space":
        cfg.task_space.command_axes["vx"].edges = [i * 0.5 for i in range(6)]
        cfg.task_space.command_axes["wz"] = CommandAxisCfg(mode="magnitude_bins", edges=[i * 0.5 for i in range(7)])
        cfg.task_space.command_axes["vy"] = CommandAxisCfg(mode="continuous", range=(-0.5, 0.5))
    return cfg
