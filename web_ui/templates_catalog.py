"""Read the repository's current RI-4438 defaults for the migration form."""

from copy import deepcopy
from functools import lru_cache
import importlib
import inspect


TEMPLATES = {
    "ppo": ("PPO", "ri_4438_ppo", "Ri4438Cfg", "Ri4438CfgPPO"),
    "him": ("HIM + PPO", "ri_4438_him", "Ri4438HimCfg", "Ri4438CFGHimPPO"),
    "him_symmetry": ("HIM Symmetry", "ri_4438_him_symmetry", "Ri4438HimCfg", "Ri4438CFGHimPPO"),
    "him_lpacrl": ("HIM LP-ACRL", "ri_4438_him_lpacrl", "Ri4438HimLpacrlCfg", "Ri4438HimLpacrlCfgPPO"),
}
LP_PRESETS = [
    {"suffix": "", "label": "默认（Rough 832 组 / Flat 16 组）", "group": "默认"},
    *({"suffix": name, "label": name, "group": "Compact"}
      for name in ("Compact", "Compact-Coverage", "Compact-Uniform")),
    *({"suffix": name, "label": name, "group": "历史对照"}
      for name in ("A0", "A1", "A2", "A3", "Random-Uniform")),
]
CONTRACT_FIELDS = {
    "num_actions", "num_action", "num_one_step_obs", "history_size", "num_observations",
    "num_obs_history", "num_obs_one_step", "num_obs", "rough_num_privileged_obs",
    "flat_num_privileged_obs", "num_privileged_obs", "history_term_dims", "history_order",
    "estimator_velocity_slice", "estimator_target_slices",
}


def class_values(cls):
    values = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        value = getattr(cls, name)
        if inspect.isclass(value):
            values[name] = class_values(value)
        elif not callable(value):
            values[name] = deepcopy(value)
    return values


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()
                if _serializable(item)}
    raise TypeError(type(value).__name__)


def _serializable(value):
    try:
        _plain(value)
        return True
    except TypeError:
        return False


def _fields(values, prefix=""):
    result = []
    labels = {"control": "控制", "init_state": "初始姿态", "environment": "环境",
              "commands": "速度指令", "randomization": "随机化", "rewards": "奖励",
              "observations": "观测", "learning": "训练", "reset": "重置",
              "terrain": "地形", "curriculum": "课程"}
    for name, value in values.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict) and value:
            result.extend(_fields(value, path))
        else:
            entry = {"path": path, "label": labels.get(name, name),
                     "type": ("boolean" if isinstance(value, bool) else "integer"
                              if isinstance(value, int) else "number" if isinstance(value, float)
                              else "array" if isinstance(value, list) else "object"
                              if isinstance(value, dict) else "string"),
                     "default": value, "source": "模板默认",
                     "readonly": name in CONTRACT_FIELDS or path == "learning.runner.experiment_name"
                     or path.startswith("lpacrl.height_scan.")}
            if name in ("sim_dt", "episode_length_s"):
                entry["unit"] = "s"
            if name in ("env_spacing", "pos"):
                entry["unit"] = "m"
            if name == "nconmax":
                entry["source"] = "迁移预设"
                entry["description"] = "新模型的接触缓冲区保守初值，至少 256；可按小规模仿真实际接触数调整。"
            if name == "njmax":
                entry["source"] = "迁移预设"
                entry["description"] = "新模型的约束缓冲区保守初值，至少 1500；可按小规模仿真实际约束数调整。"
            if name.startswith("delay_") and name.endswith(("lag", "period")):
                entry["unit"] = "仿真步" if path.startswith("control.") else "控制步"
            result.append(entry)
    return result


@lru_cache(maxsize=1)
def _catalog():
    result = []
    for key, (label, package, env_name, learning_name) in TEMPLATES.items():
        cfg_file = "ri_4438_config" if key == "ppo" else (
            "ri_4438_him_lpacrl_config" if key == "him_lpacrl" else "ri_4438_him_config")
        module = importlib.import_module(f"src.config.ri_4438.{cfg_file}")
        env_defaults = class_values(getattr(module, env_name))
        learning = class_values(getattr(module, learning_name))
        env_module = importlib.import_module(f"src.tasks.locomotion.{package}.config.env_cfgs")
        live = env_module.ri_4438_rough_env_cfg()
        defaults = {
            "control": env_defaults["control"], "init_state": env_defaults["init_state"],
            "environment": {"num_envs": live.scene.num_envs, "env_spacing": live.scene.env_spacing,
                            "episode_length_s": live.episode_length_s,
                            "sim_dt": live.sim.mujoco.timestep,
                            "nconmax": max(256, live.sim.nconmax or 0),
                            "njmax": max(1500, live.sim.njmax or 0),
                            "contact_sensor_maxmatch": live.sim.contact_sensor_maxmatch},
            "commands": env_defaults.get("comaman", env_defaults.get("commands", {})),
            "randomization": {"enabled": True, **env_defaults.get("domain_rand", {})},
            "reset": env_defaults.get("reset", {}),
            "rewards": {name: {"weight": term.weight, "params": _plain(term.params)}
                        for name, term in live.rewards.items()},
            "observations": {
                "enable_corruption": live.observations["actor"].enable_corruption,
                "delay_min_lag": 0, "delay_max_lag": 1 if key != "ppo" else 0,
                "delay_hold_prob": 0.3 if key != "ppo" else 0.0,
                "delay_update_period": 10 if key != "ppo" else 0,
                "noise": {name: {field: getattr(term.noise, field) for field in ("n_min", "n_max")}
                          for name, term in live.observations["actor"].terms.items()
                          if term.noise is not None and hasattr(term.noise, "n_min")},
            },
            "learning": {name: learning[name] for name in ("policy", "algorithm", "runner")},
            "terrain": env_defaults.get("terrain", {}),
            "curriculum": {"enabled": True, **env_defaults.get("curriculum", {})},
        }
        defaults["control"].setdefault("friction", None)
        defaults["control"].setdefault("passive_damping", None)
        if key != "him_lpacrl" and live.scene.terrain.terrain_generator is not None:
            generator = live.scene.terrain.terrain_generator
            defaults["terrain"] = {name: _plain(getattr(generator, name))
                                   for name in ("num_rows", "num_cols", "size", "border_width",
                                                "curriculum", "difficulty_range") if hasattr(generator, name)}
        if key == "him_lpacrl":
            defaults["lpacrl"] = {name: env_defaults[name] for name in
                                  ("compact", "behavior_monitor", "evaluation", "height_scan")}
        defaults = _plain(defaults)
        result.append({"key": key, "label": label, "terrains": ["Flat", "Rough"],
                       "description": "复制当前 RI-4438 完整实现，并用显式绑定适配新机器人。",
                       "source": f"src/tasks/locomotion/{package}",
                       "defaults": defaults, "fields": _fields(defaults),
                       "presets": deepcopy(LP_PRESETS) if key == "him_lpacrl" else [],
                       "contract": ({"name": "phase47_v1", "actor_frame": 47,
                                     "history_size": 6, "actor_width": 282,
                                     "critic_width": 244, "estimator_width": 50,
                                     "actions": 12} if key != "ppo" else {"actions": 12})})
    return result


def catalog():
    """Return independent JSON-compatible template descriptions and defaults."""
    return deepcopy(_catalog())
