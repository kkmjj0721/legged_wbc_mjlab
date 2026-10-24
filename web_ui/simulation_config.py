"""Validate the MuJoCo options exposed by the migration workbench."""

import math

ENUMS = {"integrator": {"euler", "implicitfast"}, "cone": {"pyramidal", "elliptic"},
         "jacobian": {"auto", "dense", "sparse"}, "solver": {"newton", "cg"}}
COUNTS = {"iterations", "ls_iterations", "ccd_iterations"}
POSITIVE = {"impratio", "tolerance", "ls_tolerance"}
OPTIONS = set(ENUMS) | COUNTS | POSITIVE | {"gravity", "disableflags", "enableflags"}


def validate_options(options):
    import mujoco

    if not isinstance(options, dict) or set(options) - OPTIONS:
        raise ValueError("environment.mujoco 包含未支持的仿真选项")
    for key, value in options.items():
        label = f"environment.mujoco.{key}"
        if key in ENUMS and (not isinstance(value, str) or value not in ENUMS[key]):
            raise ValueError(f"{label} 必须为 {sorted(ENUMS[key])}")
        if key in COUNTS and (type(value) is not int or not 1 <= value <= 10000):
            raise ValueError(f"{label} 必须为 1–10000 的整数")
        if key in POSITIVE and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{label} 必须为有限正数")
        if key == "gravity" and (not isinstance(value, (list, tuple)) or len(value) != 3
                or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
            raise ValueError(f"{label} 必须为三个有限数值")
        if key in ("disableflags", "enableflags"):
            enum, prefix = ((mujoco.mjtDisableBit, "mjDSBL_") if key == "disableflags"
                            else (mujoco.mjtEnableBit, "mjENBL_"))
            supported = {name.removeprefix(prefix).lower() for name in dir(enum) if name.startswith(prefix)}
            # The installed MuJoCo-Warp backend supports fewer flags than CPU MuJoCo.
            if key == "enableflags":
                supported &= {"energy", "invdiscrete", "sleep"}
            else:
                supported -= {"midphase", "autoreset"}
            if (not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value)
                    or len(value) != len(set(value)) or set(value) - supported):
                raise ValueError(f"{label} 必须为不重复的当前 GPU 后端支持的标志名称：{sorted(supported)}")
    if "sleep" in options.get("enableflags", ()) and options.get("solver", "newton") != "newton":
        raise ValueError("environment.mujoco.enableflags 的 sleep 要求 solver=newton")
