# Architecture Context

## System State

- Repository profile: Python robotics/RL simulation project rooted at `/home/kk/legged_wbc_mjlab`.
- Detected runtime stack: MuJoCo/mjlab-based simulation code with local `rsl_rl` package.
- No STM32 `.ioc`, Keil project, Windows batch workflow, or ROS `catkin`/`ament` CMake profile detected during initial scan.

## Active Interface Contracts

- Actuator configuration is provided through `mjlab.actuator` dataclass-style `ActuatorCfg` subclasses.
- Robot-specific actuator groups are declared under `src/assets/robots/*/*_constants.py` and matched to targets by `target_names_expr`.

## Technical Debt

- To be populated as implementation and validation agents identify concrete issues.

## Troubleshooting

- To be populated with reproducible failures, fixes, and validation commands.
