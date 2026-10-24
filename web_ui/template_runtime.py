"""Compatibility imports for existing workbench preview and PD tools.

Runtime implementation is shared by project tasks in src.utils.robot_config.
"""

from src.utils.robot_config import (
    MigrationJointPositionAction,
    MigrationJointPositionActionCfg,
    exact_names,
    joint_value,
    class_to_dict,
    parameters_from_config,
    action_scales,
    bound_joints,
    model_spec,
    create_robot_cfg,
    apply_terrain_parameters,
    adapt_env,
    adapt_runner,
    get_migration_metadata,
)

__all__ = [
    "MigrationJointPositionAction", "MigrationJointPositionActionCfg",
    "exact_names", "joint_value", "class_to_dict", "parameters_from_config",
    "action_scales", "bound_joints", "model_spec", "create_robot_cfg",
    "apply_terrain_parameters", "adapt_env", "adapt_runner", "get_migration_metadata",
]
