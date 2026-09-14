"""Register Unitree Go2 HIM locomotion tasks."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.locomotion.go2_him.rl import HIMOnPolicyRunner

from .env_cfgs import (
  unitree_go2_him_flat_env_cfg,
  unitree_go2_him_rough_env_cfg,
)
from .rl_cfg import unitree_go2_him_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2-HIM-Rough",
  env_cfg=unitree_go2_him_rough_env_cfg(),
  play_env_cfg=unitree_go2_him_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_him_runner_cfg(),
  runner_cls=HIMOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-HIM-Flat",
  env_cfg=unitree_go2_him_flat_env_cfg(),
  play_env_cfg=unitree_go2_him_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_him_runner_cfg(),
  runner_cls=HIMOnPolicyRunner,
)
