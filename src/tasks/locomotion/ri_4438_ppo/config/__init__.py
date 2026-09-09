from mjlab.tasks.registry import register_mjlab_task
from src.tasks.locomotion.ri_4438_ppo.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  ri_4438_rough_env_cfg,
  ri_4438_flat_env_cfg,
)
from .rl_cfg import ri_4438_ppo_runner_cfg

register_mjlab_task(
  task_id = "Ri-4438-Rough",
  env_cfg = ri_4438_rough_env_cfg(),
  play_env_cfg = ri_4438_rough_env_cfg(play=True),
  rl_cfg = ri_4438_ppo_runner_cfg(),
  runner_cls = VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id = "Ri-4438-Flat",
  env_cfg = ri_4438_flat_env_cfg(),
  play_env_cfg = ri_4438_flat_env_cfg(play=True),
  rl_cfg = ri_4438_ppo_runner_cfg(),
  runner_cls = VelocityOnPolicyRunner,
)
