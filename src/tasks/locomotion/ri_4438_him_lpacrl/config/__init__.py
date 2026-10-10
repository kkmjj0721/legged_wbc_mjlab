"""Register only the default flat/rough presets; custom spaces use rl.train."""
from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import ri_4438_flat_env_cfg, ri_4438_rough_env_cfg
from .rl_cfg import ri_4438_him_runner_cfg
from ..rl.runner import LPACRLRunner

for task_id, factory in (("Ri-4438-HIM-LPACRL-Flat", ri_4438_flat_env_cfg),
                         ("Ri-4438-HIM-LPACRL-Rough", ri_4438_rough_env_cfg)):
    register_mjlab_task(task_id=task_id, env_cfg=factory(), play_env_cfg=factory(play=True),
                       rl_cfg=ri_4438_him_runner_cfg(), runner_cls=LPACRLRunner)
