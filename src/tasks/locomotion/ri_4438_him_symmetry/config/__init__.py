"""Register the Rough and Flat entries for ri_4438_him_symmetry."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.locomotion.ri_4438_him_symmetry.rl.runner import HimSymmetryRunner
from .env_cfgs import ri_4438_rough_env_cfg, ri_4438_flat_env_cfg
from .rl_cfg import ri_4438_him_runner_cfg

for terrain, factory in (("Rough", ri_4438_rough_env_cfg), ("Flat", ri_4438_flat_env_cfg)):
    register_mjlab_task(
        task_id=f"Ri-4438-HIM-Symmetry-{terrain}",
        env_cfg=factory(), play_env_cfg=factory(play=True),
        rl_cfg=ri_4438_him_runner_cfg(), runner_cls=HimSymmetryRunner,
    )
