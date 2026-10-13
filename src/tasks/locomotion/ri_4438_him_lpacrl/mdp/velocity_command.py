"""Expose task velocity targets; heading mode updates feedback wz every step."""

from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


class TaskVelocityCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self):
        return self.vel_command_b

    def reset(self, env_ids):
        self._resample_command(env_ids)
        self.command_counter[env_ids] = 1
        return {}

    def compute(self, dt, env_ids=None):
        from .curriculums import get_controller

        del dt  # 由 episode_length_buf 索引；terminal recorder 的重复调用是幂等的。
        controller = get_controller(self._env)
        ids = controller._ids(env_ids)
        controller.update_commands(ids)
        self.vel_command_b[ids] = controller.commands[ids]

    def _resample_command(self, env_ids):
        from .curriculums import get_controller

        self.vel_command_b[env_ids] = get_controller(self._env).commands[env_ids]

    def _update_command(self, env_ids=None):
        pass

    def _update_metrics(self):
        pass


@dataclass(kw_only=True)
class TaskVelocityCommandCfg(CommandTermCfg):
    def build(self, env):
        return TaskVelocityCommand(self, env)
