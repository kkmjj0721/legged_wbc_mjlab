from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp import (
  UniformVelocityCommand as _MjlabUniformVelocityCommand,
  UniformVelocityCommandCfg as _MjlabUniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HeadingHoldVelocityCommand(_MjlabUniformVelocityCommand):
  """对 Viser 人工命令增加航向保持，不改变策略输入维度。"""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.commanded_displacement_w = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self.episode_start_pos_w = torch.zeros(
      self.num_envs, 2, device=self.device
    )
    self._ui_heading_ref = torch.zeros(self.num_envs, device=self.device)
    self._ui_heading_hold = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._ui_forward_active = torch.zeros_like(self._ui_heading_hold)
    self._ui_last_env_idx = -1

  def reset(
    self, env_ids: torch.Tensor | slice | None
  ) -> dict[str, float]:
    extras = super().reset(env_ids)
    self.commanded_displacement_w[env_ids] = 0.0

    self._ui_heading_ref[env_ids] = 0.0
    self._ui_heading_hold[env_ids] = False
    self._ui_forward_active[env_ids] = False
    return extras

  def _integrate_command(
    self,
    dt: float | torch.Tensor,
    env_ids: torch.Tensor | None,
  ) -> None:
    heading = self.robot.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)

    vx = self.vel_command_b[:, 0]
    vy = self.vel_command_b[:, 1]

    # 先转换到世界坐标，再累计位移向量。
    vel_w = torch.stack(
      [
        cos_h * vx - sin_h * vy,
        sin_h * vx + cos_h * vy,
      ],
      dim=-1,
    )
    step_w = vel_w * torch.as_tensor(
      dt, device=self.device
    ).reshape(-1, 1)

    ids = slice(None) if env_ids is None else env_ids
    self.commanded_displacement_w[ids] += step_w[ids]

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:

    # 在重新采样前，累计刚刚执行的那条指令。
    self._integrate_command(dt, env_ids)

    # reset 后已经完成 forward，可以读取真实出生位置。
    fresh = (self._env.episode_length_buf == 0).unsqueeze(-1)
    self.episode_start_pos_w = torch.where(
      fresh,
      self.robot.data.root_link_pos_w[:, :2],
      self.episode_start_pos_w,
    )

    super().compute(dt, env_ids)

    # 没有启用人工控制时，保持原训练命令逻辑。
    if self._joystick_enabled is None or not self._joystick_enabled.value:
      self._ui_heading_hold.zero_()
      self._ui_forward_active.zero_()
      self._ui_last_env_idx = -1
      return

    assert self._joystick_get_env_idx is not None
    idx = self._joystick_get_env_idx()

    if self._ui_last_env_idx != idx:
      old_idx = self._ui_last_env_idx
      if old_idx >= 0:
        self._ui_heading_ref[old_idx] = 0.0
        self._ui_heading_hold[old_idx] = False
        self._ui_forward_active[old_idx] = False
      self._ui_heading_ref[idx] = 0.0
      self._ui_heading_hold[idx] = False
      self._ui_forward_active[idx] = False
      self._ui_last_env_idx = idx

    # 读取原始人工输入，避免把上帧自动修正当成人工转向。
    raw_vx = float(self._joystick_sliders[0].value)
    raw_vy = float(self._joystick_sliders[1].value)
    raw_wz = float(self._joystick_sliders[2].value)

    self.vel_command_b[idx, 0] = raw_vx
    self.vel_command_b[idx, 1] = raw_vy

    forward = raw_vx >= (
      0.10 if bool(self._ui_forward_active[idx].item()) else 0.15
    )
    self._ui_forward_active[idx] = forward

    yaw = self.robot.data.heading_w[idx]
    manual_turn = abs(raw_wz) > 1.0e-4
    yaw_valid = bool(torch.isfinite(yaw).item())

    if not forward or manual_turn or not yaw_valid:
      self._ui_heading_hold[idx] = False
      self.vel_command_b[idx, 2] = raw_wz
      return

    # 开始保持或人工转向结束时，锁存当前航向。
    if not bool(self._ui_heading_hold[idx].item()):
      self._ui_heading_ref[idx] = yaw.detach()
      self._ui_heading_hold[idx] = True

    error = wrap_to_pi(self._ui_heading_ref[idx] - yaw)
    self.heading_error[idx] = error
    self.vel_command_b[idx, 2] = torch.clamp(
      self.cfg.heading_control_stiffness * error,
      min=self.cfg.ranges.ang_vel_z[0],
      max=self.cfg.ranges.ang_vel_z[1],
    )


@dataclass(kw_only=True)
class HeadingHoldVelocityCommandCfg(_MjlabUniformVelocityCommandCfg):

  def build(self, env: ManagerBasedRlEnv) -> HeadingHoldVelocityCommand:
    return HeadingHoldVelocityCommand(self, env)
