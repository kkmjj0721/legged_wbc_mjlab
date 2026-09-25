"""Capture fresh HIM supervision before mjlab automatically resets an episode."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import gc

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

from .observations import phase


ESTIMATOR_TERMS = (
  "base_ang_vel", "projected_gravity", "command", "phase",
  "joint_pos", "joint_vel", "actions", "base_lin_vel",
)


def configure_estimator_observations(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Expose the critic's first 50 features without history or raycast dependencies.

  Actor/critic inputs and the estimator's existing feature slices stay unchanged.
  A separate group avoids storing unused height scans as next-step supervision.
  """
  cfg.observations["estimator"] = ObservationGroupCfg(
    terms={
      name: replace(
        deepcopy(cfg.observations["critic"].terms[name]),
        history_length=0, delay_min_lag=0, delay_max_lag=0, noise=None,
      )
      for name in ESTIMATOR_TERMS
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )
  if not play:
    cfg.recorders["him_terminal"] = RecorderTermCfg(func=TerminalEstimatorRecorder)


class TerminalEstimatorRecorder(RecorderTerm):
  """Refresh terminal kinematics and preserve estimator targets before reset.

  The pre-reset hook follows integration and push events. Its derived state is
  otherwise one physics substep old, so reading the existing observation buffer
  here would supervise HIM with the wrong time step. Position/velocity kernels
  refresh the IMU and gravity features without collision solving or raycasting.
  The normal post-reset forward/sense pipeline still runs for the next action.
  """

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._refresh_graph = None
    self._source_graph = None
    self._recorded = False
    group = env.observation_manager.cfg["estimator"]
    if group.history_length or any(
      term.history_length or term.delay_max_lag for term in group.terms.values()
    ):
      raise ValueError("Terminal estimator observations must have no history or delay")

    # Retain correctness if a task customizes a term to require other sensors.
    supported = {
      mdp.builtin_sensor, mdp.projected_gravity, mdp.generated_commands,
      mdp.joint_pos_rel, mdp.joint_vel_rel, mdp.last_action, phase,
    }
    self._kinematics_only = all(term.func in supported for term in group.terms.values())
    for term in group.terms.values():
      if term.func is mdp.builtin_sensor:
        sensor = env.sim.mj_model.sensor(term.params["sensor_name"])
        if sensor.type[0] not in (
          mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtSensor.mjSENS_VELOCIMETER,
        ):
          self._kinematics_only = False

  def _refresh_kinematics(self) -> None:
    sim = self._env.sim
    mjwarp.kinematics(sim.wp_model, sim.wp_data)
    mjwarp.com_pos(sim.wp_model, sim.wp_data)
    mjwarp.com_vel(sim.wp_model, sim.wp_data)
    mjwarp.sensor_vel(sim.wp_model, sim.wp_data)

  def _refresh_terminal_state(self) -> None:
    sim = self._env.sim
    if not self._kinematics_only:
      sim.forward()
      sim.sense()
      return
    with wp.ScopedDevice(sim.wp_device):
      if not sim.use_cuda_graph:
        self._refresh_kinematics()
        return
      # Model field expansion replaces GPU arrays and recreates sim graphs.
      if self._refresh_graph is None or self._source_graph is not sim.step_graph:
        self._refresh_graph = None
        self._source_graph = sim.step_graph
        # As in Simulation.create_graph(), keep graph destructors out of capture.
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
          with wp.ScopedCapture(device=sim.wp_device) as capture:
            self._refresh_kinematics()
          self._refresh_graph = capture.graph
        finally:
          if gc_enabled:
            gc.enable()
      wp.capture_launch(self._refresh_graph)

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    env = self._env
    self._refresh_terminal_state()
    # Match the manual path: the terminal phase uses this step's command update.
    # The framework subsequently resets these timers and advances them by dt=0.
    env.command_manager.compute(dt=env.step_dt, env_ids=env_ids)
    terminal = env.observation_manager.compute_group("estimator")
    assert isinstance(terminal, torch.Tensor)
    env.extras["terminal_observations"] = TensorDict(
      {"estimator": terminal[env_ids].clone()}, batch_size=[env_ids.numel()],
    )
    self._recorded = True

  def record_post_reset(self, env_ids: torch.Tensor) -> None:
    del env_ids
    # Explicit reset() starts outside the pre-reset hook.
    if not self._recorded:
      self._env.extras.pop("terminal_observations", None)

  def record_post_step(self) -> None:
    if not self._recorded:
      self._env.extras.pop("terminal_observations", None)
    self._recorded = False
