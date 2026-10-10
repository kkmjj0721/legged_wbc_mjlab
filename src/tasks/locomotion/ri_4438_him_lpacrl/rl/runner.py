# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause
# Local HIM export adapter and LP-ACRL lifecycle additions.
import os
import time
from tensordict import TensorDict
from rsl_rl.utils.numerics import TrainingNumericsGuard
import inspect
from pathlib import Path
import tempfile
import warnings

import onnx
import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.runners.him_on_policy_runner import HIMOnPolicyRunner as RslRlHIMOnPolicyRunner
from rsl_rl.utils.wandb_log_writer import WandbLogWriter
from rsl_rl.utils.numerics import tensor_items
from ..mdp.curriculums import get_controller
from .wrapper import TimeoutMaskWrapper


def _single_file_onnx_kwargs() -> dict:
  """Request inline ONNX weights when the installed torch supports it."""
  try:
    parameters = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}
  if "external_data" in parameters:
    return {"external_data": False}
  if "use_external_data_format" in parameters:
    return {"use_external_data_format": False}
  return {}


class LPACRLRunner(MjlabOnPolicyRunner, RslRlHIMOnPolicyRunner):
  """Add MjLab export metadata to the RSL-RL HIM runner."""

  env: RslRlVecEnvWrapper


  def _numerical_contract(self) -> dict:
    actor = self.alg.get_policy()
    return {
      "version": 1,
      "action_clip": actor.action_clip,
      "observation_clip": actor.observation_clip,
      "action_observation_slice": list(actor.action_observation_slice or ()),
      "history_size": actor.history_size,
      "frame_size": actor.num_one_step_obs,
      "export_history_order": "frame_major_current_first",
      "last_action": "clipped_policy_action_before_joint_scale_and_offset",
    }

  def _validate_numerics(self) -> None:
    """Local check: save/export is called on rank zero only."""
    actor = self.alg.get_policy()
    wrapper_clip = getattr(self.env, "clip_actions", actor.action_clip)
    if wrapper_clip != actor.action_clip:
      raise ValueError(f"Environment action clip {wrapper_clip} differs from policy {actor.action_clip}")
    env = getattr(self.env, "unwrapped", None)
    observation_cfg = getattr(getattr(env, "observation_manager", None), "cfg", {})
    for group_name in ("actor", "critic", "estimator"):
      if group_name not in observation_cfg:
        continue
      for name, term in observation_cfg[group_name].terms.items():
        bound = actor.action_clip if name == "actions" else actor.observation_clip
        expected = (-bound, bound) if bound is not None else None
        if term.clip != expected:
          raise ValueError(f"{group_name}/{name} observation clip {term.clip} differs from policy {expected}")
    for label in ("optimizer", "estimator"):
      owner = getattr(self.alg, label, None)
      optimizer = getattr(owner, "optimizer", owner)
      for name, tensor in tensor_items(getattr(optimizer, "state", {})):
        if not torch.isfinite(tensor).all():
          raise ValueError(f"Invalid {label} optimizer state: {name}")
    for label, model in (("actor", actor), ("critic", self.alg.critic)):
      for name, tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all():
          raise ValueError(f"Invalid {label} tensor: {name}")
      normalizer = getattr(model, "obs_normalizer", None)
      if normalizer is None or not hasattr(normalizer, "std"):
        continue
      if (normalizer._var < 0).any() or (normalizer._std < 0).any() or normalizer.count < 0:
        raise ValueError(f"Invalid {label} normalizer variance/std")
      if actor.action_observation_slice is None:
        continue
      start, stop = actor.action_observation_slice
      mean, std = normalizer.mean, normalizer.std
      if label == "actor":
        mean, std = mean.reshape(actor.history_size, -1), std.reshape(actor.history_size, -1)
      if actor.action_clip is not None and (
        (std[..., start:stop] > actor.action_clip + 1e-4).any()
        or (mean[..., start:stop].abs() > actor.action_clip + 1e-4).any()
      ):
        raise ValueError(
          f"{label} last_action normalizer exceeds ±{actor.action_clip}; "
          "use a checkpoint with valid normalization statistics."
        )




  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export normalization, estimator, and deterministic actor as one graph."""
    self._validate_numerics()
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    input_names = list(onnx_model.input_names)
    output_names = list(onnx_model.output_names)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".policy_", suffix=".onnx", dir=path)
    os.close(descriptor)
    try:
      self._export_onnx(onnx_model, temporary_path, input_names, output_names, verbose)
      use_wandb = isinstance(self.logger.writer, WandbLogWriter) and wandb.run is not None
      metadata = get_base_metadata(self.env.unwrapped, wandb.run.name if use_wandb else "local")
      metadata.update({f"him_{key}": str(value) for key, value in self._numerical_contract().items()})
      attach_metadata_to_onnx(temporary_path, metadata)
      onnx.checker.check_model(temporary_path)
      os.replace(temporary_path, os.path.join(path, filename))
    finally:
      if os.path.exists(temporary_path):
        os.unlink(temporary_path)

  @staticmethod
  def _export_onnx(onnx_model, filename, input_names, output_names, verbose):
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      filename,
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=input_names,
      output_names=output_names,
      dynamic_axes={name: {0: "batch"} for name in (*input_names, *output_names)},
      dynamo=False,
      **_single_file_onnx_kwargs(),
    )



  # Training lifecycle copied from rsl_rl HIM runner (BSD-3-Clause).
  def _learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
      """Collect rollouts, reset done environments, and update HIMPPO."""
      self._validate_numerics()
      self._validate_observation_contract()
      self._maybe_evaluate()
      init_at_random_ep_len = False

      obs = self.env.get_observations().to(self.device)
      self.alg.train_mode()
      if self.is_distributed:
          print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
          self.alg.broadcast_parameters()
      self.logger.init_logging_writer()
      unwrapped = getattr(self.env, "unwrapped", None)
      manual_reset = not getattr(getattr(unwrapped, "cfg", None), "auto_reset", True)
      guard = None
      if self.cfg.get("numerics"):
          guard = TrainingNumericsGuard(self.device, self.logger.log_dir, **self.cfg["numerics"])
          self.alg.numerics_guard = guard

      def check_observations(stage, values):
          if guard is None:
              return
          raw_bad = getattr(unwrapped, "_him_numerics_bad", None)
          evidence = getattr(unwrapped, "_him_numerics_evidence", {})
          guard.check(stage, {"observations": values, "pre_clip": evidence},
                      limit=guard.cfg["raw_observation_abort"], extra_bad=raw_bad)
          if raw_bad is not None:
              raw_bad.zero_()
              for value in evidence.values():
                  value.zero_()

      start_it = self.controller.state.completed_policy_updates
      total_it = start_it + num_learning_iterations
      for it in range(start_it + 1, total_it + 1):
          start = time.time()
          action_peak = torch.zeros((), device=self.device)
          action_counts = torch.zeros(3, device=self.device)
          with torch.inference_mode():
              for step in range(self.cfg["num_steps_per_env"]):
                  if guard is not None:
                      guard.iteration, guard.step = it, step
                      guard.remember(observations=obs)
                  check_observations("before_action", obs)
                  actions = self.alg.act(obs)
                  if guard is not None:
                      guard.remember(actions=actions)
                      guard.check("raw_action", {"actions": actions, "distribution": self.alg.actor.output_distribution_params})
                      action_peak = torch.maximum(action_peak, actions.abs().max())
                      action_counts[0] += (actions.abs() > self.alg.actor.action_clip).sum()
                      action_counts[1] += actions.numel()
                      action_counts[2] += (actions.abs() > guard.cfg["raw_action_warn"]).sum()
                  next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                  check_observations("after_step", {
                      "next": next_obs, "terminal": extras.get("terminal_observations"),
                  })
                  if guard is not None:
                      guard.check("step_result", {"rewards": rewards, "dones": dones})
                  elif self.cfg.get("check_for_nan", True):
                      from rsl_rl.utils import check_nan

                      check_nan(next_obs, rewards, dones)
                  next_obs = next_obs.to(self.device)
                  rewards = rewards.to(self.device)
                  dones = dones.to(self.device)

                  # Keep a copy because ``reset()`` replaces the environment's
                  # log dictionary when auto_reset is disabled.
                  step_extras = dict(extras)
                  step_log = dict(step_extras.get("log") or {})

                  # HIMPPO stores next_obs before reset. With auto_reset=False
                  # this is the true terminal physics state for done envs.
                  if manual_reset:
                      step_extras["terminal_observations"] = next_obs
                  self.alg.process_env_step(next_obs, rewards, dones, step_extras)
                  reset_extras: dict = {}
                  if manual_reset:
                      reset_ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
                      if reset_ids.numel() > 0:
                          next_obs, reset_extras = self._reset_done_envs(next_obs, reset_ids)
                  if manual_reset:
                      check_observations("after_reset", next_obs)
                  # With auto_reset=False, MjLab emits Episode_Reward/*,
                  # Episode_Metrics/*, and Episode_Termination/* from the
                  # explicit reset call rather than from env.step(). Merge
                  # those values before handing the step to Logger.
                  log_extras = dict(step_extras)
                  reset_log = reset_extras.get("log")
                  if isinstance(reset_log, dict):
                      step_log.update(reset_log)
                  log_extras["log"] = step_log
                  self.logger.process_env_step(rewards, dones, log_extras, self.alg.intrinsic_rewards)
                  obs = next_obs

              collect_time = time.time() - start
              start = time.time()
              self.alg.compute_returns(obs)

          loss_dict = self.alg.update()
          self.controller.state.completed_policy_updates += 1
          self._maybe_evaluate()
          if guard is not None:
              # These buffers were created outside inference_mode and can
              # safely be reduced once per rollout, including with NCCL.
              action_peak = action_peak.clone()
              if self.is_distributed:
                  torch.distributed.all_reduce(action_peak, op=torch.distributed.ReduceOp.MAX)
                  torch.distributed.all_reduce(action_counts)
              loss_dict["numerics/raw_action_max"] = action_peak.item()
              loss_dict["numerics/action_clip_fraction"] = (action_counts[0] / action_counts[1].clamp_min(1)).item()
              loss_dict["numerics/raw_action_warn_fraction"] = (action_counts[2] / action_counts[1].clamp_min(1)).item()
              policy = self.alg.get_policy()
              if policy.obs_normalization and policy.action_observation_slice is not None:
                  slice_start, slice_stop = policy.action_observation_slice
                  action_std = policy.obs_normalizer.std.reshape(policy.history_size, -1)[:, slice_start:slice_stop]
                  loss_dict["numerics/last_action_std_max"] = action_std.max().item()
          learn_time = time.time() - start
          self.current_learning_iteration = it

          self.logger.log(
              it=it,
              start_it=start_it,
              total_it=total_it,
              collect_time=collect_time,
              learn_time=learn_time,
              loss_dict=loss_dict,
              learning_rate=self.alg.learning_rate,
              action_std=self.alg.get_policy().output_std,
              rnd_weight=None,
          )

          if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
              self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

      if self.logger.writer is not None:
          self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
          self.logger.stop_logging_writer()

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    from ..config.experiment_cfg import compile_experiment
    from ..config.rl_cfg import ri_4438_him_runner_cfg
    from ..mdp.task_catalog import fingerprint
    from dataclasses import asdict
    from ..evaluation.evaluator import TaskEvaluator
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
      raise ValueError("LPACRL v1 supports single-GPU training only")
    self.controller = get_controller(env.unwrapped)
    expected = asdict(ri_4438_him_runner_cfg(self.controller.cfg))
    for key in ("actor", "critic", "algorithm", "clip_actions", "obs_groups", "num_steps_per_env", "numerics"):
      if fingerprint(train_cfg.get(key)) != fingerprint(expected[key]):
        raise ValueError(f"Agent {key} changed after compilation; configure it through LPACRL rl.train")
    super().__init__(TimeoutMaskWrapper(env), train_cfg, log_dir, device)
    self.compiled = compile_experiment(self.controller.cfg)
    self.evaluator = TaskEvaluator(self.compiled, device)

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    try:
      return self._learn(num_learning_iterations, False)
    finally:
      self.close()

  def _validate_observation_contract(self):
    obs = self.env.get_observations()
    expected = {"actor": 282, "critic": 244, "estimator": 50}
    for name, size in expected.items():
      if name not in obs or obs[name][0].numel() != size:
        raise ValueError(f"HIM {name} observation width must be {size}")
    if self.env.num_actions != 12:
      raise ValueError("HIM requires 12 actions")

  def _maybe_evaluate(self):
    import json
    from ..evaluation.evaluator import export_result
    from ..mdp.task_catalog import plain
    state = self.controller.state
    update = state.completed_policy_updates
    due = state.mean_returns is None or update - state.last_evaluated_update >= self.controller.cfg.evaluation.interval_updates
    if not due:
      return
    try:
      result = self.evaluator.evaluate(self.alg.get_policy())
      # Export before commit: output failure must not partially advance the clock.
      if self.logger.log_dir:
        export_result(result, Path(self.logger.log_dir) / "evaluations" / f"update_{update}",
                      self.controller.catalog, self.compiled.fingerprints)
      self.controller.commit_evaluation(result, update)
    except Exception as exc:
      state.consecutive_failures += 1
      # A failed environment can contain pending resets or invalid state.
      self.evaluator.close()
      warnings.warn(f"LPACRL evaluation failed at update {update}: {exc}; state retained")
      if state.consecutive_failures >= self.controller.cfg.evaluation.max_consecutive_failures:
        raise RuntimeError("LPACRL consecutive evaluation failure limit reached") from exc
      return
    if self.logger.log_dir:
      data = dict(update=update, mean_returns=state.mean_returns, lp=state.lp,
                  probabilities=state.probabilities, starts=self.controller.starts,
                  completed=self.controller.completed, slot_steps=self.controller.slot_steps,
                  macro=result["macro"], diagnostics=state.diagnostics())
      with (Path(self.logger.log_dir) / "curriculum.jsonl").open("a") as handle:
        handle.write(json.dumps(plain(data), allow_nan=False) + "\n")
    if self.logger.writer is not None:
      for key, value in {**result["macro"], **state.diagnostics()}.items():
        self.logger.writer.add_scalar("LPACRL/" + key, value, update)

  def save(self, path, infos=None):
    self._validate_numerics()
    payload = {**(infos or {}), "him_numerics": self._numerical_contract(),
               "lpacrl": self.controller.state_dict()}
    # Base Mjlab runner persists common_step_counter as well.
    super().save(path, payload)
    self.export_policy_to_onnx(os.path.dirname(path), "policy.onnx")

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
    infos = checkpoint.get("infos") or {}
    if "lpacrl" not in infos:
      raise ValueError("Checkpoint has no LPACRL history; use --warm-start explicitly")
    if infos.get("him_numerics") != self._numerical_contract():
      raise ValueError("Checkpoint HIM numerical contract differs")
    if load_cfg is not None:
      raise ValueError("Partial resume is unsupported; use warm_start")
    self.controller.load_state_dict(infos["lpacrl"])
    self.alg.load(checkpoint, None, strict)
    self._validate_numerics()
    self.current_learning_iteration = self.controller.state.completed_policy_updates
    self.env.unwrapped.common_step_counter = infos.get("env_state", {}).get("common_step_counter", 0)
    self.env.reset()
    return infos

  def warm_start(self, path):
    checkpoint = torch.load(path, weights_only=False, map_location=self.device)
    contract = (checkpoint.get("infos") or {}).get("him_numerics")
    if contract is not None and contract != self._numerical_contract():
      raise ValueError("Warm-start HIM numerical contract differs")
    if contract is None:
      warnings.warn("Legacy HIM weights lack a clipping contract; validating model statistics")
    checkpoint.pop("optimizer_state_dict", None)
    checkpoint.pop("estimator_optimizer_state_dict", None)
    self.alg.load(checkpoint, {"actor": True, "critic": True, "estimator": True,
                              "optimizer": False, "iteration": False}, strict=True)
    self._validate_numerics()
    self.current_learning_iteration = 0
    self.env.unwrapped.common_step_counter = 0
    self.env.reset()

  def close(self):
    self.evaluator.close()
