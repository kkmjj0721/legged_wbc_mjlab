import os
import inspect
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
from src.config.ri_4438.numerics import ACTION_OBSERVATION_SLICE


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


class HIMOnPolicyRunner(MjlabOnPolicyRunner, RslRlHIMOnPolicyRunner):
  """Add MjLab export metadata to the RSL-RL HIM runner."""

  env: RslRlVecEnvWrapper

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    self._validate_numerics()
    return super().learn(num_learning_iterations, init_at_random_ep_len)

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
      start, stop = ACTION_OBSERVATION_SLICE
      mean, std = normalizer.mean, normalizer.std
      if label == "actor":
        mean, std = mean.reshape(actor.history_size, -1), std.reshape(actor.history_size, -1)
      if actor.action_clip is not None and (
        (std[..., start:stop] > actor.action_clip + 1e-4).any()
        or (mean[..., start:stop].abs() > actor.action_clip + 1e-4).any()
      ):
        raise ValueError(
          f"{label} last_action normalizer exceeds ±{actor.action_clip}; "
          "use a clean checkpoint (e.g. model_11500.pt), not contaminated statistics."
        )

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    infos = super().load(path, load_cfg, strict, map_location)
    saved_contract = (infos or {}).get("him_numerics")
    if saved_contract is None:
      warnings.warn("Legacy HIM checkpoint has no clipping contract; validating statistics before resume.")
    elif saved_contract != self._numerical_contract():
      raise ValueError("HIM checkpoint numerical contract differs from the current configuration")
    self._validate_numerics()
    return infos

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

  def save(self, path: str, infos: dict | None = None) -> None:
    self._validate_numerics()
    super().save(path, {**(infos or {}), "him_numerics": self._numerical_contract()})
    policy_path = os.path.dirname(path)
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    onnx_path = os.path.join(policy_path, filename)
    use_wandb = (
      isinstance(self.logger.writer, WandbLogWriter) and wandb.run is not None
    )
    if use_wandb and self.cfg.get("upload_model", True):
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))
