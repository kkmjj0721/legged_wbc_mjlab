import os
import inspect

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

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export normalization, estimator, and deterministic actor as one graph."""
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()
    os.makedirs(path, exist_ok=True)
    input_names = list(onnx_model.input_names)
    output_names = list(onnx_model.output_names)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=input_names,
      output_names=output_names,
      dynamic_axes={name: {0: "batch"} for name in (*input_names, *output_names)},
      dynamo=False,
      **_single_file_onnx_kwargs(),
    )
    # Keep the live training model on its original device after exporting.
    self.alg.get_policy().to(self.device)

  def save(self, path: str, infos: dict | None = None) -> None:
    super().save(path, infos)
    policy_path = os.path.dirname(path)
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    onnx_path = os.path.join(policy_path, filename)
    use_wandb = (
      isinstance(self.logger.writer, WandbLogWriter) and wandb.run is not None
    )
    run_name = wandb.run.name if use_wandb else "local"
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if use_wandb and self.cfg.get("upload_model", True):
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))
