import inspect
import os

import torch
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.runners.him_on_policy_runner import (
  HIMOnPolicyRunner as RslRlHIMOnPolicyRunner,
)
from rsl_rl.utils.wandb_log_writer import WandbLogWriter


def _onnx_export_kwargs_single_file() -> dict:
  """Request a single-file ONNX export across supported torch versions."""
  try:
    params = inspect.signature(torch.onnx.export).parameters
  except (TypeError, ValueError):
    return {}

  if "external_data" in params:
    return {"external_data": False}
  if "use_external_data_format" in params:
    return {"use_external_data_format": False}
  return {}


def _inline_external_onnx_data(onnx_path: str) -> None:
  """Merge external tensor data into the ONNX model when torch creates it."""
  data_path = f"{onnx_path}.data"
  if not os.path.exists(data_path):
    return

  try:
    import onnx

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    if os.path.exists(data_path):
      os.remove(data_path)
    print(f"[INFO]: Inlined external ONNX data into single file: {onnx_path}")
  except Exception as exc:
    print(f"[WARN]: Failed to inline ONNX external data for {onnx_path}: {exc}")


class HIMOnPolicyRunner(MjlabOnPolicyRunner, RslRlHIMOnPolicyRunner):
  """MjLab runner integration for HIM training and policy export."""

  env: RslRlVecEnvWrapper

  def _export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Export the complete HIM inference graph to a single ONNX file.

    The policy-provided export wrapper contains observation normalization, the
    HIM estimator, and the actor, so deployment only needs to provide the raw
    flattened observation history.
    """
    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()

    input_names = list(onnx_model.input_names)  # type: ignore[attr-defined]
    output_names = list(onnx_model.output_names)  # type: ignore[attr-defined]
    dynamic_axes = {
      name: {0: "batch"} for name in (*input_names, *output_names)
    }

    os.makedirs(path, exist_ok=True)
    onnx_path = os.path.join(path, filename)
    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[attr-defined]
      onnx_path,
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=input_names,
      output_names=output_names,
      dynamic_axes=dynamic_axes,
      dynamo=False,
      **_onnx_export_kwargs_single_file(),
    )
    _inline_external_onnx_data(onnx_path)

  def export_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    """Expose HIM ONNX export through the standard runner API."""
    self._export_policy_to_onnx(path, filename, verbose)

  def save(self, path: str, infos=None) -> None:
    """Save the checkpoint and refresh its deployable HIM ONNX policy."""
    super().save(path, infos)

    policy_path = os.path.dirname(path)
    filename = "policy.onnx"
    self._export_policy_to_onnx(policy_path, filename)

    onnx_path = os.path.join(policy_path, filename)
    writer = self.logger.writer
    use_wandb = isinstance(writer, WandbLogWriter) and wandb.run is not None
    run_name: str = wandb.run.name if use_wandb else "local"  # type: ignore[assignment]
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    _inline_external_onnx_data(onnx_path)

    if use_wandb and self.cfg.get("upload_model", True):
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))

