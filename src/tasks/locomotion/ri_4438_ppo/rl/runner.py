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
from rsl_rl.utils.wandb_log_writer import WandbLogWriter


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    onnx_path = os.path.join(policy_path, filename)
    writer = self.logger.writer
    use_wandb = (
      isinstance(self.logger.writer, WandbLogWriter)
      and wandb.run is not None
    )
    run_name: str = wandb.run.name if use_wandb else "local"
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if use_wandb:
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))