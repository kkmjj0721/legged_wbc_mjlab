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

# from rsl_rl.rsl_rl.runners.on_policy_runner import OnPolicyRunner

# class _OnnxPolicyWrapper(torch.nn.Module):
#   """
#   ONNX 导出包装器。
#   将观测值归一化逻辑（obs_normalizer）直接打包进 ONNX 模型中。
#   这样 C++ 部署端只需输入原始观测值，无需再手动实现归一化逻辑。
#   """
#   def __init__(self, actor_critic, obs_normalizer=None):
#       super().__init__()
#       self.actor_critic = actor_critic
#       self.obs_normalizer = obs_normalizer

#   def forward(self, obs):
#       if self.obs_normalizer is not None:
#           obs = self.obs_normalizer(obs)
#       # 外部 rsl_rl 中，actor_critic 的推理函数通常为 act_inference
#       return self.actor_critic.act_inference(obs)
  
# class CustomOnPolicyRunner(OnPolicyRunner):
#   """
#   适配外部 rsl_rl 的自定义 Runner
#   """
#   # 强制声明 env 为 Mjlab 的 Wrapper 类型
#   env: RslRlVecEnvWrapper

#   def _export_policy_to_onnx(self, path: str, filename: str = "policy.onnx"):
#       """
#       重写外部 rsl_rl 的 ONNX 导出方法，将 Normalizer 封装进去
#       """
#       policy = self.alg.policy
#       obs_normalizer = None

#       # 提取 Empirical Normalizer 并转至 CPU
#       if hasattr(self, "empirical_normalization") and self.empirical_normalization:
#           obs_normalizer = self.obs_normalizer
#           obs_normalizer.to("cpu")
#           obs_normalizer.eval()

#       wrapper = _OnnxPolicyWrapper(policy, obs_normalizer)
#       wrapper.to("cpu")
#       wrapper.eval()

#       num_obs = policy.actor[0].in_features
#       dummy_input = torch.zeros(1, num_obs)
#       os.makedirs(path, exist_ok=True)
      
#       onnx_path = os.path.join(path, filename)
      
#       # 导出 ONNX 
#       torch.onnx.export(
#           wrapper,
#           dummy_input,
#           onnx_path,
#           export_params=True,
#           opset_version=18,
#           input_names=["obs"],
#           output_names=["actions"],
#           dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
#       )

#       # 导出后，将 policy 和 normalizer 移回训练设备
#       policy.to(self.device)
#       if obs_normalizer is not None:
#           obs_normalizer.to(self.device)

#   def save(self, path: str, infos=None):
#       """
#       重写 save 方法，使其在保存 `.pt` 权重的同步导出 ONNX 并注入 Metadata
#       """
#       # 1. 调用外部 rsl_rl 的标准保存逻辑
#       super().save(path, infos)
      
#       # 2. 确定 ONNX 导出路径
#       policy_path = path.split("model")[0]
#       filename = "policy.onnx"
#       onnx_path = os.path.join(policy_path, filename)
      
#       # 3. 导出包含 Normalizer 的 ONNX
#       self._export_policy_to_onnx(policy_path, filename)
      
#       # 4. 获取并注入 Mjlab 专属 Metadata（部署强依赖）
#       run_name = wandb.run.name if (hasattr(self, "logger_type") and self.logger_type == "wandb" and wandb.run) else "local"
#       metadata = get_base_metadata(self.env.unwrapped, run_name)
#       attach_metadata_to_onnx(onnx_path, metadata)
      
#       # 5. 上传至 WandB (可选)
#       if hasattr(self, "logger_type") and self.logger_type == "wandb":
#           wandb.save(onnx_path, base_path=os.path.dirname(policy_path))




class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
