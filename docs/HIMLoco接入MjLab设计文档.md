# RI-4438 HIMLoco 接入 MjLab 设计说明

## 1. 结论

参考 AMP 工程的分层方式接入 HIM，但不要复制 AMP 的旧版 runner：

```text
task registry
  -> env cfg：定义 actor history 和 critic privileged observation
  -> rl cfg：选择 HIMActorModel + HIMPPO
  -> HIMRolloutStorage：保存 obs_t、obs_(t+1)、done
  -> HIMPPO：PPO loss + velocity estimation loss + swap loss
  -> task runner：checkpoint、ONNX、metadata
```

当前 RI-4438/Go2 设计中：

```text
num_obs_step = 47
history_size = 6
num_actor_obs = 47 * 6 = 282
```

47 维包含 2 维相位，不应删除 phase。

---

## 2. AMP 接入方式中需要借鉴的部分

AMP 工程 `/home/sunteng/projects/AMP_mjlab/src/tasks/amp_loco` 完成了以下闭环：

| 层 | AMP 做法 | HIM 对应做法 |
|---|---|---|
| 环境 | `actor`、`critic`、`amp` 三组观测 | `actor` history、`critic` privileged observation |
| RL 配置 | 扩展 runner cfg，指定 `AMPPPO` | 扩展 model/algorithm cfg，指定 `HIMActorModel`、`HIMPPO` |
| Storage | PPO rollout + AMP replay | PPO rollout + next observation |
| 辅助学习 | discriminator | estimator encoder/target/prototype |
| Runner | 建立 AMP 数据流并保存模型 | 保持标准 rollout，处理终止 transition 和导出 |
| Registry | 显式绑定 `AMPOnPolicyRunner` | 显式绑定 `HIMOnPolicyRunner` |

不能直接复制 AMP runner，原因是版本不同：

| 工程 | MjLab | RSL-RL |
|---|---:|---:|
| AMP 参考工程 | 1.2.0，并覆盖 ObservationManager 补丁 | 2.3.1 旧接口 |
| 当前工程 | 1.6.0 | 5.4.2 TensorDict 接口 |

AMP runner 中的 `_migrate_train_cfg()`、`_unpack_obs()`、`eval()`、完整 learn/log 循环，都是旧接口适配，
不应进入当前 HIM 实现。

---

## 3. 47 维单帧观测

依据以下现有代码：

- `src/tasks/locomotion/go2_ppo/go_ppo_env_cfg.py`
- `src/tasks/locomotion/go2_ppo/mdp/observations.py`
- `src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py`
- `src/tasks/locomotion/ri_4438_ppo/mdp/observations.py`

Go2 和 RI-4438 的 actor term 顺序一致：

```python
actor_terms = {
  "base_ang_vel": ...,
  "projected_gravity": ...,
  "command": ...,
  "phase": ...,
  "joint_pos": ...,
  "joint_vel": ...,
  "actions": ...,
}
```

对应切片为：

| 切片 | 宽度 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | base angular velocity |
| `[3:6]` | 3 | projected gravity |
| `[6:9]` | 3 | command `(vx, vy, wz)` |
| `[9:11]` | 2 | phase `(sin, cos)` |
| `[11:23]` | 12 | joint position relative to default |
| `[23:35]` | 12 | joint velocity |
| `[35:47]` | 12 | previous action |

总维度：

```text
3 + 3 + 3 + 2 + 12 + 12 + 12 = 47
```

因此旧配置中的：

```python
num_obs_one_step = 3 + 3 + 3 + num_action * 3  # 45
```

应按实际任务语义改为：

```python
num_obs_one_step = 3 + 3 + 3 + 2 + num_action * 3  # 47
num_obs = num_obs_one_step * 6                     # 282
```

---

## 4. Phase 的代码语义

现有 `phase()` 返回 2 维周期编码：

```python
global_phase = (env.episode_length_buf * env.step_dt) % period / period
phase[:, 0] = torch.sin(global_phase * 2.0 * torch.pi)
phase[:, 1] = torch.cos(global_phase * 2.0 * torch.pi)
```

当前配置：

```text
period = 0.6 s
```

静止命令下 phase 被置零：

```python
stand_mask = torch.linalg.norm(command, dim=1) < 0.1
phase = torch.where(stand_mask.unsqueeze(1), 0, phase)
```

它与 `feet_gait` reward 使用同一个 `0.6 s` 周期。RI-4438 机器人配置又把四腿 offset 设置为：

```text
[0.0, 0.5, 0.5, 0.0]
```

对应对角腿同相的 trot 引导：

```text
FL/RR 同相
FR/RL 同相
两组相差半周期
```

所以 phase 不是无关的附加量，而是 actor 的步态时钟输入。HIM history 中每一帧都应包含该帧 phase。

---

## 5. Actor history 布局

目标布局应与 HIM 推理逻辑一致：

```text
history_t = [o_t, o_(t-1), ..., o_(t-5)]
history_t.shape = [B, 282]
```

但 MjLab 1.6 有两个事实：

1. 多 term history 直接 flatten 时默认是 term-major。
2. `CircularBuffer.buffer` 的时间顺序是 oldest -> newest。

因此不能直接使用：

```python
history_length=6
flatten_history_dim=True
```

否则结果类似：

```text
[ang_vel 六帧, gravity 六帧, command 六帧, phase 六帧, ...]
```

当前 `HIMActorModel` 却把前 47 维当成当前完整帧，语义错误。

### 推荐实现

环境保留三维 history：

```python
ObservationGroupCfg(
  terms=actor_terms,
  concatenate_terms=True,
  enable_corruption=not play,
  history_length=6,
  flatten_history_dim=False,
)
```

MjLab 输出：

```text
obs["actor"].shape = [B, 6, 47]
time order = [oldest, ..., newest]
```

HIM actor 内统一转换：

```python
def _flatten_actor_history(value: torch.Tensor) -> torch.Tensor:
  if value.ndim == 2:
    return value
  if value.ndim != 3:
    raise ValueError(f"Expected [B,D] or [B,H,D], got {value.shape}")

  # [oldest, ..., newest] -> [newest, ..., oldest]
  return value.flip(dims=(1,)).flatten(start_dim=1)
```

`HIMActorModel.get_observation_tensor()`：

```python
def get_observation_tensor(self, obs: TensorDict) -> torch.Tensor:
  groups = [_flatten_actor_history(obs[group]) for group in self.obs_groups]
  return torch.cat(groups, dim=-1)
```

`_get_obs_dim()` 同时支持 2D 和 3D：

```python
def _get_obs_dim(self, obs: TensorDict) -> int:
  obs_dim = 0
  for group in self.obs_groups:
    value = obs[group]
    if value.ndim == 2:
      obs_dim += value.shape[-1]
    elif value.ndim == 3:
      obs_dim += value.shape[-2] * value.shape[-1]
    else:
      raise ValueError(f"Unsupported HIM observation shape: {value.shape}")
  return obs_dim
```

转换后：

```text
obs_history.shape = [B, 282]
obs_history[:, 0:47] = 当前帧
```

---

## 6. Critic observation 和 estimator target

现有 PPO critic 在 actor terms 后添加：

```text
base_lin_vel(3)
height_scan(187，rough only)
foot_height(4)
foot_air_time(4)
foot_contact(4)
foot_contact_forces(12)
```

保持现有 actor term 顺序时：

```text
critic[:, 0:47]  = actor 当前单帧
critic[:, 47:50] = true base linear velocity
```

维度为：

```text
rough critic = 47 + 3 + 187 + 4 + 4 + 4 + 12 = 261
flat critic  = 47 + 3 + 4 + 4 + 4 + 12       = 74
```

HIM estimator 的速度监督必须改为：

```python
velocity_target = next_critic_obs[:, 47:50]
```

### Target network 输入不能继续使用单一 `(3, 50)` 切片

原始 HIM 的原则是：

```text
从 one-step observation 中去掉 command，再加入 true base linear velocity。
```

在当前 Go2/RI-4438 顺序中，command 位于 `[6:9]`，不是最前面的 `[0:3]`。

如果直接把旧配置改成：

```python
target_slice = (3, 50)
```

实际效果会是：

```text
错误地删除 base_ang_vel
错误地保留 command
```

正确的 47 维 target input 应为：

```python
target_input = torch.cat(
  (
    next_critic_obs[:, 0:6],   # base_ang_vel + projected_gravity
    next_critic_obs[:, 9:47],  # phase + joint_pos + joint_vel + actions
    next_critic_obs[:, 47:50], # true base_lin_vel
  ),
  dim=-1,
)
```

维度：

```text
6 + 38 + 3 = 47
```

推荐把 algorithm config 从单一 slice 扩展为：

```python
estimator_velocity_slice = (47, 50)
estimator_target_slices = ((0, 6), (9, 47), (47, 50))
```

然后由 HIMPPO 或 HIMEstimator 按多个切片拼接。

另一个可选方案是把 command 重排到 one-step 最前面，然后继续使用 `(3, 50)`；但这会改变现有
Go2/RI-4438 observation 顺序。既然目标是参考现有 Go2 设计，推荐保留当前顺序并显式拼接 target。

---

## 7. 环境配置实现

第一阶段直接从已工作的 RI-4438 PPO rough/flat 环境派生，只修改 observation history：

```python
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg

from src.tasks.locomotion.ri_4438_ppo.config.env_cfgs import (
  ri_4438_flat_env_cfg,
  ri_4438_rough_env_cfg,
)

HIM_HISTORY_SIZE = 6
HIM_ONE_STEP_OBS = 47

HIM_ACTOR_TERM_ORDER = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
)


def _configure_him_observations(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
) -> ManagerBasedRlEnvCfg:
  source_actor = cfg.observations["actor"].terms
  source_critic = cfg.observations["critic"].terms

  actor_terms = {
    name: deepcopy(source_actor[name]) for name in HIM_ACTOR_TERM_ORDER
  }
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    enable_corruption=not play,
    history_length=HIM_HISTORY_SIZE,
    flatten_history_dim=False,
  )

  critic_terms = {
    name: deepcopy(source_critic[name]) for name in HIM_ACTOR_TERM_ORDER
  }
  critic_terms["base_lin_vel"] = deepcopy(source_critic["base_lin_vel"])
  for name, term in source_critic.items():
    if name not in critic_terms:
      critic_terms[name] = deepcopy(term)

  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
    flatten_history_dim=True,
  )
  return cfg


def ri_4438_him_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _configure_him_observations(ri_4438_rough_env_cfg(play), play)


def ri_4438_him_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return _configure_him_observations(ri_4438_flat_env_cfg(play), play)
```

这样保留：

- 现有 phase observation。
- `feet_gait` 的 `period=0.6`。
- RI-4438 四腿 offset `[0.0, 0.5, 0.5, 0.0]`。
- 已验证的 robot、action、reward、event、terrain 和 play 配置。
- rough/flat 不同的 critic extras。

当前 `ri_4438_him/mdp` 不需要在第一阶段重复接线。先复用 PPO 环境把 HIM 算法跑通，之后再拆分
HIM 专属 reward 或 curriculum。

---

## 8. RL 配置实现

建议增加 HIM 专属 dataclass：

```python
from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class RslRlHIMActorCfg(RslRlModelCfg):
  class_name: str = "rsl_rl.models.him_actor_model:HIMActorModel"
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True
  distribution_cfg: dict[str, Any] | None = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )
  num_one_step_obs: int = 47
  history_size: int = 6
  estimator_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "enc_hidden_dims": (128, 64, 16),
      "tar_hidden_dims": (128, 64),
      "activation": "elu",
      "learning_rate": 1.0e-3,
      "max_grad_norm": 10.0,
      "num_prototype": 32,
      "temperature": 3.0,
      "sinkhorn_eps": 0.05,
      "sinkhorn_iters": 3,
    }
  )


@dataclass
class RslRlHIMPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "rsl_rl.algorithms.him_ppo:HIMPPO"
  estimator_learning_rate: float | None = 1.0e-3
  estimator_max_grad_norm: float | None = 10.0
  estimator_obs_groups: tuple[str, ...] = ("critic",)
  estimator_velocity_slice: tuple[int, int] = (47, 50)
  estimator_target_slices: tuple[tuple[int, int], ...] = (
    (0, 6),
    (9, 47),
    (47, 50),
  )


@dataclass
class RslRlHIMRunnerCfg(RslRlOnPolicyRunnerCfg):
  class_name: str = "HIMOnPolicyRunner"
  actor: RslRlHIMActorCfg = field(default_factory=RslRlHIMActorCfg)
  critic: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=None,
    )
  )
  algorithm: RslRlHIMPPOAlgorithmCfg = field(
    default_factory=RslRlHIMPPOAlgorithmCfg
  )


def ri_4438_him_ppo_runner_cfg() -> RslRlHIMRunnerCfg:
  return RslRlHIMRunnerCfg(
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    logger="tensorboard",
    experiment_name="ri_4438_him",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=20000,
  )
```

这里使用完全限定类名，避免当前 `rsl_rl.algorithms.__init__` 未正确 import `HIMPPO` 导致短名解析失败。

---

## 9. Storage 实现

当前 `HIMRolloutStorage` 把 `next_observations` 传给基础 `RolloutStorage.Batch`，但基础 Batch 没有
这个参数，第一次 update 会报错。

需要定义 HIM Batch：

```python
class Batch(RolloutStorage.Batch):
  def __init__(
    self,
    *args,
    next_observations: TensorDict | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.next_observations = next_observations
```

generator 同时返回 dones：

```python
dones = self.dones.flatten(0, 1)

batch = HIMRolloutStorage.Batch(
  observations=observations[batch_idx],
  actions=actions[batch_idx],
  values=values[batch_idx],
  advantages=advantages[batch_idx],
  returns=returns[batch_idx],
  old_actions_log_prob=old_actions_log_prob[batch_idx],
  old_distribution_params=tuple(
    p[batch_idx] for p in old_distribution_params
  ),
  dones=dones[batch_idx],
  next_observations=next_observations[batch_idx],
)
```

---

## 10. HIMPPO estimator 更新

### 10.1 拼接非连续 target

`HIMPPO` 应从 critic tensor 显式构造 target：

```python
def _get_estimator_targets(
  self,
  next_observations: TensorDict,
) -> tuple[torch.Tensor, torch.Tensor]:
  critic_obs = torch.cat(
    [next_observations[name] for name in self.estimator_obs_groups],
    dim=-1,
  )
  velocity = critic_obs[:, 47:50]
  target_input = torch.cat(
    (
      critic_obs[:, 0:6],
      critic_obs[:, 9:47],
      critic_obs[:, 47:50],
    ),
    dim=-1,
  )
  return velocity, target_input
```

更通用的实现应读取 `estimator_velocity_slice` 和 `estimator_target_slices`，不要在算法中写死数字。

对应地，`HIMEstimator.update()` 最好直接接收已经构造好的：

```text
velocity_target [N, 3]
target_input     [N, 47]
```

而不是让 estimator 猜测 critic observation 的布局。

### 10.2 屏蔽自动 reset 的 transition

MjLab 默认 `auto_reset=True`，done 时返回 reset 后新 episode 的 observation。不能把它当成旧 episode
的 next observation 训练 estimator。

第一版直接屏蔽 done transition：

```python
valid = ~batch.dones.squeeze(-1).bool()

if torch.any(valid):
  history = self.actor.get_observation_tensor(batch.observations)[valid]
  history = self.actor.obs_normalizer(history)
  velocity_target, target_input = self._get_estimator_targets(
    batch.next_observations
  )
  estimation_loss, swap_loss = self.estimator.update(
    history,
    velocity_target[valid],
    target_input[valid],
    gradient_reducer=(
      self._reduce_estimator_gradients if self.is_multi_gpu else None
    ),
  )
```

后续若要完全复现原始 HIMLoco，再让环境在 reset 前通过 extras 返回 terminal critic observation。

---

## 11. Task 注册

`src/tasks/locomotion/ri_4438_him/config/__init__.py`：

```python
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.locomotion.ri_4438_him.rl import HIMOnPolicyRunner

from .env_cfgs import (
  ri_4438_him_flat_env_cfg,
  ri_4438_him_rough_env_cfg,
)
from .rl_cfg import ri_4438_him_ppo_runner_cfg


register_mjlab_task(
  task_id="Ri-4438-HIM-Rough",
  env_cfg=ri_4438_him_rough_env_cfg(),
  play_env_cfg=ri_4438_him_rough_env_cfg(play=True),
  rl_cfg=ri_4438_him_ppo_runner_cfg(),
  runner_cls=HIMOnPolicyRunner,
)

register_mjlab_task(
  task_id="Ri-4438-HIM-Flat",
  env_cfg=ri_4438_him_flat_env_cfg(),
  play_env_cfg=ri_4438_him_flat_env_cfg(play=True),
  rl_cfg=ri_4438_him_ppo_runner_cfg(),
  runner_cls=HIMOnPolicyRunner,
)
```

不要复用现有 `Ri-4438-Rough` / `Ri-4438-Flat` ID，否则会与 PPO task 冲突。

---

## 12. 当前代码必须修正的项目

| 优先级 | 文件 | 问题 | 正确结果 |
|---|---|---|---|
| P0 | `ri_4438_him/config/*.py` | 都是空文件 | env、RL cfg、registry 接通 |
| P0 | `rsl_rl/algorithms/__init__.py` | `HIMPPO` 只写进 `__all__`，未 import | 可解析 HIMPPO，或配置使用限定名 |
| P0 | `him_rollout_storage.py` | 基础 Batch 不接受 next observation | HIM Batch 包含 next obs 和 dones |
| P0 | `him_actor_model.py` | 只接受 2D，且默认前 47 维为当前帧 | 接受 `[B,6,47]` 并转为 current-first 282 维 |
| P0 | `him_ppo.py` | target 只支持连续 slice | 支持 `((0,6),(9,47),(47,50))` |
| P0 | `him_ppo.py` | done transition 使用 reset 后 next obs | estimator 屏蔽 done transition |
| P1 | `ri_4438_him_config.py` | 仍按 45 维计算 | 改为 47 和 282 |
| P1 | `rsl_rl/runners/__init__.py` | 重复版权头和 import block | 合并成单一导出块 |
| P1 | HIM runner load | 旧 checkpoint 不会完整迁移 estimator | 增加 HIM 专属 state_dict 映射 |

---

## 13. 验收标准

### Observation

```text
actor.shape == [num_envs, 6, 47]
critic.shape == [num_envs, 261]  # rough
critic.shape == [num_envs, 74]   # flat
```

必须验证：

```text
actor current frame [0:3]   = base_ang_vel
actor current frame [3:6]   = projected_gravity
actor current frame [6:9]   = command
actor current frame [9:11]  = phase
critic [47:50]               = true base_lin_vel
```

### History

```text
flattened_history.shape == [num_envs, 282]
flattened_history[:, 0:47] == newest/current frame
flattened_history[:, 47:94] == previous frame
```

### Estimator

```text
velocity_target.shape == [N, 3]
target_input.shape == [N, 47]
done transition 不参与 estimator loss
```

### Algorithm

一次 update 后：

```text
loss keys = value, surrogate, entropy, estimation, swap
所有 loss 为有限值
actor、critic、estimator 参数均按预期更新
storage.step == 0
```

### ONNX

```text
input:  obs_history [batch, 282]
output: actions     [batch, 12]
```

PyTorch 和 ONNXRuntime 对同一组 current-first history 的 action 输出应在数值容差内一致。

---

## 14. 实施顺序

```text
1. 修复 HIMPPO 类解析
2. 修复 HIM Batch 和 dones
3. actor 支持 [B,6,47]，验证 history 顺序
4. estimator 改为 47/50 切片和非连续 target 拼接
5. 屏蔽 done transition
6. 填写 HIM env/rl cfg 和 registry
7. Flat 32 env、2~5 iterations smoke test
8. 验证 checkpoint 和 282 维 ONNX
9. 再进行 Rough 大规模训练
```

这条顺序能把 shape、时序、target 和 task 注册问题分开定位。
