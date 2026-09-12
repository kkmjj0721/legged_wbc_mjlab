# RI-4438 HIM 二阶段蒸馏实施方案

> 本文档只做代码和算法分析，并给出后续实施方案；本次没有修改训练、环境、算法或部署代码。
>
> 分析基于当前工作区代码（2026-09-12）。工作区本身存在一些用户已有的未提交修改，因此文中“当前值”指本次读取到的实际源码和 registry 配置。

## 1. 先给结论

当前工程已经具备一条比较完整的 **HIM-PPO 训练链路**，但还没有具备一条可以直接启动的 **RI-4438 二阶段蒸馏链路**。

当前已有的能力可以概括为：

```text
RI-4438 MjLab 环境
  ├─ actor：本体观测的 6 帧历史（每帧 47 维）
  ├─ critic：actor 当前帧 + privileged information
  └─ reward / termination / terrain curriculum / domain randomization
          ↓
HIMActorModel
  ├─ history encoder：估计 base linear velocity + 16 维 latent
  └─ actor MLP：输入当前帧 + 速度估计 + latent，输出 12 个关节动作
          ↓
HIMPPO
  ├─ PPO surrogate loss
  ├─ value loss / entropy
  ├─ velocity estimation loss
  └─ HIM swap loss
          ↓
ONNX：输入 282 维 history，输出 12 维 action
```

仓库中另外存在通用的 `rsl_rl.algorithms.Distillation`，但是它目前只是一个通用的 MLP teacher-student 行为克隆器：

- 没有 RI-4438 蒸馏 task 的 registry 注册；
- 默认 student/teacher 都是 `MLPModel`，而当前 RI-4438 actor 是 `[B, 6, 47]` 的时间历史；
- 当前普通 `DistillationRunner` 使用标准 `OnPolicyRunner`，不处理 HIM runner 的 `auto_reset=False` 终止观测流程；
- 它只保存逐时刻的 `obs` 和 teacher action，不保存 HIM estimator 所需的 `next_observations`；
- 它没有蒸馏后的 PPO 恢复训练阶段。

因此，推荐把“二阶段蒸馏”定义为：

```text
阶段 1：用 privileged critic observation 训练强 teacher PPO
阶段 2：teacher 在线标注，训练只使用 onboard history 的 student
可选阶段 3：student 初始化后用 PPO 做短程恢复/鲁棒性微调
```

如果已经有一个收敛的 `Ri-4438-HIM-*` checkpoint，也可以把它直接当作阶段 2 的 teacher，蒸馏到更小的 student。这条路线属于“策略压缩”，不是严格意义上的“privileged teacher → deployable student”，文中会单独说明。

---

## 2. 当前仓库的模块关系

### 2.1 任务注册和启动

启动脚本的调用关系是：

```text
scripts/train.py / scripts/play.py
  └─ import src.tasks
      └─ 自动导入各任务的 config/__init__.py
          └─ register_mjlab_task(...)
              ├─ env_cfg
              ├─ play_env_cfg
              ├─ rl_cfg
              └─ runner_cls
```

当前已经注册的 HIM task 是：

- `Ri-4438-HIM-Rough`
- `Ri-4438-HIM-Flat`

两者都绑定 `src.tasks.locomotion.ri_4438_him.rl.runner:HIMOnPolicyRunner`。

训练时：

1. `scripts/train.py` 从 registry 得到环境和 agent 配置。
2. 创建 `ManagerBasedRlEnv`，再包成 `RslRlVecEnvWrapper`。
3. `HIMOnPolicyRunner` 创建 `HIMPPO`。
4. `HIMPPO` 创建 `HIMActorModel`、critic `MLPModel` 和 `HIMRolloutStorage`。
5. 每轮先采集 rollout，再进行 PPO 和 estimator 更新。
6. 保存 checkpoint 时，RI-4438 专用 runner 额外导出 `policy.onnx`。

### 2.2 文件职责

| 文件 | 实际职责 | 对蒸馏的意义 |
|---|---|---|
| `src/config/ri_4438/ri_4438_config.py` | 机器人控制、命令、reset、随机化和基础 PPO 参数 | 是参数参考源，但很多字段没有真正传到最终 cfg |
| `src/config/ri_4438/ri_4438_him_config.py` | HIM 的 47/282 维度、estimator、HIM PPO 和 runner 参数 | 定义当前 HIM actor 的结构 |
| `src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py` | 基础环境工厂：观测、动作、命令、事件、奖励、终止、地形 | 定义 teacher/student 共享的物理分布 |
| `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py` | 注入 RI-4438 机器人并生成 Rough/Flat 变体 | 最终环境配置入口 |
| `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py` | 选择 HIM actor、HIMPPO 和 HIM runner | 当前只有 RL task，没有蒸馏 task |
| `rsl_rl/models/him_actor_model.py` | 历史重排、归一化、estimator、actor head | 当前最适合直接作为 history student |
| `rsl_rl/modules/him_estimator.py` | 速度监督和 swap/prototype 表征学习 | 蒸馏阶段若继续用 HIM，需要额外处理 |
| `rsl_rl/algorithms/him_ppo.py` | PPO + HIM estimator 联合训练 | 当前阶段 1 的现成算法 |
| `rsl_rl/algorithms/distillation.py` | 通用 teacher-student 行为克隆 | 可复用思路，但不能无改动覆盖当前 history/HIM 需求 |
| `rsl_rl/runners/distillation_runner.py` | 检查 teacher 是否加载后调用普通 runner | 目前没有处理 HIM 终止观测和外部 teacher 路径 |
| `src/tasks/locomotion/ri_4438_him/rl/runner.py` | checkpoint 保存和 282 维 ONNX 导出 | HIM 部署输出的事实标准 |
| `deploy/include/*` | 当前 RI-4438 PPO 的 sim2sim/FSM/ONNX 运行时 | 当前输入协议仍是 47 维，不兼容 HIM 282 维 history |

---

## 3. 当前环境和 MDP 详细分析

### 3.1 仿真频率、动作频率和 episode

当前环境的核心时间参数是：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| MuJoCo timestep | `0.005 s` | 物理仿真 200 Hz |
| `decimation` | `4` | 每 4 个物理步执行一次 policy action |
| `step_dt` | `0.020 s` | policy/environment 50 Hz |
| episode length | `20 s` | 训练 episode 最长约 1000 个 policy step |
| HIM rollout | `100 steps/env` | 每轮每个环境约 2 s 数据 |
| HIM history | `6 frames` | 时间覆盖约 0.12 s |

蒸馏时 teacher 和 student 必须使用完全相同的 `timestep`、`decimation`、动作缩放和 action update 时刻。不要在 200 Hz 的每个 MuJoCo 子步上重复调用 teacher 或 student；应保持 50 Hz 的 policy action 频率。

### 3.2 一个非常重要的并行环境问题

配置层有多个看起来像并行数的值：

- `Ri4438HimCfg.env.num_envs = 16384`；
- 基础 `Ri4438PiperCfg.env.num_envs = 4096`；
- 但 `make_velocity_env_cfg()` 最终创建 `SceneCfg(num_envs=1)`。

所以从 registry 加载的默认 HIM 环境实际是 `scene.num_envs = 1`。训练脚本可以通过 CLI 覆盖，例如：

```bash
PYTHONPATH=.:src .venv/bin/python scripts/train.py \
  Ri-4438-HIM-Rough \
  --env.scene.num-envs 4096
```

二阶段蒸馏需要大量状态-动作对。如果不显式覆盖 `scene.num_envs`，即使算法配置看起来是大规模训练，实际数据吞吐仍然只有单环境，且 teacher/student 的行为分布估计会很差。

建议的资源递进：

```text
shape smoke test：32 env
小规模收敛检查：256~512 env
正式 teacher：2048~4096 env 起步
正式 distillation：2048~4096 env 起步
```

`16384` 不是必须值。Rough critic 单帧约 244 维，若同时存 actor history、critic、动作、日志和 MuJoCo 状态，显存/内存压力会很快上升。应以实际显存为准逐级增加。

### 3.3 动作：输出是归一化 joint target，不是力矩

当前动作项是 `JointPositionActionCfg`：

```text
policy raw action a ∈ R^12
processed action ≈ a * action_scale + default_joint_pos
q_target         = processed action（再叠加执行器/编码器相关影响）
```

当前：

- hip/thigh/calf 都有独立的正则表达式缩放；
- `action_scale = 0.15381525329142837`；
- `hip_reduction = 1.0`，因此当前三类关节基础缩放相同；
- 输出维度是 12；
- 执行器是 builtin position actuator，带 PD、力矩上限和随机延迟。

蒸馏 loss 应优先在 policy 的归一化动作空间上计算：

```text
L_action = ||a_student - a_teacher||
```

这样不会因为不同关节的弧度缩放而让某几个关节主导 loss。若后续发现 hip/thigh/calf 的物理误差不平衡，再额外评估在实际关节目标空间计算加权 loss：

```text
Δq = action_scale ⊙ a
L_q = ||Δq_student - Δq_teacher||
```

但第一版建议只使用归一化动作 loss，便于和现有 `Distillation` 的 MSE 行为保持一致。

### 3.4 命令和 curriculum

`twist` 命令返回 3 维：

```text
[lin_vel_x, lin_vel_y, ang_vel_z]
```

当前命令相关逻辑：

- 每个环境 4–8 s 重采样一次；
- 5% 概率进入 standing；
- 10% 概率进入 forward-only（与 standing 独立抽样）；
- heading command 当前关闭；
- phase 周期为 `0.6 s`；
- standing 命令下 phase 置零；
- Rough/Flat 训练仍有 command curriculum。

训练 teacher 和 student 时，必须明确使用哪一个命令分布：

1. teacher 训练阶段的 curriculum 分布；
2. distillation 阶段当前 teacher 实际运行到的分布；
3. 部署时手柄/上层控制器会产生的命令分布。

蒸馏不应只在 teacher 最容易的早期命令范围上采样。建议在蒸馏阶段覆盖 standing、慢走、侧向、转向和较大速度，并按照命令区间分桶统计动作误差。

### 3.5 Domain randomization 和 teacher 的“privileged”程度

当前启用的 startup/interval 随机化包括：

- `push_robot`：5–6 s 间隔施加根速度扰动；
- 脚部摩擦：`[0.3, 1.6]`，四脚共享；
- encoder bias：`[-0.015, 0.015]`；
- base COM：x/y/z 各 `[-0.05, 0.05] m`；
- PD gain：`[0.9, 1.1]` 缩放；
- effort limit：`[0.9, 1.1]` 缩放；
- pseudo inertia/质量相关随机化：由 `link_mass_range=[0.8, 1.2]` 转换得到。

但是，当前 critic observation 并没有直接把所有随机物理参数作为显式输入。它主要能看到真实 base linear velocity、接触/足端/地形信息等。因此严格来说，当前 critic 是“asymmetric privileged observation”，但不是“完整随机参数 oracle”。

如果阶段 1 的目标是训练一个真正的 privileged teacher，应在后续环境设计中为 teacher-only group 提供明确的质量、摩擦、COM、PD/effort 随机化参数；这属于未来代码工作，本次不执行。

### 3.6 奖励和终止

主要奖励结构：

- 线速度跟踪：权重 `1.5`；
- 角速度跟踪：权重 `0.5`；
- 姿态、body angular velocity、角动量惩罚；
- variable posture、hip deviation、stand still；
- 足端 gait、clearance、slip、soft landing；
- joint acceleration、action rate、action acceleration、torque、joint limit；
- termination penalty：`-100`。

终止条件：

- time out；
- body orientation 超过约 `70°`；
- `base_collision` 与 terrain 接触力超过 `10 N`。

teacher、student 评估时应使用相同 reward/termination。蒸馏 loss 本身不使用 reward，但 reward 是判断 student 是否真正复现 teacher 运动能力的主指标。

### 3.7 Rough 和 Flat 的 critic 维度

当前 actor 的单帧固定是 47 维：

| 切片 | 维度 | 内容 |
|---|---:|---|
| `[0:3]` | 3 | base angular velocity |
| `[3:6]` | 3 | projected gravity |
| `[6:9]` | 3 | command |
| `[9:11]` | 2 | sin/cos phase |
| `[11:23]` | 12 | relative joint position |
| `[23:35]` | 12 | joint velocity |
| `[35:47]` | 12 | previous action |

当前实际 critic term：

```text
Rough = actor 47
      + base_lin_vel 3
      + base_com 3
      + foot_contact 4
      + height_scan 187
      = 244 维

Flat  = actor 47
      + base_lin_vel 3
      + foot_height 4
      + foot_air_time 4
      + foot_contact 4
      + foot_contact_forces 12
      = 74 维
```

这里要特别注意：旧的 HIM 设计文档中曾出现 Rough critic `261` 维的说明，那是把 Flat 的若干足端项和 Rough 项混在一起的旧描述。以当前 `config/env_cfgs.py` 的 term 列表为准，当前 Rough 是 244 维、Flat 是 74 维。

HIM estimator 并不使用完整 critic 训练 target，而是通过切片取出：

```text
velocity_target = critic_next[:, 47:50]
target_input    = concat(
                    critic_next[:, 0:6],
                    critic_next[:, 9:47],
                    critic_next[:, 47:50],
                  )
                = 6 + 38 + 3 = 47 维
```

蒸馏 teacher 可以使用完整 244/74 维 critic；student 不应看到其中的 privileged 项。

---

## 4. 当前 HIM 算法的实现和数学含义

### 4.1 HIM actor 的输入输出

环境输出：

```text
obs["actor"].shape = [B, 6, 47]
时间顺序         = [oldest, ..., newest]
```

`HIMActorModel.get_observation_tensor()` 会：

1. 在最后一维拼接各 term；
2. 沿时间轴翻转；
3. flatten 成 `[B, 282]`。

所以进入 estimator 的布局是：

```text
history_t = [o_t, o_(t-1), o_(t-2), o_(t-3), o_(t-4), o_(t-5)]
```

模型先做 `EmpiricalNormalization`，再由 encoder 输出：

```text
encoder(history) -> [pred_base_lin_vel(3), latent_raw(16)]
latent = L2Normalize(latent_raw)
```

actor head 的输入维度为：

```text
current one-step obs 47
+ predicted velocity   3
+ latent              16
= 66 维
```

然后经过 `(512, 256, 128)` MLP 和 12 维 Gaussian distribution，训练时采样动作，推理/导出时取 deterministic output。

### 4.2 HIM estimator 的两个目标

`HIMEstimator` 有两个网络：

```text
encoder：6×47 history -> 3 维速度 + 16 维 source latent
target ：47 维 privileged one-step target -> 16 维 target latent
```

训练 loss：

```text
L_estimator = L_velocity + L_swap

L_velocity = MSE(pred_velocity, true_base_linear_velocity)
```

swap loss 的过程是：

1. source latent 和 target latent 分别与 32 个 prototype 做相似度；
2. 用 Sinkhorn 得到平衡 prototype assignment；
3. 用双向交叉熵让 source/target assignment 一致。

当前参数：

| 参数 | 值 |
|---|---:|
| encoder hidden | `(128, 64, 16)` |
| target hidden | `(128, 64)` |
| latent dim | `16` |
| prototypes | `32` |
| temperature | `3.0` |
| Sinkhorn epsilon | `0.05` |
| Sinkhorn iterations | `3` |
| estimator lr | `1e-3` |
| estimator max grad norm | `10` |

### 4.3 HIMPPO 每次 update 的顺序

当前 `HIMPPO.update()` 对每个 minibatch 大致执行：

```text
1. 用当前 actor/critic 前向
2. 计算 action log-prob、value、entropy
3. 从 next critic observation 构造 estimator target
4. 更新 HIM estimator（独立 optimizer）
5. 计算 PPO ratio、clipped surrogate、value loss、entropy loss
6. 只用 PPO 参数反向传播并更新
7. rollout 结束后更新 actor/critic normalization
```

estimator 参数会从 PPO optimizer 中排除，避免同一参数被两个 optimizer 重复更新。

PPO 关键值：

```text
num_learning_epochs = 5
num_mini_batches    = 4
learning_rate       = 1e-3
schedule            = adaptive
desired_kl          = 0.01
gamma               = 0.99
lambda              = 0.95
clip_param          = 0.2
entropy_coef        = 0.01
```

### 4.4 终止 transition 的特殊处理

当前 HIM runner 设定 `auto_reset=False`，这是为了在 done 时先保存真实 terminal physics observation，再手动 reset：

```text
env.step
  -> 保存 terminal observation
  -> HIMPPO.process_env_step
  -> 对 done env 调用 reset(env_ids=...)
  -> 继续下一步 rollout
```

`HIMRolloutStorage` 额外保存 `next_observations` 和 `next_observations_valid`，供 estimator 使用。

这个流程对 HIMPPO 是必要的，但普通 `DistillationRunner` 直接继承标准 `OnPolicyRunner`，没有这个手动 reset 逻辑。蒸馏任务若继续使用普通 runner，必须显式采用 `auto_reset=True`，或未来增加专用 distillation runner；不能把当前 HIM runner 和普通 DistillationRunner 混用而不验证。

### 4.5 checkpoint 和 ONNX

HIM PPO checkpoint 至少包含：

```text
actor_state_dict
critic_state_dict
optimizer_state_dict
estimator_state_dict
estimator_optimizer_state_dict
```

RI-4438 HIM runner 导出的是一个合并图：

```text
input  obs_history [B, 282]，顺序 current -> oldest
output actions     [B, 12]
```

ONNX 图内包含：

- observation normalizer；
- estimator encoder；
- deterministic actor MLP；
- Gaussian distribution 的 deterministic output。

---

## 5. 仓库现有 Distillation 的能力和限制

### 5.1 现有算法做什么

`rsl_rl/algorithms/distillation.py` 的基本流程是：

```text
rollout：
  student(obs) -> student action
  teacher(obs) -> teacher action
  环境执行 student action
  storage 保存 obs、student action、teacher action、done

update：
  student(batch.obs) -> predicted action
  behavior_loss = MSE(student_action, teacher_action)
  反向传播 student
```

它的优点是实现简单，而且 rollout 是由 student action 驱动的，因此天然带有一点 DAgger 风格：teacher 在 student 实际访问到的状态上持续提供标签。

### 5.2 与当前 RI-4438 HIM 的不匹配

#### 不匹配 1：`MLPModel` 只接受二维 observation

`MLPModel._get_obs_dim()` 明确要求每个输入 group 是 `[B, D]`。当前 actor group 是 `[B, 6, 47]`，所以不能直接把当前 actor 配置交给普通 MLP student。

可选方案：

1. 使用 `HIMActorModel` 作为 student，让它自己处理 `[B, 6, 47]`；
2. 在环境或模型前增加 history flatten adapter，把 `[B, 6, 47]` 转成 `[B, 282]`；
3. 只给 student 当前单帧 47 维，换成普通 MLP，但会失去历史信息，rough terrain 鲁棒性通常会明显下降。

推荐优先级：`1 > 2 > 3`。

#### 不匹配 2：当前环境没有 `student`/`teacher` observation set

通用 Distillation 支持通过配置让 student 和 teacher 读取不同 observation group，例如：

```text
student -> actor
teacher -> critic
```

但当前 HIM runner 配置只有：

```text
actor -> actor
critic -> critic
```

未来蒸馏 task 必须明确增加以下语义映射：

```text
student group = onboard actor history
teacher group = privileged critic/current frame
```

#### 不匹配 3：teacher checkpoint 不是现成的 HIM teacher 格式

普通 Distillation 在加载时支持：

- `teacher_state_dict`；或
- PPO checkpoint 的 `actor_state_dict`。

所以阶段 1 teacher 最好直接保存为普通 PPO actor checkpoint，或者保证 teacher 网络结构和 checkpoint 的 actor 结构完全一致。

如果把当前 HIM PPO checkpoint 当 teacher：

- teacher 必须使用 `HIMActorModel` 结构；
- 必须同时加载 estimator 相关参数；
- 它仍然读取 history actor，而不是 privileged critic；
- 这属于 HIM policy compression，不是 privileged teacher distillation。

#### 不匹配 4：普通 Distillation 不保存 `next_observations`

如果 student 是一个已经初始化好的纯 MLP，这没有问题；但如果 student 继续训练 HIM estimator，现有 Distillation storage 没有足够数据计算：

```text
velocity_target(t+1)
target_obs(t+1)
swap loss target
```

因此有三条路线：

| 路线 | estimator 处理 | 复杂度 | 建议 |
|---|---|---:|---|
| A | 从已训练 HIM checkpoint 初始化并先冻结 estimator | 低 | 第一版首选 |
| B | 保留 HIM estimator，并扩展 distillation storage/algorithm 做辅助 loss | 高 | 追求最高性能时使用 |
| C | student 不使用 HIM estimator，只做 282→12 的 history MLP | 中 | 结构简单，但可能损失 HIM 表征能力 |

#### 不匹配 5：`gradient_length` 有尾部梯度丢失风险

Distillation 的 update 以逐时间步 batch 迭代，并按 `gradient_length` 累积 loss。若：

```text
num_learning_epochs * num_steps_per_env
```

不能被 `gradient_length` 整除，最后一段 loss 会累积但不会反向传播。

例如当前 HIM 风格 `num_steps_per_env=100`，若使用默认 `gradient_length=15`：

```text
100 % 15 = 10
```

最后 10 个时间步的 loss 会被丢弃。纯 feed-forward student 建议第一版设置：

```text
gradient_length = 1
```

或者使用 `5/10/20/25/50/100` 等能够整除 rollout 的值。

#### 不匹配 6：没有蒸馏后的 RL 恢复阶段

现有 Distillation 只优化行为克隆 loss：

```text
MSE(student_action, teacher_action)
```

它不会直接优化 tracking reward、摔倒率或能耗。teacher label 有误差、student 自己进入 teacher 训练分布之外的状态时，单纯 BC 容易累积误差。因此建议蒸馏后做一小段 PPO recovery，而不是把 BC 作为最终唯一训练目标。

---

## 6. 推荐的二阶段方案

下面是针对当前代码结构最稳妥的主方案。

### 6.1 阶段 1：训练 privileged teacher

#### 目标

训练一个只用于仿真标注的强 teacher：

```text
输入：critic privileged observation（Rough 244 维 / Flat 74 维）
输出：12 维归一化关节动作
算法：标准 PPO
部署：不部署
```

teacher 不需要 HIM history；它的目的是真正利用 terrain/contact/true velocity 等信息提供动作上限。

#### 推荐 observation 设计

最小可行版本直接复用当前 `critic` group：

```text
teacher = critic
```

如果后续要做严格 privileged teacher，建议再增加 teacher-only 的随机化参数 group：

```text
teacher_input = [critic_obs, mass, friction, com_offset, kp_scale, kd_scale, motor_strength, ...]
```

student 绝不能读取这些 teacher-only 项。

#### teacher 网络

起步可使用：

```text
MLP hidden dims = (512, 256, 128)
activation       = elu
output dim       = 12
distribution     = GaussianDistribution
init std         = 1.0
obs normalization= True
```

teacher 的 actor 可以用确定性的 action mean 作为蒸馏标签。不要把 rollout 中的随机采样 action 直接作为唯一标签，否则 student 会学习到 teacher exploration noise。

#### teacher 训练顺序

建议分三步：

```text
T0. Flat，小规模，检查 tracking 和动作方向
T1. Rough，小规模，检查地形/接触终止
T2. Rough，大规模，加入完整随机化和 curriculum，得到最终 teacher
```

第一版不建议同时让 teacher curriculum、terrain curriculum、随机化全部从最强难度开始。先确认 Flat/Rough 的 tracking、fall rate、action saturation，再增加难度。

#### teacher 的验收条件

至少记录：

| 指标 | 要求 |
|---|---|
| linear velocity tracking | 按命令区间统计，不只看全局均值 |
| angular velocity tracking | stand/turn 单独统计 |
| episode return | 多随机种子稳定 |
| fall/illegal contact rate | 不随难度突然发散 |
| action saturation | 12 个关节分别统计 |
| action rate / torque | 不出现异常高频抖动 |
| terrain level | Rough curriculum 能正常升降 |

teacher 没有达到稳定行为前，不要进入阶段 2。蒸馏只能复制 teacher，不能补救 teacher 本身的退化行为。

### 6.2 阶段 2：teacher 在线标注，训练 deployable student

#### 推荐 student 结构

对当前 RI-4438，首选：

```text
student 输入：actor history [B, 6, 47]
student 内部：HIMActorModel 或等价 history encoder
student 输出：12 维 action
```

原因是当前任务存在 terrain、摩擦、COM、接触等隐藏状态，单帧 47 维不能完整恢复它们。保留 6 帧历史是当前 HIM 设计的核心，不应为了复用普通 MLP Distillation 而直接删除。

student/teacher 的逻辑关系：

```text
同一时刻 t、同一随机化环境
  actor_history_t ─────> student ─────> a_student
  critic_obs_t   ──────> teacher ─────> μ_teacher
                                      ↓
                        L_action(a_student, μ_teacher)
  环境真正执行 a_student
```

这会让 student 实际访问到的状态分布参与标注，比离线只回放 teacher trajectory 更能暴露 student 的分布偏移。

#### 方案 A：HIM student（推荐）

适合已有收敛 HIM checkpoint 的情况。

1. 使用已有 HIM actor/estimator 初始化 student。
2. 前期冻结 estimator，只蒸馏 actor head 或整个 actor 的 action output。
3. student 稳定后再解冻 estimator。
4. 若要解冻 estimator，必须补充 `next_observations` 和 velocity/swap auxiliary loss。当前 `HIMActorModel.get_latent()` 在 estimator 前向外包了 `torch.no_grad()`，所以 action BC 本身不会给 estimator 反向梯度；没有 auxiliary loss 时，estimator 实际上应保持冻结，而不是假设它会随 action loss 学好。

推荐冻结/解冻阶段：

```text
S0：0~20% distillation updates，estimator freeze，action BC
S1：20~80%，保持较小 student lr，视验证结果决定是否解冻 actor head
S2：80% 以后，若有 next-observation auxiliary target，再解冻 estimator
```

如果当前只有通用 Distillation、没有 HIM 专用 estimator loss，第一版应停留在 S0/S1，不要假设 estimator 会被正确训练。

#### 方案 B：history flatten + 普通 MLP student

如果目标是尽量复用现有通用 Distillation，可以把 student 设计成：

```text
输入：[o_t, o_(t-1), ..., o_(t-5)]，282 维
网络：MLP 512/256/128 或更小
输出：12 维 action
```

但必须有一个明确的 history flatten 适配层，使 `MLPModel` 收到 `[B, 282]`，而不是原始 `[B, 6, 47]`。当前仓库没有这个适配层，不能仅靠修改 `obs_groups` 名称解决。

这个方案的优点是：

- 不需要 estimator 的 next observation target；
- 可以直接使用通用 action MSE/Huber distillation；
- 导出为普通 MLP ONNX 较简单。

缺点是：

- 会丢失 HIM 的 velocity/latent 结构先验；
- 需要验证同等参数量下的 Rough terrain 鲁棒性；
- 282 维输入仍然不等于“单帧轻量部署”。

#### 方案 C：单帧 47 维 student

只推荐作为 Flat 或极简部署接口的 baseline：

```text
student input = current actor frame [B, 47]
student       = ordinary MLP
```

它最容易接入当前 `deploy/include/fsm.py`，因为当前部署代码就是 47 维 observation。但是它没有时间记忆，面对速度估计、摩擦变化、推搡和地形接触等部分可观测问题，通常会明显弱于 282/history student。不能默认它能达到 HIM 的 Rough 性能。

### 6.3 阶段 2 的数据采集方式

#### 每个训练 step 保存什么

至少需要：

```text
obs_actor_history_t   [B, 6, 47] 或 flatten 后 [B, 282]
obs_teacher_critic_t  [B, 244]（Rough）或 [B, 74]（Flat）
teacher_mean_action_t [B, 12]
student_action_t      [B, 12]
done_t                [B, 1]
command_t             [B, 3]
```

建议额外保存用于诊断的：

```text
reward_t
terrain level / id
domain randomization parameters
base velocity
student/teacher action saturation
```

随机化参数可以只进入日志或离线数据，不进入 student 输入。

#### teacher label 的选取

推荐：

```text
teacher label = deterministic action mean μ_teacher
```

不推荐：

```text
teacher label = teacher rollout 中采样出的随机 action
```

如果要保留 teacher 的不确定性，可以后续做 distribution distillation：匹配 mean 和 log-std，或使用 Gaussian KL；第一版不要同时引入太多目标。

#### reset 和 history warm-up

当前 HIM history 有 6 帧。蒸馏时必须确认 reset 后历史 buffer 的填充语义：

- 是用首帧重复填充；
- 还是有零填充；
- 还是需要积累 6 个真实观测。

部署端也必须完全复现同一种语义。若训练初始 history 和部署初始 history 不一致，开机前几步的动作可能严重偏移。

若能控制数据筛选，建议每次 reset 后丢弃前 `history_size-1=5` 个 policy step，再把有效 history 用于 loss；如果部署设计明确采用首帧重复，则训练也必须采用首帧重复，而不是训练时丢弃、部署时硬启动。

#### auto reset

普通 DistillationRunner 依赖标准 OnPolicyRunner，而当前 HIM train env 的 `auto_reset=False` 是为 HIM terminal target 服务的。第一版纯行为蒸馏若不保存 next observation，应采用：

```text
distillation env：auto_reset=True
```

这样 done 后标准 runner 返回新 episode observation，student/teacher 都能继续 rollout。若要保留真实 terminal observation 或训练 HIM estimator，则应实现和 `HIMOnPolicyRunner` 等价的专用 distillation runner。

### 6.4 阶段 2 的 loss 设计

#### 第一版基线：动作行为克隆

```text
L = L_action

L_action = mean( Huber(a_student, μ_teacher) )
```

如果必须和现有 Distillation 完全一致，可使用 MSE：

```text
L_action = mean( (a_student - μ_teacher)^2 )
```

建议顺序：先 MSE 跑通 shape 和行为，再比较 Huber 对异常 teacher label 的稳定性。

#### 第二版：按动作/命令加权

可以按动作误差、命令区间或接触状态加权：

```text
L_action = w_cmd * w_contact * w_joint * loss(a_student, μ_teacher)
```

但要避免一开始使用过于复杂的权重。建议先保证总 loss 和每个 joint 的 loss 都可解释，再引入：

- 转向样本权重；
- 接触切换样本权重；
- 大速度/大扰动样本权重；
- hip/thigh/calf 物理缩放权重。

#### 第三版：分布蒸馏

如果 teacher 和 student 都保留 Gaussian distribution，可以增加：

```text
L_dist = KL(π_teacher || π_student)
```

组合为：

```text
L = λ_action * L_action + λ_dist * L_dist
```

第一版不建议直接使用 KL 作为唯一目标，因为 teacher 的 exploration std 不一定代表部署时最优动作不确定性。默认应以 deterministic mean 行为为主。

#### HIM student 的辅助 loss

若 student 使用 HIM estimator，并且存储了正确的 next observation，才增加：

```text
L_him = λ_v * MSE(v_est, v_true)
      + λ_swap * L_swap

L_total = L_action + L_him
```

`λ_v`、`λ_swap` 不应一开始就大于 action loss。建议从较小权重开始，并观察：

- velocity prediction error 是否下降；
- latent 是否 collapse；
- action loss 是否反而升高；
- Rough fall rate 是否改善。

### 6.5 推荐的起始超参数

这些是第一轮实验的起点，不是对所有硬件都保证最优的固定值。

| 项目 | 建议起点 |
|---|---:|
| distillation envs | 512 smoke，2048~4096 正式 |
| steps/env | 32 或 64；若保持 HIM 风格可用 100 |
| epochs | 1~3 |
| mini-batch | 通用 Distillation 是按 timestep generator，先不要盲目增大 epochs |
| student lr | `3e-4` 起，稳定后可到 `1e-3` |
| optimizer | Adam |
| loss | MSE baseline，Huber 对比 |
| gradient clip | `1.0~10.0` |
| gradient_length | `1`，或选择能整除 rollout 的值 |
| teacher action | deterministic mean |
| student action clip | 与环境一致，通常 `[-1, 1]` |
| history | 6×47，current-first 进入 HIM |
| estimator | 第一版 freeze；有 next obs 后再解冻 |

训练轮数应以 transition 数而不是 iteration 数为主要尺度：

```text
有效样本数 = num_envs × steps_per_env × distillation_iterations
```

建议至少准备百万级有效 transition，再判断动作 MSE 是否进入平台期。只跑几十个单环境 iteration 只能验证接口，不能证明 student 学到了鲁棒行为。

---

## 7. “现有 HIM checkpoint 直接作为 teacher”的替代路线

如果阶段 1 已经有一个效果很好的 HIM PPO checkpoint，可以跳过重新训练 privileged MLP teacher：

```text
已有 HIM actor（输入 282 history）
          ↓ teacher label
更小的 history MLP / RNN / 单帧 MLP student
```

这条路线的含义是压缩/迁移当前 deployable policy，而不是从 privileged information 蒸馏出 deployable policy。

### 7.1 适用场景

- ONNX 推理延迟不够；
- 希望减少 actor hidden dims；
- 希望把 HIM actor 转换成普通 history MLP；
- 已经接受当前 HIM teacher 的观测限制。

### 7.2 需要注意的 checkpoint 结构

当前 HIM checkpoint 的 `actor_state_dict` 同时包含 actor MLP、normalizer 和 estimator。若 teacher 使用 `HIMActorModel`，必须按完整结构加载；不能只拿 `mlp.*` 而忽略 estimator 输出，因为 actor head 的输入包含 velocity + latent。

### 7.3 推荐的压缩实验顺序

```text
C0：282 history → 282 history，缩小 hidden dims
C1：282 history → 282 history，去掉 HIM estimator，比较 MLP
C2：282 history → 47 current frame，只作为弱 baseline
```

每一步都要与原 HIM teacher 做同一批 command、terrain、randomization 的闭环评估，不能只看 offline action MSE。

---

## 8. 部署接口必须重新对齐

### 8.1 当前 deployment 是 RI-4438 PPO，不是 HIM

`deploy/task/ri_4438_ppo/config/sim2sim.yaml` 当前写的是：

```text
observation_dim = 47
action_dim       = 12
phase_period     = 0.6
```

`deploy/include/fsm.py` 每次调用 policy 时也只构造当前一帧 observation：

```text
[base_ang_vel,
 projected_gravity,
 command,
 phase,
 joint_pos_rel,
 joint_vel,
 last_action]
= 47 维
```

而当前 HIM runner 导出：

```text
input = 282 维 current-first history
```

所以 HIM ONNX 不能直接替换当前 deployment 目录里的 47 维 PPO ONNX。必须二选一：

1. student 最终就是 47 维单帧 MLP，保持当前部署协议；
2. 部署端新增 6 帧 history buffer，并严格按照训练顺序拼成 282 维。

对于 Student-HIM，必须选择第 2 条。

### 8.2 部署对齐清单

至少核对：

- joint name 顺序；
- actor term 顺序；
- history 时间方向（训练 manager oldest→newest，HIM 输入 current→oldest）；
- reset 后 history 初始化；
- command 单位和限幅；
- phase 时间基准和 standing 清零逻辑；
- `default_joint`；
- action scale；
- action clipping；
- observation normalization 参数；
- ONNX 输入/输出 shape；
- 50 Hz action 更新和 200 Hz MuJoCo 子步。

还要注意当前 sim2sim YAML 的 action scale 与训练环境中的统一 `0.153815...` 并不完全相同。蒸馏结果验证时必须以训练 task 的 action manager 和最终部署协议为准，不能只因为 shape 一样就认为动作尺度一致。

### 8.3 ONNX 数值验收

对同一组 history：

```text
PyTorch student deterministic output
vs.
ONNXRuntime output
```

应满足固定容差内一致。至少测试：

- 零命令站立；
- 前进；
- 侧向；
- 转向；
- 随机 history；
- history reset 后前 10 个 policy step。

---

## 9. 未来需要做的代码工作（本次不执行）

下面只列实现边界，不在本次修改代码。

### 9.1 必须新增或调整的训练配置

需要一个独立的 distillation task/runner，不能复用 `Ri-4438-HIM-Rough` 的 HIM PPO registry ID。建议使用独立 ID，例如：

```text
Ri-4438-Distill-Rough
Ri-4438-Distill-Flat
```

概念上应配置：

```text
obs_groups = {
  "student": ("actor",),
  "teacher": ("critic",),
}
```

如果 student 是 HIMActorModel，还要选择能接受 3D history 的 student class；如果 student 是普通 MLP，必须先提供 `[B,282]` flatten adapter。

### 9.2 必须明确 teacher checkpoint 加载方式

当前 `scripts/train.py` 的 resume 逻辑主要按 experiment log directory 搜索 checkpoint。正式蒸馏时要明确：

```text
teacher_checkpoint = 哪个文件
student 是否从已有 HIM checkpoint 初始化
是否加载 optimizer
是否继承 iteration
```

阶段 2 不应加载 teacher 的 optimizer 继续训练；teacher 应始终 eval/frozen。student optimizer 应重新初始化，除非明确做断点续训。

### 9.3 若使用 HIM student，必须决定 estimator 策略

三种实现边界：

```text
freeze：现有 Distillation 基本可复用
auxiliary：扩展 storage 保存 next obs，并加入 velocity/swap loss
replace：student 改成 history MLP，彻底不使用 estimator
```

不应在文档或配置中把“使用 HIMActorModel”误认为“estimator 会自动得到正确的蒸馏监督”。现有通用 Distillation 并没有这条数据流。

### 9.4 必须决定 auto-reset 策略

```text
纯 action BC、无 next obs：可采用 auto_reset=True
保留 terminal obs/HIM auxiliary：使用 HIM 类专用 runner
```

### 9.5 必须新增部署 history 支持（如果 student 输入 282）

需要在部署端维护：

```text
deque/环形 buffer，长度 6
每个 policy step 写入一帧 47 维
按 current -> oldest 读出
done/reset/stand/passive 时按训练约定清空或填充
```

当前 `OnnxPolicy` 的 shape 检查本身可以接受不同 observation_dim，但 `LocomotionFSM.observation()` 和 YAML 的 term 协议仍是 47 维，不能只改一个数字。

---

## 10. 分阶段实施和验收门槛

### Gate 0：静态 shape 和配置检查

必须确认：

```text
actor raw shape        = [B, 6, 47]
HIM flattened history  = [B, 282]
Rough critic           = [B, 244]
Flat critic            = [B, 74]
teacher output         = [B, 12]
student output         = [B, 12]
```

并检查：

- actor 当前帧 `[0:47]` 是 newest/current；
- `[0:3]` 是 angular velocity；
- `[3:6]` 是 projected gravity；
- `[6:9]` 是 command；
- `[9:11]` 是 phase；
- `[11:23]` 是 joint position；
- `[23:35]` 是 joint velocity；
- `[35:47]` 是 last action；
- critic `[47:50]` 是 true base linear velocity。

### Gate 1：teacher 闭环质量

teacher 必须在 Flat/Rough、不同命令和随机化下闭环跑通。不要只检查 network forward 或 offline action 输出。

### Gate 2：蒸馏接口

至少验证：

- teacher checkpoint 能正确加载；
- teacher 参数没有梯度和更新；
- student 梯度有限；
- action loss 单调下降到平台；
- `gradient_length` 没有尾部 loss 丢失；
- done/reset 后 history 没有跨 episode 污染；
- student 的 environment action 真正被执行，而不是误执行 teacher action。

### Gate 3：行为等价性

建议同时比较 teacher/student：

| 类别 | 指标 |
|---|---|
| 动作 | mean absolute action error、per-joint RMSE、最大误差 |
| 任务 | linear/angular tracking |
| 稳定性 | fall rate、illegal contact rate |
| 动力学 | torque、action rate、joint limit |
| 场景 | flat、stairs、rough、obstacle |
| 扰动 | push 前/后恢复时间 |
| 命令 | standing、forward、lateral、turn、mixed |

student 的 action MSE 很低但 tracking/fall 明显变差时，不能继续只调 loss；应检查 history 顺序、动作尺度、phase 和 command 对齐。

### Gate 4：部署等价性

最后才做：

```text
PyTorch student
  -> JIT/ONNX
  -> ONNXRuntime
  -> MuJoCo sim2sim
  -> 真实机器人 shadow / low-speed
```

每一步都要用同一组初始姿态、命令、randomization 和 history 初始化。

---

## 11. 常见错误和排查优先级

### 错误 1：把当前 HIM actor 误认为 privileged teacher

当前 HIM actor 只读取 actor history；privileged 信息主要在 critic。若目标是严格 teacher-student，阶段 1 需要独立的 privileged actor。

### 错误 2：直接把 `[B,6,47]` 交给普通 MLPModel

`MLPModel` 要求二维输入。应使用 HIMActorModel 或明确 flatten adapter。

### 错误 3：history 顺序反了

MjLab history 是 oldest→newest；HIM 内部会 flip 为 current→oldest。部署端必须给 ONNX current-first，不能把环境原始顺序直接喂给导出模型。

### 错误 4：蒸馏时执行了 teacher action

行为蒸馏需要 teacher 提供 label，但环境应执行 student action，否则收集到的是 teacher 分布，student 自己的分布偏移没有暴露。

### 错误 5：把 stochastic teacher sample 当作标签

第一版应使用 teacher deterministic mean；否则 student 会拟合 exploration noise。

### 错误 6：忘记当前默认环境实际只有 1 个 env

必须通过最终 cfg/registry 或 CLI 检查 `scene.num_envs`，不能只看 `Ri4438HimCfg.env.num_envs`。

### 错误 7：普通 runner 配合 `auto_reset=False`

普通 DistillationRunner 没有 HIM runner 的手动 reset/terminal observation 逻辑。纯 BC 用 `auto_reset=True`；需要 terminal/estimator target 时使用专用 runner。

### 错误 8：只看 offline MSE

控制策略是闭环系统。最终必须看 student 自己动作驱动环境时的 tracking、摔倒率和扰动恢复。

### 错误 9：只替换 ONNX 文件，不更新部署 history

282 维 HIM ONNX 不能直接替换 47 维 PPO runtime。必须一起更新 observation buffer、reset 语义和 shape 配置。

---

## 12. 最终推荐的执行顺序

按当前仓库成熟度，建议使用下面顺序：

```text
1. 先保留当前 HIM-PPO 代码不动，完成 Gate 0 的 shape/顺序检查。
2. 用普通 PPO + critic privileged observation 训练阶段 1 teacher。
3. teacher 在 Flat/Rough 和随机化场景下完成闭环验收。
4. 新增独立 distillation task，明确 student=actor、teacher=critic。
5. 第一版 student 使用已有 HIM actor 初始化，冻结 estimator，只做 deterministic action BC。
6. distillation env 使用 auto_reset=True，gradient_length=1。
7. 按 command/terrain/contact 分桶检查 action loss 和闭环 reward。
8. 若 BC student 的闭环性能达到 teacher，再做短程 PPO recovery。
9. 若需要进一步提升 Rough 鲁棒性，再扩展 next_observations，加入 HIM velocity/swap auxiliary loss。
10. 最后决定部署目标：282 维 Student-HIM，或 47 维单帧压缩 baseline。
11. 对最终 student 重新导出 ONNX，并对齐 deploy runtime 的 history、phase、action scale 和 reset。
```

一句话总结：

> 对当前 RI-4438，最稳妥的二阶段蒸馏不是把现有 `Distillation` 类直接套上去，而是先把“privileged teacher 的输入、history student 的输入、terminal/reset 语义和 ONNX 输入协议”四件事对齐，再以 deterministic action behavior cloning 为第一版目标，最后用闭环 PPO recovery 处理纯蒸馏带来的分布偏移。
