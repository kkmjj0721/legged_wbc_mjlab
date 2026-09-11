# RI-4438 PPO 配置协同使用说明

## 1. 文档目的

本文档说明 RI-4438 PPO 速度跟踪任务中，下列配置如何协同工作：

- [`src/config/ri_4438/ri_4438_config.py`](../src/config/ri_4438/ri_4438_config.py)
- [`src/assets/robots/ri_4438/ri_4438_constants.py`](../src/assets/robots/ri_4438/ri_4438_constants.py)
- [`src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py`](../src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py)
- [`src/tasks/locomotion/ri_4438_ppo/config/env_cfgs.py`](../src/tasks/locomotion/ri_4438_ppo/config/env_cfgs.py)
- [`src/tasks/locomotion/ri_4438_ppo/config/rl_cfg.py`](../src/tasks/locomotion/ri_4438_ppo/config/rl_cfg.py)
- [`src/tasks/locomotion/ri_4438_ppo/config/__init__.py`](../src/tasks/locomotion/ri_4438_ppo/config/__init__.py)

本文档基于 2026-09-10 的当前代码。其中的“已接通”表示修改 `ri_4438_config.py` 对应字段后，重启 Python 进程可以改变最终配置；“未接通”表示下游仍在使用硬编码值或根本没有消费该字段。

## 2. 总体结论

当前工程不是把两个配置文件直接合并，而是分层构造最终任务：

```text
ri_4438_config.py
├─ Ri4438PiperCfg
│  ├─→ ri_4438_constants.py       机器人初始姿态和执行器参数
│  └─→ ri_4438_ppo_env_cfg.py     动作、命令、reset 等环境参数
└─ Ri4438CfgPPO
   └─→ config/rl_cfg.py               PPO 网络与训练器参数

ri_4438_ppo_env_cfg.py
└─→ config/env_cfgs.py
   ├─ 注入 RI-4438 机器人
   ├─ 补入 base、足端 site、碰撞 geom 名称
   ├─ 添加接触传感器
   └─ 生成 Rough / Flat 变体

config/__init__.py
├─→ Ri-4438-Rough
└─→ Ri-4438-Flat
   └─→ scripts/train.py / scripts/play.py
```

因此，`ri_4438_ppo_env_cfg.py` 只是“基础环境工厂”，它返回的配置还不完整。真正用于注册和训练的是 `env_cfgs.py` 中的 `ri_4438_rough_env_cfg()` 和 `ri_4438_flat_env_cfg()`。

## 3. 各文件的职责

### 3.1 `ri_4438_config.py`

这是项目自定义的集中参数源，包含两类配置：

- `Ri4438PiperCfg`：机器人、命令、控制、初始状态、reset 和域随机化参数。
- `Ri4438CfgPPO`：Actor/Critic、PPO 算法和 runner 参数。

`Ri4438PiperCfg` 继承了 `BaseConfig`。实例化时，`BaseConfig` 会把内嵌 class 转成实例，并深拷贝 `dict`/`list`/`set`，所以可以使用：

```python
ri_4438_cfg = Ri4438PiperCfg()
ri_4438_cfg.control.action_scale
ri_4438_cfg.comaman.ranges.lin_vel_x
```

注意：`comaman` 是当前代码里的实际字段名，应该是 `command` 或 `commands` 的拼写错误。现在上下游使用了同一错误拼写，因此仍能运行。

### 3.2 `ri_4438_constants.py`

该文件将 `Ri4438PiperCfg` 转成 MjLab 的机器人 `EntityCfg`：

- 加载 `ri_4438.xml`；
- 根据 `stiffness`/`damping`/`effort_limit` 创建 hip、thigh、calf 位置执行器；
- 设置执行器延迟参数；
- 把 `init_state` 转成机器人默认根位置和默认关节位置；
- 设置碰撞规则和软关节限位系数。

`get_ri_4438_robot_cfg()` 是机器人配置的最终出口。

### 3.3 `ri_4438_ppo_env_cfg.py`

`make_velocity_env_cfg()` 创建 Manager-based RL 环境的通用部分：

- terrain scan 传感器；
- actor/critic 观测；
- 关节位置动作；
- `twist` 速度命令；
- reset、push 和 domain randomization 事件；
- 奖励、终止条件和 curriculum；
- MuJoCo 步长、decimation、episode 时长和 viewer。

这一层使用了名为 `robot` 的占位引用，但没有将具体机器人放进 `scene.entities`。足端 site、脚部 geom 和 base body 的名称也暂时为空。

### 3.4 `config/env_cfgs.py`

该文件负责将基础环境“具体化”为 RI-4438 环境：

```python
cfg = make_velocity_env_cfg()
cfg.scene.entities = {"robot": get_ri_4438_robot_cfg()}
```

它将基础配置中的占位名称补齐为：

- base body：`base_link`；
- foot sites：`FR`, `FL`, `RR`, `RL`；
- foot collision geoms：`FR_foot_collision`, `FL_foot_collision`, `RR_foot_collision`, `RL_foot_collision`；
- illegal-contact geom：`base_collision`。

这些名称均能在 `ri_4438.xml` 中找到，与当前 MJCF 匹配。

### 3.5 `config/rl_cfg.py`

该文件将 PPO 参数转成 MjLab/RSL-RL 的：

- `RslRlModelCfg`；
- `RslRlPpoAlgorithmCfg`；
- `RslRlOnPolicyRunnerCfg`。

当前只有 Actor 和 Critic 的 `hidden_dims` 真正从 `Ri4438CfgPPO` 读取，其他字段大部分还在 `rl_cfg.py` 中硬编码。

### 3.6 `config/__init__.py`

该文件在 import 时调用 `register_mjlab_task()`，注册：

- `Ri-4438-Rough`；
- `Ri-4438-Flat`。

`scripts/train.py` 和 `scripts/play.py` 先 import `src.tasks` 触发注册，然后根据 task ID 从 registry 中取出最终环境、RL 配置和 runner。

## 4. `Ri4438PiperCfg` 参数映射表

### 4.1 已接通参数

| 参数源 | 当前消费位置 | 实际作用 |
|---|---|---|
| `control.stiffness` | `ri_4438_constants.py` | hip/thigh/calf 执行器刚度 |
| `control.damping` | `ri_4438_constants.py` | hip/thigh/calf 执行器阻尼 |
| `control.effort_limit` | `ri_4438_constants.py` | 执行器力矩限制 |
| `control.armature` | `ri_4438_constants.py` | 执行器 armature |
| `control.delay_min_lag` | `ri_4438_constants.py` | 动作延迟下限 |
| `control.delay_max_lag` | `ri_4438_constants.py` | 动作延迟上限 |
| `control.delay_hold_prob` | `ri_4438_constants.py` | 延迟保持概率 |
| `control.delay_update_period` | `ri_4438_constants.py` | 延迟更新周期 |
| `control.action_scale` | `ri_4438_ppo_env_cfg.py` | policy 动作到目标关节角的缩放 |
| `control.hip_reduction` | `ri_4438_ppo_env_cfg.py` | 单独缩小/放大 hip 动作幅度 |
| `init_state.pos` | `ri_4438_constants.py` | 机器人默认根位置 |
| `init_state.default_joint` | `ri_4438_constants.py` | 默认关节位置及 action offset |
| `comaman.resampling_time` | `ri_4438_ppo_env_cfg.py` | `twist` 命令重采样时间 |
| `comaman.ranges.lin_vel_x` | `ri_4438_ppo_env_cfg.py` | 初始前后速度范围 |
| `comaman.ranges.lin_vel_y` | `ri_4438_ppo_env_cfg.py` | 初始侧向速度范围 |
| `comaman.ranges.ang_vel_yaw` | `ri_4438_ppo_env_cfg.py` | 初始偏航角速度范围 |
| `reset.base_offset` | `ri_4438_ppo_env_cfg.py` | reset 时根位姿和速度偏移 |

命令 ranges 虽然已接通，但在训练模式中还会被 curriculum 覆盖，详见第 6 节。

### 4.2 未接通或只接了一部分的参数

| `ri_4438_config.py` 字段 | 当前运行时行为 | 结论 |
|---|---|---|
| `env.num_envs = 4096` | `SceneCfg(num_envs=1)` | 未接通 |
| `comaman.curriculum` | curriculum 始终被创建 | 未接通 |
| `comaman.max_curriculum` | 无消费者 | 未接通 |
| `comaman.num_commands` | 命令维度由 MjLab 命令类决定 | 未接通 |
| `comaman.heading_command` | 只控制 `ranges.heading` 是否为 `None`；命令项本身仍硬编码 `False` | 半接通，存在冲突 |
| `comaman.rel_standing_envs` | 环境中硬编码 `0.05` | 未接通，只是当前值相同 |
| `comaman.rel_forward_envs` | 环境中硬编码 `0.1` | 未接通，只是当前值相同 |
| `control.decimation` | `ManagerBasedRlEnvCfg` 中硬编码 `4` | 未接通，只是当前值相同 |
| `control.friction = 0.01` | 没有传入执行器、joint 或 geom | 未接通 |
| `reset.joint_offset` | reset event 中硬编码零范围 | 未接通 |
| `domain_rand.randomize_link_mass_range` | 没有质量随机化 event | 未接通 |
| `domain_rand.randomize_KpKd_factor` | `pd_gains` event 尚未实现 | 未接通 |
| `domain_rand.com_displacement_range` | COM event 使用硬编码 `(-0.05, 0.05)` | 未接通，只是当前值相同 |
| `noise` | 观测噪声在 env cfg 中直接硬编码 | 未接通 |
| `reward` | 奖励权重在 env cfg 中直接硬编码 | 未接通 |

## 5. 动作、默认姿态和 PD 执行器的关系

当前 action term 是 `JointPositionActionCfg`，且：

```python
use_default_offset = True
```

因此 policy 输出不是绝对关节角。关节位置目标大致为：

```text
processed_action = raw_action * action_scale + default_joint_pos
q_target         = processed_action - encoder_bias
```

其中：

- `default_joint_pos` 来自 `Ri4438PiperCfg.init_state.default_joint`；
- hip 的缩放是 `action_scale * hip_reduction`；
- thigh/calf 的缩放是 `action_scale`；
- `encoder_bias` 由 startup event 在 `[-0.015, 0.015]` 之间随机化；
- 最终位置目标交给 `ri_4438_constants.py` 中定义的 PD 执行器。

按当前配置，12 个关节的基础 action scale 均为 `0.15381525329142837 rad`，因为 `hip_reduction=1.0`。

`action_rate_l2` 的权重中另外写死了数值 `0.15381525329142837`。因此，如果只修改 `control.action_scale`，动作幅度会变，但该奖励的缩放补偿不会同步变化。

## 6. `twist` 速度命令的实际行为

### 6.1 命令内容

`UniformVelocityCommand` 的输出始终是 3 维：

```text
[lin_vel_x, lin_vel_y, ang_vel_z]
```

`comaman.num_commands = 4` 不会把输出变成 4 维。`heading` 是用来计算 `ang_vel_z` 的内部目标，不会作为第四个 command 维度返回。

### 6.2 重采样时间

`resampling_time = [4.0, 8.0]` 被转换为 `(4.0, 8.0)`。每个环境的命令计时器独立工作，每次重采样后都会在 4–8 s 内均匀抽取下一个持续时间。

当前物理步长和控制频率为：

```text
MuJoCo timestep = 0.005 s     物理频率 200 Hz
decimation      = 4
environment dt  = 0.020 s     policy/环境频率 50 Hz
```

因此一个命令大约持续 200–400 个 policy step。

### 6.3 `rel_standing_envs`

`rel_standing_envs=0.05` 表示：每次对一批环境重采样命令时，每个环境独立以 5% 概率被标记为 standing environment。

被标记后，该环境的命令会被强制置为：

```text
[0.0, 0.0, 0.0]
```

这不是“把采样范围缩小到 5%”，也不保证每批恰好有 5% 的环境站立；它是对每个环境做 Bernoulli 随机抽样。

站立命令还会影响其他 MDP 项：

- `phase()` 检测到命令范数小于 `0.1` 时，输出零相位；
- `foot_gait`、`foot_clearance`、`foot_slip`、`soft_landing` 等通过 command threshold 关闭或减少行走相关激励；
- `stand_still` 在小命令下约束机器人保持默认姿态。

当前 `ri_4438_config.py` 中已经有 `comaman.rel_standing_envs=0.05`，但 env cfg 仍直接写了 `0.05`。因此修改前者不会改变运行时行为。

### 6.4 `rel_forward_envs`

`rel_forward_envs=0.1` 表示每次重采样时，每个环境独立以 10% 概率进入 forward-only 模式。命令会被转换为：

```text
lin_vel_x = max(abs(sampled_lin_vel_x), 0.3)
lin_vel_y = 0.0
ang_vel_z = 0.0
```

standing 和 forward 标记是独立抽样的。如果同一环境同时抽中，最后 standing 逻辑会将命令清零。按当前概率，长期期望大致为：

- 站立命令：5%；
- 非站立的 forward-only 命令：`0.95 * 0.10 = 9.5%`；
- 非站立且非 forward-only 命令：`0.95 * 0.90 = 85.5%`。

这些是概率期望，不是每个时刻的精确环境数量。

### 6.5 heading 命令的当前冲突

当前 env cfg 中：

```python
heading_command = False
```

但 `ranges.heading` 是根据 `ri_4438_cfg.comaman.heading_command` 决定的。所以：

- 配置为 `False` 时：`heading_command=False` 且 `ranges.heading=None`，可以正常使用；
- 如果只把 `ri_4438_config.py` 改为 `True`：会形成 `heading_command=False` 但 `ranges.heading=(-3.14, 3.14)`。

MjLab 在构造 `UniformVelocityCommand` 时会拒绝第二种组合，并报出“`ranges.heading` 已设置，但 `heading_command=False`”的错误。所以当前不能只在集中配置中开启 heading。

### 6.6 命令 curriculum 对 ranges 的覆盖

训练配置中总是含有 `command_vel` curriculum：

| 阶段 | 触发条件 | `lin_vel_x` | `lin_vel_y` | `ang_vel_z` |
|---|---:|---:|---:|---:|
| 配置初始值 | curriculum 生效前 | `(-1.0, 1.0)` | `(-1.0, 1.0)` | `(-1.0, 1.0)` |
| stage 1 | `common_step_counter > 0` | `(-0.5, 1.0)` | `(-0.5, 0.5)` | `(-1.0, 1.0)` |
| stage 2 | `common_step_counter > 5000 * 24` | `(-1.0, 2.0)` | `(-1.0, 1.0)` | 保持 stage 1 的值 |

`num_steps_per_env=24`，所以 `5000 * 24 = 120000` 对应大约第 5000 次 PPO iteration 之后进入第二阶段。

这意味着：修改 `comaman.ranges` 可以改变环境初始值和 Rough play 值，但默认训练很快就会转入硬编码的 stage 1 范围。

## 7. 观测与传感器配合

### 7.1 Actor 观测

Actor 使用：

- `robot/imu_ang_vel`；
- projected gravity；
- `twist` command；
- 0.6 s 周期的 sin/cos phase；
- 相对默认姿态的 joint position；
- joint velocity；
- last action。

Actor 开启了 observation corruption，所以配置的均匀噪声会在训练模式生效。Play 模式会将 actor corruption 关闭。

### 7.2 Critic 观测

Critic 包含所有 Actor 项，并额外使用：

- `robot/imu_lin_vel`；
- terrain height scan（仅 Rough）；
- 四个足端 site 的高度；
- 足端腾空时间；
- 足端接触状态；
- 足端接触力。

Critic 不开启 corruption，属于 asymmetric actor-critic 中的 privileged observation。Flat 配置会删除 terrain scan 传感器和 `height_scan` critic term。

### 7.3 名称匹配

MJCF 中当前存在下列内置传感器：

- `imu_ang_vel`；
- `imu_lin_vel`；
- `root_angmom`。

`env_cfgs.py` 创建的 `feet_ground_contact` 覆盖四个 foot collision geom，其顺序是：

```text
FR, FL, RR, RL
```

`foot_gait.offset` 最终被设置为：

```text
[0.0, 0.5, 0.5, 0.0]
```

因此 FR/RL 同相，FL/RR 同相，表示对角小跑（trot）的相位关系。

## 8. Reset 和 domain randomization

### 8.1 Reset

`reset_base` 直接使用 `reset.base_offset`。当前所有 pose 偏移都为 0，`velocity_range` 为空，所以根状态会回到机器人默认状态。

`reset_robot_joints` 重置到默认关节姿态，但它的 position/velocity offset 目前在 env cfg 中写死为 0，没有使用 `reset.joint_offset`。

### 8.2 已启用的训练事件

- `push_robot`：每 5–6 s 通过设置根速度施加扰动；
- `foot_friction`：startup 时将四只脚的 geom friction 共享随机化到 `[0.3, 1.6]`；
- `encoder_bias`：startup 时对编码器添加 `[-0.015, 0.015]` 偏差；
- `base_com`：startup 时将 `base_link` 的 x/y/z COM 各自偏移 `[-0.05, 0.05]` m。

Play 模式会删除 `push_robot`，但仍保留 friction、encoder bias 和 COM 随机化。因此当前 Play 并不是完全确定性的环境。

还需要区分三种 friction：

- `control.friction=0.01`：当前未使用；
- MJCF 默认 joint `frictionloss=0.06`；
- 脚部 geom friction：由 collision config 设定基础值，然后在 startup 时随机化到 `[0.3, 1.6]`。

## 9. 奖励、终止与 curriculum

### 9.1 奖励的主要结构

当前奖励可以分为：

- 任务奖励：线速度跟踪、角速度跟踪；
- 姿态奖励/惩罚：躯干姿态、速度分段关节姿态、hip 偏移、站立保持；
- 动力学平滑项：关节加速度、action rate、action acceleration、身体角速度、角动量；
- 足端项：步态相位、抬脚高度、打滑、柔和落地；
- 安全项：关节限位、终止惩罚。

`pose` 奖励在 `env_cfgs.py` 中根据 standing/walking/running 分别设置了关节标准差。这些数值不在 `Ri4438PiperCfg.reward` 中，后者当前是空类。

### 9.2 终止

基础环境包含：

- `time_out`：20 s；
- `fell_over`：姿态倾角超过 70°。

RI-4438 专属配置又添加：

- `illegal_contact`：`base_collision` 与地形的接触力超过 10 N。

腿部、大腿和小腿碰到地面不会通过该 sensor 直接触发 illegal contact，因为 sensor 的 primary pattern 只匹配 `base_collision`。

### 9.3 地形 curriculum

Rough 训练使用 `terrain_levels`：

- 机器人走过地形长度的一半时升级；
- 运动距离过小时降级。

Flat 环境会删除 `terrain_levels`，但仍保留 `command_vel` 课程。

## 10. Rough、Flat、Train 和 Play 最终差异

| 组合 | 地形 | terrain scan | curriculum | push | 命令范围 |
|---|---|---|---|---|---|
| Rough Train | generator | 有 | `terrain_levels`, `command_vel` | 有 | 初始取 config，后被课程覆盖 |
| Rough Play | generator | 有 | 清空 | 删除 | 使用 config ranges |
| Flat Train | plane | 删除 | 仅 `command_vel` | 有 | 初始取 config，后被课程覆盖 |
| Flat Play | plane | 删除 | 清空 | 删除 | 强制为 x `(-0.5,1.0)`、y/z `(-0.5,0.5)` |

Play 还会：

- 把 episode 时长设为 `1e9` s；
- 关闭 actor observation corruption；
- 对 Rough terrain 关闭 curriculum，并改为 reset 时随机选地形；
- 保留 friction、encoder bias 和 COM 随机化。

## 11. PPO 配置的实际映射

### 11.1 已接通

| `Ri4438CfgPPO` 字段 | 最终配置 |
|---|---|
| `policy.actor_hidden_dims` | Actor `hidden_dims` |
| `policy.critic_hidden_dims` | Critic `hidden_dims` |

### 11.2 当前只是数值相同，但没有建立引用

下列字段在 `rl_cfg.py` 里又写了一遍常量：

- `policy.init_noise_std`；
- `policy.activation`；
- `policy.obs_normalization`；
- `algorithm.value_loss_coef`；
- `algorithm.use_clipped_value_loss`；
- `algorithm.clip_param`；
- `algorithm.entropy_coef`；
- `algorithm.num_learning_epochs`；
- `algorithm.num_mini_batches`；
- `algorithm.learning_rate`；
- `algorithm.schedule`；
- `algorithm.gamma`；
- `algorithm.lam`；
- `algorithm.desired_kl`；
- `algorithm.max_grad_norm`；
- `runner.num_steps_per_env`。

所以这些值目前看起来一致，但修改 `Ri4438CfgPPO` 不会改变最终 RL config。

### 11.3 已经不一致的 runner 参数

| 参数 | `Ri4438CfgPPO.runner` | 实际 `rl_cfg.py` |
|---|---:|---:|
| `max_iterations` | 20000 | 10001 |
| `save_interval` | 200 | 100 |
| `experiment_name` | `test` | `ri_4438_ppo` |

以实际 `rl_cfg.py` 为准，训练日志会写入：

```text
logs/rsl_rl/ri_4438_ppo/<run-time>/
```

Runner 在保存 checkpoint 时还会导出 `policy.onnx` 并附加 MjLab metadata。

## 12. 当前实际加载值

通过 MjLab registry 实际加载后，当前核心数值为：

```text
task registered             Ri-4438-Rough / Ri-4438-Flat
scene.num_envs              1
MuJoCo timestep             0.005 s
decimation                  4
environment step_dt         0.020 s
episode length              20 s (Train)
command resampling          (4.0, 8.0) s
rel_standing_envs           0.05
rel_forward_envs            0.10
heading_command             False
initial command ranges      x/y/yaw = (-1.0, 1.0)
action scale                0.15381525329142837 rad
actor hidden dims           (512, 256, 128)
critic hidden dims          (512, 256, 128)
PPO rollout length          24 steps/env
PPO max iterations          10001
checkpoint save interval    100 iterations
experiment name             ri_4438_ppo
```

20 s episode 在 50 Hz 下是 1000 个 environment step。PPO 每次 iteration 收集 24 个 step/env，即每个环境 0.48 s 的数据。

当 `num_envs=4096` 时，每次 iteration 的样本量是：

```text
4096 * 24 = 98304 transitions
```

但当前注册环境的默认值是 1，不是 `Ri4438PiperCfg.env.num_envs` 中的 4096。

## 13. 使用方式

### 13.1 用零动作检查机器人和环境

在项目根目录运行：

```bash
PYTHONPATH=.:src .venv/bin/python scripts/play.py \
  Ri-4438-Rough \
  --agent zero \
  --num-envs 1
```

Flat 环境：

```bash
PYTHONPATH=.:src .venv/bin/python scripts/play.py \
  Ri-4438-Flat \
  --agent zero \
  --num-envs 1
```

### 13.2 训练

由于 `env.num_envs` 尚未接通，可以先通过 CLI 覆盖：

```bash
PYTHONPATH=.:src .venv/bin/python scripts/train.py \
  Ri-4438-Rough \
  --env.scene.num-envs 4096
```

Flat 训练：

```bash
PYTHONPATH=.:src .venv/bin/python scripts/train.py \
  Ri-4438-Flat \
  --env.scene.num-envs 4096
```

也可以通过 CLI 覆盖最终 PPO 配置，例如：

```bash
PYTHONPATH=.:src .venv/bin/python scripts/train.py \
  Ri-4438-Rough \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 20000 \
  --agent.save-interval 200
```

### 13.3 加载 checkpoint

```bash
PYTHONPATH=.:src .venv/bin/python scripts/play.py \
  Ri-4438-Rough \
  --agent trained \
  --checkpoint-file logs/rsl_rl/ri_4438_ppo/<run-time>/<checkpoint>.pt \
  --num-envs 1
```

### 13.4 检查 registry 中的最终值

修改配置后，不要只查看源文件，可以用下列方式检查注册后的最终配置：

```bash
PYTHONPATH=.:src .venv/bin/python - <<'PY'
import src.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

env = load_env_cfg("Ri-4438-Rough")
agent = load_rl_cfg("Ri-4438-Rough")

print("num_envs:", env.scene.num_envs)
print("step_dt:", env.sim.mujoco.timestep * env.decimation)
print("command:", env.commands["twist"])
print("action scale:", env.actions["joint_pos"].scale)
print("max_iterations:", agent.max_iterations)
print("save_interval:", agent.save_interval)
PY
```

## 14. 配置修改的生效时机

`ri_4438_ppo_env_cfg.py`、`ri_4438_constants.py` 和 `rl_cfg.py` 都在模块 import 时实例化集中配置。任务注册时又会立即构造 Train/Play 配置对象。

因此正确使用方式是：

1. 修改配置源文件。
2. 停止当前训练/播放/Python 会话。
3. 启动新进程，让模块重新 import 和注册。
4. 通过 registry 或保存的 `env.yaml`/`agent.yaml` 确认最终值。

在同一 Python 进程里修改已经实例化的配置，不会自动回溯到所有已构造的 actuator、env cfg 和 registry 对象。

## 15. 当前最需要注意的问题

### 高优先级

1. `env.num_envs=4096` 没有接到 `SceneCfg.num_envs`，默认训练实际只有 1 个环境。
2. `heading_command` 只接了 ranges 部分，仅修改集中配置会导致不合法组合。
3. 命令 ranges 虽已接通，但训练 curriculum 会在第一个阶段很快将其覆盖。
4. `Ri4438CfgPPO` 除 hidden dims 外几乎未接入，而且 runner 的三个数值已经与最终配置不同。

### 中优先级

1. `rel_standing_envs` 和 `rel_forward_envs` 在两个文件里重复定义，修改集中配置无效。
2. `control.decimation`、`reset.joint_offset`、COM 范围和 action-rate 缩放未跟随集中配置。
3. link mass 和 Kp/Kd domain randomization 参数已定义，但没有对应 event。
4. Play 仍保留 startup domain randomization，如果目标是可重复的确定性评估，需要特别注意。

### 代码结构问题

1. `comaman` 拼写不清晰，但必须在所有消费位置同步重命名。
2. 配置导入同时使用 `src.config...` 和 `config...` 两种路径，有将同一源文件加载为两个 Python module 的风险。
3. `ri_4438_ppo_env_cfg.py` 先 import MjLab `mdp`，然后用本地 `mdp` 覆盖同名符号。本地 `mdp/__init__.py` 目前通过 wildcard 重新导出 MjLab MDP，所以能用，但阅读和排错时容易混淆函数的真实来源。

## 16. 后续理顺配置时的建议边界

如果后续要将 `ri_4438_config.py` 作为真正的单一参数源，建议至少统一下列映射：

- `env.num_envs -> SceneCfg.num_envs`；
- `control.decimation -> ManagerBasedRlEnvCfg.decimation`；
- `comaman.rel_standing_envs/rel_forward_envs -> UniformVelocityCommandCfg`；
- `comaman.heading_command -> heading_command` 和 `ranges.heading` 同时使用；
- `reset.joint_offset -> reset_robot_joints.params`；
- `domain_rand.com_displacement_range -> base_com.params.ranges`；
- 完整将 `Ri4438CfgPPO.policy/algorithm/runner` 映射到 `rl_cfg.py`；
- 决定 command curriculum 是由集中配置生成，还是作为 task-specific 逻辑继续留在 env cfg 中。

机器人结构相关的名称（`base_link`、foot sites、collision geoms）建议继续留在 `env_cfgs.py`，因为它们是 RI-4438 MJCF 的具体接口，不是通用 PPO 超参数。
