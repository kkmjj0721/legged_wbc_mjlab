# RI-4438 HIM Rough 训练监控与分析报告

报告日期：2026-09-15  
任务：ri_4438_him rough terrain 速度控制  
说明：本报告只读取训练日志、TensorBoard event、checkpoint 和运行参数快照，没有修改训练代码，也没有替用户停止训练进程。

## 1. 结论摘要

RI-4438 的主要问题不是 HIM estimator 发散，也不是完全学不会站立，而是控制输出、执行器能力和 command/terrain 难度之间不匹配。

当前可以得到以下结论：

1. 旧配置能够学会基本站立和行走，但线速度 tracking 长期卡在约 0.63～0.70，terrain curriculum 推进很慢。
2. 后续修改后的新 run 明显改善了角速度 tracking 和速度误差：角速度 reward 从旧 run 的约 0.29 提高到约 0.65～0.69，error_vel_xy 从约 1.2～1.4 降到约 0.56～0.65。
3. 新 run 在 iteration 9000～9900 左右达到最健康状态：线速度约 0.73～0.77，角速度约 0.66～0.69，总 reward 约 20～22。
4. 在 iteration 10000 的 command curriculum 切换后，性能再次明显回撤：总 reward 降到约 17～19，线速度降到约 0.65～0.71，跌倒和非法接触上升。
5. terrain level 上升并不等于能力变好。新 run 在 terrain level 超过 3 时，跌倒率和非法接触仍偏高，说明 terrain curriculum 当前可能比策略能力推进得更快。
6. 下一轮最应该优先处理 command curriculum 和控制尺度，不建议继续优先调整 HIM 网络结构。

## 2. 数据来源与运行时间线

### 2.1 旧 run

目录：

logs/rsl_rl/ri_4438_him/2026-09-14_20-49-46

该 run 从前一个 model_700.pt 恢复训练。5 小时重点监控区间为约 iteration 742～3573；之后训练继续运行到约 iteration 8215。

旧 run 的参数快照仍是：

    resampling_time_range = [4.0, 8.0]
    action_scale = 0.5
    effort_limit = 10.0 Nm
    max_init_terrain_level = 5
    platform_width = 2.0

### 2.2 新 run

目录：

logs/rsl_rl/ri_4438_him/2026-09-15_09-52-30

该 run 从旧 run 的 model_8200.pt 恢复，并使用了修改后的环境配置。监控从约 iteration 8306 开始，直到用户准备暂停时约 iteration 10501；因此本次监控没有达到原计划的 iteration 13000。

新 run 的参数快照确认包含：

    resampling_time_range = [8.0, 12.0]
    track_angular_velocity = src.tasks.locomotion.ri_4438_him.mdp.rewards.track_angular_velocity
    pose weight = 0.001
    hip_pos weight = -1.0
    foot_gait weight = 0.05

同时仍然保留：

    action_scale = 0.5
    effort_limit = 10.0 Nm
    max_init_terrain_level = 5
    platform_width = 2.0

## 3. 旧 run 的 5 小时监控结果

旧 run 在 iteration 742～3573 的窗口平均值：

| 指标 | 742～3573 平均 | 3573 附近 |
|---|---:|---:|
| mean reward | 18.36 | 约 20.21 |
| episode length | 984.9 / 1000 | 约 999 |
| 线速度 reward | 0.660 | 约 0.718 |
| 角速度 reward | 0.464 | 约 0.498 |
| terrain level | 1.15 | 约 1.32 |
| error_vel_xy | 1.26 | 约 1.12 |
| error_vel_yaw | 1.10 | 约 1.08 |
| fell-over | 4.6% | 约 1.8% |
| illegal contact | 0.18% | 约 0% |
| policy mean std | 0.187 | 约 0.185 |
| estimator loss | 0.00332 | 约 0.0032 |

这段时间的特点是：机器人已经基本能存活，但策略标准差过早降到约 0.18，探索不足；线速度 tracking 没有继续突破，terrain level 也只是在 1 附近缓慢波动。

### 3.1 旧 run 在 iteration 5000 之后的退化

旧 run 后续继续训练到 iteration 8215。在 iteration 5000 左右，command curriculum 从第一阶段扩大到第二阶段后，性能发生明显变化：

| 区间 | reward | 线速度 reward | 角速度 reward | terrain | fell-over |
|---|---:|---:|---:|---:|---:|
| 4500～4999 | 18.79 | 0.665 | 0.484 | 1.25 | 4.4% |
| 5000～5499 | 14.71 | 0.621 | 0.384 | 1.27 | 8.1% |
| 5500～5999 | 13.72 | 0.651 | 0.339 | 1.80 | 9.0% |
| 7000～7499 | 13.02 | 0.691 | 0.303 | 2.43 | 7.0% |
| 7500～7999 | 12.62 | 0.689 | 0.294 | 2.57 | 6.9% |
| 8000 附近 | 约 12.6 | 约 0.70 | 约 0.29 | 约 2.4 | 约 6%～10% |

旧 run 的核心问题不是彻底崩溃，而是 command 难度提高后，策略通过牺牲角速度 tracking 和 reward 来维持部分生存能力。

## 4. 新 run 的修改效果

### 4.1 iteration 8300～8999

窗口平均：

| 指标 | 平均值 |
|---|---:|
| mean reward | 20.00 |
| episode length | 964.4 |
| 线速度 reward | 0.719 |
| 角速度 reward | 0.657 |
| terrain level | 2.22 |
| error_vel_xy | 0.625 |
| error_vel_yaw | 0.958 |
| fell-over | 7.4% |
| illegal contact | 5.6% |
| policy mean std | 0.260 |
| estimator loss | 0.00669 |

相比旧 run 同阶段，速度误差和角速度 tracking 明显改善；但跌倒和非法接触仍高于旧 run，说明新奖励确实鼓励了更积极的动作，同时也提高了稳定性压力。

### 4.2 iteration 9000～9999

窗口平均：

| 指标 | 平均值 |
|---|---:|
| mean reward | 20.87 |
| episode length | 967.9 |
| 线速度 reward | 0.734 |
| 角速度 reward | 0.672 |
| terrain level | 2.53 |
| error_vel_xy | 0.600 |
| error_vel_yaw | 0.917 |
| fell-over | 5.9% |
| illegal contact | 5.9% |
| policy mean std | 0.250 |
| estimator loss | 0.00655 |

这是目前最有价值的训练阶段。代表性较好的点包括：

| iteration | reward | 线速度 | 角速度 | terrain | episode length | fell-over | illegal |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8670 | 20.47 | 0.740 | 0.668 | 2.16 | 987 | 7.9% | 2.6% |
| 8759 | 21.65 | 0.757 | 0.684 | 2.17 | 980 | 5.6% | 1.4% |
| 9289 | 22.05 | 0.766 | 0.690 | 2.51 | 975 | 4.1% | 2.7% |
| 9560 | 21.35 | 0.754 | 0.675 | 2.60 | 970 | 4.7% | 7.8% |
| 9923 | 21.36 | 0.757 | 0.687 | 2.45 | 995 | 1.5% | 3.0% |

### 4.3 iteration 10000～10501

iteration 10000 后，command curriculum 又扩大了一步：

    lin_vel_x: [-0.75, 1.0] -> [-1.0, 1.0]
    lin_vel_y: [-0.75, 0.75]
    ang_vel_z: [-0.75, 0.75]

切换后的窗口平均：

| 指标 | 10000～10501 平均 |
|---|---:|
| mean reward | 18.53 |
| episode length | 957.0 |
| 线速度 reward | 0.690 |
| 角速度 reward | 0.638 |
| terrain level | 2.83 |
| error_vel_xy | 0.661 |
| error_vel_yaw | 0.960 |
| fell-over | 8.8% |
| illegal contact | 7.2% |
| policy mean std | 0.269 |
| entropy | 1.01 |
| estimator loss | 0.00764 |

iteration 10501 附近：

    mean_reward            = 17.11
    episode_length         = 953.5
    track_linear_velocity  = 0.652
    track_angular_velocity = 0.607
    terrain_level          = 3.09
    error_vel_xy           = 0.703
    error_vel_yaw          = 0.989
    fell_over              = 9.7%
    illegal_contact        = 8.1%
    policy_mean_std        = 0.278
    estimator_loss         = 0.0079

和 iteration 10000 前相比，退化方向很清楚：

- 线速度 reward 约下降 0.04～0.08；
- 角速度 reward 约下降 0.04～0.08；
- episode length 下降约 1 秒；
- fell-over 和 illegal contact 增加；
- mean std 和 entropy 增加，策略重新进入高探索状态；
- estimator loss 从约 0.0065 升到约 0.0076～0.0080。

这次退化比旧 run 在 iteration 5000 时的退化轻，但仍说明 command 扩展速度超过了当前策略的适应能力。

## 5. 对各项修改的判断

### 5.1 resampling_time 从 [4, 8] 改为 [8, 12]

判断：有效。

更长的 command 持续时间给策略留下了更多时间形成稳定步态。它不是唯一原因，但和新 run 的 tracking 改善一致，建议保留。

### 5.2 使用项目内自定义角速度 reward

判断：有效，但 reward 数值需要重新建立基线。

新 run 使用了项目内的 track_angular_velocity。它与 Go2 使用同一套较轻的 roll/pitch 角速度处理方式，因此新 run 的角速度 reward 从约 0.3 回到约 0.65～0.69。这个修改消除了之前 4438 和 Go2 之间的 reward 实现差异，建议保留。

### 5.3 pose weight 从 0 改为 0.001

判断：影响较小，当前不是主要矛盾。

该项权重很小，暂时不需要优先调整。应该先固定 command 和控制尺度，再观察它的独立作用。

### 5.4 hip_pos 从 -0.1 改为 -1.0

判断：可能过强，需要谨慎。

它会更强地限制髋关节偏离默认姿态。新 run tracking 变好，但 rough 环境中的非法接触和跌倒仍较高，强髋偏置惩罚可能让策略在需要横向调整时缺少自由度。

下一轮不建议继续加大该惩罚；可以考虑回到中间值，例如 -0.3～-0.5，但应单独做对照实验。

### 5.5 foot_gait 从 0.01 改为 0.05

判断：可能促进步态，但不能继续无条件增大。

它可能帮助策略形成更清晰的交替步态，但同时会鼓励更积极的抬脚/落脚行为。新 run 仍有较高接触和跌倒，因此下一轮应先保持 0.05，不要继续提高。

### 5.6 分阶段 command curriculum

判断：方向正确，但每阶段跨度仍偏大。

你已经把原本较粗的阶段拆细，这是有效的；不过 iteration 10000 仍一次性把 lin_vel_x 负方向扩到 -1.0。对于当前策略，这个改变仍然太突然。

## 6. 当前最可能的根本原因

### 6.1 动作尺度与力矩上限不匹配

当前仍是：

    stiffness = 32.5
    action_scale = 0.5
    effort_limit = 10 Nm

thigh/calf 仅比例项的理论力矩就可能达到：

    32.5 × 0.5 = 16.25 Nm

这超过了 10 Nm 的 effort limit。实际动作如果经常打到上限，策略会出现：

- 目标关节位置变化很大；
- 实际关节跟不上；
- 速度误差降不下来；
- rough 地形上纠偏能力不足；
- PPO 学会保守动作来避免摔倒。

这与旧 run 的低速平台以及新 run 在高 command 阶段的接触/跌倒增加相吻合。

### 6.2 terrain curriculum 与真实能力脱钩

terrain level 在新 run 中从约 2.5 上到 3.1，但 reward 和 episode length 同时下降，说明 terrain level 不能作为唯一成功标准。

当前 terrain_levels_vel 主要按 episode 结束时的净位移决定升级/降级。对于会摔倒、滑动或短时间冲出起点的环境，这种指标可能把 terrain 等级推高得过快。

### 6.3 adaptive PPO learning rate 频繁变化

新 run 在 10000 前多次出现：

    learning_rate ≈ 1e-5 ～ 3.8e-4

这说明 KL 约束频繁触发，优化器不断缩小或恢复更新步长。它不是根因，但说明策略在新 command 阶段对更新非常敏感。

### 6.4 estimator 没有发散，但承受更大压力

旧 run estimator loss 约 0.003；新 run 通常约 0.0065～0.008。这不表示 HIM estimator 已经坏掉，但说明修改后的 observation/reward/command 分布更难预测。

## 7. 暂停时建议保留的 checkpoint

### 7.1 新 run 候选

优先检查：

    model_8750.pt 附近
    model_9200.pt
    model_9400.pt
    model_9500.pt
    model_9800.pt
    model_9900.pt

实际文件以 run 目录中存在的 checkpoint 为准。当前最值得优先做 deterministic play/evaluation 的是：

    model_9400.pt
    model_9500.pt
    model_9900.pt

其中：

- model_9400.pt、model_9500.pt：tracking 和稳定性较平衡；
- model_9900.pt：10000 阶段切换前的最新策略；
- 10000 之后的 checkpoint：可以作为高 command 难度实验结果，但不建议直接当作最优部署策略。

### 7.2 评估时不要只看总 reward

应至少同时记录：

    track_linear_velocity
    track_angular_velocity
    error_vel_xy
    error_vel_yaw
    episode_length
    fell_over
    illegal_contact
    terrain_level

当前 reward 峰值可能来自角速度、步态或存活项改善，不能单独代表线速度 tracking 变好。

## 8. 下一轮修改建议

建议新开实验，不要直接从当前 model_10500.pt 左右继续，因为 command/reward 分布已经变化，且当前策略正在高探索、较不稳定状态。

### 8.1 第一优先级：平滑 command curriculum

保留：

    resampling_time = [8.0, 12.0]

建议阶段不要同时扩大所有方向。可以采用类似：

    阶段 1:
    lin_vel_x [-0.5, 1.0]
    lin_vel_y [-0.5, 0.5]
    ang_vel_z [-0.5, 0.5]

    阶段 2:
    lin_vel_x [-0.75, 1.0]
    lin_vel_y [-0.6, 0.6]
    ang_vel_z [-0.6, 0.6]

    阶段 3:
    lin_vel_x [-0.9, 1.0]
    lin_vel_y [-0.75, 0.75]
    ang_vel_z [-0.7, 0.7]

    最终阶段:
    lin_vel_x [-1.0, 1.0]
    lin_vel_y [-1.0, 1.0]
    ang_vel_z [-1.0, 1.0]

尤其不要在一个阶段里同时把负向前进、横向和 yaw 范围全部放大。

### 8.2 第二优先级：单独验证 action scale

当前尚未验证 action_scale=0.5 是否导致大比例动作饱和。建议单独开一个对照 run：

    action_scale = 0.25
    hip_reduction = 0.5

第一步先不改 stiffness、damping 和 effort limit，这样可以判断是否确实是动作尺度问题。

如果 0.25 过弱，再测试 0.30，不建议直接恢复到 0.5 后继续堆 reward 权重。

### 8.3 第三优先级：让 terrain curriculum 更保守

bootstrap 阶段建议：

    max_init_terrain_level = 0 或 1
    platform_width = 3.0

当前 run 快照使用 platform_width=2.0，比 Go2 的 3.0 更苛刻。4438 腿更短、执行器余量更小，建议先用 3.0 验证基本 locomotion，再恢复更难平台。

### 8.4 第四优先级：先减弱初期随机化

建议 bootstrap 阶段使用较窄随机范围：

    link_mass_range = [0.9, 1.1]
    KpKd_factor_range = [0.95, 1.05]
    motor_strength_range = [0.95, 1.05]
    delay_max_lag = 2

可以暂时关闭 push，等 tracking 稳定后再恢复。

### 8.5 reward 权重建议

建议保留：

    自定义 track_angular_velocity
    foot_gait = 0.05

建议暂时不要继续增大：

    foot_gait
    hip_pos 惩罚

hip_pos=-1.0 建议做一个独立对照，测试 -0.3～-0.5 是否能降低非法接触，同时保持速度 tracking。

## 9. 下一轮验收标准

在 iteration 400～600 或一个相当的 bootstrap 阶段内，建议使用窗口平均而不是单点判断：

    track_linear_velocity   > 0.72
    track_angular_velocity  > 0.65
    episode_length           > 900
    fell_over                < 5%
    illegal_contact          < 5%
    error_vel_xy             < 0.70
    error_vel_yaw            < 1.0

terrain level 只作为辅助指标，不能超过 tracking 和稳定性指标的优先级。

如果出现：

    episode_length 很高
    但线速度 reward 长期约 0.65

应优先检查 actuator torque saturation 和实际关节动作幅度，而不是继续调整 estimator、网络宽度或 PPO 学习率。

## 10. 推荐实验顺序

为了保留因果关系，建议按以下顺序做小规模对照：

### 实验 A：只验证 command 持续时间

    resampling_time = [8, 12]
    action_scale = 0.5
    保留现有 reward
    延后 command 扩展

观察是否能稳定保持 0.72+ 的线速度 reward。

### 实验 B：只验证 action scale

    resampling_time = [8, 12]
    action_scale = 0.25
    其余保持实验 A

重点观察速度 reward、动作幅度、跌倒率和 torque saturation。

### 实验 C：稳定性与 terrain 对照

在 A 或 B 中加入：

    max_init_terrain_level = 0/1
    platform_width = 3.0
    较窄 domain randomization

只有在实验 C 稳定后，才恢复更高 terrain 难度和完整 command 范围。

## 11. 最终判断

这次修改是有价值的：它成功把 4438 从“能存活但速度/角速度 tracking 很差”的状态，提升到了“tracking 明显可用但高难度阶段不稳定”的状态。

当前最好的训练阶段在 9000～9900 iteration，而不是 10000 之后。10000 之后的退化说明 command curriculum 仍然过快，并且当前 action_scale=0.5 与 10 Nm effort limit 的组合仍值得重点怀疑。

建议暂停并保留 9400～9900 附近 checkpoint，下一轮优先验证：

    更平滑的 command curriculum
    action_scale = 0.25
    max_init_terrain_level = 0/1
    platform_width = 3.0
    初期较弱的 domain randomization

在这些因素得到验证前，不建议继续用当前配置盲目训练到更高 iteration，也不建议仅根据 terrain level 上升判断策略已经超过 Go2。

