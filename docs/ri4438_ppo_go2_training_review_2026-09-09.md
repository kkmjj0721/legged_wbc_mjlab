# RI-4438 PPO 与当前 Go2 训练审查

审查日期：2026-09-09（Asia/Shanghai）

审查范围：

- 当前仓库的 RI-4438 / Go2 环境、PPO 配置、机器人资产与执行器；
- `logs/rsl_rl/ri_4438_ppo` 和 `logs/rsl_rl/go2_ppo` 中的 TensorBoard event、运行时 `params/*.yaml` 与 git diff 快照；
- 参考项目 `/home/sunteng/projects/DreamWaQ` 的 RI-4438 配置、CENet/PPO、奖励、终止、执行器和完整训练日志；
- CPU 小环境构造与显式 reset/zero-action smoke。没有启动正式训练，也没有修改训练源码。

## 1. 结论

RI-4438 的“速度一直上不去”不是一个孤立的 PPO 超参数问题，而是以下问题叠加形成的：

1. 当前 actor 没有机身线速度、只有单帧观测，也没有 DreamWaQ 的 5 帧历史/CENet 速度估计；速度反馈对 actor 基本不可观测。
2. 奖励存在很强的静止局部最优：全程 `pose=+1` 和固定相位 `foot_gait=+0.5`，而移动会承担 action-rate、净空、打滑和终止风险。
3. 非足接触范围过宽：base、hip、thigh、calf 中任意一个在 4 帧历史内超过 10 N 都会终止；DreamWaQ 只让 base 接触终止，thigh/calf 只受轻度碰撞惩罚。
4. `is_terminated=-200` 在 `dt=0.02` 下等价于每次非 timeout 终止约 `-4`；DreamWaQ 是 `-1 * 0.02=-0.02`。当前失败代价约为参考实现的 200 倍。
5. 旧 run 的 action-rate 太重，策略学成保守；把它降到按 action scale 补偿后的值后，adaptive-KL PPO 又把策略标准差推到约 1.4～1.5，碰撞、打滑和动作加速度一起上升，训练进入另一种不稳定状态。
6. 当前运行的是普通 47 维 MLP PPO，不是 DreamWaQ。物理执行器、命令、地形、奖励和 curriculum 也都不等价，因此不能把 DreamWaQ 的曲线当成只差训练轮数的基线。

最符合所有日志的机制是：

```text
单帧 actor 看不到真实速度
        +
pose / 固定 gait 的静止收益
        +
小 action scale、all-nonfoot termination、单次 -4 失败代价
        |
        +--> action-rate=-0.05：学会生存/站立，但速度跟踪停在低水平
        |
        `--> action-rate=-0.003785：探索方差升高，接触与抖动增加，回报再次回落
```

所以继续单独调 action-rate 或把训练从 1k 延长到 10k，成功概率不高。应先修正/消融观测、终止和奖励目标，再调 PPO。

## 2. 日志事实

下表是审查时冻结的快照。`Episode_Termination/illegal_contact` 是 reset 子集中的 `count_nonzero`，不是百分比，不能跨不同 reset batch 当成概率比较；但同一 run 内的趋势仍有意义。

| Run | 关键运行时差异 | 最后 update | mean reward | episode length / 1000 | track linear | error vel xy | illegal contact | policy std |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| RI `11-02-14` | action-rate `-0.05`，clearance `0.10`，Go2 MDP 路径 | 1486 | 12.58 | 662 | 0.286 | 0.821 | 2.25 | 0.723 |
| RI `11-26-39` | action-rate `-0.003785`，clearance `0.08`，Go2 MDP 路径 | 362 | -2.36 | 237 | 0.074 | 0.343 | 12.83 | 1.479 |
| RI `11-38-20` | action-rate `-0.003785`，clearance `0.10`，Go2 MDP 路径 | 22 | -2.82 | 51 | 0.013 | 0.080 | 44.46 | 1.081 |
| RI `11-52-17` | RI MDP 路径、zero root reset；event 仍在增长 | 1017 | 0.21 | 366 | 0.119 | 0.444 | 10.08 | 1.382 |
| Go2 `11-44-18` | 当前 Go2 Flat baseline | 439 | 51.51 | 996 | 0.779 | 0.530 | 0.043 | 0.236 |
| DreamWaQ RI 完整 run | Rough、CENet、H=5、固定退火 | 20000 | 20.84 | 990 | 0.874 | 无同名量 | reset contact 0.005 | 0.063 |

### 2.1 RI 旧 run 学到的是“生存”，不是速度跟踪

`11-02-14` 从 update 0 到 1486：

- episode length 从约 7 上升到约 662，说明确实学会了更长时间不终止；
- `track_linear_velocity` 最后只有约 0.286，`error_vel_xy` 长期约 0.8～0.9 m/s；
- `pose≈0.592`、`foot_gait≈0.226`，两项合计 0.818，已经明显大于 `track_linear≈0.286`；
- `action_rate≈-0.370`，移动探索的直接代价很高。

总回报上升并不等价于速度跟踪变好。该 run 的主要进步是少碰撞、延长 episode、保持姿态。

### 2.2 降低 action-rate 后不是变快，而是探索发散

`11-26-39` 和 `11-52-17` 都把 action-rate 调为 `-0.003785`。与旧 run 相比：

- policy std 从 1.0 增长到约 1.4～1.5，而不是逐步收敛；
- entropy 从约 17 增长到约 21；
- `mean_action_acc` 增长到约 3.5，slip 和 landing force 同时上升；
- illegal contact 保持在较高水平，episode length 只有约 200～400；
- track linear 仍只有约 0.07～0.14，回报短暂上升后再次回落。

这说明 `-0.05` 的确抑制了 RI 的动作，但它不是唯一问题。去掉这层约束后，当前 adaptive LR、entropy、终止和动力学组合不能把探索转化成稳定步态。

### 2.3 Go2 pipeline 本身能学，问题更集中在 RI 迁移

Go2 Flat run 在约 440 update 已经达到：

- episode length 约 996/1000；
- track linear 约 0.78，track angular 约 0.90；
- illegal contact 接近 0；
- policy std 已下降到约 0.24。

这证明当前 rsl_rl 训练主循环、TensorBoard、checkpoint 和基本 PPO pipeline 不是全面失效。RI 与 Go2 的关键物理差异是：

- RI action scale 是所有腿关节 `0.1538`；Go2 是 hip `0.25`、thigh/calf `0.5`；
- RI torque cap 是所有关节 `10 Nm`；Go2 为 hip/thigh `23.7 Nm`、calf `45.43 Nm`；
- RI 在小动作范围和较低力矩下更依赖可靠的速度反馈、合理的探索和接触策略。

但 Go2 当前曲线仍然只证明 Flat + 低速阶段有效，不能证明 Rough 或 1～2 m/s 高速阶段有效。

## 3. 高优先级问题

### P0-1：actor 缺少可用的速度反馈

当前 actor terms 只有：角速度、重力投影、命令、2 维 phase、关节角、关节速度和上一动作；`base_lin_vel` 只给 critic。actor 和 critic 的 `history_length` 都是 1：

- [`ri_4438_ppo_env_cfg.py:61`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:61)
- [`ri_4438_ppo_env_cfg.py:90`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:90)
- [`ri_4438_ppo_env_cfg.py:120`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:120)

实际构造出的 actor 是 47 维普通 MLP。机身平移速度不是单帧 IMU/关节状态可以稳定辨识的量，策略没有直接的跟速误差反馈。

DreamWaQ 的契约不同：单帧是 45 维，但用 5 帧历史（225 维）经 CENet 产生 3 维速度估计和 16 维 context，再和当前观测一起进入 actor：

- [`DREAMWAQ_PROJECT_PAPER_SUMMARY.md:91`](/home/sunteng/projects/DreamWaQ/DREAMWAQ_PROJECT_PAPER_SUMMARY.md:91)
- [`actor_critic_DWAQ.py:24`](/home/sunteng/projects/DreamWaQ/rsl_rl-1.0.2/rsl_rl/modules/actor_critic_DWAQ.py:24)

这不是小超参数差异，而是任务可观测性和 policy architecture 的差异。若目标只是 plain PPO，至少也要把“actor 怎样获得速度/历史信息”作为首个独立变量验证；若目标是复现 DreamWaQ，则需要恢复 H=5/CENet/privileged velocity supervision，不能只复制它的奖励权重。

### P0-2：终止范围过宽，并叠加 200 倍失败代价

当前 `nonfoot_ground_touch` 匹配除四只脚外的所有 `.*_collision`，因此包含 base、hip、thigh 和 calf，并保留 4 帧接触历史：

- [`env_cfgs.py:53`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/config/env_cfgs.py:53)
- [`env_cfgs.py:112`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/config/env_cfgs.py:112)

实际解析的是 mjlab 自带 `illegal_contact`。当存在 `force_history` 时，它对所有 body/slot/history 做 `any(force > 10N)`，一次腿部擦地可能在历史窗口中持续触发终止。显式 reset 后的 zero-action smoke 没有初始误终止，因此问题不是“默认站姿必然穿地”，而是运动中的 thigh/calf/hip 触地被当成致命失败。

DreamWaQ RI 的配置是：

- `terminate_after_contacts_on = ["base"]`；
- `penalize_contacts_on = ["thigh", "calf"]`。

见 [`ri_4438_config.py:57`](/home/sunteng/projects/DreamWaQ/legged_gym/legged_gym/envs/ri_4438/ri_4438_config.py:57)。这种设计允许学习初期出现腿部擦地，并通过小 penalty 改善，而不是立即截断 rollout。

当前 `is_terminated` 权重还是 `-200`：

- [`ri_4438_ppo_env_cfg.py:299`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:299)

mjlab 默认按 `dt` 缩放奖励，控制周期为 `0.005 * 4 = 0.02 s`，所以一次非 timeout 失败的 impulse 约为 `-4`。DreamWaQ 的 termination scale 是 `-1`，同样乘 `dt` 后约 `-0.02`。当前的失败代价约为参考实现的 200 倍。

这两个因素必须一起审查；只缩小接触集合但保留极大 terminal cost，或只减 terminal cost但仍让所有腿部接触终止，都不能明确定位问题。

### P0-3：奖励目标允许“不动也拿高分”

当前奖励中：

- `track_linear_velocity=+1.0`；
- `pose=+1.0`，在所有移动速度下都生效；
- `foot_gait=+0.5`，按 episode 固定 0.6 s 相位匹配触地状态；
- 另有 action-rate、foot clearance、slip、landing 和 terminal penalty。

见：

- [`ri_4438_ppo_env_cfg.py:260`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:260)
- [`ri_4438_ppo_env_cfg.py:276`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:276)
- [`ri_4438_ppo_env_cfg.py:303`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:303)

定量例子：命令 `vx=0.5 m/s`、实际静止且 `vy=vz=0` 时，线速度奖励约为 `exp(-0.5^2 / 0.25)=0.368`；默认姿态的 pose 奖励是 1。策略只要站住，就能从 pose 和静态触地与固定 gait 的部分匹配中拿到比跟速更强的正反馈。移动则立刻面临接触终止和动作代价。

DreamWaQ RI 没有全程 `pose` 正奖励，只有 exact-idle 下的 stand-still 约束；固定相位 `foot_gait` 也被换成基于实际 touchdown、stride period、duty、diagonal phase 的 gait quality penalty，并根据 rough/flat 地形门控。

建议将 `pose` 和固定 `foot_gait` 分开做单变量消融。不要同时改，否则无法判断是静态姿态收益还是固定 gait 相位在吞掉速度梯度。

### P0-4：adaptive-KL 与高 entropy 在 RI 上形成不稳定反馈

实际 RL config 是：

- `learning_rate=1e-3`；
- `schedule="adaptive"`；
- `desired_kl=0.01`；
- `entropy_coef=0.01`；
- 每个 update 5 epoch × 4 minibatch。

见 [`rl_cfg.py:33`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/config/rl_cfg.py:33)。

当前 PPO 会在每个 minibatch 后按 KL 将 LR 乘/除 1.5，范围为 `1e-5`～`1e-2`：

- [`ppo.py:241`](/home/sunteng/projects/legged_wbc_mjlab/rsl_rl/algorithms/ppo.py:241)

`11-02-14` 和 `11-52-17` 的 LR 都实际跑过约 `1e-5`～`5.06e-3`。后两个低 action-rate run 中 policy std 和 entropy 上升、回报下降，说明探索没有被稳定收敛。

DreamWaQ RI 使用 `schedule='fixed'`，并按绝对 update 做 piecewise LR/entropy 退火：LR 从 `1e-3` 逐步降到 `5e-5`，entropy 在 10k 前降到 0。其完整日志最后 std 约 0.063。

当前日志没有 KL、clip fraction 和 explained variance，只有最终 minibatch 后的 LR，所以无法进一步区分是 KL estimator、advantage 分布还是 value fitting 导致 scheduler 抖动。固定 LR/显式退火应作为一个独立消融，并补齐这些审计指标后再决定是否恢复 adaptive。

## 4. 配置与 DreamWaQ 不一致

### 4.1 当前并不是 DreamWaQ PPO

当前 runner 是普通 `RslRlOnPolicyRunnerCfg + MLPModel + PPO`，无 CENet、estimation/reconstruction/KL loss、AdaBoot 或 5 帧历史：

- [`rl_cfg.py:13`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/config/rl_cfg.py:13)

DreamWaQ 完整 run 有 `CENet/*` 指标、19 维 context code、H=5 历史和 286 维 privileged critic。当前只把部分环境和 PPO 超参数移植到了 mjlab。二者不是同一个算法，也不是同一个 MDP。

### 4.2 命令配置被硬编码覆盖

`Ri4438PiperCfg` 和 `Go2Cfg` 都把命令类拼成了 `comaman`，其中的 `heading_command=False`、range 和 curriculum 不会被环境读取：

- [`ri_4438_config.py:8`](/home/sunteng/projects/legged_wbc_mjlab/src/config/ri_4438/ri_4438_config.py:8)
- [`go2_config.py:15`](/home/sunteng/projects/legged_wbc_mjlab/src/config/go2/go2_config.py:15)

真正生效的是 env factory 中硬编码的：

- `heading_command=True`；
- `rel_heading_envs` 采用 mjlab 默认 1.0，即所有非 standing env 都是随机 heading 任务；
- `rel_standing_envs=0.05`；
- resampling time 3～8 s；
- x/y/yaw 初始范围分别为 `[-1,2]`、`[-1,1]`、`[-1,1]`。

见 [`ri_4438_ppo_env_cfg.py:169`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:169)。实际 class 还来自 `mjlab.tasks.velocity.mdp.UniformVelocityCommandCfg`，各任务目录中的本地 `mdp/velocity_command.py` 没有被这里使用。

DreamWaQ RI 使用 direct yaw（heading false）、约 10 s 命令持续时间，并保留 10% exact-idle。这些差异会改变角速度目标、站立样本覆盖和同一命令下可积累的稳定 rollout 长度。

### 4.3 当前 command curriculum 不是性能课程

当前代码按固定 global step 切 command range：

- stage 0：x `[-0.5,1.0]`、y `[-0.5,0.5]`；
- 到 `5000 * 24` env step，也就是约 PPO update 5000，才扩展到 x `[-1,2]`、y `[-1,1]`。

见 [`ri_4438_ppo_env_cfg.py:371`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/ri_4438_ppo_env_cfg.py:371)。

由于判断条件使用严格 `>`，初始 reset 仍可能先按 env factory 的 full range 采样，第一次后续 reset 才落回 stage 0。更重要的是，所有现有 RI run 都远小于 5000 update，所以它们根本没有系统训练 1～2 m/s 区间。若用户说“高速一直上不去”，当前 curriculum 在时间上本来就没给出高速样本。

DreamWaQ 参考代码的旧 command curriculum 是当 tracking reward 超过最大值的 80% 才扩展；当前完整 RI config 则默认关闭这一课程。二者都不等于当前硬时间阈值。

### 4.4 执行器没有移植 DreamWaQ 的 TN torque-speed envelope

当前 RI 只配置了固定 PD、10 Nm effort cap、action delay、armature：

- [`ri_4438_config.py:20`](/home/sunteng/projects/legged_wbc_mjlab/src/config/ri_4438/ri_4438_config.py:20)
- [`ri_4438_constants.py:33`](/home/sunteng/projects/legged_wbc_mjlab/src/assets/robots/ri_4438/ri_4438_constants.py:33)

编译后的 MuJoCo actuator 是标准 `<position>` servo：Kp 32.5065、Kd 2.0694、force range ±10 Nm。DreamWaQ RI 还启用了：

- `tn_curve_enabled=True`；
- `a=-0.128416, b=-0.699618, c=19.833274`；
- `max_torque=10 Nm, max_velocity=20 rad/s`；
- motoring 时按速度降低 torque cap，braking 保持完整 torque。

见：

- [`DreamWaQ ri_4438_config.py:44`](/home/sunteng/projects/DreamWaQ/legged_gym/legged_gym/envs/ri_4438/ri_4438_config.py:44)
- [`DreamWaQ legged_robot.py:845`](/home/sunteng/projects/DreamWaQ/legged_gym/legged_gym/envs/base/legged_robot.py:845)

这更直接影响动力学等价性和 sim-to-real，而不一定单独导致训练速度上不去；但如果目标是复现 DreamWaQ RI，当前执行器不是同一物理系统，必须在最终对齐前解决。

### 4.5 `Ri4438CfgPPO` / `GO2CfgPPO` 是 shadow/dead config

两份 `config/*_config.py` 都写了 `max_iterations=20000` 等 PPO 参数，但任务实际使用的 `config/rl_cfg.py` 又硬编码了一份配置，并且只设置 `max_iterations=10001`。导入得到的 `ri_4438_ppo_cfg` / `go2cfg` 没有用于构造 runner。

因此应以每个 run 的 `params/agent.yaml` 为准，不能以 `Ri4438CfgPPO` 或 `GO2CfgPPO` 判断实际训练行为。

## 5. Flat / Rough 与地形问题

### 5.1 现有 RI 和 Go2 日志全部是 Flat

所有当前 run 的 `params/env.yaml` 都是：

```yaml
terrain_type: plane
terrain_generator: null
```

例如 [`11-02-14 env.yaml:102`](/home/sunteng/projects/legged_wbc_mjlab/logs/rsl_rl/ri_4438_ppo/2026-09-09_11-02-14/params/env.yaml:102)。这说明实际任务是 `Ri-4438-Flat` 和 `Unitree-Go2-Flat`，不是 DreamWaQ 的 rough RI run。

Flat/Rough 虽然 task ID 不同，但各自机器人共享相同 `experiment_name`，所以仅看日志目录名无法分辨。以后每个 run 必须以 `params/env.yaml` 验证 terrain，并给 run_name 写入 `flat/rough + seed + 消融项`。

### 5.2 当前 rough 的 foot clearance 使用绝对世界高度

当前 `foot_height` 和 `feet_clearance` 都直接使用 `site_pos_w[..., 2]`，没有减去足下局部地形高度：

- [`observations.py:13`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/mdp/observations.py:13)
- [`rewards.py:125`](/home/sunteng/projects/legged_wbc_mjlab/src/tasks/locomotion/ri_4438_ppo/mdp/rewards.py:125)

在 Flat 上只是定义偏差；在 Rough/Stairs 上是明确的地形坐标错误。当前 `ROUGH_TERRAINS_CFG` 的不同 patch origin z 可相差米级，而 target 固定为 0.08/0.10 m。足端水平速度越大，这个错误 penalty 越大，会形成“不要加速”的梯度；critic 的绝对 foot height 也会和 terrain tile 的 world-z offset 绑定。

DreamWaQ 先计算足下局部 terrain height，再以相对地面的 clearance 计算 penalty：

- [`legged_robot.py:2384`](/home/sunteng/projects/DreamWaQ/legged_gym/legged_gym/envs/base/legged_robot.py:2384)

在进入 Rough 正式训练前，这项必须先做 terrain-relative 验证或消融。

### 5.3 rough terrain 分布也不相同

mjlab `ROUGH_TERRAINS_CFG` 包括 20% flat、40% 上/下 pyramid stairs、20% 正/反 slope、10% random rough、10% wave；DreamWaQ 使用另一套斜坡、粗糙坡、楼梯和 discrete obstacle 比例及难度曲线。两边虽然都是 10×20、8 m tile，但不是同一课程。

因此“当前 rough 是否达到 DreamWaQ rough reward”只能在先对齐地形/奖励语义后比较。

## 6. 当前 Go2 的审查结论

Go2 Flat 当前明显比 RI 健康，短期不应把它当成失败 run。它已经证明 plain PPO pipeline 可工作。但还有以下审查项：

1. 仍然是 47 维单帧 actor、没有 base linear velocity/history；建议按 command bin 统计误差，确认不是只在低速/heading 收敛。
2. 当前 curriculum 要到 update 5000 才开放 1～2 m/s，约 440 update 的日志不能回答高速跟踪问题。
3. 同样硬编码 `heading_command=True`，而 `Go2Cfg.comaman.heading_command=False` 不生效。
4. Go2 的 angular tracking 权重是 1.0，RI 是 0.5；两者总 reward 不可直接比较。
5. Go2 Flat/Rough 共用 `experiment_name=go2_ppo`，必须用 run_name/params 区分。
6. 当前 Go2 log 的 `agent.multi_gpu=null`，只证明单进程训练；没有多 GPU 正确性或扩展效率证据。
7. Go2/RI 的 `get_spec()` 都没有填充 `MjSpec.assets`。源码目录直接训练可成功，但场景导出、移动目录、policy bundle/deploy 时 mesh 可能丢失。已有更完整审查见 [`go2_mjlab_rsl_rl_multi_algorithm.md:116`](/home/sunteng/projects/legged_wbc_mjlab/docs/go2_mjlab_rsl_rl_multi_algorithm.md:116)。

## 7. 推荐验证顺序

不要同时改多项。使用固定 seed、相同 command 序列、相同环境数和相同训练 update，按下面顺序跑最小矩阵。

### 第一阶段：先证明 RI Flat 能稳定跟速

1. `baseline-current`：冻结当前配置，保存完整 params/git diff。
2. `termination-base-only`：只对齐 contact termination 范围，并单独审计 terminal scale；记录 base/hip/thigh/calf 每类接触率。
3. `no-pose`：只关闭全程 pose 正奖励；保留其他项不变。
4. `no-fixed-gait`：只关闭固定 0.6 s phase gait 奖励。
5. `direct-yaw`：只改为 heading false/direct yaw，命令范围不动。
6. `fixed-lr`：只把 adaptive KL 换成固定/显式退火；记录 KL、clip fraction、explained variance、grad norm。
7. `velocity-observable`：比较单帧 MLP、actor history 和 H=5/CENet/速度估计方案。

最先只允许选两个实验时，优先：

- `termination-base-only + 合理 terminal scale`；
- `no-pose`。

随后再做 actor history/CENet。原因是前两项最容易验证当前“移动立即付出高代价、站立收益更高”的局部最优；观测架构则决定最终能否做高精度跟速。

### 第二阶段：再开放高速度

只有当 Flat 的以下条件稳定后，才开放 1～2 m/s：

- episode length 接近 1000；
- track linear 明显持续上升；
- error vel xy 持续下降，而不是只靠 survival/pose 增加总回报；
- policy std 逐步下降；
- non-base contact 可以出现但不会大量截断 episode；
- command bin 中每个速度段都有独立误差统计。

### 第三阶段：最后进入 Rough

进入 Rough 前先完成：

- foot height / clearance 改为 terrain-relative 或关闭做对照；
- fixed gait 的 terrain gate/消融；
- terrain level、terrain type、reset 原因、base height 的日志；
- 检查是否需要 DreamWaQ 的 `terminate_below_height` 安全项；
- 明确是复现 DreamWaQ 地形，还是使用 mjlab 自己的 rough benchmark。

## 8. 日志和复现建议

当前每个 run 都在 dirty worktree 上运行，且训练期间源码确实发生过变化：前三个 RI run 的 reward 函数路径写成 `src.tasks.locomotion.go2_ppo.mdp`，`11-52-17` 才变为 RI 路径；reset pose 也从 x/y/yaw 随机变成全零。虽然两套 MDP 文件当前内容相同，这仍说明 run 之间不是严格可比实验。

后续每个实验至少固定并记录：

- task ID、flat/rough、seed、num_envs、GPU 数；
- commit hash、dirty diff、run_name；
- command range 与实际命令直方图；
- actual vx/vy/yaw、分 command bin tracking error；
- 每个 body 类别的 contact/termination 率；
- reward 各分项占比；
- KL、clip fraction、policy std、entropy、explained variance；
- action/target/torque saturation 比例与 joint velocity。

另外，当前 `Episode_Termination/*` 记录的是 raw count，日志中应另加归一化 reset fraction，否则不同 batch/reset 数之间不容易比较。

## 9. 最终判断

当前最可能的根因不是“RI 需要更多 iteration”，而是：

1. actor 缺少速度历史/估计；
2. pose + fixed gait 让静止策略收益过高；
3. all-nonfoot termination + `-200` 让探索代价过大；
4. action-rate 调小后 adaptive LR/entropy 又导致方差发散；
5. RI 执行器和 DreamWaQ 的物理语义未对齐。

Go2 Flat 已经能够学习，说明训练框架本身可用；它也把排查范围收窄到了 RI 的动作尺度、力矩能力、接触/终止、速度可观测性和奖励设计。先把 RI Flat 做成可稳定、可解释的速度跟踪任务，再谈 Rough 和 DreamWaQ 等价复现，是风险最低的路径。
