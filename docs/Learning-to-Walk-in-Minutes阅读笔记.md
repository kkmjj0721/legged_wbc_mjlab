# Learning to Walk in Minutes 阅读笔记

# 前言：

- 这篇论文的核心目标，是把 legged RL 里一个以前非常慢、非常依赖调参耐心的训练流程，压缩到**几分钟到二十分钟级别**。

- 论文标题是 **Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning**。它不是提出一个很复杂的新 policy 结构，也不是靠 imitation 或 motion prior 让机器人学出某种特定步态，而是问了一个更工程的问题：

    - **如果我们把仿真、reward、observation、policy inference、PPO update 全部尽量放到 GPU 上，并一次并行跑几千个机器人，四足机器人 locomotion 训练到底能快到什么程度？**

- 这篇论文很重要的原因，是它后来基本变成了 `legged_gym` / `rsl_rl` 这条路线的工程基准。我们现在很多配置里熟悉的东西，比如：

    - `num_envs = 4096`；

    - `num_steps_per_env = 24`；

    - PPO 的 `desired_kl = 0.01` 自适应学习率；

    - `send_timeouts = True`，也就是 timeout 和真正摔倒 termination 分开处理；

    - 地形 curriculum 里根据机器人走出的距离升降 terrain level；

    都能在这篇论文里找到非常直接的来源。

- 所以这篇论文真正值得学的不是“4096 个环境很快”这一句表面结论，而是它背后的训练系统判断：**大规模并行不是简单把环境数量拉满，而是要重新平衡 batch size、rollout horizon、GAE、timeout bootstrapping、terrain curriculum、contact cost 和 sim-to-real randomization。**

- 未执行/未验证：这份笔记基于 `/home/kk/下载/2109.11978v3.pdf` 和项目内同名 PDF `/home/kk/legged_wbc_mjlab/paper/2109.11978v3.pdf` 的文本抽取，本地参考笔记 `/home/kk/飞书文档/论文学习部分.md`、`/home/kk/飞书文档/AMP_for_hardware学习记录.md`、`/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`、项目内已有阅读笔记的写法整理，以及当前项目源码的静态只读核对；没有跑作者代码，没有重新训练策略，没有验证真实 ANYmal 部署效果。

# 论文核心信息：

- **论文标题：** Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

- **标题翻译：** 使用大规模并行深度强化学习，在几分钟内学会行走

- **作者：** Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter

- **机构：** ETH Zurich and NVIDIA, NVIDIA, ETH Zurich

- **会议/版本：** CoRL 2021 Proceedings；arXiv:2109.11978v3，PDF metadata 创建时间为 2022-08-22

- **关键词：** Reinforcement Learning, Legged Robots, Sim-to-real

- **项目页 / 代码：** https://leggedrobotics.github.io/legged_gym/

- **主要机器人平台：** ANYbotics ANYmal C；仿真里还测试 ANYmal B、带固定机械臂的 ANYmal C、Unitree A1、Cassie

- **训练仿真器：** NVIDIA Isaac Gym，仿真和训练都尽量放在单 workstation GPU 上

- **核心训练规模：** uneven terrain policy 使用 4096 个并行机器人、batch size 98304、1500 次 policy update，在 20 分钟以内完成训练

- **硬件配置：** 论文脚注给出训练机器为 i9-11900k CPU + NVIDIA RTX A6000 GPU

- **控制方式：** policy 输出期望关节位置，底层 PD controller / actuator network 转为 motor torque

- **论文类型：** massively parallel RL / legged locomotion / PPO engineering / terrain curriculum / sim-to-real

# 原文摘要翻译：

- 作者提出并研究了一套训练配置，用单台工作站 GPU 上的大规模并行仿真，快速生成可用于真实机器人任务的策略。

- 论文分析了在 massively parallel regime 下，不同训练算法组件对最终性能和训练时间的影响。

- 作者提出了一个 game-inspired curriculum，它适合几千个机器人并行训练，能够自动根据机器人表现调节地形难度。

- 实验任务是训练 ANYmal 四足机器人在复杂地形上行走，包括 uneven terrain、slopes、stairs 和 obstacles。

- 并行训练可以在 4 分钟以内训练平地策略，在 20 分钟以内训练 uneven terrain 策略。相比之前很多需要十几小时甚至上百小时的 locomotion 训练，这相当于多个数量级的加速。

- 最后，作者把策略迁移到真实机器人上，验证这个训练流程不是只在仿真 benchmark 上快，而是能生成可部署的真实机器人策略。

# 研究问题：

- 论文要回答的问题可以拆成三层：

    - **训练速度问题：** 真实机器人可用的 legged locomotion policy，能不能不再以“天”为单位训练，而是在分钟级完成？

    - **并行尺度问题：** PPO 这种 on-policy 算法，在一次跑几千甚至上万个环境时，batch size、rollout length、mini-batch、GAE 和 timeout 该怎么改？

    - **真实部署问题：** 这么快训练出来的策略，能不能仍然通过 sim-to-real，上真实 ANYmal C 走楼梯和障碍？

- 这个问题看起来像“只要 GPU 更快就行”，但实际不是。

- 如果我们只是盲目增加 `num_envs`，会遇到几个新问题：

    - batch size 可能变得太大，每次 update 前收集了大量重复或边际收益很低的数据；

    - 如果为了保持 batch size 不变而减少每个环境的 rollout step，GAE 的时间范围会太短；

    - episode timeout 会频繁跨过多个 PPO batch，如果把 timeout 当成 terminal，critic 会学错；

    - 几千个机器人不能每次 reset 都生成新 terrain mesh，否则地形生成和 contact 计算会拖慢吞吐；

    - contact detection、terrain representation、robot collision geometry 会成为新的性能瓶颈。

- 所以论文真正问的是：**在 GPU 并行仿真成为主要生产力以后，legged RL 的训练系统应该怎样重构？**

# 研究背景：

## 第一种：传统 CPU 并行 / 多进程仿真路线

- 早期很多 RL pipeline 是多个 CPU 进程分别跑仿真，然后把数据传给 GPU 做神经网络 update。

- 这条路线直觉上很自然：CPU 跑物理，GPU 跑网络。

- 但问题是，policy inference、observation、reward、simulation、storage 之间会反复搬数据。论文特别强调 PCIe 数据传输可能是 GPU 加速里的弱链路，数据拷贝可以比 GPU 本身计算慢很多。

- 对 locomotion 来说，这个问题更明显。一个 step 不是只算网络，还要算 contact、关节状态、奖励项、terrain scan、reset 条件。如果这些都在 CPU/GPU 之间来回倒，GPU 并行度很难真正用起来。

## 第二种：分布式超大 batch 路线

- 也有一些工作使用几千个 CPU worker 或分布式系统。它们通常是每个 worker 跑一份环境，然后跨 worker 平均梯度。

- 这种方式可以得到几百万级别的大 batch，对梯度噪声有帮助，但不一定优化总训练时间。

- 原因是：如果每个 agent 仍然提供很多 trajectory samples，总 batch 变大了，policy update 也会变重。它解决的是“数据量更多”，不一定解决“单位时间完成一个可用 policy”。

## 第三种：复杂 perceptive locomotion 路线

- 在这篇论文之前，复杂地形 locomotion 已经有很多强工作，但训练时间通常很长。

- 论文引用的对比里，blind rough terrain policy 有 12 小时训练时间，perceptive locomotion over challenging terrain 有 82、88、120 小时级别训练时间。

- 这些工作更追求鲁棒性或感知能力；这篇论文的目标不是直接超过它们所有指标，而是证明：**训练时间本身可以被大幅压缩，并且压缩后仍然能上真实机器人。**

# 核心直觉：

- 这篇论文的核心直觉很简单：**如果机器人控制是靠试错学出来的，那我们就应该让试错并行发生。**

- 以前我们可能是一只机器人在仿真里走，摔倒，reset，再走。即使并行，也可能是几十个或几百个环境。

- 这里变成几千个机器人同时走：

    - 有的机器人在平地上学基础稳定；

    - 有的机器人正在爬坡；

    - 有的机器人卡在上楼梯；

    - 有的机器人已经能跨障碍；

    - curriculum 根据每个机器人的表现，把它移动到更难或更简单的地形格子。

- 这就像一个游戏关卡系统。每个机器人不是全都排队等同一个任务，而是在同一张大地图上分布到不同难度。机器人能过关，就进入下一关；走得太差，就回到低一关；最高关卡过了以后，再随机回到低关卡复习，避免 catastrophic forgetting。

- 关键点在于：**几千个机器人本身就构成了 policy 当前能力分布的采样。** 我们不需要额外训练一个 generator 或 particle filter 去估计“现在应该给多难的 terrain”，直接看这些机器人在课程地图上的分布就能知道训练进度。

# 方法主线：

- 论文的方法可以拆成五段来看：

    - **第一段：端到端 GPU pipeline。** Isaac Gym 同时在 GPU 上跑仿真和训练，减少 CPU/GPU 数据搬运。

    - **第二段：PPO 在大规模并行下的改造。** 重新选择 `nrobots`、`nsteps`、batch size、mini-batch size，并处理 timeout bootstrapping。

    - **第三段：game-inspired terrain curriculum。** 把所有 terrain type 和 difficulty level 平铺成一张大地图，用机器人自己的进度调难度。

    - **第四段：简单但硬件友好的 locomotion MDP。** observation/action/reward 都尽量简单，不强制 gait，不引入 motion primitive。

    - **第五段：sim-to-real additions。** domain randomization、观测噪声、random push、actuator network，让快速训练出来的 policy 能迁移到真实 ANYmal。

- 这五段里，最容易被低估的是第二段和第三段。因为它们不像新网络结构那样显眼，但如果缺少它们，大规模并行很容易变成“跑得很快，但学不稳”。

# Massive Parallel RL 到底改变了什么：

## 数据收集不再是 CPU 主导

- 在 on-policy PPO 里，一个训练 iteration 通常有两步：

    - collect rollout；

    - update policy。

- Policy update 本来就适合 GPU，因为它是 neural network backprop。

- 真正麻烦的是 rollout collection。每一步要做：

    - policy inference；

    - physics simulation；

    - reward computation；

    - observation construction；

    - termination / reset handling。

- 如果这些环节分散在 CPU 和 GPU 之间，速度会被数据传输拖住。论文选择 Isaac Gym，就是为了让这些工作尽量在 GPU 端闭环。

## 并行不是越多越好

- PPO 的 batch size 可以写成：

$$
B = n_{robots} \times n_{steps}
$$

- 这里：

    - $n_{robots}$ 是并行机器人数量；

    - $n_{steps}$ 是每个机器人在一次 policy update 前连续走多少步；

    - $B$ 是一次 update 使用的样本数。

- 如果我们把 $n_{robots}$ 从 128 拉到 4096 或 16384，而 $n_{steps}$ 不变，batch 会爆炸。

- 如果我们为了固定 batch size，把 $n_{steps}$ 压得太小，又会丢掉时间连续性。这个问题对 GAE 很关键，因为 advantage 估计不是只看一步 reward，它需要一段连续 reward/value 序列。

- 论文的经验结论是：在这个任务里，少于大约 25 个连续 step，算法会明显难以收敛到好策略。按 50 Hz policy frequency 算，25 steps 约等于 0.5 s。

- 注意：这里的 $n_{steps}$ 不是 episode length。论文的 episode timeout 是 20 s，一个 episode 会跨过很多次 PPO update。这个区分非常重要，因为 timeout 处理也正是由这个设置触发出来的。

## Mini-batch 也要变大

- 常规 PPO 里，我们可能习惯比较小的 mini-batch。

- 但在这篇论文的大规模并行设置下，作者发现 tens of thousands 级别的 mini-batch 反而有利于稳定学习，而且不会明显增加总训练时间。

- 测试策略的参数表里：

    - batch size = 98304，也就是 $4096 \times 24$；

    - mini-batch size = 24576，也就是 $4096 \times 6$；

    - num epochs = 5。

- 这说明它不是简单“rollout 短一点”。它是在保持大 batch 的同时，用更大的 mini-batch 去让 PPO update 在 GPU 上高效稳定。

# Time-out Bootstrapping：为什么只看 `done` 不够

- 这篇论文里最值得工程上记住的点之一，就是 timeout 和真正 failure termination 必须分开。

- PPO critic 预测的是一个 discounted future return。对真正摔倒或任务失败来说，episode 终止是环境状态导致的，critic 可以学习“快摔了，所以后面没有价值”。

- 但 timeout 不一样。Timeout 只是因为我们人为设置了 episode 最大长度，比如 20 s。机器人并不是物理上失败了，只是这一段 rollout 到时间了。

- 如果把 timeout 也当成 terminal，critic 会被迫相信：

    - 状态本来很好；

    - 但因为时间到了；

    - 后面价值突然变成 0。

- 这对 critic 是错误监督。更麻烦的是，论文的大规模并行设置中，每次 PPO update 的 $n_{steps}$ 很短，而 episode length 很长，一个 episode 会跨很多个 batch。我们不能假设 timeout 只发生在 batch 最后一刻。

## 公式直觉

- 标准 GAE 里，一步 TD error 通常写成：

$$
\delta_t = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)
$$

- 然后 advantage 递推为：

$$
A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
$$

- 如果 $d_t=1$ 表示真实 terminal，后面的 bootstrap 断掉是合理的。

- 但如果 $d_t=1$ 只是 timeout，就不应该把后面的价值砍成 0。论文的做法是检测 timeout，并用 critic 自己的预测去补上未来 discounted return，也就是对 timeout 做 bootstrapping。

- 这可以理解成：

    - 真摔倒：这一局真的结束，价值可以截断；

    - 时间到：只是裁判吹停这一段录像，机器人本来还能继续走，所以要把“还能继续走”的价值补回来。

## 当前项目里的对应实现

- 代码观察：`/home/kk/legged_wbc_mjlab/rsl_rl/env/vec_env.py` 在 `extras` 文档里明确要求环境可以返回 `time_outs`，并说明它表示由时间限制导致的 termination，而不是真正 terminal state。

- 代码观察：`/home/kk/legged_wbc_mjlab/rsl_rl/algorithms/ppo.py:141-158` 里，`process_env_step()` 先 clone reward，然后在 `extras` 包含 `time_outs` 时，把 `gamma * transition.values * time_outs` 加到 reward 里。

- 代码观察：`/home/kk/legged_wbc_mjlab/src/config/go2/go2_config.py:12-13` 里保留了 `send_timeouts = True` 和 `episode_length_s = 20`，这和论文里强调的 timeout 处理逻辑是对齐的。

- 这里要特别注意：timeout bootstrapping 不是一个可有可无的小 trick。论文 Appendix A.2 里对比了有无 timeout bootstrapping，结果显示不做 bootstrapping 时 critic loss 更高，总 reward 更低；加入后在 flat 和 rough terrain 上 reward 大约提升 10% 到 20%。

# Game-Inspired Curriculum：把训练变成闯关

## Terrain 设计

- 论文使用五类 procedural terrain：

    - flat terrain；

    - sloped terrain；

    - randomly rough terrain；

    - discrete obstacles；

    - stairs。

- Terrain 被组织成 8 m 边长的方形 tile。机器人从 tile 中心开始，接收 randomized heading 和 linear velocity command，然后尝试穿过对应地形。

- 其中几个具体难度范围很重要：

    - rough terrain 高度变化约 0.1 m；

    - slope 最大到 25 deg；

    - stairs 宽度 0.3 m，高度最高 0.2 m；

    - discrete obstacles 高度最高到 $\pm 0.2$ m。

- Slopes 和 stairs 采用 pyramid 形式组织，使机器人可以从不同方向 traversable。

## 升降级规则

- 每个机器人都有一个 terrain type 和一个 difficulty level。

- 如果机器人成功走过当前 terrain tile 的边界，就说明当前 level 太简单，下次 reset 时升到更难 level。

- 如果 episode 结束时，机器人移动距离小于目标速度要求距离的一半，就说明当前 level 太难，下次 reset 时降到更简单 level。

- 如果机器人已经能解决最高 level，它会被 loop back 到一个随机 level。这个设计不是惩罚，而是为了保持数据多样性，避免策略只在最高难度附近过拟合，忘掉简单地形上的稳定行为。

- 用一个最小例子看：

    - Setup：terrain tile 边长是 8 m，机器人从中心出发。

    - Known values：如果 command 要求它以 0.5 m/s 前进，episode 是 20 s，那么目标路程约 10 m。

    - Operation：如果机器人走出 tile 边界，升难度；如果只走了不到 5 m，也就是目标路程的一半，降难度。

    - Interpretation：curriculum 不需要人工判断“现在该给 10 cm 还是 15 cm 台阶”，它直接用机器人完成度调节。

    - Generalization：几千个机器人并行时，这套规则天然形成一张能力分布图，训练进度能从机器人在地图上的分布看出来。

## 当前项目里的对应实现

- 代码观察：`/home/kk/legged_wbc_mjlab/src/tasks/locomotion/go2_ppo/mdp/curriculums.py:49-67` 的 `terrain_levels_vel()` 基本就是这个思想：

    - 先计算机器人当前位置和环境原点之间的平面距离；

    - `move_up = distance > terrain_generator.size[0] / 2`；

    - `move_down = distance < ||command_xy|| * max_episode_length_s * 0.5`；

    - 然后调用 `terrain.update_env_origins(env_ids, move_up, move_down)` 更新地形等级。

- 代码观察：`/home/kk/legged_wbc_mjlab/src/tasks/locomotion/go2_ppo/config/env_cfgs.py:75-76` 在 rough terrain 配置中打开 `terrain_generator.curriculum = True`。

- 这说明当前项目不是只在 README 里说自己参考了 legged_gym，而是在地形课程训练和 timeout 处理这些关键训练 mechanics 上，都保留了这篇论文的思路。

# Observation、Action、Reward：简单接口反而是重点

## Observation

- policy 的 observation 由两类信息组成：

    - robot proprioception；

    - base 周围的 terrain information。

- 论文列出的 observation 包括：

    - base linear velocity；

    - base angular velocity；

    - gravity vector measurement；

    - joint positions；

    - joint velocities；

    - previous actions；

    - base 周围采样的 108 个 terrain height measurements。

- 每个 terrain height measurement 是 terrain surface 到 robot base height 的距离。

- 这意味着这篇论文不是 DreamWaQ 那种 purely proprioceptive 路线。它是 perceptive locomotion，只是 perception 接口非常轻量：给 policy 一个局部 height scan，而不是完整复杂视觉 pipeline。

## Action

- policy 输出的是 desired joint positions。

- 底层 motor controller 再通过 PD controller 产生 torque。

- 对 ANYmal 的 series elastic actuator，作者还用了一个 actuator network 去近似复杂 actuator dynamics。论文里说他们把 actuator model 简化成只吃当前 measurement 的 LSTM，而不是像之前工作那样拼很多固定历史步给 feed-forward network。

- 这个选择很工程化：policy 不直接输出 torque，可以降低真实硬件上的高频不稳定风险；actuator network 处理 motor dynamics，让仿真和真实执行器之间更接近。

## Reward

- 总 reward 是 9 个 terms 的 weighted sum。核心目标是 tracking commanded velocities，同时惩罚不希望的身体速度、过大力矩、关节加速度、action change 和 collision。

- Appendix A.3 里给出 reward terms。记 $\phi(x) = \exp(-\|x\|^2 / 0.25)$，主要包括：

    - linear velocity tracking：$\phi(v^*_{b,xy} - v_{b,xy})$，权重 $1dt$；

    - angular velocity tracking：$\phi(\omega^*_{b,z} - \omega_{b,z})$，权重 $0.5dt$；

    - vertical linear velocity penalty：$-v_{b,z}^2$，权重 $4dt$；

    - roll/pitch angular velocity penalty：$-\|\omega_{b,xy}\|^2$，权重 $0.05dt$；

    - joint motion penalty：$-\|\ddot q_j\|^2 - \|\dot q_j\|^2$，权重 $0.001dt$；

    - joint torques：$-\|\tau_j\|^2$，权重 $0.00002dt$；

    - action rate：惩罚 target joint position 的变化率，权重 $0.25dt$；

    - collisions：$-n_{collision}$，权重 $0.001dt$；

    - feet air time：鼓励更长 step，权重 $2dt$。

- 膝盖、小腿、脚和垂直面的异常接触算 collision；base 接触算 crash，并触发 reset。

- 论文特别强调两点：

    - reward function 和 action space 没有 gait-dependent elements；

    - 单一 policy 在所有 terrain 上使用同一套 reward。

- 这点非常关键。作者不是靠“告诉机器人应该小跑”来拿结果，而是让策略在简单 reward 和地形压力下自己收敛出 gait。实验中最终常常收敛到 trotting gait，但这个 gait 不是显式写死的。

# Sim-to-Real Additions：快速训练不等于忽略真实差距

- 论文虽然强调训练很快，但并没有忽略 sim-to-real。

- 为了让策略能上真实 ANYmal，作者加了几类训练扰动：

    - ground friction randomization：每个机器人 friction coefficient 从 $[0.5, 1.25]$ 均匀采样；

    - observation noise：噪声尺度基于真实机器人数据；

    - random push：每 10 s 推一次机器人，在 x/y 方向给 base 最多 $\pm 1$ m/s 的扰动速度；

    - actuator network：近似 ANYmal series elastic actuator 的复杂动力学。

- Appendix A.5 里列出的 observation noise 大致包括：

    - joint positions：$\pm 0.01$ rad；

    - joint velocities：$\pm 1.5$ rad/s；

    - base linear velocity：$\pm 0.01$ m/s；

    - base angular velocity：$\pm 0.2$ rad/s；

    - projected gravity：表格抽取为 $\pm 0.05$；

    - measured terrain heights：$\pm 0.1$ m。

- 注意：这不是“训练快，所以不需要 randomization”。恰好相反，训练快让作者可以更频繁地迭代 randomization 和 reward 权重，最后保留必要组件。

# PPO 训练参数：为什么这些数字后来很熟悉

- Appendix A.4 给出的 PPO hyper-parameters 是这篇论文和后续 legged_gym 工程最直接的连接点之一。

- 测试 policy 使用：

    - batch size：98304，也就是 $4096 \times 24$；

    - mini-batch size：24576，也就是 $4096 \times 6$；

    - number of epochs：5；

    - clip range：0.2；

    - entropy coefficient：0.01；

    - discount factor $\gamma$：0.99；

    - GAE discount factor $\lambda$：0.95；

    - desired KL-divergence：0.01；

    - learning rate：adaptive。

- adaptive learning rate 的规则是：

$$
kl \leftarrow KL(\pi_{new}, \pi_{old})
$$

- 如果：

$$
kl > 2kl^*
$$

- 就把学习率降低：

$$
\alpha \leftarrow \max(10^{-5}, \alpha / 1.5)
$$

- 如果：

$$
kl < 0.5kl^*
$$

- 就把学习率升高：

$$
\alpha \leftarrow \min(10^{-2}, 1.5\alpha)
$$

- 直觉上，这就是一个 PPO 的“步子大小控制器”：

    - KL 太大，说明新策略离旧策略太远，容易训练不稳，所以降学习率；

    - KL 太小，说明更新太保守，浪费样本，所以升学习率；

    - `desired_kl = 0.01` 就是这个控制器的目标区间。

- 当前项目里的对应点很直接：

    - 代码观察：`/home/kk/legged_wbc_mjlab/src/config/go2/go2_config.py:96-112` 使用 `clip_param=0.2`、`entropy_coef=0.01`、`num_learning_epochs=5`、`num_mini_batches=4`、`schedule='adaptive'`、`gamma=0.99`、`lam=0.95`、`desired_kl=0.01`、`num_steps_per_env=24`。

    - 代码观察：`/home/kk/legged_wbc_mjlab/src/tasks/locomotion/go2_ppo/config/rl_cfg.py:33-52` 的新配置接口同样保留了这些 PPO 参数。

    - 代码观察：`/home/kk/legged_wbc_mjlab/rsl_rl/algorithms/ppo.py:241-263` 里实现了基于 KL 的 adaptive learning rate 逻辑。

# Simulation Throughput：快在哪里，瓶颈在哪里

## 吞吐来源

- Appendix A.1 里，作者把 environment step 的耗时拆开看。

- 主要结论是：

    - simulation 是最耗时的部分，并且随机器人数量缓慢增加；

    - observation 和 reward computation 是第二慢的部分，也随机器人数量增加；

    - policy inference 和 actuator network inference 的耗时几乎保持常数；

    - 增加并行机器人数量会减少收集固定样本数所需的时间，但 learning step 和机器人数量基本无关。

- 这说明一个很关键的工程事实：当网络推理不再是瓶颈时，**contact、terrain、reset、reward/obs vectorization 才是大规模 legged RL 的真正成本中心。**

## VRAM 成本

- 对 batch size $B = 98304$ 的 rough terrain 训练，论文报告：

    - 4096 robots + rendering enabled 需要约 9 GB VRAM；

    - 4096 robots + no rendering 需要约 6 GB VRAM；

    - flat terrain 下约为 7 GB 和 5 GB。

- 这个数字的工程意义很强：4096 env 不是只能在超级集群上跑，单张高端 GPU 就能做。训练速度变快以后，算法调参从“等一天看结果”变成“半小时内反复试”。

## Time step 和 contact handling

- 论文在 A.1.1 提到，policy step 运行在 50 Hz，每个 policy step 内要跑多个 actuator/simulator step 来保证稳定。最终设置对应 0.005 s simulation step，也就是每个 50 Hz policy step 有 4 个内部 step。

- 这里 PDF 文本抽取中的英文表述容易让人误读，但工程含义很清楚：inner simulation step 的数量不能随便减少，否则 actuator network 或仿真稳定性会出问题；但 inner step 太多又会直接拖慢训练。

- Contact handling 也被专门优化：

    - 机器人模型只保留必要 collision bodies，比如 feet、shanks、knees、base；

    - terrain 用低分辨率 height field 转 triangle mesh，并修正 vertical surfaces，避免高分辨率 height field 带来的接触成本；

    - PhysX 即使忽略机器人之间的 contact，也仍然会检测 potential contact pairs，所以机器人在地形中的空间摆放会影响计算量；

    - 训练初期机器人集中、摔倒多、contact 多，后期机器人分散、base/knee contact 少，simulation time 可以有约 2 倍差异。

- 这解释了为什么大规模并行不是只改 `num_envs`。如果 collision geometry、terrain mesh、机器人间距没有处理好，4096 env 可能只是让 contact bottleneck 更快暴露出来。

# 实验结果：

## Massive parallelism 的 trade-off

- 作者先用一个 baseline：$n_{robots}=20000$，$n_{steps}=50$，batch size 约 1M samples。

- 这个超大 batch 能得到最好 policy，但训练时间更长。

- 接着他们固定 batch size，逐步增加机器人数量。这样每个机器人每次 update 前走的 step 数会减少。

- 实验看到两个现象：

    - robot 数量太高时，performance 会突然下降，因为每个机器人的 time horizon 太短，GAE 和时序信息不够；

    - robot 数量太低时，training time 更长，而且样本之间相似度更高，因为同一机器人连续步太多，数据多样性不足。

- 这就是一个典型的中间最优区间。论文结论是：对这个任务来说，2048 到 4096 robots，配合约 100k 或 200k batch size，是训练时间和最终性能之间最好的 trade-off。

- 注意这个结论不是普适常数。换机器人、换仿真器、换 terrain、换 reward、换 policy architecture，都可能改变最优点。但它给了我们一个很好的起点：不要一上来追求最多环境，也不要退回很小并行度。

## 复杂地形仿真表现

- 最终 simulation/deployment policy 使用：

    - 4096 robots；

    - batch size 98304；

    - 1500 policy updates；

    - under 20 minutes training time。

- 仿真评估使用 robustness 和 traversability tests。机器人接收 0.75 m/s forward velocity command，side velocity 在 $[-0.1, 0.1]$ m/s 内随机。

- 成功定义是：机器人能穿过 terrain，同时 base 不发生接触。

- 结果大致是：

    - stairs：最高训练难度 0.2 m step 附近仍接近 100% success rate；

    - randomized obstacles：更难，success rate 随高度增加持续下降；

    - slopes：超过 25 deg 后基本不能继续向上爬，但仍能以中等 success rate 滑下去。

- 论文还提到，policy 自由采用任何 gait，最终通常收敛到 trotting gait。但如果 reward 权重没调好，会出现拖腿、base height 过高或过低等 artifacts。经过 reward tuning 后，作者得到可以迁移到真实机器人的策略。

## 跨机器人泛化

- 作者还验证同一训练设置在多个机器人上的适用性：

    - ANYmal C with a fixed arm：增加约 20% 重量，不改 reward 或 PPO 超参数也能训练出类似表现；

    - ANYmal B：尺寸相近但动力学和运动学不同，同样能不改 reward/算法超参数训练；

    - Unitree A1：尺寸更小、重量约低 4 倍、腿部构型不同，需要去掉 ANYdrive actuator model、降低 PD gains 和 torque penalties、改变默认关节姿态；

    - Cassie：双足机器人，需要额外加入鼓励单脚站立的 reward，才能形成 walking gait。

- 这个实验说明方法有一定通用性，但也说明“同一套配置完全无脑迁移到所有机器人”并不成立。机器人形态差异越大，需要调整的物理先验越多。

## 真实机器人部署

- 真实 ANYmal C 部署时，policy 是固定的。系统从机器人传感器计算 observation，送入 policy，然后直接把 action 作为 target joint positions 发给 motors。

- 论文明确说，没有使用额外 filtering 或 constraint satisfaction checks。

- Terrain height measurements 来自机器人用 LiDAR scan 构建的 elevation map。

- 真实部署中最大问题来自 height map 不完美。尤其在高速时，terrain mapping 或 state estimation drift 会降低鲁棒性。因此真实硬件上作者把最大线速度 command 降到 0.6 m/s。

- 真实实验展示了机器人可以上下楼梯，并动态处理障碍。

- 但作者也很诚实地说，像文献 [19] 那样使用 teacher-student setup 可以处理 imperfect terrain mapping 和 state estimation drift，并获得更强鲁棒性。未来工作会考虑把两条路线结合。

# 这篇论文最值得注意的几个点：

## 1. 它不是“PPO 参数调快一点”，而是训练系统整体重构

- 如果只看结果，很容易把论文理解成：Isaac Gym 很快，所以 PPO 训练变快。

- 但真正有价值的是，作者把整个训练 pipeline 都按 GPU 并行重新组织了：

    - 数据收集在 GPU 上批量做；

    - reward/observation vectorized；

    - terrain 预生成成大地图；

    - curriculum 只移动机器人位置，不频繁重建 terrain；

    - PPO 重新平衡 env 数和 horizon；

    - timeout 单独 bootstrapping；

    - contact geometry 专门优化。

- 所以这篇论文的贡献不是单点 trick，而是把 legged RL 训练变成了一个高吞吐工程系统。

## 2. `num_envs` 太高会让 on-policy 学习失去时间感

- 很多人会觉得并行环境越多越好，因为样本更多。

- 论文提醒我们：对 on-policy PPO 来说，如果 batch size 固定，`num_envs` 越大，每个 env 的连续 rollout 就越短。

- 连续 rollout 太短时，GAE 估计会失效，策略只看到很多短片段，却看不到动作在 0.5 s 或更长时间内如何影响身体动态。

- 对 locomotion 来说，这一点非常关键。走路不是一帧一帧独立决策，接触、摆腿、落脚、身体回正都有时间结构。只给极短片段，actor/critic 很难学到稳定节奏。

## 3. Timeout bootstrapping 是 critic 语义问题，不是代码细节

- 在 Gym 老接口里，`done=True` 同时混合了成功、失败、时间到等含义。

- 这在很多短 episode benchmark 里可能还能糊过去，但在 legged locomotion 里会直接污染 value learning。

- 论文的做法是修改标准 Gym interface，以便检测 timeout 并做 bootstrap。

- 对我们写环境来说，这意味着：

    - `terminated` / crash / illegal contact 要明确；

    - `truncated` / timeout 要明确；

    - algorithm 层必须能收到 timeout 信息；

    - storage 和 return computation 不能只依赖一个模糊的 `done`。

- 这点和后来的 Gymnasium API 把 `terminated` 和 `truncated` 分开，其实是同一个方向。

## 4. Curriculum 的强点是低开销，而不是复杂

- 这篇论文的 curriculum 并不复杂。它没有训练一个 generator，也没有用复杂指标估计地形难度。

- 它就是用：

    - 走出 tile，升难度；

    - 走不到目标距离一半，降难度；

    - 最高难度过了，随机回流。

- 但这个设计非常适合几千个机器人。因为每个 robot reset 时只要换一个 origin 或 terrain level，不需要为它重新生成一张地形。

- 对大规模并行系统来说，低开销往往比“很聪明”更重要。一个每步都要复杂计算的 curriculum，可能会把 Isaac Gym 好不容易省出来的时间又吃回去。

## 5. 快速训练让 reward 调参方式变了

- 论文结论里有一句很关键：因为训练很快，作者可以做很多 training runs，简化 setup，只保留 essential components。

- 这意味着训练速度不是单纯节省时间，而是改变了研究和工程迭代方式。

- 以前一次训练要 2 天，我们会不敢动 reward，不敢系统做 ablation。

- 现在 20 分钟能得到一个 rough terrain policy，我们就可以：

    - 快速测试 reward 权重；

    - 快速测试 terrain 难度；

    - 快速排除无效 randomization；

    - 针对特定环境 scan 定制训练。

- 这也是为什么这篇论文影响很大：它把 legged RL 从“慢实验”推向了“快速工程循环”。

# 和我之前笔记里几个方法的关系：

## 和 HIMLoco / DreamWaQ 的关系

- HIMLoco 和 DreamWaQ 更关注 POMDP 下的隐藏状态恢复。

- 它们会问：

    - 没有完整地形和外界扰动信息时，policy 怎么从历史观测里恢复内部状态？

    - velocity、latent、context reconstruction、contrastive learning 这些表征怎么帮助 proprioceptive locomotion？

- 这篇论文问的问题更底层：

    - 我们怎样把一个 legged locomotion policy 快速训练出来？

    - PPO 在几千环境下应该怎么跑？

    - terrain curriculum 和 timeout 怎么处理？

- 关系可以这样理解：

    - `Learning to Walk in Minutes` 提供高吞吐训练地基；

    - DreamWaQ/HIMLoco 在这个地基上进一步处理“部署时看不到地形/扰动”的信息恢复问题。

- 论文自己也在 sim-to-real transfer 部分提到，未来可以和 teacher-student setup 结合，以处理 imperfect terrain mapping 和 state estimation drift。这和 DreamWaQ/HIMLoco 的方向是接上的。

## 和 AMP_for_hardware 的关系

- AMP_for_hardware 关心的是：不用复杂手写 reward，能不能用运动先验让真实机器人动作自然且可部署。

- 这篇论文反过来证明：即使用比较直接的 tracking/regularization reward，只要训练系统足够快，仍然可以在短时间内得到可部署 locomotion policy。

- 两者共同点是都非常重视真实硬件：

    - AMP 用 discriminator style reward 限制动作自然性；

    - 这篇论文用 torque/action/collision/air-time 等 reward 加上 actuator model 和 randomization 限制硬件不可行行为。

- 区别是：

    - AMP 的核心是 motion prior；

    - 这篇论文的核心是 training throughput 和 PPO/curriculum engineering。

## 和 CTS-MoE / 多任务地形方法的关系

- CTS-MoE 这类方法更关注“多种地形/多任务 reward 之间如何不互相干扰”。

- 这篇论文的 terrain curriculum 虽然覆盖多个 terrain types，但它仍然训练一个单一 policy，用同一套 reward 处理所有地形，没有显式 task-specific multi-critic 或 MoE router。

- 所以它更像一个 baseline 地基：先证明单策略 + 自动课程 + 大规模并行可以很强；再往后，如果地形类型更多、任务差异更大，就可能需要 CTS-MoE 那种显式分工机制。

## 和 BFM-Zero 的关系

- BFM-Zero 关心的是 promptable behavioral foundation model，也就是用 latent prompt 表达 motion tracking、goal reaching、reward optimization 等多种任务。

- 这篇论文没有引入 prompt 或 foundation model 概念，但它提供了一个非常重要的前提：**底层行为策略训练可以被压缩到很快。**

- 如果未来要训练更通用的 behavior foundation policy，训练吞吐仍然是硬瓶颈。BFM-Zero 需要更复杂的数据和目标，而这篇论文告诉我们：先把 simulator/policy/reward/storage 的吞吐做好，才有资格谈更大规模的行为学习。

# 对当前 legged_wbc_mjlab 的工程启发：

## 1. 不要随便改掉 timeout 通道

- 当前项目中，`send_timeouts = True` 和 PPO 里的 timeout reward bootstrap 是关键机制。

- 如果后续换环境接口、换 `mjlab` wrapper、换 runner，一定要确认 `extras['time_outs']` 是否还正确传到 algorithm。

- 最危险的情况是：环境还能训练，看起来没有报错，但 timeout 被当成普通 done。这样 critic 会悄悄变差，reward 曲线可能变慢、变低或更不稳定。

## 2. `4096 x 24` 是一个合理起点，不是神圣常数

- 当前项目 Go2 配置里也用了 4096 env 和 24 steps。

- 这个设置和论文 Table 3 对齐，但工程上仍要按当前仿真器和机器人调整。

- 如果训练不稳定，可以优先检查：

    - 总 batch size 是否足够；

    - 每个 env 的 horizon 是否太短；

    - mini-batch size 是否过小；

    - KL adaptive LR 是否频繁触发上下震荡；

    - timeout 是否正确 bootstrap；

    - reward 是否因为 terrain curriculum 难度变化而不能直接横向比较。

## 3. Curriculum 先保持简单

- 当前项目 `terrain_levels_vel()` 的升降级规则已经很接近论文。

- 如果要扩展 terrain curriculum，建议先不要一上来加复杂策略。

- 更稳的顺序是：

    - 先保证每个 terrain type 的 level 单调可控；

    - 再检查 `move_up` / `move_down` 的统计分布；

    - 再看最高 level 是否有 loopback 或足够随机化；

    - 最后才考虑 task-specific curriculum 或更复杂的采样器。

- Curriculum 的目标不是让训练曲线一直涨。因为 policy 变强后会被送到更难地形，episode reward 可能暂时下降。所以评估时要把 curriculum 关掉或使用固定难度测试集。

## 4. Reward 不要只追速度 tracking

- 论文能上真实 ANYmal，不是只靠 velocity tracking reward。

- torque、joint acceleration、action rate、collision、base crash、feet air time 这些项都在约束动作形态。

- 对真实机器人尤其要保留硬件友好的 regularization：

    - 限制力矩；

    - 限制关节加速度；

    - 限制 action 抖动；

    - 明确非法接触；

    - 把 base contact 当成 crash。

- 如果只追 tracking reward，仿真里可能跑得很快，但真实硬件上容易出现发热、抖动、打滑、结构碰撞。

## 5. 地形感知链路是部署短板

- 这篇论文真实部署最大问题来自 terrain height map 不完美。

- 这对当前工程很有提醒：如果 actor 依赖 height scan，那么真实部署时就必须审查：

    - scan frame 是否和 base frame 对齐；

    - height map 延迟多大；

    - terrain scan 是否有 blind spot；

    - 机器人快速运动时 elevation map 是否漂移；

    - height noise 和训练 randomization 是否覆盖真实误差。

- 如果这些问题解决不了，就要考虑 DreamWaQ/HIMLoco 那种 proprioceptive latent 或 teacher-student/adaptation 路线，把外部感知误差对底层 policy 的影响降下来。

## 6. 接触和碰撞几何会决定训练速度

- 论文 Appendix A.1.2 很适合当前项目调性能时参考。

- 如果训练吞吐不理想，不要只看 neural network。还要检查：

    - robot XML/MJCF 里 collision geom 是否过多；

    - foot、shank、knee、base 之外的碰撞体是否必要；

    - terrain mesh resolution 是否过高；

    - 多环境间距是否导致不必要 contact detection；

    - reset 初期是否因为大量摔倒导致 contact 计算暴涨。

- 对 legged RL 来说，仿真性能优化很多时候就是 contact 性能优化。

# 如果要在当前项目里复现这篇论文思想，大概有哪些模块：

## Environment / MDP

- 需要有一个 velocity tracking locomotion task。

- 输入至少包括：

    - base linear/angular velocity；

    - projected gravity；

    - joint position / velocity；

    - previous action；

    - command；

    - rough terrain 下的 local height scan。

- 输出是 12 维 joint position target，对应四条腿每条 3 个关节。

- Termination 必须区分：

    - illegal contact / crash；

    - timeout。

## PPO / Storage

- Storage shape 要能稳定支持 `[num_steps_per_env, num_envs, ...]`。

- 对当前论文设置，就是 `[24, 4096, ...]`。

- PPO 要支持：

    - clipped surrogate objective；

    - value loss；

    - entropy bonus；

    - GAE；

    - adaptive LR by KL；

    - timeout bootstrapping。

- 当前项目已经有这些基础：

    - `/home/kk/legged_wbc_mjlab/rsl_rl/storage/rollout_storage.py` 用 `[num_transitions_per_env, num_envs]` batch_size 组织 rollout；

    - `/home/kk/legged_wbc_mjlab/rsl_rl/algorithms/ppo.py` 实现 GAE、adaptive KL、timeout reward bootstrap。

## Terrain Curriculum

- 需要在 terrain generator 里把 terrain type 和 level 铺成二维地图。

- 每个 env reset 时，根据 env 的 terrain level 放置 origin。

- Curriculum term 读取机器人移动距离和 command，计算 move_up/move_down。

- 评估时要能关闭 curriculum，用固定 terrain difficulty 测 success rate。

## Sim-to-real / Events

- 至少需要：

    - friction randomization；

    - observation noise；

    - random push；

    - actuator delay 或 actuator model；

    - torque/action smoothness penalty；

    - 硬件部署前的速度上限和安全限制。

- 未验证：我没有运行当前项目训练，也没有确认 `mjlab` 当前 rough terrain/event 配置已经完整覆盖论文的所有 sim-to-real additions。

# 局限和风险：

- **第一点：目标不是最强鲁棒性。** 作者自己说，这篇论文目的不是得到绝对最强、最鲁棒的 policy，而是证明可部署策略可以在极短时间内训练出来。如果要追求极端真实鲁棒性，还需要引入更多技术。

- **第二点：依赖高质量仿真吞吐。** 方法强依赖 Isaac Gym 端到端 GPU pipeline。如果换成 CPU-heavy simulator，或者 contact/reward/obs 不能 vectorize，训练时间优势会明显下降。

- **第三点：perceptive policy 依赖 height map。** 真实部署里 height map 不完美已经造成鲁棒性下降，并迫使作者把真实最大速度降到 0.6 m/s。这说明外部感知链路仍然是系统短板。

- **第四点：reward tuning 仍然存在。** 论文虽然说 reward/action space 简单，但也承认会出现拖腿、base height 不合理等 artifacts，需要调 reward weights 才能得到可迁移 policy。

- **第五点：4096 env 的最佳性不是普适结论。** 2048 到 4096 robots 是该任务、该硬件、该仿真器、该 PPO 设置下的 trade-off。换到 MuJoCo/MJLab、Go2、带机械臂平台或不同 terrain，仍然要重新测吞吐和学习曲线。

- **第六点：没有形式化安全保证。** 论文真实部署时没有额外 filtering 或 constraint satisfaction checks，但这不等于真实硬件部署可以省掉安全工程。真实机器人仍然需要 torque/velocity/current limits、急停、watchdog、NaN/Inf 保护、通信超时和分阶段验证。

# 论文的核心贡献：

- **第一点：证明 real-world legged locomotion policy 可以分钟级训练。** Flat terrain policy under 4 minutes，uneven terrain policy under 20 minutes，并且能迁移到真实 ANYmal。

- **第二点：系统分析 massively parallel PPO 的关键 trade-off。** 论文不是只报一个快的结果，而是分析了 robot 数量、batch size、rollout horizon、训练时间和最终性能之间的关系。

- **第三点：提出低开销 game-inspired terrain curriculum。** 通过 terrain level 升降和 loopback，适配几千机器人并行训练，不需要额外 tuning 或复杂生成器。

- **第四点：强调 timeout bootstrapping。** 在短 rollout、多 batch、长 episode 的设置下，timeout 处理直接影响 critic loss 和 reward。

- **第五点：给出可复用的 legged_gym 工程基线。** 论文开源训练代码，后续很多 legged locomotion 项目都沿用了它的配置风格和训练结构。

# 我觉得最值得学的地方：

## 1. 训练速度本身是一种算法能力

- 很多时候我们会把训练速度当成工程优化，把 policy performance 当成算法结果。

- 这篇论文提醒我们：训练速度会反过来改变算法开发方式。

- 当一次训练从 2 天变成 20 分钟，我们能做的事情完全不同：

    - 可以系统 sweep reward；

    - 可以做更多 ablation；

    - 可以为特定真实场景快速定制 terrain；

    - 可以更快发现 sim-to-real 缺口。

- 所以 high-throughput training 不是“跑得快一点”，而是让研究闭环变短。

## 2. 大规模并行的关键不是 sample quantity，而是 sample geometry

- 同样的 batch size，可以来自：

    - 少量机器人，每个走很长；

    - 大量机器人，每个走很短。

- 两者给 PPO 的数据几何完全不同。

- 少量机器人会让样本时间相关性太强，多样性不够；大量机器人会让单个 trajectory 太短，GAE 失去时间结构。

- 论文真正做的是在这两者之间找平衡，而不是单纯最大化 samples per second。

## 3. 好的 curriculum 应该让训练进度可见

- Fig. 3 里 4000 个机器人在 terrain map 上从低难度走到高难度，这个可视化本身就很有价值。

- 它让我们能直接看到：

    - 哪类 terrain 学得快；

    - 哪类 terrain 卡住；

    - policy 是否已经覆盖最高难度；

    - 是否出现某些地形被遗忘。

- 一个好的 curriculum 不只是自动调难度，还应该提供诊断信号。

## 4. 简单 reward 不等于没有先验

- 论文强调没有 gait-dependent action/reward，也没有 motion primitives。

- 但系统里仍然有很多先验：

    - velocity command tracking；

    - vertical/base angular velocity penalty；

    - torque/action smoothness；

    - collision/crash 定义；

    - feet air time；

    - terrain curriculum；

    - actuator model；

    - domain randomization。

- 这些不是坏事。对真实机器人来说，完全无先验往往只是把安全和硬件约束藏进失败实验里。更好的做法是把必要先验写清楚，然后让 RL 在这些边界内学习。

# 作者 claim、代码观察和我的推断：

- **作者 claim：** 论文明确声称，单台 workstation GPU 上的大规模并行训练可以在 4 分钟以内得到 flat terrain policy，在 20 分钟以内得到 uneven terrain policy，并能迁移到真实 ANYmal C 上做楼梯和障碍行走。

- **作者 claim：** 论文明确声称，massively parallel on-policy PPO 不能只把 `nrobots` 拉满；需要同时控制 $B=n_{robots}n_{steps}$、保留足够短时 horizon、使用较大 mini-batch，并把 timeout 和真正 terminal 分开。

- **作者 claim：** 论文明确声称，game-inspired curriculum 不需要额外 tuning，可以根据机器人是否越过 terrain tile 边界、是否走到目标距离一半以上，自动升降 terrain level；最高 level 通过后随机回流到较低 level，用来增加多样性并缓解 catastrophic forgetting。

- **代码观察：** 当前项目中 `rsl_rl/env/vec_env.py`、`rsl_rl/algorithms/ppo.py`、`src/config/go2/go2_config.py`、`src/tasks/locomotion/go2_ppo/mdp/curriculums.py`、`src/tasks/locomotion/go2_ppo/config/rl_cfg.py` 的静态只读检查，能看到 timeout 通道、PPO 自适应 KL、`4096 x 24` rollout 和 terrain level curriculum 与论文思路对齐。

- **这里是推断：** 我把这篇论文称为后续 `legged_gym` / `rsl_rl` 路线的“地基论文”，是基于它开源训练代码、参数组合和当前项目中保留的相似 mechanics 做出的工程归纳；这不是论文原文中的措辞。

- **这里是推断：** 对 `legged_wbc_mjlab` 来说，最应该优先保留 timeout 语义、terrain curriculum 和 contact/collision 性能优化，是根据论文结论和当前项目结构推出的工程建议；真实收益还需要跑训练曲线、吞吐统计和部署前安全验证。

# 最容易写错的地方：

- **第一，别把 `nsteps` 写成 episode length。** 论文测试策略用的是 `4096 x 24` batch，但 episode timeout 是 20 s；一个 episode 会跨很多个 PPO update。

- **第二，别把 timeout 当成 crash。** Timeout 是人为时间上限，不是机器人失败；把 timeout 混进普通 `done` 会错误截断 critic 的 bootstrap target。

- **第三，别说 4096 env 永远最优。** 论文结论是对该任务、Isaac Gym、RTX A6000、rough terrain、PPO 参数组合而言，2048 到 4096 robots 配合约 100k/200k batch 是较好 trade-off。

- **第四，别把这篇论文说成 proprioception-only。** Policy observation 里有 108 个 terrain height measurements；它是轻量 perceptive locomotion，不是 DreamWaQ 那种只靠 proprioception history 的路线。

- **第五，别漏掉 mini-batch 的变化。** 论文不只是把 rollout 变短，还使用 24576 这种 tens-of-thousands 级别 mini-batch，并跑 5 个 epoch。

- **第六，别把简单 reward 理解成没有先验。** Reward 没有 gait-dependent term，但仍包含 tracking、base velocity penalty、joint motion、torque、action rate、collision、feet air time、domain randomization 和 actuator model。

- **第七，别把真实部署说成完全无安全层结论。** 论文说部署时 policy 输出直接发为 target joint positions，没有额外 filtering 或 constraint satisfaction checks；这只说明作者实验设置，不等于工程上可以省掉限幅、watchdog、急停和分阶段验证。

# 一句话总结：

- **这篇论文的核心不是“4096 个机器人并行训练很快”，而是证明 legged locomotion 的训练系统可以被重构成一个高吞吐、可调参、可部署的 GPU 闭环；只要同时处理好 PPO horizon、timeout bootstrapping、terrain curriculum、contact 性能和 sim-to-real randomization，真实机器人可用的策略可以在分钟级生成。**

# 我的笔记：

- 我觉得这篇论文是后面很多 legged RL 项目的“地基论文”。它不像 DreamWaQ/HIMLoco 那样提出一个很明确的新 representation，也不像 AMP 那样用运动数据改变 reward 设计，而是把训练这件事本身做成了一个成熟工程系统。

- 对我们当前项目来说，最应该直接吸收的是三个东西：

    - **timeout 语义必须干净。** `done` 不能混用 timeout 和 crash。

    - **curriculum 先简单可诊断。** 能通过机器人分布看出训练进度，比写一个复杂但不可解释的课程采样器更实用。

    - **训练吞吐要和接触建模一起看。** 如果 MJCF collision geom、terrain mesh、contact sensor 配置很重，神经网络再小也救不了训练速度。

- 这篇论文还有一个很实际的态度：作者没有说自己做到了最强鲁棒性，而是说“我们把可用策略训练到足够快”。这个边界很重要。快速训练不是终点，但它让后续所有更复杂的方法都更容易试、更容易错、更容易修。

# 参考来源：

- `/home/kk/下载/2109.11978v3.pdf`

- `/home/kk/legged_wbc_mjlab/paper/2109.11978v3.pdf`

- `/home/kk/飞书文档/论文学习部分.md`

- `/home/kk/飞书文档/AMP_for_hardware学习记录.md`

- `/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`

- `/home/kk/legged_wbc_mjlab/docs/BFM-Zero阅读笔记.md`

- `/home/kk/legged_wbc_mjlab/docs/DreamWaQ阅读笔记.md`

- `/home/kk/legged_wbc_mjlab/docs/Deep-WBC阅读笔记.md`

- 代码观察路径：`/home/kk/legged_wbc_mjlab/rsl_rl/algorithms/ppo.py`、`/home/kk/legged_wbc_mjlab/rsl_rl/env/vec_env.py`、`/home/kk/legged_wbc_mjlab/src/config/go2/go2_config.py`、`/home/kk/legged_wbc_mjlab/src/tasks/locomotion/go2_ppo/mdp/curriculums.py`、`/home/kk/legged_wbc_mjlab/src/tasks/locomotion/go2_ppo/config/rl_cfg.py`
