# CAPS阅读笔记

# 前言：

- 这篇论文的核心目标，是解决深度强化学习控制器在真实硬件上经常出现的一个很实际的问题：**动作输出不够平滑，控制信号里有明显的高频抖动**。

- 论文标题是 **Regularizing Action Policies for Smooth Control with Reinforcement Learning**，方法名叫 **CAPS, Conditioning for Action Policy Smoothness**。它不是重新设计一个更复杂的 reward，也不是在策略输出后面简单接一个 low-pass filter，而是在训练 actor 的时候直接约束策略映射本身，让相近时间、相近状态下的 action 不要跳得太厉害。

- 这个问题对真实机器人很重要。因为在仿真里，一个 policy 即使动作抖一点，也可能照样拿到不错的 reward；但放到真实电机上，高频动作会变成电机发热、功耗上升、机械磨损、控制响应不稳定，严重时甚至会损坏硬件。

- CAPS 的思路很直接：

    - **temporal smoothness：** 相邻时刻的状态 $s_t$ 和 $s_{t+1}$，经过同一个 policy 后不应该输出差异很大的动作；

    - **spatial smoothness：** 原状态 $s_t$ 和一个轻微扰动后的相近状态 $\bar{s}_t$，经过 policy 后也不应该输出差异很大的动作。

- 论文最有价值的地方不是说“动作平滑很重要”这个结论本身，而是把 action smoothness 从 reward engineering 里抽出来，变成了一个可以加到 DDPG、TD3、SAC、PPO 这类连续控制算法里的 policy regularization。

- 这篇论文虽然实验平台是 quadrotor attitude control，不是 legged locomotion，但它和足式机器人控制关系很近。我们平时在 legged RL 里加的 `action_rate`、`joint acceleration`、`torque change`、`smoothness` 惩罚，本质上也在处理同一个硬件问题：**不要让策略为了刷 reward 学出仿真里能跑、硬件上伤机器的动作。**

- 未执行/未验证：这份笔记只基于 `/home/kk/下载/ICRA21_1616_FI.pdf` 的阅读整理，以及本地样例笔记 `/home/kk/飞书文档/论文学习部分.md`、`/home/kk/飞书文档/AMP_for_hardware学习记录.md`、`/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`、`/home/kk/legged_wbc_mjlab/docs/DreamWaQ阅读笔记.md` 的风格参考；没有跑作者代码、没有复现实验，也没有验证真实无人机硬件部署。

# 论文核心信息：

- **论文标题：** Regularizing Action Policies for Smooth Control with Reinforcement Learning

- **方法名称：** CAPS, Conditioning for Action Policy Smoothness

- **作者：** Siddharth Mysore, Bassel Mabsout, Renato Mancuso, Kate Saenko

- **机构：** Boston University Department of Computer Science；Kate Saenko 同时 affiliated with MIT-IBM Watson AI Lab

- **会议：** ICRA 2021

- **任务类型：** continuous control / deep reinforcement learning / robot control / sim-to-real policy transfer

- **主要实验平台：**

    - 1D toy goal tracking environment；

    - OpenAI Gym 连续控制 benchmark：Pendulum-v0、LunarLanderContinuous-v2、Reacher-v2、Ant-v2；

    - 真实 quadrotor drone attitude control。

- **使用到的 RL 算法：** DDPG、TD3、SAC、PPO

- **真实机器人任务：** 训练低层姿态控制器，让无人机跟踪 pilot 输入的三轴期望角速度 command

- **对比对象：** vanilla RL policy、PID controller、Neuroflight controller、只加 temporal smoothing 的 PPO、只加 spatial smoothing 的 PPO

- **关键结果：** 论文报告，在真实无人机上，PPO + CAPS 相比 Neuroflight 将平均电流从 22.87A 降到 4.86A，接近 80% 功耗下降；smoothness score 从 $4.3 \times 10^{-3}$ 降到 $0.16 \times 10^{-3}$，约 96% 改善。

# 研究问题：

- 这篇论文要回答的问题是：

    - **强化学习训练出来的连续控制 policy，能不能在不依赖复杂 reward engineering 和后处理滤波器的情况下，直接学成一个动作平滑、硬件友好的控制器？**

- 这里的关键词是“直接”。传统做法一般有两种：

    - 在 reward 里加入动作变化、能耗、振荡、姿态误差等一堆惩罚项，让 agent 间接学会平滑；

    - 训练完以后在 action 输出后面接滤波器，把高频成分滤掉。

- 但这两种方法都有明显问题。

- **第一，reward engineering 是间接约束。** Policy 真正输出的是 action，但 reward 通常只在环境层面反馈“任务做得好不好”。如果我们想要的是 action mapping 平滑，却把这个要求绕到 reward 里，网络需要通过 surrogate value function 间接理解这个行为偏好。强化学习本来就对 reward scale、初始化、normalization、implementation detail 很敏感，所以这个信号传递不一定稳定。

- **第二，后处理滤波器会改变闭环动力学。** 对经典 PID 来说，滤波器是控制工程里的常用工具。但神经网络控制器通常是在“没有滤波器”的闭环系统里训练出来的。部署时突然加一个 filter，相当于把 policy 熟悉的 dynamics 改掉，它可能会过度补偿，甚至出现更严重的振荡或失控。

- 所以 CAPS 的问题重构是：

    - 不再问“怎么写一个复杂 reward 让动作看起来平滑”；

    - 也不先问“输出后面接什么滤波器”；

    - 而是问：**能不能直接让 actor 的状态到动作映射本身变平滑？**

# 研究背景：

## 为什么 RL controller 容易抖：

- 标准强化学习目标通常只关心累计回报：

    - 到达目标快不快；

    - tracking error 小不小；

    - 是否摔倒；

    - 是否完成任务。

- 但它不天然关心“相邻两帧 action 是否连续”。

- 对连续控制来说，这个漏洞尤其明显。因为 action 是连续值，policy 可以在很小的状态变化下输出完全不同的 motor command。仿真器可能只看到 body tracking 还不错，但真实电机看到的是高频 PWM / thrust / torque 抖动。

- 论文里强调，这类振荡会带来几个硬件后果：

    - 控制响应看起来有高频震荡；

    - 电机功耗上升；

    - 电机过热；

    - 结构件和传动件磨损；

    - 极端情况下出现硬件 failure。

- 这和 legged robot 很像。四足机器人仿真里如果 action 抖动，可能只是 joint target 在高频跳；真实机器人上就会变成电机电流波动、减速器冲击、足端接触不稳、机身小幅震荡。

## 为什么 reward engineering 不够：

- 在很多机器人 RL 工作里，我们会通过 reward 惩罚 action rate、torque、joint acceleration 或 body oscillation。

- 这当然有效，但它有两个限制。

- **第一个限制：reward 需要环境信息。** 有些 smoothness 指标可能依赖环境状态、动力学响应或额外日志。训练环境如果是 black-box，或者接口没有暴露这些量，就很难把它写进 reward。

- **第二个限制：reward 到 actor 是绕路的。** 对 policy gradient 或 actor-critic 来说，actor 不是直接看到“你的映射不平滑”，而是通过 value / advantage / Q function 间接获得优化信号。这个 surrogate objective 和真实任务目标之间可能存在偏差。

- 所以论文的观点是：如果一个行为属性可以直接用 action policy 来定义，那就不要全部绕到 reward 里。平滑控制正好属于这种情况。

## 为什么不是简单加滤波器：

- 直觉上，如果 action 有高频抖动，接一个 low-pass filter 好像就行了。

- 但论文专门做了一个 toy problem 说明：对 neural network controller 来说，滤波器并不一定安全。

- 原因是 policy 训练时学到的是某个闭环系统：

    - 它输出一个 action；

    - 环境立刻按这个 action 响应；

    - 下一帧 observation 再反馈给 policy。

- 如果部署时把 action 先经过 filter，再送进环境，那么 policy 以为自己输出的动作已经作用到系统上，实际系统却只执行了滤波后的动作。这个差异会改变状态转移，policy 可能为了补偿延迟和幅值变化输出更激烈的动作。

- 注意这里不是说 filter 永远不能用，而是说：**filter 不能被当作训练后随手加上的万能补丁。** 如果真的要用 filter，最好在训练闭环里一起建模，并且给 policy 足够的 history，让它知道滤波器引入的动态状态。

- 但这样又会带来输入维度上升、Markov property 处理、表示复杂度增加等问题。CAPS 的价值就是：很多时候我们不必把问题推到输出后处理，而是在 actor 学习阶段就把映射压平滑。

# 核心直觉：

- CAPS 的核心直觉可以用一句话概括：

    - **如果两个状态在时间上很接近，或者在状态空间里很接近，那么 policy 输出的 action 也应该接近。**

- 这其实是在控制 policy 的局部 Lipschitz 行为。

- 对一个 policy：

$$
a = \pi_\theta(s)
$$

- 我们不希望出现这种情况：

$$
s_1 \approx s_2, \quad \text{but} \quad \pi_\theta(s_1) \not\approx \pi_\theta(s_2)
$$

- 因为这意味着 policy 对小扰动过于敏感。

- 在真实机器人上，小扰动永远存在：

    - IMU 噪声；

    - 电机响应延迟；

    - 状态估计误差；

    - 接触模型误差；

    - sim-to-real dynamics gap；

    - command 或 observation 的离散采样误差。

- 如果 policy 对这些小变化反应过激，就会出现“身体还没怎么变，电机命令先乱跳”的现象。

- 所以 CAPS 不是让机器人永远慢吞吞，也不是禁止快速动作，而是要求：**快速变化应该来自真实需要，而不是来自 policy mapping 的不连续和过敏。**

# 问题建模：

- 标准 RL 里，policy $\pi_\theta$ 把状态映射成动作：

$$
a_t = \pi_\theta(s_t)
$$

- 强化学习要优化的是某个 policy objective $J_{\pi_\theta}$。

- 对 DDPG、TD3、SAC 这类基于 Q-learning / actor-critic 的方法，policy optimization 通常和 Q value 有关：

$$
J_{\pi_\theta} \propto Q^{\pi_\theta}(s, \pi_\theta(s))
$$

- 对 PPO / TRPO 这类 policy gradient 方法，优化目标通常和 advantage 以及 policy likelihood 有关：

$$
J_{\pi_\theta} \propto \log(\pi_\theta(a \mid s)) A^{\pi_\theta}(s, a)
$$

- CAPS 不去改 value function 的学习方式，而是在 actor objective 里加入两个 regularization term：

$$
J_{\pi_\theta}^{CAPS} = J_{\pi_\theta} - \lambda_T L_T - \lambda_S L_S
$$

- 这里需要注意符号：论文写的是 maximize objective，所以 smoothness penalty 前面是减号。如果在代码里写成 loss minimization，通常就是把 $\lambda_T L_T + \lambda_S L_S$ 加到 actor loss 上。

# CAPS 方法主线：

## Temporal smoothness：

- temporal smoothness 约束相邻时间步的 action 不要变化过大：

$$
L_T = D_T(\pi_\theta(s_t), \pi_\theta(s_{t+1}))
$$

- 这里 $D_T$ 是 action space 里的距离函数。论文实际使用的是欧氏距离：

$$
D(a_1, a_2) = \|a_1 - a_2\|_2
$$

- 直觉上，这个 loss 在问：

    - 当前状态输出的 motor command 是多少？

    - 下一帧状态输出的 motor command 是多少？

    - 如果系统没有发生巨大变化，为什么 action 会突然跳一下？

- 对机器人控制来说，这个约束非常自然。真实系统有惯性，电机和机体响应都不可能无限快，所以 action 也不应该在每个 control tick 上大幅跳变。

- 但注意：temporal smoothness 不是越强越好。如果 $\lambda_T$ 太大，policy 会变得保守，遇到突发扰动时不敢快速修正，tracking error 可能上升。

## Spatial smoothness：

- spatial smoothness 约束状态空间附近的 action mapping 不要太敏感：

$$
L_S = D_S(\pi_\theta(s_t), \pi_\theta(\bar{s}_t)), \quad \bar{s}_t \sim \phi(s_t)
$$

- 其中 $\bar{s}_t$ 是从 $s_t$ 附近采样出来的扰动状态。论文里使用正态分布：

$$
\phi(s) = \mathcal{N}(s, \sigma)
$$

- $\sigma$ 可以理解成我们认为合理的 measurement noise 或 state tolerance。

- 这一步很像在问：

    - 如果 IMU 噪声让 observation 稍微偏了一点；

    - 如果状态估计器输出有一点误差；

    - 如果仿真和真实动力学不完全一致；

    - policy 是否还会输出大致一致的动作？

- temporal smoothness 更偏“时间连续性”，spatial smoothness 更偏“局部鲁棒性”。

- 论文的假设是：

    - 如果仿真动力学非常准确，只靠 temporal smoothness 可能已经足够让控制信号连续；

    - 但 sim-to-real 里一定有未建模动态和测量噪声，所以 spatial smoothness 对真实迁移很关键。

## 两个项为什么要一起用：

- 只用 temporal smoothing，policy 可能在仿真里动作连续，但遇到真实噪声和 domain shift 时仍然敏感。

- 只用 spatial smoothing，policy 对扰动比较钝感，但可能把 action 压到几个离散 band 里，tracking 变差。

- 两个项一起用，才比较接近论文想要的行为：

    - 相邻时刻不要乱跳；

    - 相近状态不要过敏；

    - 真实部署时不要因为一点 dynamics gap 就输出高频抖动。

# 一个最小例子：

- 论文里用了一个 1D toy goal tracking 问题。

- agent 看到的是当前状态和目标状态之间的差：

$$
s_t = g_t - c_t
$$

- action 会直接影响系统下一步响应。理想情况下，最简单的控制就是：

$$
a_t^* = s_t
$$

- 也就是说，差多少就补多少，得到的是一个平滑的线性映射。

- 但 vanilla RL agent 很容易学成类似 binary step response 的策略：

    - 离目标远一点就用很大的 action 冲过去；

    - 冲过头了再反向大 action 拉回来；

    - 最终 tracking reward 可能还行，但 action 在高频振荡。

- 加了 CAPS 后，policy 更接近理想线性映射，控制信号也更平滑。

- 这个 toy problem 的意义是：**动作不平滑并不一定来自复杂动力学。即使在非常简单的系统里，只要 RL objective 没有直接约束 action mapping，它也可能学出 aggressive、oscillatory 的策略。**

# Smoothness 指标：

- 只看曲线图可以直观看出抖不抖，但论文需要一个可比较的 smoothness metric。

- 传统控制信号分析有时会看频谱里的高频 peak，但神经网络控制器的问题不一定表现为某一个特别突出的 peak，而可能是整个高频段都有不小的能量。

- 所以论文定义了一个基于 FFT 的 smoothness measure：

$$
S_m = \frac{2}{n f_s} \sum_{i=1}^{n} M_i f_i
$$

- 其中：

    - $M_i$ 是第 $i$ 个频率成分的 amplitude；

    - $f_i$ 是第 $i$ 个频率；

    - $f_s$ 是 sampling frequency；

    - $n$ 是频率成分数量。

- 这个指标可以理解为：**按 amplitude 加权后的平均归一化频率。**

- 如果控制信号里高频成分很多，$M_i f_i$ 的贡献就大，$S_m$ 就高；如果控制信号主要集中在低频，$S_m$ 就低。

- 注意：论文也提醒，这个 smoothness score 更适合同一个任务内部比较，不一定适合跨任务直接比较。因为不同任务的 action scale、sampling frequency 和 dynamics 都不同。

# OpenAI Gym 实验：

- 论文先在 4 个常见连续控制环境上测试 CAPS：

    - Pendulum-v0；

    - LunarLanderContinuous-v2；

    - Reacher-v2；

    - Ant-v2。

- 每个环境里测试 DDPG、SAC、TD3、PPO，并比较加 CAPS 前后的 reward 和 $S_m$。

- 结果的总体趋势很清楚：**加 CAPS 后，所有测试任务里的 smoothness score 都下降，也就是动作更平滑。**

- 几个代表性结果：

    - Pendulum 上，TD3 的 $S_m \times 10^3$ 从 43.9 降到 5.92；DDPG 从 47.6 降到 7.09；

    - LunarLanderContinuous 上，TD3 从 37.9 降到 16.7；DDPG 从 34.9 降到 16.7；

    - Reacher 上，PPO 从 4.49 降到 3.38；TD3 从 5.70 降到 4.63；

    - Ant 上，PPO 从 6.09 降到 1.60；DDPG 从 2.73 降到 1.31。

- 但 reward 不总是提升。Pendulum 和 LunarLander 上，某些算法加 CAPS 后 reward 会变差一些。

- 这并不奇怪。因为一个更平滑的 controller 往往不会用最激进的 action 去最快达到目标。它牺牲一点 reward 或 tracking speed，换来更低的高频动作和更友好的硬件行为。

- 对真实机器人来说，这个 trade-off 很重要。仿真 reward 高一点，不一定比“电机不过热、动作不抽搐、能稳定飞/走”更重要。

# 真实无人机实验：

注意：本节真实飞行数据均为论文作者报告结果，本笔记未执行代码复现、未做实机验证，也不构成无人机上机参数或安全流程建议。

## 实验设置：

- 论文真正有说服力的部分，是把 CAPS 放到 quadrotor drone 的 sim-to-real attitude control 上。

- 任务是训练一个低层姿态控制器，跟踪 pilot 输入的三轴角速度 command。

- 论文基于 Neuroflight 的架构和部署 pipeline，对比：

    - tuned PID controller；

    - Neuroflight 里表现最好的已训练 agent；

    - 只用 temporal smoothing 的 PPO；

    - 只用 spatial smoothing 的 PPO；

    - 同时使用 temporal + spatial 的 PPO + CAPS。

- Neuroflight 的问题是：它在仿真里能有不错 tracking，但迁移到真实无人机时会出现明显高频控制信号，导致电机发热、耗电高，并且不是每次训练出来的 agent 都能真实飞行，需要 cherry-pick。

## 关键结果：

- 论文表 II 的真实飞行结果如下：

| Agent | MAE (deg/s) 越低越好 | Current (Amps) 越低越好 | $S_m \times 10^3$ 越低越好 |
| --- | ---: | ---: | ---: |
| PID | 5.01 | 8.07 | 0.4 |
| Neuroflight | 5.19 | 22.87 | 4.3 |
| PPO + Temporal | $7.82 \pm 2.42$ | $7.59 \pm 2.24$ | $1.10 \pm 0.32$ |
| PPO + Spatial | $14.85 \pm 6.85$ | $4.59 \pm 2.70$ | $0.37 \pm 0.22$ |
| PPO + CAPS | $9.28 \pm 2.31$ | $4.86 \pm 2.32$ | $0.16 \pm 0.02$ |

- 这里可以看到一个很典型的工程 trade-off：

    - Neuroflight 的 tracking MAE 是 5.19 deg/s，比 CAPS 的 9.28 deg/s 更低；

    - 但 Neuroflight 的平均电流是 22.87A，而 CAPS 是 4.86A；

    - Neuroflight 的 smoothness score 是 4.3，而 CAPS 是 0.16。

- 也就是说，CAPS 没有拿到最小 tracking error，但它把控制信号从“能飞但非常耗电、抖动明显”变成了“tracking 误差仍在可接受范围内，同时电机负担显著下降”。

- 论文报告，CAPS 训练出的 agents 都是 flight-worthy，并且训练 pipeline 达到 100% repeatability。这里的 repeatability 指的是在这个任务设置下，独立训练出来的 agents 都能达到可飞状态，而不是像 Neuroflight 那样需要挑选少数可用模型。

- 另外，CAPS 在这个任务里 1 million time-steps 内完成训练，相比 Neuroflight 有 90% data intensity reduction 和 8 倍 wall-time speedup。

## Temporal 和 Spatial 的消融：

- 只看表 II，很容易看到两个 regularizer 的性格差异。

- **PPO + Temporal：**

    - tracking 还不错，MAE 是 $7.82 \pm 2.42$；

    - 但真实飞行的 smoothness score 仍有 $1.10 \pm 0.32$；

    - 说明它能让动作时间上更连续，但对真实噪声和 dynamics shift 仍然不够鲁棒。

- **PPO + Spatial：**

    - 电流很低，平均 $4.59 \pm 2.70$A；

    - smoothness 也不错，是 $0.37 \pm 0.22$；

    - 但 MAE 变成 $14.85 \pm 6.85$，tracking 明显变差。

- **PPO + CAPS：**

    - MAE 保持在 10 deg/s 以下；

    - 电流接近 spatial-only；

    - smoothness score 最低。

- 所以这篇论文的经验不是“随便加一个平滑项就行”，而是：**时间连续性和状态局部鲁棒性解决的是两个不同问题，真实 sim-to-real 控制里最好一起考虑。**

# 为什么 CAPS 对 sim-to-real 有用：

- sim-to-real 的核心问题是：训练时的 transition dynamics 和真实系统不完全一样。

- 如果 policy 的 action mapping 非常尖锐，那么一点点 dynamics gap 就可能让 observation 落到另一个局部区域，policy 输出完全不同的 action。这样就会产生连锁反应：

    - action 抖动；

    - 系统响应偏离训练分布；

    - policy 继续过度修正；

    - 控制信号进入高频振荡。

- spatial smoothness 相当于提前训练 policy：

    - “附近状态都算同一类，不要因为一点点测量误差就大幅改变动作。”

- temporal smoothness 相当于提前训练 policy：

    - “真实系统有惯性，连续时刻的 action 要有连续性，不要每一帧都重新激进决策。”

- 这两个约束合起来，就让 policy 更像一个连续控制器，而不是一个每个采样点都独立输出的黑盒分类器。

# 和 reward smoothness 的区别：

- 很多 locomotion 项目里会写类似这样的 reward：

$$
r_{action\_rate} = -\|a_t - a_{t-1}\|^2
$$

- 这和 CAPS 的 temporal smoothness 看起来很像，但优化路径不一样。

- reward 版本的逻辑是：

    - environment 观察到 action change；

    - reward 变小；

    - critic / advantage 学到这个变化；

    - actor 通过 RL objective 间接受到影响。

- CAPS 的逻辑是：

    - actor 当前就计算 $\pi(s_t)$ 和 $\pi(s_{t+1})$；

    - 直接把两者距离作为 policy regularization；

    - 不需要环境额外返回一个 smoothness reward。

- 所以 CAPS 更像是 supervised regularization 或 representation regularization：它直接管网络函数的形状。

- 这里不是说 reward 版本不好。对 legged RL 来说，reward 里的 action rate penalty 仍然很有用，因为它直接和 rollout 里的实际 action 序列绑定，也能表达接触、命令变化、关节负载这些环境相关因素。

- 但 CAPS 给了一个额外视角：**有些硬件友好性约束可以不只写在 reward 里，还可以写在 actor mapping 的函数正则化里。**

# 和滤波器的关系：

- 论文有一个很有意思的问题：为什么不直接 filter action？

- 对经典控制器来说，filter 是很自然的工具。比如 PID 输出有高频噪声，加滤波器可以降低噪声进入 actuator。

- 但神经网络 policy 不是普通线性控制器。它可能已经把训练环境的响应方式“记”进了网络参数里。

- 如果部署时突然加 filter，policy 看到的是：

    - 我明明输出了某个 action；

    - 但下一帧状态没有按我预期变化；

    - 那我需要更用力地修正。

- 这就可能导致过度补偿。

- 论文在 toy problem 上测试了 Median、EMA、FIR 等 filter，发现有的 filter 会导致 overshoot 和 oscillation，FIR 甚至会导致 catastrophic loss of control。

- 更理论一点看，如果 action filter 有内部状态，而 policy observation 里没有包含这段历史，那么 agent 看到的 observation 就不再满足原来的 Markov assumption。要修正这个问题，就需要把 state history 加进 observation，让 policy 知道 filter dynamics。

- 但这样又把问题变复杂了。

- 所以更稳妥的工程理解是：

    - filter 可以作为控制系统设计的一部分；

    - 但不要把它当成训练完成后临时接上的补丁；

    - 如果 policy 最终要带 filter 部署，最好训练时就把 filter 和它的状态一起纳入闭环；

    - CAPS 的优势是不用改变环境 dynamics，就可以先把 actor 本身压平滑。

# 对 legged RL 的工程映射：

- 这篇论文虽然做的是无人机，但对足式机器人很有启发。

- 我们可以把 CAPS 映射到 legged locomotion 的几个位置。

## 1. 对 action target 做 temporal regularization：

- 如果 policy 输出的是 12 维 desired joint position offset，那么 temporal loss 可以写成：

$$
L_T = \|\mu_\theta(o_t) - \mu_\theta(o_{t+1})\|_2^2
$$

- 这里 $\mu_\theta$ 可以理解为 Gaussian policy 的 mean action。

- 注意最好不要跨 episode reset、fall termination 或 command discontinuity 强行算这个 loss。比如机器人摔倒后 reset 到初始状态，如果还把 reset 前后的两帧 action 拉近，就会给 actor 错误约束。

## 2. 对 observation noise 做 spatial regularization：

- spatial loss 可以写成：

$$
\bar{o}_t = o_t + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma)
$$

$$
L_S = \|\mu_\theta(o_t) - \mu_\theta(\bar{o}_t)\|_2^2
$$

- 这对真实机器人上的 IMU 噪声、关节速度噪声、状态估计偏差很有意义。

- 但要注意 observation 的不同维度尺度差异很大：

    - base angular velocity；

    - projected gravity；

    - joint position；

    - joint velocity；

    - command velocity；

    - previous action。

- 所以 $\sigma$ 不能随便设成同一个常数。更合理的做法是基于 normalized observation 空间加噪，或者按传感器物理噪声分别设置。

## 3. 不要把 command 也无脑平滑掉：

- legged policy 往往是 command-conditioned，比如输入期望线速度、角速度、高度或 gait command。

- 如果 spatial smoothing 对 command 维度也加很强扰动，可能会让 policy 学成“不同 command 下动作也差不多”。这会损害 command tracking。

- 所以工程上可以考虑：

    - 对 proprioception 维度加较强 spatial noise；

    - 对 command 维度加较弱 noise，或者不加；

    - 对历史 action 维度谨慎处理，因为它本身已经带有时间信息。

## 4. 接触切换时不能过度追求平滑：

- 足式机器人和无人机不一样，腿式系统有明显的 contact switching。

- 有些动作变化是合理的：

    - swing leg 落地；

    - stance leg 支撑切换；

    - 被推后快速恢复；

    - 踩到台阶边缘时突然调整足端轨迹。

- 如果平滑项太强，policy 可能不敢做必要的快速修正，最后反而更容易摔。

- 所以在 legged locomotion 里用 CAPS，需要把它当作 regularization，不是 hard constraint。它应该限制无意义的高频抖动，而不是抹掉接触相位和扰动恢复所需的动作变化。

# 如果要在 PPO 里实现：

- 这里给一个工程级理解，不是作者代码复现。

- 对 PPO 来说，原始 actor loss 大致来自 clipped surrogate objective。假设代码里是 loss minimization，那么可以把 CAPS 写成：

$$
L_{actor} = L_{PPO} + \lambda_T L_T + \lambda_S L_S
$$

- 需要特别区分：论文原式使用的是欧氏距离 $\|\cdot\|_2$；这里为了表达常见工程 loss 写成平方范数 $\|\cdot\|_2^2$，只是实现层面的变体，不代表原文公式。

- 一个比较自然的实现位置是在 PPO update 阶段，而不是环境 step 阶段。

- 需要的数据：

    - batch observation $o_t$；

    - batch next observation $o_{t+1}$；

    - terminal mask / timeout mask；

    - 当前 actor 对 observation 输出的 action mean $\mu_\theta(o)$。

- temporal loss：

$$
L_T = \mathbb{E}\left[m_t \cdot \|\mu_\theta(o_t) - \mu_\theta(o_{t+1})\|_2^2\right]
$$

- 其中 $m_t$ 是 mask，用来避免跨真正 termination 或 reset 的样本。

- spatial loss：

$$
L_S = \mathbb{E}\left[\|\mu_\theta(o_t) - \mu_\theta(o_t + \epsilon)\|_2^2\right]
$$

- 这里的关键不是公式本身，而是几个工程细节：

    - 最好在 normalized observation 空间里加噪；

    - loss 作用在 actor mean 上通常更稳定，而不是对采样 action 直接做距离；

    - action 如果经过 tanh 或 clip，需要明确是在 pre-squash action、post-squash action，还是实际 PD target 上计算；

    - $\lambda_T$ 和 $\lambda_S$ 要从小开始扫，不要一上来拉满；

    - 真实机器人部署前仍然要看 joint target、joint velocity、torque/current、foot contact 和 body oscillation，而不是只看 reward。

- 这里是推断：在 `legged_gym` / `rsl_rl` 这类框架里，如果 rollout storage 没有保存 `next_obs`，就需要在 storage 里额外存下一帧 observation，或者在 mini-batch 构造时从连续 rollout 中取相邻 index。这个改动要小心 timeout 和 episode boundary。

# 和我之前看的几类方法的关系：

## 和 AMP_for_hardware 的关系：

- AMP_for_hardware 解决的是“不要靠复杂 reward 手写自然动作”，它用 discriminator 从参考动作里提供 style reward。

- CAPS 解决的是“不要让 RL policy 输出高频抖动”，它用 action policy regularization 直接约束 actor mapping。

- 两者都在处理硬件友好性，但角度不同：

    - AMP 更像是从 motion prior 里学习“动作应该像什么”；

    - CAPS 更像是从函数正则化角度限制“动作不应该怎么乱跳”。

- 所以两者理论上可以互补：AMP 拉住动作风格，CAPS 拉住局部平滑性。

## 和 DreamWaQ / HIMLoco 的关系：

- DreamWaQ 和 HIMLoco 更关注 POMDP 下的隐藏状态恢复：

    - 怎么从 proprioception history 里估计速度、地形、扰动或 latent；

    - 怎么让 actor 在没有 privileged information 的真实部署中仍然知道当前环境大概是什么。

- CAPS 关注的是另一个层面：

    - 即使 actor 已经有了比较好的内部状态表征，它输出的 action mapping 也可能不平滑。

- 换句话说：

    - DreamWaQ / HIMLoco 在回答“policy 应该知道什么”；

    - CAPS 在回答“policy 知道以后，输出动作时应该保持怎样的局部连续性”。

- 对真实足式机器人来说，这两件事都需要。只解决 hidden-state recovery，policy 可能仍然输出高频关节目标；只解决 smoothness，policy 可能仍然不知道地面滑不滑。

## 和 CTS-MoE 的关系：

- CTS-MoE 这类方法通过多个 expert 和 router 处理多地形、多任务之间的策略分工。

- 但 MoE 也可能带来一个额外问题：router 权重变化如果不连续，不同 expert 的动作输出差异又比较大，那么最终混合 action 也可能出现不平滑。

- 所以 CAPS 的 spatial smoothness 对 MoE actor 也有潜在意义：

    - 相近 observation 下，router 不要突然把权重从 expert A 切到 expert B；

    - 即使 router 权重变化，混合后的 action 也应该连续。

- 这里是推断：如果未来把 CAPS 加到 MoE locomotion policy 里，除了对最终 action 做 smoothness，也可以考虑对 router probability 做额外 regularization。但这已经超出本文实验范围。

# 论文的核心贡献：

- **第一，提出 CAPS regularization。** 它把 action policy smoothness 直接写进 actor optimization，包括 temporal smoothness 和 spatial smoothness 两项。

- **第二，给出一个 FFT-based smoothness metric。** 这个指标不是只看某个高频 peak，而是看控制信号频谱中 amplitude-weighted frequency，能更稳定地评价 NN controller 的高频成分。

- **第三，在多个 RL 算法和 benchmark 上验证通用性。** DDPG、TD3、SAC、PPO 加 CAPS 后，在测试任务中都能降低 smoothness score。

- **第四，在真实无人机上展示硬件收益。** PPO + CAPS 虽然 tracking error 不是最小，但显著降低 current draw 和高频控制成分，使训练出的 controllers 更稳定可部署。

- **第五，指出 naive filtering 对 NN controller 可能有风险。** 这点很重要，因为它提醒我们不能把经典控制里的工具不加训练闭环地直接接到 learned policy 后面。

# 我觉得最值得注意的点：

## 1. 它把“动作平滑”从 reward 里拿出来了：

- 这篇论文最有启发的地方，是把 smoothness 看成 policy function 的性质。

- 以前我们说动作平滑，容易直接想到 reward：

$$
-\|a_t - a_{t-1}\|^2
$$

- 但 CAPS 说得更底层：policy 本身就是一个函数 $\pi(s)$，如果这个函数在局部不连续、不稳定，那么你再怎么调 reward，都可能只是间接修补。

- 这对于机器人控制很重要。因为真实硬件最终不关心你的 reward 写得多漂亮，它只关心每个控制周期收到的 motor command 是不是合理。

## 2. Smoothness 和 performance 不是同一个指标：

- 论文里有些 benchmark 加 CAPS 后 reward 会下降，但控制信号更平滑。

- 这提醒我们：在机器人上，不能只看 episode return。

- 一个 policy 可能 reward 很高，但动作高频抖动、功耗大、关节冲击大；另一个 policy reward 稍低，但真实硬件上更稳定、更省电、更耐用。

- 所以做 sim-to-real 时，至少要同时看：

    - tracking error；

    - action rate / action acceleration；

    - torque 或 current；

    - power；

    - body oscillation；

    - contact stability；

    - failure / fall rate。

## 3. Spatial smoothness 比我一开始想的更重要：

- temporal smoothness 很直观，相邻 action 不要跳。

- 但 spatial smoothness 对 sim-to-real 更关键。因为真实部署最大的问题之一就是 observation 和 dynamics 不会完全等于仿真。

- 如果 policy 对 observation 小扰动很敏感，domain randomization 也不一定完全救得回来。

- Spatial CAPS 本质上是在训练 actor：**把局部邻域看成同一个控制语义，不要对传感器噪声过拟合。**

## 4. 不能迷信 filter：

- 我之前看到 action 抖动时，第一反应也可能是“加个滤波器”。

- 但这篇论文提醒了一点：learned controller 的行为是和训练闭环绑定的。

- 如果 filter 没有在训练时出现，部署时突然加入，就不是简单地“去噪”，而是在改 policy 面对的系统动力学。

- 对 legged robot 来说，这个问题同样存在。比如部署时对 joint target 做强低通，可能会让 swing foot 落点滞后、扰动恢复变慢，policy 反而输出更激进的补偿动作。

# 局限和风险：

- **第一，过强 smoothness 会让 policy 变保守。** 对需要快速反应的任务，比如避障、抗推扰、落脚修正、空中姿态恢复，action 不能被过度压平。

- **第二，状态扰动分布 $\phi(s)$ 需要认真设计。** 如果 $\sigma$ 太小，spatial smoothing 没效果；如果太大，就会把本来不同语义的状态强行拉成相同 action。

- **第三，不同 observation 维度不能随便同尺度加噪。** IMU、joint velocity、command、history action 的单位和含义都不同，必须结合 normalization 或物理噪声建模。

- **第四，论文主要验证的是 quadrotor attitude control。** 它没有证明 CAPS 在所有 contact-rich legged locomotion 任务上都一定有效。足式机器人有接触切换、冲击、地形突变，smoothness 约束需要更小心。

- **第五，smoothness metric 不能替代硬件指标。** $S_m$ 低说明频谱高频成分少，但真实硬件还要看电流、温度、关节力矩、接触冲击、结构振动和长时间稳定性。

- **第六，CAPS 不解决所有 sim-to-real 问题。** 它不替代 domain randomization、系统辨识、延迟建模、状态估计、contact modeling、动作限幅和安全 watchdog。

# 对工程实现的启发：

- 如果我们要在 legged locomotion 里借鉴 CAPS，我觉得可以按一个比较保守的顺序来做。

- **第一步：先只记录，不改训练。** 先在现有 policy 上统计 action FFT、action rate、joint target difference、torque/current proxy、base acceleration。不要一上来改算法，否则不知道问题到底来自 reward、policy architecture、PD gain 还是 observation noise。

- **第二步：先加很小的 temporal CAPS。** 从 $\lambda_T$ 很小的值开始，只看是否能降低无意义 action jitter，同时不明显损害 tracking 和抗扰。

- **第三步：再加 spatial CAPS。** 只对 proprioception 的 normalized observation 加小噪声，先不要强扰动 command。观察 policy 对 IMU / joint velocity noise 是否更稳。

- **第四步：检查 contact phase。** 看 swing/stance 切换附近是否被 smoothing 压坏。如果落脚变慢、抗推变差，说明 regularization 过强或 mask 设计不对。

- **第五步：真实部署前必须做安全门槛。** 包括动作限幅、速度限幅、力矩/电流限制、NaN/Inf 检查、watchdog、急停、低速小范围测试。CAPS 让动作更平滑，不等于自动安全。

- 这里最重要的是不要把 CAPS 当成一个“加上就变好”的 trick。它更像一个诊断和约束工具：当我们确认 policy 的主要问题是 action mapping 太敏感、控制信号高频多，再用它会比较合理。

# 一句话总结：

- CAPS 的核心思想是：**与其通过复杂 reward 或部署后滤波器间接修补动作抖动，不如在训练 actor 时直接约束 policy 的时间连续性和状态局部鲁棒性，让强化学习控制器从函数映射层面变得更平滑、更适合真实硬件。**

# 我的笔记：

- 我觉得这篇论文对 legged RL 最大的提醒是：我们不能只把 smoothness 当作 reward 里的一个小 penalty。

- 对真实机器人来说，action 本身就是硬件接口。Policy 输出的每一次跳变，最后都会变成电机、电流、减速器、足端接触和机体振动上的真实代价。

- 所以评估一个 locomotion policy 时，reward、速度 tracking、是否摔倒当然重要，但还不够。一个真正能上机器的 policy，应该同时满足：

    - 能完成任务；

    - 对 observation 噪声不过敏；

    - 对 sim-to-real gap 不会立刻输出高频补偿；

    - 动作变化符合机械系统的时间尺度；

    - 接触切换时还能保留必要的快速反应能力。

- CAPS 正好卡在这个位置：它不负责教机器人“怎么走”，也不负责恢复隐藏地形状态；它负责让已经学到的控制策略别以一种伤硬件的方式输出动作。

- 如果把 DreamWaQ / HIMLoco 这类 history representation 方法看成“让 policy 看懂当前环境”，那么 CAPS 可以看成“让 policy 看懂以后不要手抖”。这两个方向组合起来，才更接近真实机器人上可靠的 learning-based control。
