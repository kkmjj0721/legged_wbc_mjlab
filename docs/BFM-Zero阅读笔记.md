# BFM-Zero阅读笔记

# 前言：

- 这篇论文的核心目标，是把 **Behavioral Foundation Model（BFM，行为基础模型）** 这个概念真正落到真实人形机器人全身控制上。

- 传统 humanoid whole-body control 里面，很多工作本质上还是“为一个任务训练一个策略”：

    - 追踪某一批 mocap 动作，就训练一个 motion tracking policy；

    - 做 locomotion，就手工写速度、姿态、脚滑、能耗等奖励；

    - 做 reaching 或 sitting，又要重新组织任务奖励或者重新蒸馏策略。

- BFM-Zero 想解决的问题是：**能不能训练一个单一的、可 prompt 的底层控制策略，让它在不重新训练网络参数的情况下，根据不同形式的 prompt 去做 motion tracking、goal reaching、reward optimization，甚至还能通过少量仿真交互在 latent space 里快速适配？**

- 论文的关键点不是“又训练了一个更大的 PPO tracking policy”，而是把 **off-policy unsupervised RL + Forward-Backward representation + mocap regularization** 组合起来，让任务、目标姿态、奖励函数和运动序列都能被映射到同一个 latent space $Z$ 里。这个 latent space 就像一个“技能遥控器”：不同的 $z$ 不是直接代表某一个动作片段，而是代表一种可以驱动 policy 的任务意图。

- 未执行/未验证：这份笔记只基于 `/home/kk/legged_wbc_mjlab/paper/BFM-Zero.pdf` 的阅读整理，没有跑作者代码、没有复现实验，也没有验证真实机器人部署效果。

# 论文核心信息：

- **论文标题：** BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning

- **作者：** Yitang Li, Zhengyi Luo, Tonghe Zhang, Cunxi Dai, Andrea Tirinzoni, Anssi Kanervisto, Haoyang Weng, Kris Kitani, Mateusz Guzek, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta, Guanya Shi

- **机构：** Carnegie Mellon University, Meta

- **项目页：** https://lecar-lab.github.io/BFM-Zero/

- **机器人平台：** 主要是 Unitree G1，Appendix 里还验证了 Booster T1

- **训练仿真器：** IsaacLab，训练频率 200 Hz，控制频率 50 Hz

- **运动数据：** LAFAN1 retarget 到 Unitree G1，主训练集包含 40 段 several-minute-long motions；OOD 测试使用 AMASS/CMU 子集

- **论文类型：** humanoid whole-body control / unsupervised RL / behavior foundation model / sim-to-real

# 研究问题：

- 论文要回答的是一个很直接但很难的问题：

    - **我们能不能不用 task-specific reward 和两阶段蒸馏，直接训练一个通用 humanoid policy，让它通过 latent prompt 完成多种全身控制任务？**

- 这里的 prompt 不是语言模型里的自然语言 prompt，而是一个连续向量 $z \in Z$。不同任务会先被编码成 $z$，然后 policy 执行：

    - motion tracking：把一段目标 motion sequence 编码成随时间变化的 latent；

    - goal reaching：把目标 pose 编码成 latent；

    - reward optimization：把任意 reward function 投影到 latent；

    - few-shot adaptation：在 latent space 里做 CEM 或 sampling-based optimization，而不是更新网络权重。

- 这个问题的难点在于：

    - **人形机器人是强接触、强欠驱动、强部分可观测系统。** 只在仿真里学到的策略，放到真实 G1 上很容易因为摩擦、质量、关节偏置、传感噪声等差异而失效。

    - **mocap 数据没有 actuator-level action labels。** 所以不能像 VLA/BC 那样直接学“给定状态应该输出什么动作”。

    - **传统 PPO tracking 很强，但接口太窄。** 它通常擅长“追踪这一段动作”，但不天然支持“给我一个 reward，我直接算出一个 latent 去优化它”。

    - **unsupervised RL 以前多在 virtual character 或较理想环境里成立。** 论文想证明 off-policy unsupervised RL 不只是动画角色的玩具，也可以做真实 humanoid sim-to-real。

# 研究背景：

- 在这之前，真实人形机器人全身控制大致有几条常见路线。

## 第一种：单任务 RL / motion tracking 路线

- 这条路线很直接：给一段 mocap，写 tracking reward，让机器人学会跟上目标关节角、根节点姿态、速度等。

- 优点是工程上容易理解，也确实在近几年把 humanoid 的跳舞、走路、跑步、起身等任务推到了很高水平。

- 但缺点也明显：

    - policy 很容易变成“动作播放器”，会追踪，但不一定理解任务；

    - 如果新任务不是一个干净的 reference motion，就要重新设计 reward 或重新蒸馏；

    - 多个 tracking policy 最后经常还要压到一个 student 里，训练链路变长。

## 第二种：两阶段 foundation policy / skill distillation 路线

- 这类方法通常先训练很多 skill policy 或 tracking policy，再把它们蒸馏到一个 conditional policy / latent policy 里。

- 它的问题是：

    - 第一阶段 policy 的质量决定了第二阶段上限；

    - 如果底层 tracking policy 对噪声、低质量 motion 或 OOD pose 不够鲁棒，蒸馏出来的 skill space 也会继承这些问题；

    - 训练不是一个统一的 unsupervised objective，而是先做技能，再做压缩。

## 第三种：VLA / behavior cloning 路线

- VLA 在 manipulation 上很自然，因为可以收大量视觉、语言、动作标签。

- 但 humanoid whole-body control 不一样：

    - 真实 humanoid 很难收大规模高质量 teleoperation；

    - actuator-level labels 不容易获得；

    - 人类视频/mocap 可以给“身体怎么动”，但不会告诉你每个电机下一步应该输出什么 PD target。

- 所以，BFM-Zero 的出发点是：**既然没有动作标签，就不要把它当成 supervised behavior cloning；我们让机器人在仿真里自己探索，用 mocap 只作为行为先验和 regularization。**

# 核心直觉：

- 我们可以把 BFM-Zero 理解成三个东西的组合：

    - **一个会动的底层 policy：** $\pi(o_{t,H}, z)$，输入历史观测和 latent prompt，输出 29 DoF 的 PD controller target；

    - **一个会把状态/目标压成任务向量的 backward map：** $B(s)$ 或 $B(o,s)$，它负责把目标 pose、motion 或 reward 变成 latent；

    - **一个会估计“某个 latent 下长期会去哪里”的 forward map：** $F(s,a,z)$，它类似 successor feature，告诉我们在当前动作之后，长期访问哪些行为特征。

- 如果用一个简单比喻：

    - 普通 tracking policy 像“照着舞蹈视频一帧一帧跟”；

    - BFM-Zero 更像“学会了一套身体技能坐标系”。你给它一个坐标 $z$，它就会往对应的行为模式上走。

- 关键点是 $z$ 不只来自 mocap：

    - 一个 reward function 可以被变成 $z_r$；

    - 一个 goal pose 可以被变成 $z_g$；

    - 一段 motion sequence 可以被变成 $z_t$ 序列；

    - 一个困难任务也可以从初始 $z$ 出发，在仿真里通过优化得到更好的 $z^\star$。

- 这就让 policy 的接口从“我要追踪这个动作”变成了“我要完成这个目标”。这个抽象层级比单纯 tracking 高很多。

# 问题建模：

- 论文把真实 humanoid control 建模成 POMDP：

$$
(S, O, A, P, \gamma)
$$

- 对 Unitree G1：

    - 动作 $a \in \mathbb{R}^{29}$，表示 29 个 DoF 的 PD controller targets；

    - privileged state $s \in \mathbb{R}^{463}$，包含 root height、body pose、body rotation、linear/angular velocities 等仿真里可得信息；

    - 可部署观测 $o_t \in \mathbb{R}^{64}$，由关节位置、关节速度、root angular velocity、projected gravity 等组成：

$$
o_t = \{q_t - \bar q, \dot q_t, \omega^{root}_t / 4, g_t\}
$$

    - policy 不只看单帧，而是看历史：

$$
o_{t,H}=\{o_{t-H}, a_{t-H}, \ldots, o_t\} \in \mathbb{R}^{93H+64}
$$

- 这里有一个和 HIMLoco 很接近的思想：**真实部署时，actor 不能吃 privileged state，所以必须靠 history 来弥补 partial observability。**

- 但 BFM-Zero 和 HIMLoco 的区别也很明显：

    - HIMLoco 更像是在 actor 前面加一个 estimator，专门从 history 里恢复 velocity + latent；

    - BFM-Zero 的 latent $z$ 是任务/行为 prompt，不是单纯环境隐变量；

    - 它的核心不是“估计当前地形或扰动”，而是“学一个可以表达目标、奖励和 motion 的行为空间”。

# 方法主线：

- BFM-Zero 的整体流程可以拆成三段：

    - **Pre-training：** 在仿真里进行 online off-policy unsupervised RL，同时使用 unlabeled mocap motion regularize 行为；

    - **Zero-shot inference：** 把 reward / goal / tracking motion 编码成 latent prompt，直接喂给同一个 policy；

    - **Few-shot adaptation：** 如果 zero-shot 效果不够好，就不改网络参数，只在 latent space 里做优化。

- 论文 Fig. 2 的图可以理解成：

    - 左边是训练期：policy 和环境交互，产生 replay buffer；FB representation 学 latent space；discriminator 用 mocap 约束行为风格；auxiliary critic 处理安全和物理约束；

    - 右上是 zero-shot：reward、goal、tracking motion 都能变成 prompt；

    - 右下是 few-shot：在 latent prompt 上做采样优化，处理更难的 single pose 或 motion sequence。

# FB Representation 到底在干什么：

- 这一部分是论文最关键也最抽象的地方。

- 普通 RL 里，我们通常先指定 reward，再训练 policy 去最大化这个 reward。

- FB / successor-feature 系列方法反过来想：**能不能先学一个通用表示，使得很多 reward 都可以写成这个表示的线性组合？**

- 它会学三个对象：

    - $\phi:S\rightarrow \mathbb{R}^d$：把状态映射成 task feature；

    - $\pi_z:S\rightarrow A$：给定 latent $z$ 的 policy；

    - $F_z$：latent-conditioned successor feature，近似在 policy $\pi_z$ 下未来会累计访问到哪些 feature。

- 直觉上，$F_z$ 不是只看当前一步，而是在估计：**如果我现在按这个 latent 去行动，未来一段时间内会经过哪些状态特征？**

## FB 的核心分解

- 论文使用 Forward-Backward representation，把长期转移动态写成一个低秩分解：

$$
M^{\pi_z}(ds'|s,a) \simeq F(s,a,z)^\top B(s')\rho(ds')
$$

- 这里：

    - $M^{\pi_z}$ 是 successor measure，可以理解为“从 $(s,a)$ 出发，按 $\pi_z$ 走，未来折扣访问到 $s'$ 附近的概率总量”；

    - $F(s,a,z)$ 是 forward map，看当前状态、动作和 latent；

    - $B(s')$ 是 backward map，把未来状态压到同一个表示空间；

    - $\rho$ 是训练状态分布。

- 这有点像把一个巨大的“未来访问概率表”拆成两个向量的内积：

    - $F$ 负责当前出发点和任务；

    - $B$ 负责目标状态的表示；

    - 两者内积越大，说明当前行为越容易导向那个目标区域。

## 为什么它能做 reward prompt

- FB 理论里可以得到：

$$
\phi(s) = \left(\mathbb{E}_\rho[B(s)B(s)^\top]\right)^{-1}B(s)
$$

- 然后 reward 可以写成：

$$
r_z(s)=\phi(s)^\top z
$$

- 对应 policy 的 Q-value 可以写成：

$$
Q^{\pi_z}_{r_z}(s,a)=F(s,a,z)^\top z
$$

- 这一步非常关键。它说明 $z$ 不只是一个“风格编码”，而是一个任务方向：policy $\pi_z$ 学到的是如何最大化 $\phi(s)^\top z$ 这类 reward。

- 换句话说，$z$ 像一个“奖励函数的压缩版”。以前我们要为每个 reward 单独训练，现在希望把 reward 投影到 $Z$ 里，然后直接用同一个 policy。

# Mocap regularization 和 Discriminator：

- 单纯做 unsupervised RL 有一个很现实的问题：机器人可能会学出能在数学目标上得分、但动作很奇怪的行为。

- 这和你之前 AMP_for_hardware 笔记里的问题非常接近：如果只给“向前走”这种简单 reward，机器人可能会学出抽搐、畸形但仿真里有效的动作。

- BFM-Zero 也用了 discriminator，但它和 AMP 的使用方式有区别：

    - AMP 里 discriminator 主要给 style reward，推动 policy 像专家动作；

    - BFM-Zero 里 discriminator 是 **latent-conditioned discriminator**，它不仅判断动作像不像 mocap，还要判断“在这个 latent $z$ 下，当前状态是否像对应 motion 的状态”。

## expert motion 的 latent 怎么来

- 对一段 motion $\tau$，论文用 backward map 把它编码成 imitation embedding：

$$
z_\tau = \frac{1}{l(\tau)}\sum_{(o,s)\in \tau} B(o,s)
$$

- 这就是把一整段 mocap motion 的状态表示做平均，得到一个代表这段 motion 风格/目标的 latent。

- 注意：mocap 数据是 action-free trajectories，也就是只有状态序列，没有电机动作标签。BFM-Zero 不需要知道“专家在这个状态下输出了什么动作”，只需要知道“专家动作经过了哪些状态”。

## discriminator loss

- discriminator 的目标是区分两类样本：

    - 来自 mocap motion $\tau$ 的真实状态，配上它自己的 $z_\tau$；

    - 来自 online replay buffer 的机器人状态，配上训练时采样的 $z_i$。

$$
\mathcal{L}(D)=
-\mathbb{E}_{\tau\sim M,(o,s)\in\tau}[\log D(o,s,z_\tau)]
-\mathbb{E}_{(o,s,z)\sim \mathcal{D}}[\log(1-D(o,s,z))]
$$

- 然后用 discriminator 生成一个风格奖励：

$$
r^D(o,s,z)=\log D(o,s,z)-\log(1-D(o,s,z))
$$

- 这和 GAN 的 logit reward 思路很像：

    - $D$ 越相信这个状态像专家，$r^D$ 越大；

    - $D$ 越相信它是 policy 产生的假样本，$r^D$ 越小。

- 关键点：**这个 reward 不是最终任务 reward，而是行为先验。** 它把探索过程拉回人类 motion 的分布附近，避免 off-policy unsupervised RL 学出过于怪异、硬件不友好的动作。

# 为什么它能 sim-to-real：

- BFM-Zero 不是只把 FB-CPR 原封不动搬到 humanoid 上。论文强调了 4 个对真实机器人很关键的设计。

## A. Asymmetric Training

- actor 输入部署可得的历史观测 $o_{t,H}$；critic / forward map / auxiliary critic 等训练模块可以使用 privileged information $(o_{t,H},s_t)$。

- 这个设计和非对称 AC 的直觉一致：

    - Actor 是“上场干活的人”，部署时只能看真实传感器；

    - Critic 是“训练场教练”，仿真里可以看上帝视角，用来降低 value estimate 的方差。

- 注意它还用了 history。原因很简单：真实 robot 是 POMDP，单帧观测看不出摩擦、扰动、延迟、刚才脚有没有滑，必须从一段历史响应里恢复信息。

## B. Massively Parallel Off-policy RL

- 论文受到 FastTD3 这类 large-batch off-policy humanoid RL 的启发，在 1024 个并行环境里训练，使用大 replay buffer 和较高 update-to-data ratio。

- 训练配置里几个重要数字：

    - history length $H=4$；

    - episode length $T=500$；

    - $N_{env}=1024$；

    - batch size $N_{batch}=1024$；

    - update-to-data ratio $N_{ups}=16$；

    - total gradient steps $N_{grad}=3M$；

    - total environment steps 约 $192M$；

    - replay buffer 约 $5M$ transitions；

    - discount factor $0.98$；

    - latent dimension $d=256$。

- 这意味着它不是小规模 PPO 训练，而是一个很重的 off-policy pretraining 系统。

## C. Domain Randomization

- 为了让真实 G1 能跑，训练环境里随机化了很多物理参数：

    - COM offset：$U([-0.02,0.02])$ m；

    - link mass：$U([0.95,1.05])$；

    - friction：表中给的是 $U([-0.5,1.25])$；

    - default joint position offset：$U([-0.02,0.02])$；

    - push robots：$U([0,0.5])$ m/s。

- 还加入了观测噪声：关节位置、关节速度、projected gravity、root angular velocity 都会加扰动。

- 这部分和一般 legged sim-to-real 的经验一致：**不要让 policy 只会在一个“完美仿真世界”里走路。**

## D. Reward Regularization

- 因为真实机器人有硬件约束，论文额外加了 auxiliary penalty rewards，并训练一个 auxiliary critic $Q_R$。

- Appendix 里列出的 regularization 包括：

    - DoF Limit，权重 $-10$；

    - Action Rate，权重 $-0.1$；

    - Self Contact，权重 $-1$；

    - Feet Orientation，权重 $-0.4$；

    - Ankle Roll，权重 $-4$；

    - Feet Slip，权重 $-2$。

- 这个地方很重要：BFM-Zero 虽然强调 unsupervised RL，但并不是完全不要 reward shaping。它不要的是“每个任务都手工写一个主 reward”，但仍然需要安全、硬件、姿态和接触层面的 regularization。

# Actor Loss 怎么组合：

- BFM-Zero 的 actor 不是只最大化一个 reward，而是同时看三种信号：

    - FB critic 给出的 task-centric value：$F(o_{t,H},s_t,a_t,z)^\top z$；

    - discriminator critic $Q_D$，保证动作像人类 motion；

    - auxiliary critic $Q_R$，保证动作不撞硬件约束。

- 论文里的 actor loss 可以简化理解为：

$$
\mathcal{L}(\pi)=
-\mathbb{E}\left[
F(o_{t,H},s_t,a_t,z)^\top z
+ \lambda_D Q_D(o_{t,H},s_t,a_t,z)
+ \lambda_R Q_R(o_{t,H},s_t,a_t,z)
\right]
$$

- 因为 loss 前面有负号，所以优化时是在最大化括号里的三项。

- 我们可以把这三项理解成三个老师：

    - **FB 老师：** 你要往 latent $z$ 对应的任务方向走；

    - **Discriminator 老师：** 你走得要像人，不要像仿真怪动作；

    - **Regularization 老师：** 你别碰关节极限、别脚滑、别自碰撞、别动作抖得太厉害。

- 这也是论文能部署到真实机器人上的核心工程折中：它没有放弃 unsupervised RL 的通用性，也没有放弃传统 legged control 里那些必要的安全先验。

# Zero-shot inference：

- 训练完成后，网络参数固定。接下来不同任务只需要换 prompt latent $z$。

## 1. Reward Optimization

- 给任意 reward function $r(s)$，论文用下面这个方式把 reward 投影成 latent：

$$
z_r = \mathbb{E}_{s\sim \rho}[B(s)r(s)]
$$

- 实际实现中用采样估计：

$$
z_r \approx \frac{1}{N}\sum_i r(s_i)B(s_i)
$$

- 论文的 real-test 配置里，reward inference 使用 400,000 个样本。

- 直觉解释：

    - 如果某些状态 $s_i$ 在 reward 下得分高，那么它们的 $B(s_i)$ 会被更大权重加进 latent；

    - 最终 $z_r$ 就指向“那些高 reward 状态所在的行为方向”；

    - policy 拿到 $z_r$ 后，就会倾向于产生能到达这些状态的行为。

- 这比手工调一个完整 RL 训练过程快很多，因为这里只是在算一个 latent，不是在重新训练 policy。

## 2. Goal Reaching

- 对 goal pose，直接用 backward map：

$$
z_g = B(s_g)
$$

- 如果实现里 $B$ 接收 $(o,s)$，也可以理解成 $z_g=B(o_g,s_g)$。

- 这意味着 goal pose 不需要 interpolation，不需要 tracking reward，也不需要专门训练 reaching policy。目标 pose 自己就是 prompt。

## 3. Motion Tracking

- 对一段 motion $\tau=\{s_1,\ldots,s_n\}$，论文构造一串随时间变化的 latent：

$$
z_t = \sum_{t'=t}^{t+H} B(s_{t'})
$$

- 这里 $H$ 是 look-ahead horizon。真实机器人上 tracking look-ahead 设为 3，仿真里用 sequence length。

- 这一步很像“往前看几帧的 motion intention”，而不是只盯着当前一帧。这样 policy 能提前知道接下来要转身、抬手、落脚还是恢复姿态。

# Few-shot adaptation：

- Zero-shot 不一定总是足够，尤其是：

    - 目标 pose 很难；

    - 真实机器人多了额外负载；

    - 地面摩擦变化很大；

    - motion 里有高度动态片段。

- BFM-Zero 的做法不是 fine-tune 网络，而是在 latent space $Z$ 里优化 prompt。

## Single-pose adaptation

- 初始 latent 来自 goal pose：

$$
z_0 = B(s_g,o_g)
$$

- 然后用 Cross-Entropy Method（CEM）做 20 轮优化，目标是最大化任务奖励并扣掉 auxiliary regularization。

- 论文例子是：真实 G1 躯干上绑了 4 Kg 负载，让它单腿站立。

- 结果：

    - 未适配的 $z_{init}$ 会在 5 秒内发生环境碰撞；

    - 优化后的 $z^\star$ 能保持单腿平衡超过 15 秒。

- 这里的工程意义很强：**适配的是 prompt，不是网络参数。** 这比在真实机器人上重新训练 policy 安全很多。

## Trajectory adaptation

- 对 trajectory，论文用 DIAL-MPC 风格的 dual-loop annealing sampling-based optimization。

- 具体设置：

    - particle count $N=2048$；

    - temperature schedules $\beta_1=0.85$、$\beta_2=0.9$；

    - optimization iterations $M=6$。

- 测试任务是 altered ground friction 下的 leaping motion。

- 结果是 tracking error 降低约 $29.1\%$。

# 实验部分：

## 训练和评估设置

- 训练平台：Unitree G1 的 IsaacLab 仿真版本。

- 行为数据：LAFAN1 retarget 到 G1，40 段 several-minute-long motions。

- 控制频率：50 Hz。

- 仿真频率：200 Hz。

- OOD 测试：AMASS 的 CMU 子集中随机选 175 段 motion，以及从 AMASS 中手工选 10 个 pose。

- Reward evaluation：共 24 个 reward，episode 长度 $T=500$。

## 指标

- Tracking 和 Goal Reaching 都使用平均 joint position error：

$$
E_{mpjpe}(e,g)=\frac{1}{|e|}\sum_{t=1}^{|e|}\|q_t(e)-q(g)\|_2
$$

- 对 motion tracking，则把目标 pose 换成目标 motion 中对应时刻的关节位置：

$$
E_{mpjpe}(e,m)=\frac{1}{|e|}\sum_{t=1}^{|e|}\|q_t(e)-q_t(m)\|_2
$$

- 注意：

    - tracking / pose 的 $E_{mpjpe}$ 越低越好；

    - reward return 越高越好。

# 仿真实验结果：

- 论文比较了一个 idealized privileged version 和可部署版本：

| Model | Test env | Test data | Track | Reward | Pose |
| --- | --- | --- | ---: | ---: | ---: |
| BFM-Zero-priv | Isaac no DR | LAFAN1 | 1.0749 | 299.3 | 1.0291 |
| BFM-Zero | Isaac DR | LAFAN1 | 1.1015 | 221.9 | 1.1387 |
| BFM-Zero | Mujoco DR | LAFAN1 | 1.0789 | 207.3 | 1.1041 |
| BFM-Zero | Mujoco DR | AMASS | 1.0342 | - | 1.4735 |

- 作者总结：部署版 BFM-Zero 相比 privileged idealized 版本，在 tracking、reward、pose 上分别差 $2.47\%$、$25.86\%$、$10.65\%$。

- 这个结果的意思是：

    - 对 tracking 和 pose reaching，历史观测 + DR 以后性能掉得不多；

    - reward optimization 掉得比较明显，说明 reward inference 对数据分布、采样质量和 DR 更敏感；

    - 这也符合直觉：tracking/pose 有更明确的目标状态，而某些 reward 比较 sparse，采样估计 $z_r$ 时更容易被噪声影响。

- Sim-to-sim 结果也比较重要：把 IsaacLab 里训练的模型放到 Mujoco 测试，性能差异基本小于 $7\%$。这说明 domain randomization 和 history-based actor/critic 对动力学变化确实有帮助。

# 真实机器人结果：

## Tracking

- 真实 G1 上，BFM-Zero 可以 tracking：

    - styled walking；

    - highly dynamic dances；

    - fighting / sports motions；

    - monocular video retarget 后质量不完美、有遮挡和不连续的 motion。

- 论文特别强调了一个现象：即使机器人不稳定或摔倒，它也会表现出比较自然的 recovery，然后继续 tracking。

- 这里作者认为，这不只是 disturbance training 的结果，还来自：

    - TD-based off-policy training；

    - GAN-based human-likeness reward；

    - regularization terms；

    - rich skill library。

- 我自己的理解是：如果 policy 的 latent space 里真的包含很多“人类式恢复动作”的邻域，那么摔倒以后它不是只会机械地拉回 reference pose，而是能临时绕到一个更可行的 recovery behavior，再回到原来的 prompt。

## Goal Reaching

- 论文从目标状态里去掉 velocity，只保留 pose，然后让 policy 连续到达多个随机目标姿态。

- 结果表现为：

    - 即使目标姿态不连续，轨迹也能平滑过渡；

    - 即使目标 pose 不完全可行，机器人也会收敛到一个自然、接近目标的 configuration；

    - 从任意姿态到 T-pose 也不需要显式插值。

- 这说明 latent space 不是一堆离散技能按钮，而更像一个连续坐标系。两个目标之间没有硬切换，而是可以沿着 latent space 平滑过渡。

## Reward Optimization

- 真实机器人 reward optimization 测了三类任务：

    - locomotion reward：指定 base velocity / angular velocity；

    - arm-movement reward：指定 wrist height；

    - pelvis-height reward：让机器人 sitting、crouch 或低姿态移动。

- Reward definition 在 Appendix C 里分成 6 类：

    - Standing；

    - Locomotion；

    - Rotation；

    - Ground poses；

    - Arm raise；

    - Combined rewards。

- 最有意思的是 combined reward。作者把不同 reward 做线性组合，比如“后退 + 抬手”，policy 能直接表现出组合行为。

- 这意味着 BFM-Zero 的 reward prompt 接口有一定可组合性：不是只会执行训练集中某个固定 motion，而是可以把多个目标方向加起来。

## Disturbance Rejection

- 真实测试包括踢腿、推搡、甚至被拖倒。

- 论文描述里比较突出的现象是：机器人不是剧烈反应，而是会用比较顺滑、human-like 的方式恢复。

- 例如受到强推以后，机器人会收臂、快速后退或进入类似 running-like 的 recovery pose，然后逐渐减速并回到原来的姿态。

- 这点和普通 tracking controller 很不一样：普通 controller 可能只会死命追 reference，反而在大扰动下更危险；BFM-Zero 因为有行为先验和更广的 latent skill space，可能会“先活下来，再回任务”。

# Appendix 里的训练细节：

## Motion sampling priority

- 训练 episode 初始状态是一个混合分布：

    - 随机 falling positions；

    - motion dataset $M$ 中的状态。

- 作者还做了 motion prioritization：根据 agent 对某段 motion 的 tracking 能力，用 earth mover's distance（EMD）更新采样优先级。

- LAFAN1 的 40 段 motion 会被切成 10 秒 chunks，以便更细粒度地调整采样。

- 这个设计的意义是：不要让训练一直采简单片段，也不要让困难片段完全学不到。它更像 curriculum sampling，但依据是当前 agent 的 tracking 表现。

## 网络结构

- Actor 和 critics 使用 residual architecture，结构类似 transformer block 里的 residual connection + layer normalization + Mish activation。

- Critics 使用两个网络组成 ensemble。

- Discriminator 和 backward map $B$ 使用普通 MLP + ReLU。

- 主论文真实测试用的模型规模：

    - actor $\pi$：31.9M 参数；

    - $Q_R$：134.8M；

    - $Q_D$：134.8M；

    - $F$：135.9M；

    - discriminator $D$：2.9M；

    - backward map $B$：201k；

    - total：440.5M。

- 这个规模说明 BFM-Zero 已经不是传统 legged_gym 里几十万到几百万参数的小 policy，而更接近“大型控制基础模型”。

## 预训练算法流程

- Algorithm 1 可以简化成下面几步：

    - 初始化 online replay buffer $D_{online}$；

    - 初始化 expert buffer $M$，里面是 action-free mocap trajectories；

    - 每一步为多个并行环境采样 latent $z_t$；

    - 执行 $a_t\sim \pi(o_{t,H},z_t)$，把 transition 存入 $D_{online}$；

    - 从 $D_{online}$ 采样 transition batch；

    - 从 mocap buffer 采样 sequence batch；

    - 用 $B$ 把 expert sequence 编码成 $z_j$ 并归一化；

    - 更新 discriminator；

    - 更新 $F$ 和 $B$，让 $F^\top B$ 逼近 successor measure；

    - 计算 discriminator reward $r_i^D$；

    - 更新 $Q_D$、$Q_R$；

    - 更新 actor；

    - 更新 target networks。

- 注意这个流程里，mocap 不负责给动作标签。它负责提供“人类运动状态分布”和“latent-conditioned style reference”。

# 和我之前笔记里几个方法的关系：

## 和 AMP_for_hardware 的关系

- BFM-Zero 和 AMP 都用了 discriminator，所以直觉上可以放在一起理解。

- AMP 里 discriminator 更像一个“裁判”：你当前动作像不像狗/人/专家动作，像就给 style reward。

- BFM-Zero 里的 discriminator 更进一步：它是 latent-conditioned 的。

    - 不是只问“你像不像人”；

    - 而是问“在这个 $z$ 代表的行为目标下，你现在这个状态像不像对应的人类 motion”。

- 所以 BFM-Zero 的 discriminator 不只是美化动作，它还参与塑造 latent space。

## 和 HIMLoco 的关系

- HIMLoco 的核心问题是：POMDP 下怎么从历史观测恢复对控制有用的 hidden state。

- BFM-Zero 也承认这个问题，所以 actor 输入的是 observation history，critics 使用 privileged state。

- 但两者 latent 的语义不一样：

    - HIMLoco 的 latent 更偏向“从历史里估计环境/扰动/内部响应”；

    - BFM-Zero 的 $z$ 更偏向“任务和行为 prompt”；

    - HIMLoco 解决的是“我现在处在什么隐藏物理条件下”；

    - BFM-Zero 解决的是“我应该往哪个行为目标走”。

## 和 CTS-MoE / multi-critic 思路的关系

- CTS-MoE 主要用 MoE 和 multi-critic 去处理多地形、多任务下的梯度冲突和 value interference。

- BFM-Zero 没有显式 MoE router，也不是把任务拆成多个专家 policy。

- 它处理多任务的方式是：

    - 用 latent $z$ 条件化同一个 policy；

    - 用 FB representation 把不同 reward / goal / motion 投影到同一个空间；

    - 用 $Q_D$ 和 $Q_R$ 分别管理 style 和安全约束。

- 所以它更像“统一坐标系”路线，而不是“专家分工”路线。

# 论文的核心贡献：

- **第一点：把 off-policy unsupervised RL 真正推到真实 humanoid。** 论文声称这是第一个可以部署到真实人形机器人上的 behavioral foundation model。

- **第二点：给了一个统一 prompt 接口。** reward、goal、motion tracking 都能变成 latent prompt，而不是每个任务单独训练。

- **第三点：把 FB representation 和 mocap regularization 结合。** FB 提供 task-centric latent，discriminator 把行为拉回 human-like distribution。

- **第四点：针对 sim-to-real 做了必要工程补丁。** 包括 domain randomization、history-dependent asymmetric learning、auxiliary reward regularization、大规模并行 off-policy training。

- **第五点：展示了 prompt-level few-shot adaptation。** 困难任务可以通过 latent optimization 改善，而不是 fine-tune 网络。

# 我觉得最值得注意的点：

## 1. 它不是“更强 motion tracking”，而是“更强任务接口”

- 很多 humanoid paper 的主要卖点是动作更快、更稳、更像人。

- BFM-Zero 的核心卖点是接口更统一：同一个 policy 可以接受不同任务形式。

- 这点很重要，因为真实机器人最终不可能只有 tracking 需求。我们更希望上层 planner / language model / task planner 能说“把手举高，同时低姿态向左移动”，底层 policy 能把这种目标组合执行出来。

## 2. Reward inference 是很漂亮但也很脆弱的地方

- $z_r=\mathbb{E}[B(s)r(s)]$ 这个公式非常优雅，但它依赖采样分布。

- 论文自己也观察到：reward tasks 相比 privileged 版本掉得更多，而且某些 reward inference 重复采样会出现很差的 instance。

- Appendix D 还提到，使用 online replay buffer 或 training motion set 做 reward inference，效果会不同；使用 motion dataset 特别是 LAFAN1 时更好。

- 这说明一个实际工程问题：**reward prompt 的质量不是只由 reward function 决定，还由你拿什么状态集来估计 $z_r$ 决定。**

## 3. “无监督”并不等于“不需要先验”

- 论文标题强调 unsupervised RL，但系统里其实有很多强工程先验：

    - mocap motion data；

    - discriminator style regularization；

    - domain randomization；

    - auxiliary rewards；

    - action-rate / joint-limit / self-contact / feet-slip penalties；

    - fall initialization 和 motion prioritization。

- 所以不要把它理解成“随便给一个 robot，它自己从零探索出所有人类动作”。更准确的说法是：**主任务空间通过 unsupervised RL 学，行为形态和硬件边界仍然靠数据和 regularization 拉住。**

## 4. Latent space 的连续性是关键资产

- 论文用 t-SNE 和 spherical linear interpolation（SLERP）展示 latent space：相似 motion 会聚在一起，两个 latent 之间插值能产生语义上合理的中间技能。

- 这说明 $Z$ 不只是一个查表式 skill id，而是一个连续空间。

- 对后续工程很重要：如果 latent space 是平滑的，那么上层 planner 可以在 $Z$ 里做搜索、插值、优化和组合；如果 latent space 是乱的，那么 prompt optimization 会很不稳定。

# 局限和风险：

- **真实机器人结果偏定性。** 论文展示了很多 G1 视频和现象，但真实硬件上的大规模量化对比不多。尤其是扰动恢复、视频 retarget tracking、combined reward 等，更多是 qualitative validation。

- **行为范围受 motion dataset 影响。** 作者在 Discussion 里也承认，BFM-Zero 表达行为的范围和性能与训练 motions 强相关。未来需要研究 dataset size、motion quality、architecture 和 performance 的 scaling law。

- **Reward inference 对样本分布敏感。** 如果用 DR 后的 replay buffer，reward prompt 可能更噪；如果 reward sparse，估计 $z_r$ 会更容易失败。

- **模型和训练成本很高。** 主模型 440.5M 参数，训练约 192M environment steps，还需要 1024 并行环境和大 replay buffer。普通实验室想完整复现会有压力。

- **Few-shot adaptation 还只是初步验证。** 单腿站立和 leaping motion 很有说服力，但还不能说明所有复杂任务都能靠 latent optimization 解决。

- **没有解决高层语义到 reward 的完整链路。** 论文提到 reward interface 对语言 prompt 友好，但真正从自然语言生成稳定 reward function 仍然是另一个问题。

# 对工程实现的启发：

- 如果要在自己的 legged / humanoid 项目里借鉴 BFM-Zero，不建议一上来全量复现。可以先从最小闭环开始：

    - 先做一个 fixed robot + fixed simulator 的 FB representation toy version；

    - 只验证 goal reaching latent 是否能收敛；

    - 再加入 mocap discriminator；

    - 再做 reward inference；

    - 最后才上 domain randomization、history actor、auxiliary critic 和 real robot。

- 这里有几个实现检查点：

    - **observation/action 维度必须固定清楚。** 论文里 actor action 是 29 DoF PD targets，observable state 是 64 维，history 是 $93H+64$；

    - **actor 和 critic 的信息边界要分开。** actor 只能吃部署可得 history，critic/F/QR/QD 可以吃 privileged state；

    - **mocap 不需要 action label，但 retarget 质量会影响 behavior prior。** 如果 retarget 很抖，discriminator 会把坏动作也当专家；

    - **reward regularization 不是可选装饰。** joint limit、feet slip、self contact 这些项直接关系到能不能上硬件；

    - **reward inference 要记录采样来源。** 用 training motions、online replay buffer、DR buffer 得到的 $z_r$ 可能差很多；

    - **few-shot adaptation 先在仿真里做。** 不要直接在真实机器人上让 CEM 随机试危险动作。

- 如果把它和 WBC/MJLab 的思路结合，我觉得更现实的切入点不是立刻替换现有 controller，而是先把 BFM-Zero 当成一个 **latent skill proposal module**：

    - BFM policy 提供目标姿态、速度或动作趋势；

    - WBC / safety layer 负责接触约束、力矩限制和硬件保护；

    - 上层 planner 在 $Z$ 里搜索或组合技能。

# 一句话总结：

- **BFM-Zero 的价值在于，它把 humanoid 控制从“为每个动作训练一个 tracking policy”，推进到“训练一个可 prompt 的行为空间”，让 motion、goal 和 reward 都能通过同一个 latent interface 驱动真实人形机器人。**

# 我的笔记：

- 这篇论文我觉得最核心的是两个字：**接口**。

- 很多时候我们看 legged RL，会把注意力放在 reward、网络结构、DR 参数、sim-to-real trick 上。但 BFM-Zero 真正有启发的是：它在底层控制和高层任务之间放了一个统一的 latent interface。

- 以前的接口是 reference motion 或 velocity command，很窄；现在的接口是 $z$，它可以来自 reward、goal、motion，也可以被优化。这让底层 policy 更像一个可以被调用的 foundation controller。

- 当然，这个接口还没有完全成熟。它现在还依赖 mocap 数据、采样估计、很大的训练资源和大量工程 regularization。但方向是很清楚的：未来 humanoid 的底层控制很可能不是一堆分散 policy，而是一个连续、可组合、可优化的行为空间。

- 对我们自己做项目来说，最值得先学的不是 440M 参数的大模型，而是这三个思想：

    - 用 history 和 privileged critic 处理真实部署的 POMDP；

    - 用 discriminator / motion prior 把探索行为拉回自然动作分布；

    - 用 latent prompt 把 goal、reward、motion tracking 统一成一个控制接口。
