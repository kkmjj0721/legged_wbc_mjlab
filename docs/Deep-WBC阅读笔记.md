# Deep-WBC阅读笔记

# 前言：

- 这篇论文的核心目标，是让一个**带机械臂的四足机器人**不再把“走路”和“操作”拆成两个互相妥协的模块，而是用一个统一的 RL policy 同时控制腿和机械臂。

- 论文标题是 **Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion**。这里的 Whole-Body Control 不是传统意义上只写一套优化式 WBC，把接触约束、末端任务、质心任务全部放进 QP 里解；它更像是：

    - 用强化学习直接学一个统一策略；

    - policy 输出腿部和机械臂的 joint position target；

    - 底层 PD controller 再把这些 target 转成 torque；

    - 训练时靠 privileged information 和 domain randomization 跨过 sim-to-real；

    - 部署时只用机器人自己能拿到的历史观测和当前状态。

- 这篇论文最有意思的地方不是“Go1 上面绑了一个 WidowX 机械臂”，而是它把 legged manipulation 里一个很容易被工程上拆开的东西重新合起来：**机械臂能不能到达目标，并不只取决于机械臂本身；腿部姿态、机身 pitch/roll/yaw、支撑状态和整机重心都会影响末端工作空间。**

- 换句话说，机器人不是“底盘负责移动，机械臂负责操作”。如果末端目标在脚边，机械臂单独够不到，腿必须下蹲、机身必须前倾；如果机械臂向侧面伸得很远，腿和躯干也要改变姿态去平衡。这个时候再把 locomotion controller 和 manipulation controller 分开，就很容易出现 IK 奇异、自碰撞、动作不自然、模块之间互相甩锅的问题。

- 论文提出两个核心训练技巧：

    - **Advantage Mixing：** 利用动作空间里“手臂动作主要影响 manipulation、腿部动作主要影响 locomotion”的因果偏置，先降低 credit assignment 难度，再逐渐让两个任务互相耦合。

    - **Regularized Online Adaptation：** 让 privileged encoder 和 adaptation module 在线共同训练，并用正则项缩小两者 latent 的 realizability gap，避免传统 RMA/teacher-student 里 student 学不到 teacher latent 的问题。

- 未执行/未验证：这份笔记只基于 `/home/kk/legged_wbc_mjlab/paper/Deep-Whole-Body-Control.pdf` 的 PDF 文本抽取，以及本地参考笔记 `/home/kk/飞书文档/论文学习部分.md`、`/home/kk/飞书文档/AMP_for_hardware学习记录.md`、`/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`、项目内已有 `BFM-Zero阅读笔记.md` / `DreamWaQ阅读笔记.md` 的写法整理；没有跑作者代码，没有复现实验，也没有验证真实 Go1/WidowX 部署效果。

# 论文核心信息：

- **论文标题：** Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion

- **作者：** Zipeng Fu, Xuxin Cheng, Deepak Pathak

- **机构：** Carnegie Mellon University

- **会议：** 6th Conference on Robot Learning, CoRL 2022, Auckland, New Zealand

- **项目页：** https://maniploco.github.io

- **机器人平台：** Unitree Go1 quadruped + Interbotix WidowX 250s 6-DoF arm + parallel gripper

- **传感器：** RealSense D435 / D435i RGB camera mounted near the gripper

- **真实部署计算：** policy 和 adaptation module 在 Go1 onboard computation 上以 50 Hz 运行；Supplement 里写到 policy/adaptation inference 在 Raspberry Pi 4 上，WidowX 软件栈在 Nvidia TX2 上，通过 UDP 通信

- **训练仿真器：** IsaacGym

- **控制方式：** policy 输出 18 维 joint position target，包含 12 维腿部目标和 6 维机械臂目标，底层 PD controller 转成 torque

- **论文类型：** legged mobile manipulation / learning-based whole-body control / sim-to-real RL / online adaptation

- **代码链接：** PDF 文本中没有看到作者主代码仓库链接；Supplement 只提到 WidowX 官方 ROS 软件栈链接

- **DOI：** PDF metadata 和正文抽取中没有看到 DOI

# 原文摘要翻译：

- 外接机械臂可以显著扩大足式机器人的应用范围，让机器人去做很多轮式或履带式平台很难完成的移动操作任务。

- 传统 legged manipulator 的控制 pipeline 通常会把系统拆成 manipulation controller 和 locomotion controller。这个拆法看起来清楚，但在真实系统里并不理想：需要大量工程规则来协调手臂和腿，模块误差会互相传播，最后动作容易变得不平滑、不自然。

- 论文提出的做法是：用强化学习直接学习一个 unified policy，让它作为 legged manipulator 的 whole-body controller，同时控制机械臂和四足底盘。

- 为了让高 DoF 全身控制从仿真迁移到真实世界，作者提出 **Regularized Online Adaptation**；为了让统一策略在训练时不要陷入只会 manipulation 或只会 locomotion 的局部最优，作者提出 **Advantage Mixing**，利用 arm / leg action space 和 manipulation / locomotion objective 之间的因果偏置来降低 credit assignment 难度。

- 论文还给出一个低成本真实硬件方案，并展示统一策略可以在擦白板、拿杯子、按门禁按钮、放置物体、扔杯子、拥挤环境抓取等任务里产生动态的腿臂协同行为。

# 创新点：

- **统一控制接口：** 不再把腿和机械臂拆成两个 controller，而是用一个 neural policy 同时输出 leg joint position targets 和 arm joint position targets。

- **Advantage Mixing：** 在 PPO policy gradient 里显式区分 arm action 和 leg action，让 $A_{manip}$ 与 $A_{loco}$ 先按动作分支承担主要责任，再通过 $\beta$ curriculum 逐渐混合成全身任务。

- **Regularized Online Adaptation：** 在线训练 privileged encoder $\mu$ 和 adaptation module $\phi$，并用 regularization 约束 $z^\mu$ 不要跑到 $z^\phi$ 无法从历史观测恢复的 latent 区域。

- **低成本真实平台：** 用 Unitree Go1 + WidowX 250s 搭出约 6K USD 的 fully untethered legged manipulator，而不是只在高价封闭平台或仿真中验证。

- **真实任务验证：** 通过 teleoperation、AprilTag vision-guided tracking、open-loop demo replay 等任务展示身体姿态确实参与了 reaching 和 manipulation，不只是机械臂单独追 IK 解。

# 研究问题：

- 论文要回答的问题是：

    - **能不能训练一个统一策略，让带机械臂的四足机器人同时完成 locomotion 和 manipulation，而不是用分层/分模块 pipeline 把手和腿硬拆开？**

- 这个问题表面上像“控制维度变高了”，但真正麻烦的是：

    - 腿部动作会改变机械臂末端可达空间；

    - 机械臂伸出去以后会改变整机质心和姿态稳定性；

    - manipulation reward 和 locomotion reward 可能互相冲突；

    - 训练时有仿真完整状态，部署时只有 onboard history；

    - 如果策略只学会“伸手”，可能不愿意探索会暂时破坏末端精度的腿部动作；如果只学会“站稳/走路”，又可能不愿意为了末端目标改变身体姿态。

- 所以 Deep-WBC 不只是把两个 action head 拼起来。它真正想解决的是：

    - **统一控制带来的 credit assignment 问题；**
    - **高 DoF 动态系统的 sim-to-real 问题；**
    - **腿臂协同对末端工作空间和稳定性的提升问题。**

- 这里有一个和传统 WBC 很不一样的地方：传统 WBC 往往先把任务优先级、接触约束、质心目标、末端目标写清楚，再通过优化器求解动作；Deep-WBC 则让 policy 在仿真中自己学会这些耦合关系。好处是工程调参可能少一些，坏处是可解释性和安全边界需要额外小心。

# 研究背景：

- 在 Deep-WBC 之前，legged mobile manipulation 大致可以分成几条路线。

## 第一种：传统模型式 whole-body control / optimization 路线

- 这条路线会显式建模机器人动力学、接触约束、末端任务、姿态约束，然后通过 QP、MPC、inverse dynamics 或 operational space control 来算控制输入。

- 优点是：

    - 约束清楚；
    - 安全边界比较容易写进优化问题；
    - 对已知任务和受控环境，调通以后可解释性强。

- 但在带机械臂的四足机器人上，问题会变得很硬：

    - 系统是高 DoF、强接触、非光滑动力学；
    - 腿部接触状态一变，机械臂末端可达性也会变；
    - 机械臂运动带来的质心变化会反过来影响步态；
    - 想手写所有耦合项，需要非常多 domain expertise。

- 论文的出发点不是说模型式控制没用，而是说：如果我们希望低成本平台在各种真实任务里做动态操作，纯靠手工分层控制会越来越重。

## 第二种：腿部 RL + 机械臂 IK/MPC 的半耦合路线

- 这类方法更接近工程直觉：

    - 腿部用一个 locomotion policy 或 MPC 保持平衡和速度；
    - 机械臂用 IK 或 operational space controller 追末端目标；
    - 上层再把两个模块接起来。

- 这比完全手写整机 WBC 简单一些，但有一个致命问题：**模块边界刚好切在耦合最强的地方。**

- 举个例子：杯子在机器人前脚附近，机械臂单独 IK 可能够不到；如果腿部控制器不主动下蹲或前倾，机械臂 controller 只能在局部可达空间里挣扎，最后可能出现 IK failure 或 self-collision。

- 再比如机械臂向右侧伸出去，整机需要 roll 或调整支撑来抵消重心偏移。如果 locomotion controller 不知道末端目标，它只会尽量维持自己的稳定目标，结果机械臂动作和底盘稳定性互相打架。

- 所以这类半耦合 pipeline 的失败不是某个模块太弱，而是：**每个模块都在优化局部正确的目标，但全身任务需要全局协同。**

## 第三种：直接训练一个大统一策略

- 最简单粗暴的想法是：既然分开不行，那就把 base、leg、arm、commands 全部喂给一个大 policy，直接输出 18 维动作。

- 这条路方向上没错，但训练会遇到局部最优：

    - 初期随机腿部动作会让 base 晃动，base 一晃末端误差就变大；
    - policy 很容易发现“别乱动腿，先把手臂末端跟住”是一个更容易的局部解；
    - 一旦腿部不探索，locomotion command following 就学不起来；
    - 反过来，如果只强调走路，机械臂又可能变成扰动项。

- Deep-WBC 的 Advantage Mixing 就是在解决这个问题：先让手臂动作主要对 manipulation advantage 负责，让腿部动作主要对 locomotion advantage 负责；等两个子能力都有了，再逐步让两个动作分支都看到混合 advantage。

# 核心直觉：

- 我们可以把 Deep-WBC 理解成三个层次：

    - **身体层面：** 机器人不是“一个底盘 + 一个机械臂”，而是一个 18 维受控 action 的整体；腿部姿态本身就是机械臂 workspace 的一部分。

    - **训练层面：** manipulation 和 locomotion 不是完全独立任务，但训练早期又不能让它们全部耦合，否则 credit assignment 太难。

    - **部署层面：** 仿真里可以知道质量、摩擦、payload、motor strength 等 privileged extrinsics；真实机器人只能从历史观测中估计这些变化，所以需要 adaptation module。

- 如果用一个简单比喻：

    - 传统分层控制像“让一个人只管走路，另一个人只管伸手，两个人通过电话协商”；

    - Deep-WBC 更像“同一个身体在学一个协调动作”。

- 这也是论文里 Figure 1/5/7 的重点：机器人为了让末端到达目标，会自然地 bend、stretch、pitch、roll，而不是机械臂孤立地追踪一个 IK 解。

- 关键点：Deep-WBC 的 whole-body coordination 不是显式写死的规则，例如“目标低就蹲下”。它是通过 unified reward、action coupling、Advantage Mixing 和 adaptation 共同学出来的。

# 问题建模：

- 论文把策略写成一个统一 neural network：

$$
\pi(a_t \mid s_t, a_{t-1}, z_t, command)
$$

- 输入大致包含：

    - base state $s_t^{base} \in \mathbb{R}^5$：roll、pitch、base angular velocity；

    - arm state $s_t^{arm} \in \mathbb{R}^{12}$：6 个 arm joint 的 position 和 velocity；

    - leg state $s_t^{leg} \in \mathbb{R}^{28}$：12 个 leg joint 的 position、12 个 velocity、4 个 foot contact indicator；

    - last action $a_{t-1} \in \mathbb{R}^{18}$；

    - end-effector pose command $[p^{cmd}, o^{cmd}]$；

    - base velocity command $[v_x^{cmd}, \omega_{yaw}^{cmd}]$；

    - environment extrinsics latent $z_t \in \mathbb{R}^{20}$。

- 输出是 18 维 joint position target：

$$
a_t = [a_t^{leg}, a_t^{arm}], \qquad a_t^{leg}\in\mathbb{R}^{12},\quad a_t^{arm}\in\mathbb{R}^{6}
$$

- 这些 target 再由 PD controller 转成 torque。论文强调使用 joint-space position control，而不是直接对机械臂使用 operational space control。这个选择看起来保守，但工程意义很强：

    - policy 可以自己学会避开 self-collision；
    - 不需要在线求 IK；
    - sim-to-real gap 相对小；
    - 对腿和臂统一成同一种 action interface，训练起来更简单。

- 注意一个细节：Go1 有 12 个可控腿部 DoF，WidowX 250s 是 6-DoF arm，gripper 不属于 policy action。论文有时说系统总 DoF 为 19，通常是把 gripper 也算进机械结构；但 policy 的动作输出是 18 维。

# 数据与任务定义：

- 这篇论文的训练数据不是离线 demonstration dataset，而是 policy 在 IsaacGym 中与 randomized environment 交互产生的 rollout。

- 任务输入可以分成两类 command：

    - **manipulation command：** end-effector position / orientation command $[p^{cmd},o^{cmd}]$；
    - **locomotion command：** forward velocity $v_x^{cmd}$ 和 yaw velocity $\omega_{yaw}^{cmd}$。

- 任务输出不是末端速度、末端 wrench，也不是 operational-space command，而是 joint-space target：

$$
a_t=[a_t^{leg},a_t^{arm}],\qquad a_t^{leg}\in\mathbb{R}^{12},\quad a_t^{arm}\in\mathbb{R}^{6}
$$

- 训练时随机化的环境参数包括 base extra payload、end-effector payload、center of base mass、arm motor strength、leg motor strength 和 friction。这些参数在仿真里可直接访问，真实部署时则需要 adaptation module 从 history 里估计其影响。

- 这里的任务边界很重要：论文没有把 grasp planning、视觉识别、语义任务规划、gripper open/close 全部塞进同一个 policy，而是把上层任务先抽象成 EE command + base velocity command。这样底层 whole-body controller 的接口比较干净，后续可以接 joystick、vision 或 demonstration replay。

# 方法主线：

- Deep-WBC 的方法可以拆成四段来看：

    - **Unified Policy：** 一个 policy 同时输出 leg action 和 arm action，不使用独立腿部/手臂策略。

    - **Reward 分解：** reward 分成 manipulation reward 和 locomotion reward，但最终 policy 是统一训练的。

    - **Advantage Mixing：** 在 PPO 更新时，不是直接把总 advantage 无差别地作用到所有动作维度，而是对 arm/leg action 分支做 curriculum-like mixing。

    - **Regularized Online Adaptation：** 训练 privileged encoder 和 adaptation module，让真实部署时可以从历史观测估计环境 latent。

- 这里最容易误解的是 Advantage Mixing。它不是多训练几个 critic head 这么简单，也不是把 reward 权重调一下。它是改 policy gradient 的信用分配方式：

    - arm action 的 log probability 先主要乘 manipulation advantage；
    - leg action 的 log probability 先主要乘 locomotion advantage；
    - 随着 $\beta$ 增大，arm/leg 都逐渐接收混合后的全任务 advantage。

- 这意味着论文不是在否认“腿和手要耦合”，而是在说：**训练早期先降低问题难度，训练后期再让耦合真正发生。**

# Reward Function：

- 论文把总 reward 分成两部分：

$$
r_t = r_{manip} + r_{loco}
$$

- 两部分都遵循同一个结构：

$$
r = r_{following} + r_{energy} + r_{alive}
$$

## Manipulation reward

- manipulation 的核心是 end-effector pose tracking：

$$
r_{manip, following}=0.5\cdot \exp\left(-\left\|[p,o]-[p^{cmd},o^{cmd}]\right\|_1\right)
$$

- 直观理解：

    - $p,o$ 是当前末端位置和姿态；
    - $p^{cmd},o^{cmd}$ 是目标末端位置和姿态；
    - 误差越小，指数项越接近 1；
    - 误差变大，奖励快速下降。

- arm energy penalty 为：

$$
r_{manip, energy}= -0.004\cdot \sum_{j\in arm\ joints}|\tau_j \dot{q}_j|
$$

- 这里惩罚的是正机械能消耗的绝对值。它的工程含义很直接：不要为了追末端目标让机械臂高速抽搐或用很大的力矩硬顶。

## Locomotion reward

- locomotion 的 tracking 包含前向速度和 yaw rate：

$$
r_{loco, following}\approx -0.5\cdot |v_x-v_x^{cmd}| + 0.15\cdot \exp(-|\omega_{yaw}-\omega_{yaw}^{cmd}|)
$$

- 注意：这里公式来自 PDF Table 1 的文本抽取，原表排版有一部分符号在抽取时错位。笔记按上下文理解为前向速度误差惩罚和 yaw tracking 指数奖励。

- leg energy penalty 使用二次能耗项：

$$
r_{loco, energy}= -0.00005\cdot \sum_{i\in leg\ joints}|\tau_i \dot{q}_i|^2
$$

- 论文特意说明，腿部能耗用二次项，是为了同时降低平均能耗和各腿关节之间的能耗方差。这个地方很工程：四足机器人真实部署时，不只是总功耗重要，某个电机长期过载也会导致发热和保护停机。

- alive reward 为：

$$
r_{loco, alive}=0.2+0.5\cdot v_x^{cmd}
$$

- 这里可以理解为鼓励机器人在前向速度命令下持续存活，不要通过“站着不动”获得一个看似低能耗的假好解。

# Command 和 Goal Sampling：

- Deep-WBC 里的 command 同时包含 base velocity command 和 end-effector command。

- base 侧主要是：

    - forward velocity $v_x^{cmd}$；
    - yaw angular velocity $\omega_{yaw}^{cmd}$。

- 末端位置 command 使用球坐标：

$$
(l,p,y)
$$

- 其中：

    - $l$ 是末端目标到 arm base 的半径；
    - $p$ 是 pitch；
    - $y$ 是 yaw；
    - 球坐标原点设在 arm base；
    - 目标不依赖 torso height、roll、pitch，这一点是为了让 command 本身不被当前机身姿态污染。

- 末端目标不是一步跳到新位置，而是在 $T_{traj}$ 时间内从当前末端位置插值到目标位置：

$$
p_t^{cmd}=\frac{t}{T_{traj}}p + \left(1-\frac{t}{T_{traj}}\right)p^{end},\qquad t\in[0,T_{traj}]
$$

- 这里 PDF 抽取出的公式方向看起来有点反直觉：如果 $t=0$，上式给出 $p^{end}$；如果 $t=T_{traj}$，给出当前 $p$。按文字描述，作者想表达的是在当前位置 $p$ 和随机采样目标 $p^{end}$ 之间插值。工程实现时必须回到代码核对 start/end 的方向，不能只依赖抽取文本。

- 论文还会在采样目标时检查：

    - 目标是否导致 self-collision；
    - 目标是否碰到地面；
    - 如果不满足，就重新采样 $p^{end}$。

- 训练和测试 command 范围大致如下：

| Command | Training Range | Test Range |
|---|---:|---:|
| $v_x^{cmd}$ | $[0,0.9]$ | $[0.8,1.0]$ |
| $\omega_{yaw}^{cmd}$ | $[-1.0,1.0]$ | $[-1,-0.7]\ \&\ [0.7,1]$ |
| $l$ | $[0.2,0.7]$ | $[0.6,0.8]$ |
| $p$ | $[-2\pi/5,2\pi/5]$ | $[-2\pi/5,2\pi/5]$ |
| $y$ | $[-3\pi/5,3\pi/5]$ | $[-3\pi/5,3\pi/5]$ |
| $T_{traj}$ | $[1,3]$ | $[0.5,1]$ |

- 这里可以看到，测试时作者故意把 forward velocity、goal length、trajectory time 放到更难的范围里，主要是看策略是否能在 OOD command 下仍然协调。

# Advantage Mixing：

## 为什么需要 Advantage Mixing

- 如果我们直接用普通 PPO 训练统一策略，最自然的目标是：

$$
J(\theta_\pi)=\frac{1}{|D|}\sum_{(s_t,a_t)\in D}\log\pi(a_t\mid s_t)A(s_t,a_t)
$$

- 但问题在于，$a_t$ 里面有 arm action，也有 leg action；$A$ 里面既包含 manipulation return，又包含 locomotion return。

- 训练初期，随机腿部动作会让 base 不稳。base 不稳以后，末端 pose tracking 会变差。于是 policy 可能会学到一个错误直觉：

    - “动腿会让末端误差变大，所以少动腿”；
    - “只要先用机械臂追末端目标，reward 下降没那么严重”；
    - “locomotion command 暂时不管也可以”。

- 结果就是策略卡在一个局部最优：会伸手，但不会好好走；或者会站稳，但不会为了末端目标主动改变身体姿态。

## 公式

- Deep-WBC 把 advantage 拆成两部分：

$$
A_{manip}\quad\text{和}\quad A_{loco}
$$

- 对 diagonal Gaussian policy，动作可以按 arm/leg 分支理解。论文把 objective 写成：

$$
J(\theta_\pi)=\frac{1}{|D|}\sum_{(s_t,a_t)\in D}
\left[
\log\pi(a_t^{arm}\mid s_t)(A_{manip}+\beta A_{loco})
+
\log\pi(a_t^{leg}\mid s_t)(\beta A_{manip}+A_{loco})
\right]
$$

- 其中 curriculum 参数：

$$
\beta=\min(t/T_{mix},1)
$$

- 当 $\beta=0$ 时：

    - arm action 主要由 $A_{manip}$ 更新；
    - leg action 主要由 $A_{loco}$ 更新；
    - 这相当于先让两个身体部分各自学会自己的主任务。

- 当 $\beta\rightarrow1$ 时：

    - arm action 也接收 locomotion advantage；
    - leg action 也接收 manipulation advantage；
    - 最终两者都为全身任务负责。

## 直观解释

- Advantage Mixing 可以理解成一种很轻量的 curriculum，不需要手工设计一堆 terrain difficulty、goal difficulty、payload difficulty。

- 它只控制一个东西：**跨任务 credit assignment 的强度。**

- 早期先把问题切开：

    - 手先学会追末端；
    - 腿先学会跟速度、保持稳定。

- 后期再把问题合起来：

    - 手臂伸出去会影响稳定，所以手臂动作也要考虑 locomotion；
    - 腿部姿态能扩大末端工作空间，所以腿部动作也要考虑 manipulation。

- 这和我之前看 HIMLoco / DreamWaQ 时的一个训练经验很像：**不要一上来全拉满。** 复杂系统里，很多模块最终必须耦合，但训练初期如果耦合太强，梯度会直接混乱。先让每个子能力站住，再逐步增加耦合，往往比硬训一个全耦合目标更稳。

# Regularized Online Adaptation：

## 传统 RMA / Teacher-Student 的问题

- 传统 sim-to-real 里很常见的方案是两阶段 teacher-student：

    - 第一阶段：teacher 在仿真里看 privileged information，例如质量、摩擦、payload、motor strength，训练出一个强策略；
    - 第二阶段：student 只看 onboard observation history，模仿 teacher 的 action 或 latent；
    - 部署时只用 student。

- 这条路线很强，但 Deep-WBC 指出两个问题：

    - **Realizability gap：** teacher 看到的信息太完整，可能学出一个 student 根本无法从历史观测中预测的 latent。
    - **Pipeline gap：** student 训练必须等 teacher 收敛以后才能开始，训练链路更长，也更容易出现分布不匹配。

- 这个问题在带机械臂的四足机器人上会更严重。因为 payload、arm pose、base dynamics、ground friction 都会一起影响观测历史；如果 privileged latent 太“上帝视角”，student 可能无论怎么拟合都学不到。

## 架构

- Deep-WBC 使用两个模块：

    - privileged information encoder $\mu$：输入仿真中的 environment extrinsics $e_t$，输出 $z_t^\mu$；
    - adaptation module $\phi$：输入最近 history，输出 $z_t^\phi$。

- 训练时 policy 可以使用 $z_t^\mu$ 或 $z_t^\phi$：

$$
a_t=\pi(s_t,a_{t-1},z_t)
$$

- 部署时没有 privileged information，只使用 adaptation module：

$$
z_t=z_t^\phi=\phi(s_{t-10:t-1}^{arm},s_{t-10:t-1}^{leg},s_{t-10:t-1}^{base},a_{t-11:t-2})
$$

- 注意：这里的 history 长度是 10 步，policy 和 adaptation 都以 50 Hz 运行。也就是说，history 大概覆盖最近 0.2 秒的系统响应。它不是长时记忆，而是短窗口动态适应。

## Loss

- 论文把整体 loss 写成：

$$
L(\theta_\pi,\theta_\mu,\theta_\phi)
=-J(\theta_\pi,\theta_\mu)
+\lambda\left\|z^\mu-sg[z^\phi]\right\|^2
+\left\|sg[z^\mu]-z^\phi\right\|^2
$$

- 其中：

    - $J(\theta_\pi,\theta_\mu)$ 是前面的 Advantage Mixing RL objective；
    - $sg[\cdot]$ 是 stop-gradient；
    - $\lambda$ 是 regularization strength；
    - 第一项 $-J$ 让 policy 和 privileged encoder 为 RL return 服务；
    - 第二项 $\lambda\|z^\mu-sg[z^\phi]\|^2$ 约束 privileged encoder，不要跑到 adaptation module 学不到的 latent 空间；
    - 第三项 $\|sg[z^\mu]-z^\phi\|^2$ 训练 adaptation module 去模仿 privileged encoder。

- 这个 loss 的设计很关键。它不是单向蒸馏，而是双向拉近：

    - $\phi$ 要学 $\mu$；
    - $\mu$ 也不能完全不管 $\phi$ 的可实现性。

- 所以 Regularized Online Adaptation 的本质不是“再加一个 adaptation loss”，而是把 teacher latent 的表达能力限制在 student 可预测的范围内。这样真实部署时，student/adaptation module 才不会被要求预测一个不可能从 onboard history 恢复的上帝变量。

## Algorithm 1 的训练节奏

- Supplement 里的 Algorithm 1 给了更具体的流程：

    - 初始化 privileged encoder $\mu$、adaptation module $\phi$、unified policy $\pi$；
    - 每轮迭代采样 rollout；
    - 如果 $itr \bmod H=0$，使用 $z_t^\phi$ 执行动作，并更新 $\theta_\phi$；
    - 否则使用 $z_t^\mu$ 执行动作，并更新 $\theta_\pi,\theta_\mu$；
    - 每轮清空 buffer；
    - $\lambda$ 按 linear curriculum 增大。

- 作者设置：

$$
H=20
$$

- 正则系数：

$$
\lambda=\min\left(\max\left(\frac{itr-5000}{5000},0\right),1\right)
$$

- 这意味着前 5000 个 iteration，$\lambda=0$，先让 expert/unified policy 学到基本能力；从 5000 到 10000 之间逐步把 regularization 拉到 1。

- 这里依然是一个“不要一上来全拉满”的训练设计：

    - 早期 adaptation 还不可靠，强行要求 privileged latent 可模仿，可能拖慢主策略学习；
    - 中后期 policy 已经有基本能力，再把 latent 空间压到可部署模块能预测的区域；
    - 最后部署时只保留 unified policy + adaptation module。

# Domain Randomization 和 Environment Extrinsics：

- 论文训练时随机化了一组 environment parameters，并把它们编码成 $z_t$：

| Env Param | Training Range | Test Range |
|---|---:|---:|
| Base extra payload | $[-0.5,3.0]$ | $[5.0,6.0]$ |
| End-effector payload | $[0,0.1]$ | $[0.2,0.3]$ |
| Center of base mass | $[-0.15,0.15]$ | $[0.20,0.20]$ |
| Arm motor strength | $[0.7,1.3]$ | $[0.6,1.4]$ |
| Leg motor strength | $[0.9,1.1]$ | $[0.7,1.3]$ |
| Friction | $[0.25,1.75]$ | $[0.05,2.5]$ |

- 可以看到，测试范围故意比训练范围更难，尤其是：

    - base extra payload 从训练的最高 3.0 到测试的 5.0-6.0；
    - end-effector payload 从 0.1 到 0.2-0.3；
    - friction 从 0.25-1.75 扩展到 0.05-2.5；
    - motor strength 也更宽。

- 这说明作者不是只测 IID randomization，而是在看 adaptation module 面对 OOD dynamics 时能不能维持性能。

- 但这里也要注意：OOD 仍然是作者定义的参数范围，不等于真实世界无限泛化。真实部署还会有线缆、延迟、关节 backlash、电池电压、通信抖动、相机误差等因素，论文没有把它们全部量化。

# Simulation 和 Training 设置：

- 仿真平台：IsaacGym。

- 机器人模型：Unitree Go1 URDF + Interbotix WidowX URDF，自定义刚性连接。

- terrain：使用 fractal noise 生成 rough terrain。参数包括：

    - number of octaves = 2；
    - fractal lacunarity = 2.0；
    - fractal gain = 0.25；
    - frequency = 10 Hz；
    - amplitude = 0.15 m。

- 作者提到，rough terrain 会自然逼出 foot clearance，从而替代 flat terrain 下需要手写的复杂抬脚 reward。这一点很有工程价值：与其在平地上写很多“抬脚要漂亮”的 reward，不如让地形本身制造必须抬脚的物理压力。

- episode：最多 1000 steps，满足特定姿态/高度条件会 early terminate：

    - robot height 低于 0.28 m；
    - body roll 超出与 EE command yaw 方向相关的阈值；
    - body pitch 超出与 EE command pitch 方向相关的阈值。

- 频率：

    - policy control frequency = 50 Hz；
    - simulation frequency = 200 Hz。

- PD 参数：

    - leg joints stiffness $K_p=50$；
    - arm joints stiffness $K_p=5$；
    - leg damping $K_d=1$；
    - arm damping $K_d=0.5$。

- 默认腿部目标关节位置：

$$
[-0.1,0.8,-1.5,\ 0.1,0.8,-1.5,\ -0.1,0.8,-1.5,\ 0.1,0.8,-1.5]
$$

- arm 默认目标为 zeros。

- action delta range：

    - leg target joint position delta range = 0.45；
    - arm target joint position delta range = $[2.1,1.0,1.0,2.1,1.7,2.1]$。

- policy network：

    - MLP 输入 current state $s_t\in\mathbb{R}^{75}$，拼接 environment extrinsics $z_t\in\mathbb{R}^{20}$；
    - 第一层 hidden dim = 128；
    - 后面 split 成两个 head；
    - 每个 head 有 2 层 hidden dim = 128；
    - 最后输出 leg action $\mathbb{R}^{12}$ 和 arm action $\mathbb{R}^{6}$，再拼成 18 维。

- PPO 超参数：

| Hyper-parameter | Value |
|---|---:|
| PPO clip range | 0.2 |
| Learning rate | $2\times10^{-4}$ |
| Discount factor $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Number of environments | 5000 |
| Env steps per training batch | 40 |
| Learning epochs per batch | 5 |
| Mini-batches per batch | 4 |
| Minimum policy std | 0.2 |

- 训练总量：

    - 10000 iterations / training batches；
    - 约 2 billion samples；
    - 约 200k gradient updates。

- 这里和我们当前项目的关系很直接：Deep-WBC 的数字可以作为参考，但不能直接搬。当前 `legged_wbc_mjlab` 是 mjlab manager-based API，和论文 legacy `legged_gym + rsl_rl` 结构不同。真正迁移时要重新核对 observation term、action ordering、frame、history、privileged group 和 RL config。

# 仿真实验结果：

## Baselines 和 Metrics

- 论文在仿真里比较了几个 baseline：

    - **Separate policies：** 腿和机械臂分别有独立策略；
    - **Uncoordinated policy：** 一个 policy 看 aggregate state，但 arm action 只由 manipulation reward 训练，leg action 只由 locomotion reward 训练；
    - **RMA：** 两阶段 teacher-student adaptation baseline；
    - **Expert policy：** 使用 privileged encoder $z^\mu$ 的 unified policy；
    - **Domain Randomization：** 没有 environment extrinsics $z$ 的 unified policy。

- 指标包括：

    - Survival percentage；
    - Base Accel：base angular acceleration；
    - Vel Error：base velocity command 和实际 velocity 的 L1 error；
    - EE Error：end-effector command 和实际 EE pose 的 L1 error；
    - Total Energy：legs + arm 总能耗。

- 所有仿真实验测试 3 个随机初始化网络，每个 1000 episodes，指标按 episode length normalize。

## Unified Policy 优于分离和非协同策略

| Method | Survival ↑ | Base Accel ↓ | Vel Error ↓ | EE Error ↓ | Tot. Energy ↓ |
|---|---:|---:|---:|---:|---:|
| Unified (Ours) | $97.1\pm0.61$ | $1.00\pm0.03$ | $0.31\pm0.03$ | $0.63\pm0.02$ | $50\pm0.90$ |
| Separate | $92.0\pm0.90$ | $1.40\pm0.04$ | $0.43\pm0.07$ | $0.92\pm0.10$ | $51\pm0.30$ |
| Uncoordinated | $94.9\pm0.61$ | $1.03\pm0.01$ | $0.33\pm0.01$ | $0.73\pm0.02$ | $50\pm0.28$ |

- 这张表说明的不是“统一网络参数更多所以更强”，而是：

    - Unified policy 在几乎相同能耗下，有更高 survival、更低 velocity error 和更低 EE error；
    - Separate policies 的 base accel 明显更大，说明腿臂互相干扰后身体晃动更强；
    - Uncoordinated policy 虽然比 separate 好一些，但 EE error 仍然高于 unified。

- 这里的关键是 reward 和 observation 都让腿/臂看到整机状态，且 action 分支最终为总任务负责。只把两个 policy 放在一起，不等于 whole-body coordination。

## Whole-body coordination 扩大机械臂 workspace

| Method | Arm Workspace $(m^3)$ ↑ | Survival under perturb ↑ |
|---|---:|---:|
| Unified (Ours) | $0.82\pm0.02$ | $0.87\pm0.04$ |
| Separate | $0.58\pm0.10$ | $0.64\pm0.06$ |
| Uncoordinated | $0.65\pm0.02$ | $0.77\pm0.06$ |

- 论文用 1000 个 sampled EE poses 的 convex hull volume 估算 arm workspace，再减掉包住 quadruped 本体的 cube 体积。

- Unified policy 的 workspace 是 $0.82m^3$，Separate 只有 $0.58m^3$。这个差异很重要，因为它说明腿不是单纯在“维持底盘”，而是真的参与了 manipulation：

    - 目标低时，腿会 bend 让机械臂够到更低的位置；
    - 目标高时，腿会 stretch 让机械臂够到更高的位置；
    - 侧向目标会诱导机身 roll 来帮助 reach。

- Survival under perturb 也更高，说明机械臂不是纯扰动项。统一策略学到的是整机姿态协调，而不是“机械臂动了以后腿部被动救火”。

## Advantage Mixing 的效果

- Figure 4 / Supplement Figure 9 的结论是：没有 Advantage Mixing 时，统一策略很容易先学 EE command following，但忽略 locomotion command。

- 论文里的解释很合理：

    - 初期探索 leg action 会 destabilize base；
    - base 不稳会伤害 manipulation；
    - policy 因此不愿意探索 leg action space；
    - 最后卡在“只追末端，不好好走”的 local minima。

- Advantage Mixing 的作用就是先降低 credit assignment 复杂度，让 policy 学会 walk 和 grasp 两个基本能力，再逐步合并成全身控制。

- 这里我觉得比传统 curriculum 更干净：它没有设计复杂 task schedule，而是在 policy gradient 里加入了动作分支和任务 advantage 的结构先验。

## Regularized Online Adaptation 的 OOD 性能

| Method | Realizability Gap $\|z^\mu-z^\phi\|^2$ ↓ | Survival ↑ | Base Accel ↓ | Vel Error ↓ | EE Error ↓ | Tot. Energy ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Domain Randomization | - | $95.8\pm0.2$ | $0.44\pm0.00$ | $0.46\pm0.00$ | $0.40\pm0.00$ | $21.9\pm0.53$ |
| RMA | $0.31\pm0.01$ | $95.2\pm0.2$ | $0.54\pm0.02$ | $0.44\pm0.00$ | $0.26\pm0.04$ | $27.3\pm0.95$ |
| Regularized Online Adaptation | $2\times10^{-4}\pm0.00$ | $97.4\pm0.1$ | $0.51\pm0.02$ | $0.39\pm0.01$ | $0.21\pm0.00$ | $25.9\pm0.56$ |
| Expert w/ Reg. | - | $97.8\pm0.2$ | $0.52\pm0.02$ | $0.40\pm0.01$ | $0.21\pm0.00$ | $25.8\pm0.49$ |
| Expert w/o Reg. | - | $98.3\pm0.2$ | $0.51\pm0.02$ | $0.39\pm0.00$ | $0.21\pm0.00$ | $25.6\pm0.30$ |

- 这张表最核心的数字不是 survival，而是 realizability gap：

    - RMA 的 gap 是 $0.31$；
    - Regularized Online Adaptation 是 $2\times10^{-4}$；
    - EE Error 从 RMA 的 $0.26$ 降到 $0.21$，约 20% improvement。

- 这说明 regularization 没有明显伤害 expert policy，却让 adaptation module 更容易追上 expert latent。

- Domain Randomization 的 Base Accel 和 Tot. Energy 看起来更低，但论文指出它在困难环境里经常只是站着不动，所以这是一个假优势。这个点很重要：真实评估不能只看能耗低，也要看任务是否真的完成。

# 真实机器人部署：

## Robot System Setup

- 硬件：

    - Unitree Go1 quadruped，12 个可控腿部 DoF；
    - Interbotix WidowX 250s，6-DoF arm，带 parallel gripper；
    - RealSense D435/D435i 安装在 gripper 附近；
    - Go1 电池给 Go1 和 WidowX 供电；
    - 整机 fully untethered。

- 计算和通信：

    - policy 和 adaptation module 固定权重后直接部署；
    - 两者都以 50 Hz 运行；
    - Supplement 写到 inference 在 Raspberry Pi 4 上；
    - WidowX 软件栈在 Nvidia TX2 上；
    - Pi 和 TX2 之间用 UDP 通信；
    - gripper open/close 不是 policy 的一部分。

- 论文用 MPC+IK 作为真实世界 baseline：

    - Go1 使用内置 MPC controller；
    - WidowX 用 IK solver 做 operational space control。

## Teleoperation

- teleoperation 中，末端位置 command 更新为：

$$
p_{t+1}^{cmd}=p_t^{cmd}+\Delta p
$$

- 其中：

$$
\Delta p=(\Delta l,\Delta p,\Delta y)
$$

- 由两个 joystick 指定。

- Figure 5 展示了一个很关键的现象：quad body rotation 和 EE command 有明显相关性。

    - 当 command EE pitch 幅度大时，quadruped 会 pitch upward/downward 帮助手臂；
    - 当 command EE yaw 靠近侧向时，quadruped 会 roll 更多；
    - 当 yaw 为 0 时，它更多通过 pitch downward 来帮助手臂。

- 这说明统一策略不是单纯让机械臂自己伸，而是在用底盘姿态扩展末端能力。

## Vision-Guided Tracking

- vision-guided picking 中，作者把 RealSense 放在 gripper 附近，用 AprilTag 得到目标相对 gripper/camera 的位置：

$$
p_{tag}=[x_{tag},y_{tag},z_{tag}]^T
$$

- 然后用一个简单 position feedback controller 设置当前 EE position command：

$$
p_t^{cmd}=K^T p_{tag}
$$

- 其中 gain vector：

$$
K=[-1.5,-1.5,0.1]^T
$$

- 相机参数：

| Item | Value |
|---|---:|
| Resolution | $640\times400$ |
| Frequency | 10 Hz |
| Tag/Cam offset | $(-0.02,-0.03,0.12)$ |

- Table 7 / Table 10 的真实实验结果：

| Task | Method | Success Rate ↑ | TTC ↓ | IK Failure Rate ↓ | Self-Collision Rate ↓ |
|---|---|---:|---:|---:|---:|
| Easy, 3 points | Ours | 0.8 | 5s | - | 0 |
| Easy, 3 points | MPC+IK | 0.3 | 17s | 0.4 | 0.3 |
| Hard, 5 points | Ours | 0.8 | 5.6s | - | 0 |
| Hard, 5 points | MPC+IK | 0.1 | 22.0s | 0.2 | 0.5 |

- 每个设置平均 10 次真实实验。

- 作者分析失败模式：

    - Deep-WBC 的失败主要来自实际杯子位置和 AprilTag 位置不匹配；
    - Supplement 里用两个 AprilTag 并取平均 pose 可以缓解；
    - MPC+IK 在 easy task 有时成功，但会出现 IK singularity 和 self-collision；
    - hard task 中杯子太靠近身体，MPC+IK 更容易自碰撞；
    - MPC+IK 的 TTC 更长，因为在线 IK + operational space control 更耗计算。

- 这里我觉得真实实验比仿真表更有说服力。因为 hard pick-up task 本质上就是“机械臂单独够不到，需要身体配合”。如果只看普通 reaching，MPC+IK 可能还能撑住；一旦目标贴近前脚，分模块 pipeline 的边界就暴露出来了。

## Open-loop Demonstration Replay

- 论文还做了一个 open-loop demonstration replay：

    - 起始末端位置：$p=(0.5,-0.5,-1.2)$；
    - 目标末端位置：$p^{end}=(0.55,-0.9,0.4)$；
    - $T_{traj}=2.5s$；
    - robot 同时接收 forward velocity command：$v_x^{cmd}=0.35$。

- 现象是：

    - 一开始末端 command 比较高，机器人能用自然步态向前走；
    - 当 EE command 移到 torso 下方和接近地面时，机器人开始 pitch down、roll left、yaw right，帮助手臂够到目标；
    - 实验在 uneven grass terrain 上展示，说明它不是只在仿真或平地上成立。

- 这段实验说明了一个更动态的能力：机器人不是“先停下再伸手”，而是可以边走边做较大幅度机械臂运动。

# 关键结果：

- **统一策略有效：** Unified policy 在 survival、base acceleration、velocity error、EE error 上整体优于 separate policy 和 uncoordinated policy，而且不是靠显著增加 total energy 换来的。

- **身体协同确实扩大 workspace：** Unified policy 的 arm workspace 从 separate policy 的 $0.58m^3$ 提升到 $0.82m^3$，扰动下 survival 从 $0.64$ 提升到 $0.87$。

- **Advantage Mixing 主要解决训练早期局部最优：** 不加 mixing 时，policy 容易停在“末端跟踪还行，但不愿意探索腿部动作”的局部解；加 mixing 后 velocity error 下降更快。

- **ROA 缩小 realizability gap：** RMA 的 $\lVert z^\mu-z^\phi\rVert^2$ 是 $0.31$，Regularized Online Adaptation 降到 $2\times10^{-4}$，这说明部署时使用的 history latent 更接近 privileged latent。

- **真实 picking 明显优于 MPC+IK：** Easy tasks 中 success rate 是 $0.8$ vs $0.3$，Hard tasks 中是 $0.8$ vs $0.1$，同时 TTC 更短、自碰撞更少。

# 图表索引：

- **Fig. 1：** 展示 legged manipulator 通过 bend / stretch 扩展 workspace，并给出 wiping、picking、button pressing、placing、throwing 等真实任务图。

- **Fig. 2：** 展示训练期和部署期的数据流；训练期有 privileged encoder $\mu$、adaptation module $\phi$ 和 unified policy $\pi$，部署期只保留 $\phi$ 和 $\pi$。

- **Table 1：** 给出 manipulation / locomotion reward 的 following、energy、alive 三类项。

- **Table 2：** 给出 command variables 的训练采样范围和测试范围。

- **Table 3：** 给出 environment parameters 的训练范围和 OOD 测试范围。

- **Table 4：** 比较 unified、separate、uncoordinated policy 的仿真性能。

- **Table 5：** 展示 unified policy 如何扩大 arm workspace，并提高扰动下 survival。

- **Table 6：** 比较 Regularized Online Adaptation、RMA、Domain Randomization 和 Expert policy，重点看 realizability gap 和 OOD performance。

- **Table 7 / Table 10：** 展示真实 pick-up 任务中 Ours 与 MPC+IK 的 success rate、TTC、IK failure rate、self-collision rate。

- **Algorithm 1：** 给出 Regularized Online Adaptation 的交替更新流程，以及 $\lambda$ curriculum。

# 论文的核心贡献：

- **第一点：提出了统一 leg-arm policy。** 不再把 locomotion 和 manipulation 拆成两个独立 controller，而是让一个 policy 同时输出腿部和机械臂 target。

- **第二点：提出了 Advantage Mixing。** 通过 arm/leg action 与 manipulation/locomotion advantage 的分支混合，降低训练早期 credit assignment 难度，再逐渐恢复全任务耦合。

- **第三点：提出了 Regularized Online Adaptation。** 在线训练 privileged encoder 和 adaptation module，用 regularization 缩小 $z^\mu$ 与 $z^\phi$ 的 realizability gap，改善 RMA 式两阶段蒸馏的问题。

- **第四点：给出低成本 legged manipulator 实物系统。** 使用 Go1 + WidowX 的组合，硬件成本约 6K USD，相比 Spot Arm / ANYmal arm 这种 100K USD 级别平台更适合学术实验室复现思路。

- **第五点：做了真实世界任务验证。** 包括 joystick teleoperation、vision-guided pick-up、button pressing、whiteboard wiping、cup throwing、open-loop demo replay 等，展示了腿臂协同不是只在仿真 reward 里成立。

# 深度分析：

- 如果只看论文标题，Deep-WBC 很容易被理解成“又一个端到端 RL 控制器”。但真正有价值的是，它把 whole-body control 里的几个矛盾分别处理了：表达上用统一策略，训练上先降低 arm/leg credit assignment 难度，部署上再让 privileged latent 保持可由 history module 恢复。

- 这三个环节缺一个都不太稳：只做 unified policy，训练容易陷入局部最优；只做 Advantage Mixing，但没有 sim-to-real adaptation，真实动力学差异会直接打穿策略；只做 adaptation，但腿和机械臂仍然分层，hard reaching 里的 workspace 和 self-collision 问题还是绕不过去。

- 所以这篇论文的工程价值，不在于某一个公式特别复杂，而在于几个朴素选择组合得很顺：joint-space action、双 reward stream、advantage mixing、online latent regularization、简单 command interface、真实任务验证。

- 这里是推断：如果后续要把这条路线扩展到更复杂的 manipulation，比如遮挡抓取、软物体、工具使用或需要持续接触力的任务，关键新增模块大概率不是更复杂的 leg controller，而是更强的 perception / task command abstraction，以及能把 gripper 和 contact-rich manipulation 纳入统一 action / reward 的接口。

# 我觉得最值得注意的点：

## 1. Whole-body 的关键不是“动作维度拼接”，而是 workspace 被身体姿态重新定义

- 如果只从网络结构看，Deep-WBC 好像就是一个 18 维 action 的 policy。

- 但真正重要的是它改变了末端操作的边界：机械臂 workspace 不再只由 arm kinematics 决定，而是由整个 quadruped posture 决定。

- 传统移动机械臂经常把 base 当成一个移动平台，把 arm 当成执行器。但四足机器人不一样：

    - base 可以 pitch；
    - base 可以 roll；
    - 腿可以 bend/stretch；
    - 支撑状态可以改变末端可达区域；
    - 这些动作本身就是 manipulation strategy 的一部分。

- 所以 Deep-WBC 的一个核心贡献是把“身体姿态”变成了机械臂能力的一部分。

## 2. Advantage Mixing 是一个很实用的 credit assignment 工程补丁

- 这篇论文里我最喜欢的是 Advantage Mixing，因为它没有试图用一个特别复杂的算法解决所有问题。

- 它承认一个简单事实：

    - arm action 对末端误差更直接；
    - leg action 对速度跟踪和稳定性更直接；
    - 但最终二者又会互相影响。

- 所以训练时先利用这个近似因果结构，再逐渐放开耦合。

- 这和 HIMLoco 里 estimator/actor 分工、DreamWaQ 里 AdaBoot 的思想很接近：复杂系统最终要端到端，但训练早期不能把所有难度同时压到一个 loss 里。

## 3. Regularized Online Adaptation 解决的是“teacher latent 可不可以被 student 学到”

- 很多 teacher-student 方法默认 teacher 越强越好。

- 但 Deep-WBC 这篇论文提醒我们：teacher 太强也可能是问题。因为 teacher 使用 privileged information 学出来的 latent，可能包含 student 从 onboard history 里根本恢复不了的细节。

- 这就是 realizability gap。

- Regularized Online Adaptation 的思路很讲究：

    - 不是只让 student 追 teacher；
    - 也让 teacher/privileged encoder 不要离 student 可预测空间太远。

- 这对真实部署很重要。因为部署时真正执行的是 $z^\phi$，不是 $z^\mu$。如果 $z^\mu$ 在仿真里表现再好，但 $z^\phi$ 永远追不上，真实机器人还是会崩。

## 4. Joint-space control 不是退步，而是降低 sim-to-real gap 的选择

- 机械臂控制里，很多人天然会想到 operational space control 或 IK。

- 但 Deep-WBC 选择输出 joint position target，这个选择很现实：

    - 不在线求 IK；
    - 不把 IK singularity 暴露成主失败模式；
    - 避免 operation-space controller 和 legged base dynamics 的接口不一致；
    - 让 policy 直接在 joint/action 空间里学习 self-collision avoidance。

- Table 7 里 MPC+IK 的失败模式正好反过来证明了这个选择：IK failure 和 self-collision 在 hard tasks 中很明显。

## 5. 真实实验真正证明的是“协同动作”，不是视觉能力

- Vision-guided tracking 实验里用了 AprilTag 和一个很简单的 controller。视觉本身不是论文重点。

- 论文重点是：当 AprilTag 给出目标位置以后，Deep-WBC policy 能用全身动作完成 reaching，而 MPC+IK 在近身目标上容易失败。

- 所以不要把这篇论文理解成视觉 manipulation 论文。它更像是底层 whole-body motor control 论文：上层可以是 joystick、vision、demo replay，底层 unified policy 负责把 command 变成协调身体动作。

# 局限和风险：

- **第一点：安全约束不是显式保证。** 论文通过训练和 reward 学到了较好的 self-collision avoidance 和稳定性，但这不是形式化 safety proof。真实硬件部署仍然需要 torque/velocity/current limit、急停、watchdog、通信超时、NaN/Inf 保护等工程边界。

- **第二点：任务仍然是 command-following，不是通用 manipulation。** 论文展示了 picking、wiping、pressing、throwing 等任务，但 policy 本质上追的是 EE command 和 base command。抓取遮挡物体、软物体操作、复杂接触操作还没有被统一进 policy。

- **第三点：感知链路比较简单。** Vision-guided task 依赖 AprilTag，而不是复杂视觉识别或闭环 grasp planning。失败还会来自 cup 和 tag 的 pose mismatch。

- **第四点：OOD 仍然是预定义参数范围内的 OOD。** base payload、EE payload、friction、motor strength 的测试范围比训练更难，但真实世界还包含更多未建模因素。

- **第五点：统一策略的可解释性有限。** 我们能从 Figure 5 看到 body pitch/roll 和 command 的相关性，但无法直接知道 policy 内部如何分配腿臂协同。

- **第六点：平台规模有限。** 论文平台是 Go1 + WidowX 250s，任务偏轻量。更重的机械臂、更强接触力、更高速度或更复杂地形，可能需要显式动力学约束和安全层。

- **第七点：训练成本不低。** 5000 env、2 billion samples、200k gradient updates 对个人机器并不轻。当前项目如果要复现，需要先做最小闭环，不要直接照搬完整训练规模。

# 和我之前笔记里几个方法的关系：

## 和 HIMLoco 的关系

- HIMLoco 的核心是从历史 proprioception 中恢复 velocity + latent，用 estimator 解决 POMDP hidden state recovery。

- Deep-WBC 也有 history adaptation module，但关注点不同：

    - HIMLoco 更偏 locomotion 中的环境/扰动估计；
    - Deep-WBC 更偏 leg-arm whole-body coordination 和 sim-to-real extrinsics adaptation。

- 两者共同点：

    - 部署时不能直接吃 privileged information；
    - 都需要从历史观测中估计某种 latent；
    - actor/policy 需要依赖这个 latent 适应动力学变化。

- 区别在于：

    - HIMLoco 的 estimator 输出更像“当前脚感/扰动状态”；
    - Deep-WBC 的 $z_t$ 更像“环境 extrinsics 对当前整机动力学的影响”；
    - Deep-WBC 还额外处理 manipulation/locomotion 的 action-space credit assignment。

- 如果把两者结合，我觉得合理方向是：

    - 用 HIMLoco 风格的 history encoder 提升 $z^\phi$ 的表示能力；
    - 保留 Deep-WBC 的 Advantage Mixing 处理手腿任务 credit assignment；
    - 在当前 Go2+Piper 里先把 history/action/privileged shape 固定，再考虑更复杂 latent objective。

## 和 AMP_for_hardware 的关系

- AMP_for_hardware 解决的是动作风格和 reward engineering 问题：让 discriminator 从动物 motion 里提供 style reward，减少手工设计复杂自然运动奖励。

- Deep-WBC 解决的是腿臂协同和 whole-body command following 问题。

- 两者的共同工程目标是：仿真里学出的动作必须能上真实硬件。

- 但它们防止“仿真投机动作”的方式不同：

    - AMP 用 motion prior / discriminator 把动作拉回自然分布；
    - Deep-WBC 用 energy penalty、joint-space action、PD target、domain randomization、online adaptation 和真实任务验证拉回硬件可执行范围。

- 如果未来结合，AMP 可以给 Deep-WBC 的腿臂动作加一个 style prior，避免 policy 为了 reach 学出奇怪的高频姿态。但这个要小心，因为 arm manipulation 本身可能需要偏离动物步态，不能把 style reward 权重拉太高。

## 和 CTS-MoE 的关系

- CTS-MoE 关注多地形、多任务下的 expert 分工、router 和 multi-critic。

- Deep-WBC 没有 MoE，也没有地形分类器。它选择的是单一 unified policy + action branch + advantage mixing。

- 两者都在处理“一个策略要覆盖多个模式”这个问题，但切法不同：

    - CTS-MoE 是在网络结构上分专家；
    - Deep-WBC 是在 action/advantage 上分 credit；
    - CTS-MoE 的 router 决定听哪个专家；
    - Deep-WBC 的 $\beta$ 决定跨任务 advantage 混合到什么程度。

- 我觉得这两个方法可以互补：

    - 如果 Go2+Piper 的任务以后变成平地 reach、粗糙地形 reach、button press、pick-up、throwing 多种模式，MoE 可以处理模式容量；
    - Advantage Mixing 可以处理腿臂 credit assignment；
    - multi-critic 可以进一步避免 manipulation reward 和 locomotion reward 的 value interference。

## 和 DreamWaQ 的关系

- DreamWaQ 的关键是 proprioception-only implicit terrain imagination，用 CENet 从历史本体观测里估计 velocity + context latent。

- Deep-WBC 的 adaptation module 也从历史中估计 latent，但目标是替代 privileged extrinsics encoder。

- 两者都说明一件事：

    - 真实机器人部署时，history 不是可有可无的附加项；
    - history 是 partial observability 下恢复隐藏动力学条件的主要证据。

- 区别是：

    - DreamWaQ 主要解决复杂地形 locomotion；
    - Deep-WBC 主要解决带臂四足的整机操作；
    - DreamWaQ 的 context latent 更像地形/接触响应；
    - Deep-WBC 的 extrinsics latent 包含 mass、payload、friction、motor strength 等域随机化参数。

## 和 BFM-Zero 的关系

- BFM-Zero 更关注“任务 prompt”：把 reward、goal、motion 都映射到 latent，让一个 humanoid policy 能做不同任务。

- Deep-WBC 更关注“身体协同”：把 manipulation 和 locomotion 合成一个底层 motor policy。

- 两者都在向一个方向走：

    - 不再为每个任务训练一个孤立 policy；
    - 不再只让底层 policy 做单一 tracking；
    - 通过 latent/command/reward 接口让同一个身体控制器处理更多任务。

- 但 Deep-WBC 的接口仍然比较具体：EE pose command + base velocity command。它还不是 BFM-Zero 那种 reward/goal/motion 都可 prompt 的 foundation policy。

# 对当前 legged_wbc_mjlab 的工程启发：

- 当前项目里已经有 `deep_wbc/go2_piper` 任务方向，所以这篇论文对我们不是单纯背景阅读，而是可以直接转成实现约束。

## 1. 先借鉴数据流，不要复制 legacy 结构

- Deep-WBC 原论文/代码风格更接近 legacy `legged_gym + rsl_rl`，而当前项目是 mjlab manager-based API。

- 所以迁移时不要直接搬 `WidowGo1RoughCfg` 或 legacy env class。

- 应该先抽象成当前项目能接住的接口：

    - command manager 负责 base command 和 EE command；
    - observation terms 负责 current state、history、privileged group；
    - action manager 负责 18 维 joint target ordering；
    - reward terms 分别记录 manipulation / locomotion；
    - RL config 再决定是否支持 separated action heads、priv encoder、history encoder。

## 2. Action shape 和 ordering 必须先固定

- Deep-WBC 的 action 是：

$$
a_t=[a_t^{leg},a_t^{arm}]\in\mathbb{R}^{18}
$$

- 其中：

    - $a_t^{leg}\in\mathbb{R}^{12}$；
    - $a_t^{arm}\in\mathbb{R}^{6}$。

- 当前 Go2+Piper 实现必须先确认：

    - 12 个 Go2 joint 的顺序；
    - 6 个 Piper arm joint 的顺序；
    - action_scale；
    - default_joint_pos；
    - decimation；
    - delay；
    - clip；
    - 底层 target 到 torque 的边界。

- 如果 ordering 错了，后面 reward、policy、Advantage Mixing 全都会学到错误身体映射。

## 3. Command frame 不要随便改

- Deep-WBC 用球坐标 $(l,p,y)$ 表示 EE target，这对它的 WidowX+Go1 setup 很自然。

- 但当前项目里如果已经在使用 7-D pose 或某种 body/base frame，就不要为了贴论文直接改成 sphere command。

- 正确做法是先回答：

    - EE command 是在 world frame、base frame、arm base frame，还是某个 yaw-aligned frame？
    - orientation 用 quaternion、Euler delta，还是只控制 position？
    - reset 时 goal 如何采样？
    - step 中 goal 是否插值？
    - reward 里比较的是哪个 frame 下的误差？

- Deep-WBC 的球坐标可以作为一个 command generator 版本，但不应该悄悄替换当前 MDP 定义。

## 4. Reward 要分开记录，再统一训练

- Deep-WBC 的一个关键工程习惯是：

    - manipulation reward 有自己的 following / energy；
    - locomotion reward 有自己的 following / energy / alive；
    - 训练目标最终合并，但指标和 advantage 可以分开。

- 当前项目如果要做 Advantage Mixing，必须保留两类 return/advantage：

$$
A_{manip},\quad A_{loco}
$$

- 这意味着 rollout storage 或 PPO update 不能只保留一个 scalar reward 后就把信息丢掉。

- 最小实现可以先做 logging：

    - `rew_manip_tracking`；
    - `rew_manip_energy`；
    - `rew_loco_tracking`；
    - `rew_loco_energy`；
    - `rew_alive`。

- 等这些 term 和 episode stats 稳定后，再考虑把它们拆成两个 advantage path。

## 5. Advantage Mixing 需要算法层支持，不只是环境层配置

- Advantage Mixing 的公式在 PPO objective 里：

$$
\log\pi(a_t^{arm}\mid s_t)(A_{manip}+\beta A_{loco})
+
\log\pi(a_t^{leg}\mid s_t)(\beta A_{manip}+A_{loco})
$$

- 所以要实现它，至少需要：

    - 能分别计算/缓存 manipulation return 和 locomotion return；
    - 能得到 arm action log-prob 和 leg action log-prob，或者至少能按 action dimension 分开 log-prob；
    - PPO update 里能接收 $\beta$ curriculum；
    - actor distribution 的 shape 与 leg/arm split 一致。

- 这不是在 `rewards.py` 里加几个权重就能完成的。环境层只能提供 reward 分解，真正的 mixing 在 algorithm 层。

## 6. Regularized Online Adaptation 需要 privileged group + history group

- Deep-WBC 的 adaptation 需要两类输入：

    - privileged extrinsics $e_t$：训练中可用，如 mass、friction、payload、motor strength；
    - onboard history：部署可用，如最近 10 步 base/leg/arm state 和 previous actions。

- 当前项目如果要复现 ROA，需要先设计 observation groups：

    - actor current obs；
    - critic privileged obs；
    - history obs；
    - domain/randomization labels 或 latent target。

- 不要一开始就写复杂 adaptation loss。先确认这些张量每一步都能对齐：

$$
history: [N,H,D_{obs}],\quad z^\phi:[N,20],\quad z^\mu:[N,20]
$$

- 如果 history 和 privileged label 的时间索引错一帧，adaptation module 学到的就是错因果。

## 7. 先做最小闭环，再加完整 WBC 能力

- 建议路线：

    - 先实现 18 维 joint target action，确认 reset/step 不崩；
    - 只开 base velocity + 静态 EE target，观察 reward 和 termination；
    - 加 EE target 插值；
    - 加 domain randomization；
    - 加 history observation；
    - 加 privileged encoder / adaptation module；
    - 最后再加 Advantage Mixing。

- 不要一开始把 rough terrain、OOD payload、arm reaching、visual tracking、Advantage Mixing、ROA 全部同时打开。失败以后很难定位是 command frame、action ordering、reward scale、policy architecture 还是 adaptation loss 的问题。

# 如果要写成代码，大概有哪些模块：

- 这里不写具体代码，只整理接口，方便后续看 `legged_wbc_mjlab` 时对照。

## Command Generator

- 负责生成：

$$
v_x^{cmd},\quad \omega_{yaw}^{cmd},\quad p^{cmd},\quad o^{cmd}
$$

- 如果采用 Deep-WBC 球坐标，还要维护：

$$
l,p,y,T_{traj},p^{start},p^{end}
$$

- 必须明确：

    - 采样范围；
    - reset 时是否重采样；
    - command resampling period；
    - collision / underground check；
    - 插值方向；
    - command frame。

## Observations

- actor obs 至少包含：

    - base orientation / angular velocity；
    - leg joint position / velocity；
    - arm joint position / velocity；
    - previous action；
    - foot contact；
    - base command；
    - EE command / EE error；
    - adaptation latent 或 history encoder 输出。

- critic / privileged obs 可以包含：

    - randomized mass；
    - friction；
    - payload；
    - motor strength；
    - CoM shift；
    - terrain/domain parameters。

- 关键不是把维度凑出来，而是保证每个 term 的 frame、unit、time index 都一致。

## Actor-Critic / Policy

- 一个合理的结构可以先参考论文：

$$
input=[s_t,a_{t-1},command,z_t]
$$

- 输出：

$$
a_t^{leg}\in\mathbb{R}^{12},\qquad a_t^{arm}\in\mathbb{R}^{6}
$$

- 可以先用单 MLP + 18 维 head 做 baseline。等 baseline 跑通后，再考虑：

    - shared trunk + leg head + arm head；
    - separate log-prob split；
    - privileged encoder；
    - adaptation module；
    - Advantage Mixing。

## Rollout Storage / PPO Update

- 如果只做普通 PPO，storage 保留 scalar reward 即可。

- 如果做 Advantage Mixing，storage 需要额外保留：

$$
r_{manip},\quad r_{loco},\quad A_{manip},\quad A_{loco}
$$

- PPO update 里需要计算：

$$
\beta=\min(t/T_{mix},1)
$$

- 然后分别作用到 arm/leg log-prob。这里必须小心 distribution 的实现，如果当前 rsl_rl 只返回总 log-prob，就要确认能否按 action dimension 求和拆分。

## Adaptation Module

- 输入：最近 10 步 observation/action history。

- 输出：

$$
z_t^\phi\in\mathbb{R}^{20}
$$

- privileged encoder 输入 environment extrinsics：

$$
e_t\rightarrow z_t^\mu\in\mathbb{R}^{20}
$$

- loss：

$$
\lambda\|z^\mu-sg[z^\phi]\|^2+\|sg[z^\mu]-z^\phi\|^2
$$

- 更新节奏可以先不照搬 $H=20$，但至少要保留同一个思想：$\phi$ 在线模仿，$\mu$ 被约束到可模仿空间。

## Events / Domain Randomization

- 需要覆盖：

    - base payload；
    - EE payload；
    - CoM shift；
    - arm motor strength；
    - leg motor strength；
    - friction。

- 当前 mjlab 里应该通过 event manager 或等价机制实现，而不是把 randomization 写死在 env step 里。

## Deployment Boundary

- 真实部署前至少要检查：

    - action clip；
    - joint limit；
    - torque/current limit；
    - velocity limit；
    - self-collision approximation；
    - command bounds；
    - observation NaN/Inf；
    - communication timeout；
    - emergency stop；
    - gripper 是否由 policy 管；
    - Pi/TX2 或当前硬件之间的 latency。

- 未验证：这份笔记没有做真实硬件 safety review。Deep-WBC 论文展示的真实机器人结果不能直接替代当前项目的硬件安全验证。

# 论文最重要的边界：

- Deep-WBC 的结论不是“学习式 WBC 可以完全取代模型式控制”。

- 更准确的结论是：

    - 对轻量机械臂 + 四足底盘的动态 reaching / picking / wiping / pressing 等任务，一个统一 RL policy 可以学到比分模块 MPC+IK 更自然、更快、更能扩展 workspace 的腿臂协同；
    - Advantage Mixing 能缓解统一策略训练早期的 credit assignment 问题；
    - Regularized Online Adaptation 能让 privileged latent 更接近部署时 history module 可预测的 latent，从而提升 OOD sim-to-real 表现。

- 但它没有证明：

    - 任意复杂 manipulation 都可以端到端学出来；
    - 无需显式安全约束就能真实硬件安全；
    - AprilTag 视觉结果可以推广到一般视觉抓取；
    - OOD domain randomization 可以覆盖所有真实动力学差异；
    - 更重载、更强接触、更高速任务也能同样成立。

- 对我们当前工程来说，最该学的是数据流和训练思想，不是逐行复制：

    - action split；
    - reward split；
    - advantage split；
    - privileged/history latent；
    - command frame；
    - deployment-only observation boundary。

# 一句话总结：

- **Deep-WBC 的价值在于，它把带机械臂四足机器人的 manipulation 和 locomotion 从分模块协调问题，改写成一个统一策略的 whole-body motor control 问题，并用 Advantage Mixing 解决训练期 credit assignment、用 Regularized Online Adaptation 缩小 sim-to-real latent gap，让 Go1+WidowX 在真实任务中表现出自然的腿臂协同。**

# 我的笔记：

- 这篇论文我觉得最核心的是一句话：**机械臂的工作空间，其实是整台四足机器人的工作空间。**

- 如果只看 arm kinematics，我们会很自然地问：这个点机械臂够不够得到？IK 有没有解？会不会自碰撞？

- 但在四足平台上，这个问题应该换一种问法：

    - 腿能不能蹲低一点？
    - 身体能不能 pitch 下去？
    - 目标在侧面时，机器人能不能 roll 一点来扩大 reach？
    - 机械臂伸出去以后，腿能不能主动改变支撑来保持平衡？

- Deep-WBC 就是在把这个问题交给统一策略学习。

- Advantage Mixing 很值得记下来。它不是花哨算法，而是一个非常实用的训练期结构先验：早期让手和腿各自对最直接的任务负责，后期再让它们为全身目标共同负责。这个思想在我们自己的 Go2+Piper 里也很有用，因为一上来全耦合很容易不知道问题出在 reward、action、command 还是 policy。

- Regularized Online Adaptation 也很重要。很多时候我们说 teacher-student，只关注 student 能不能模仿 teacher；但这篇论文提醒我们，teacher 也不能学一个 student 永远预测不了的 latent。真实部署最终靠的是 history module，不是仿真 privileged encoder。

- 对当前 `legged_wbc_mjlab` 来说，我觉得最现实的路线不是马上复现完整 Deep-WBC，而是先做一个最小闭环：

    - 18 维 joint target action 对齐；
    - base velocity + EE pose command 对齐；
    - manipulation / locomotion reward 分开 logging；
    - reset/step/termination 跑通；
    - 再考虑 history adaptation；
    - 最后再做 Advantage Mixing。

- 这样做的原因很简单：whole-body learning 已经够复杂了，不要一上来把所有变量全部打开。先让机器人能稳定站、能走、能追一个简单末端目标，再逐步增加插值 goal、rough terrain、payload randomization、history latent 和优势混合。每一步只改变一个因素，失败了才知道该查哪里。

# 参考来源：

- `/home/kk/legged_wbc_mjlab/paper/Deep-Whole-Body-Control.pdf`

- `/home/kk/飞书文档/论文学习部分.md`

- `/home/kk/飞书文档/AMP_for_hardware学习记录.md`

- `/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`

- `/home/kk/legged_wbc_mjlab/docs/BFM-Zero阅读笔记.md`

- `/home/kk/legged_wbc_mjlab/docs/DreamWaQ阅读笔记.md`
