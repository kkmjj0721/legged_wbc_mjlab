# Sim-to-Real: Learning Agile Locomotion For Quadruped Robots 阅读笔记

## 前言：

- 这篇论文的核心目标，是让四足机器人在仿真里用深度强化学习学会敏捷步态，然后不经过真实机器人上的再训练，直接部署到真实硬件上。

- 论文标题是 **Sim-to-Real: Learning Agile Locomotion For Quadruped Robots**，作者使用 Ghost Robotics 的 Minitaur 作为平台，在 PyBullet / Bullet 仿真中训练，然后把策略部署到真实 Minitaur。

- 我们先把问题说清楚：这不是一篇“PPO 学会走路”的普通仿真论文。

- 它真正关心的是：**仿真里学到的反馈控制器，为什么一上真实机器人就可能崩，以及应该从哪些接口缩小 reality gap。**

- 这篇论文的时间比较早，2018 年就已经把几个后来在 legged RL 里反复出现的工程点串起来了：

    - actuator model 不能太理想化；

    - latency 必须进入训练闭环；

    - observation 不是越多越好；

    - dynamics randomization 是 trade-off，不是免费午餐；

    - 用户可以用 open-loop reference 控制 gait style，再让 feedback policy 补稳定性。

- 这里最值得我们关注的，不是 Minitaur 本身，也不是 PyBullet 本身，而是它给 `legged_wbc_mjlab` / MuJoCo / `rsl_rl` 项目提供了一个很实用的 checklist：

    - policy 输出的 action 到 motor torque 之间有没有和真实系统一致；

    - 控制延迟、传感器延迟、观测噪声是否在 rollout 中出现；

    - domain randomization 的范围是不是基于辨识和不确定性，而不是随手乱设；

    - reward 高是不是同时伴随可部署性，而不是只在仿真里好看。

- 未执行/未验证：这份笔记只基于 `/home/kk/下载/1804.10332v2.pdf` 的 PDF 文本抽取、本地写作风格参考和 `legged_wbc_mjlab` 中少量只读代码观察整理；没有运行论文代码，没有训练策略，没有复现 PyBullet 环境，也没有在真实机器人上验证任何部署结论。

## 论文核心信息：

- **论文标题：** Sim-to-Real: Learning Agile Locomotion For Quadruped Robots

- **arXiv：** arXiv:1804.10332v2 [cs.RO]

- **版本时间：** 2018-05-16

- **作者：** Jie Tan, Tingnan Zhang, Erwin Coumans, Atil Iscen, Yunfei Bai, Danijar Hafner, Steven Bohez, Vincent Vanhoucke

- **机构：** Google Brain、X、Google DeepMind

- **机器人平台：** Ghost Robotics Minitaur

- **执行器：** 8 个 direct-drive actuators，每条腿 2 个电机，在 sagittal plane 中运动

- **传感器：** motor encoders、IMU

- **板载计算：** STM32 ARM microcontroller + Nvidia Jetson TX2

- **通信链路：** TX2 与 microcontroller 之间通过 UART 通信

- **控制频率：** 真实系统约 150-200 Hz，且因为 TX2 不是实时系统，控制频率会波动

- **训练仿真器：** PyBullet / Bullet Physics Engine

- **RL 算法：** Proximal Policy Optimization, PPO

- **主要任务：** galloping 和 trotting

- **主要贡献：** 完整 sim-to-real locomotion pipeline、open-loop + feedback policy、actuator / latency / randomization / compact observation 的系统 ablation

- **论文里没有明确看到 DOI。** 所以这里不强行补 DOI。

- **会议/期刊信息：** PDF 文本抽取中没有明确给出会议或期刊入口；这份笔记只按 arXiv v2 版本处理。

## 一句话总结：

- 这篇论文的核心思想是：**先用更接近真实硬件的仿真闭环训练一个紧凑观测、硬件友好的反馈策略，再用 system identification、actuator model、latency modeling、dynamics randomization 和 perturbation 把仿真策略转到真实四足机器人上。**

- 如果用更工程一点的话说：

    - 不要只问“PPO 能不能学会跑”；

    - 要问“policy 在仿真里看到的 observation、发出的 action、经历的 actuator response、承受的 latency 和 noise，是否和真实部署足够接近”。

- 这也是它对后续 legged RL 的长期价值。

## 研究问题：

- 论文要回答的问题可以拆成两层。

- 第一层是 locomotion learning：

    - **能不能不用大量手工调参，让四足机器人自动学到敏捷步态？**

- 第二层是 sim-to-real：

    - **在仿真里学到的动态反馈控制器，能不能直接迁移到真实机器人，而不是只在仿真器里好看？**

- 为什么第二层更难？

    - 四足 locomotion 是强接触系统；

    - 每一步都有 foot contact switch；

    - 接触切换会把控制空间切成很多不连续区域；

    - 小的模型误差可能在落脚瞬间被放大；

    - 延迟、摩擦、执行器饱和、电池电压下降，都会改变闭环相位和稳定裕度。

- 所以这篇论文不是单纯追求“在 simulator 中 reward 最大”。

- 它真正想证明的是：

    - 如果仿真闭环足够贴近真实系统；

    - 如果 policy 训练时已经经历足够合理的不确定性；

    - 如果 observation space 不诱导网络过拟合仿真细节；

    - 那么深度 RL 学出的 gait 可以直接部署到真实 Minitaur 上。

- 作者 claim：论文报告 learned galloping 和 learned trotting 都可以从仿真直接部署到真实 Minitaur，不需要在真实硬件上继续训练。

- 未执行/未验证：这份笔记没有验证真实硬件部署，只记录作者论文中的实验结论。

## Baseline 的局限：

- 我们先看传统路线为什么不够。

### 第一种：手工 controller / 经验调参

- 经典四足控制可以通过手工设计 CPG、状态机、轨迹模板、PD 增益、触地逻辑、姿态补偿来做。

- 优点是：

    - 可解释；

    - 工程边界清楚；

    - 对特定平台和特定地面可以调得很稳。

- 但缺点也明显：

    - 需要很多机器人经验；

    - gait style、速度、能耗、稳定性之间要反复调；

    - 动态 gait 的接触相位复杂，人工写规则很容易变成大量特殊情况；

    - 换平台或换地面后，之前的参数未必还能用。

- 所以作者提出的问题是：能不能自动化这个 controller 设计过程。

### 第二种：只在仿真里做 deep RL

- 深度强化学习在仿真 locomotion benchmark 中很强。

- 但如果只看仿真 reward，会遇到一个核心问题：

    - 仿真器可能给了策略一个过于理想的世界。

- 例如：

    - motor command 立即生效；

    - sensor measurement 立即返回；

    - actuator torque 与控制信号线性对应；

    - contact friction 固定不变；

    - IMU 没有 bias 和 noise；

    - 质量、惯量、摩擦、控制周期都是精确常数。

- 在这种世界里学出的反馈策略，可能会利用仿真器的漏洞。

- 一上真实机器人，延迟和饱和会让相位滞后变大，policy 的补偿动作可能从“稳定”变成“振荡”。

- 这就是 reality gap。

### 第三种：直接在真实机器人上学习

- 另一个直觉是：既然仿真有 gap，那就在真实机器人上学。

- 但 locomotion 和抓取不一样。

- 真实机器人上学习 gait 的成本很高：

    - 每次摔倒都要重置；

    - 采样速度慢；

    - 高速摔倒可能损坏电机、连杆和电池；

    - 安全人员、场地和急停都要投入。

- 所以论文选择的是更实际的路线：

    - 在仿真中训练；

    - 但让仿真闭环尽量接近真实；

    - 再训练鲁棒策略。

## 方法主线：

- 这篇论文的方法可以概括成四个模块。

- **第一，POMDP + PPO。**

    - 把 locomotion 控制建模为部分可观测强化学习问题；

    - policy 只看真实机器人可获得的 observation；

    - 用 PPO 训练 neural feedback policy。

- **第二，open-loop + feedback policy。**

    - 用户可以给一个 open-loop reference gait；

    - RL policy 学习在这个 reference 上做反馈修正；

    - 如果 reference 为 0 且 feedback 输出范围大，就接近从零学 gait。

- **第三，提高 simulation fidelity。**

    - 通过 system identification 建 URDF；

    - 加入更真实的 actuator model；

    - 加入 latency modeling。

- **第四，学习 robust controller。**

    - dynamics randomization；

    - random perturbation；

    - compact observation space。

- 关键点：这四块不是互相独立的技巧。

- 它们共同围绕一个目标：**让 policy 训练时经历的因果链，尽量接近真实部署时的因果链。**

## POMDP 建模：

- 作者把 locomotion 写成一个 MDP：

$$
M=(S,A,r,D,P_{sas'},\gamma)
$$

- 这里：

    - $S$ 是完整状态空间；

    - $A$ 是动作空间；

    - $r$ 是 reward function；

    - $D$ 是初始状态分布；

    - $P_{sas'}$ 是 transition probability；

    - $\gamma\in[0,1]$ 是 discount factor。

- 但是 Minitaur 真实部署时拿不到完整状态。

- 比如：

    - base 在全局坐标中的位置不直接可见；

    - foot contact forces 没有对应传感器；

    - 地面摩擦、真实电机响应和接触状态都不是完整可观测。

- 所以实际问题是 POMDP。

- 每个控制 step，policy 看到的是 partial observation $o\in O$，不是完整 state $s\in S$。

- 策略写成：

$$
\pi:O\mapsto A
$$

- 优化目标是最大化期望累计回报：

$$
\pi^*=\arg\max_{\pi}\ \mathbb{E}_{s_0\sim D}\left[R_{\pi}(s_0)\right]
$$

- 这里 $R_{\pi}(s_0)$ 是从初始状态 $s_0$ 出发，按照策略 $\pi$ 执行得到的累计 reward。

- 为什么要强调 POMDP？

    - 因为 sim-to-real 里很多 gap 都表现为隐藏状态；

    - policy 只能通过短时响应、IMU、关节角等信息间接感知；

    - 如果 observation 设计不合理，policy 可能在仿真中依赖真实世界不稳定的信号。

- 这里和后来的 RMA、DreamWaQ、HIMLoco 有同一条主线：

    - 真实 legged locomotion 几乎永远不是完全可观测；

    - 关键不是把所有仿真 state 都塞给 actor；

    - 而是选择部署时可靠、分布差异小、对控制足够的信息。

## PPO 训练设置：

- 论文选择 PPO，主要因为它是稳定的 on-policy 方法，而且容易并行化。

- policy 和 value function 都是 fully-connected neural network。

- 每个网络都有两层 hidden layer。

- hidden size 通过 hyperparameter search 决定。

- 每次 policy update，作者并行采集 25 条 rollout。

- 每条 rollout 最长 1000 steps。

- 训练最多跑 7 million simulation steps。

- episode 结束条件：

    - 达到 1000 steps；

    - 或者 simulated Minitaur 失去平衡，base tilt 超过 0.5 rad。

- Table II 给出的训练参数是：

| Gait | Observation Dimension | Policy Net Size | Value Net Size | Learning Time |
|---|---:|---:|---:|---:|
| Trotting | 4 | (125, 89) | (89, 55) | 4.35 h |
| Galloping | 12 | (185, 95) | (95, 85) | 3.25 h |

- 这里有个很有意思的点：

    - galloping 用 12D observation；

    - trotting 最终为了更好迁移，降到了 4D observation。

- 这说明“网络更大、观测更多、仿真 reward 更高”不一定等于“真实部署更好”。

- 对 `rsl_rl` 来说，这个提醒很直接：

    - actor observation 和 critic observation 应该分开考虑；

    - actor 不要轻易吃真实部署时噪声大、延迟大、难标定的量；

    - critic 可以看更多 privileged information，但 actor 的输入最好保守。

## Observation Space：

- 作者设计了一个比较紧凑的 observation space。

- 基础 observation 包括：

    - base roll；

    - base pitch；

    - roll angular velocity；

    - pitch angular velocity；

    - 8 个 motor angles。

- 合起来是 12 维。

- 注意，作者没有把所有传感器量都塞进去。

- 例如：

    - IMU 可以提供 yaw，但 yaw measurement 很快漂移，所以不用；

    - motor velocity 可以计算，但噪声较大，所以不用。

- 这里的关键点是：

    - observation 不是越完整越好；

    - 真实部署时不稳定的信号，可能会扩大 sim-to-real gap；

    - 紧凑 observation 会减少 policy 过拟合仿真细节的机会。

- trotting ablation 里，作者进一步把 observation 降成 4D：

    - roll；

    - pitch；

    - roll angular velocity；

    - pitch angular velocity。

- 结果是：

    - 12D observation 在仿真中表现更好；

    - 4D observation 在真实机器人上表现更稳；

    - 小 observation space 的训练/测试分布更容易重合。

- 这里是推断：对现代 Go2 / ANYmal 这类机器人，我们不一定要极端压到 4D，因为现在的状态估计、关节速度和仿真器更强；但这篇论文提醒我们，actor 观测里的每一项都应该问一句：真实部署时它的噪声、延迟、坐标系和归一化是否能匹配训练。

## Action Space：

- 作者没有直接让 policy 输出 8 个 motor desired angles。

- 它选择在 leg space 中输出每条腿的 desired pose。

- 每条腿的 pose 分成两个量：

    - swing component $s$；

    - extension component $e$。

- 对同一条腿的两个电机，映射关系写成：

$$
\theta_1=e+s
$$

$$
\theta_2=e-s
$$

- 其中：

    - $\theta_1$ 和 $\theta_2$ 是控制同一条腿的两个 motor angles；

    - $s$ 控制 leg 的 swing；

    - $e$ 控制 leg 的 extension。

- 为什么不用 motor space？

    - motor space 里很多角度组合会造成 body parts self-collision；

    - valid configurations 会变成分散的、非凸的区域；

    - RL 在这种 action space 里探索更困难。

- leg space 的好处是：

    - 可以用简单的矩形 bounds 剪掉大部分 invalid actions；

    - 同时保留大多数有效姿态；

    - 对 gait style 来说也更直观。

- 注意：PDF 图注提到 extension / swing 对电机“同向/反向”旋转的物理解释，但电机安装和符号约定会影响直观方向。工程实现时不能只靠文字记忆，必须以实际 URDF/MJCF joint axis、gear sign、默认姿态和 actuator convention 重新核对。

## Reward Function：

- 论文的 reward 非常简单。

- 它鼓励机器人向目标方向跑得更远，同时惩罚能量消耗：

$$
r=(p_n-p_{n-1})\cdot d-w\Delta t|\tau_n\cdot\dot{q}_n|
$$

- 这里：

    - $p_n$ 是当前 time step 的 base position；

    - $p_{n-1}$ 是上一个 time step 的 base position；

    - $d$ 是期望奔跑方向；

    - $\Delta t$ 是 time step；

    - $\tau_n$ 是 motor torque；

    - $\dot{q}_n$ 是 motor velocity；

    - $w$ 是能耗惩罚权重。

- 作者使用 $w=0.008$，并说 PPO 对较宽范围的 $w$ 比较鲁棒，所以没有精调这个权重。

- 第一项：

    - 衡量沿目标方向的位移；

    - 本质上是 forward progress。

- 第二项：

    - 衡量 mechanical power proxy；

    - 惩罚高 torque 和高 joint velocity 的组合；

    - 避免 policy 靠大力高速乱甩腿拿速度。

- 这里很有意思：论文没有写复杂的 foot clearance、foot slip、gait phase、body height、torque rate reward。

- 它能学出 gallop，说明在合适 action space 和仿真闭环下，简单 reward 也能诱导出复杂 gait。

- 但我们不能把这个结论过度推广。

- 局限是：

    - 任务是在 flat ground 上最大化奔跑速度；

    - 不包含速度命令跟踪；

    - 不包含转向；

    - 不包含复杂地形；

    - 不包含真实外部感知。

- 所以对于 `legged_wbc_mjlab` 的 Go2 velocity task，不能直接照搬这个 reward。

- 更合理的借鉴是：

    - reward 项不要无节制堆叠；

    - 每个 reward 都要知道它对应的真实硬件问题；

    - 速度、能耗、姿态、动作平滑、脚端接触要分开记录，避免只看总 reward。

## Open-loop + Feedback Policy：

- 这篇论文非常重要的一点，是它没有把“从零学习”和“人类指定步态”对立起来。

- 作者把 controller 拆成两部分：

$$
a(t,o)=\bar{a}(t)+\pi(o)
$$

- 这里：

    - $\bar{a}(t)$ 是 open-loop component，通常是周期信号；

    - $\pi(o)$ 是 feedback component，由神经网络表示；

    - $a(t,o)$ 是最终 action。

- 为什么这样设计？

    - open-loop reference 方便人指定 gait style；

    - feedback policy 负责平衡、速度、能耗和扰动恢复；

    - 两者叠加后，可以连续调节“人类先验”和“自主学习”的比例。

- 如果想完全由用户指定：

    - 把 feedback policy 的上下界都设为 0；

    - 那么 $a(t,o)=\bar{a}(t)$。

- 如果想完全从零学：

    - 设 $\bar{a}(t)=0$；

    - 给 $\pi(o)$ 足够大的 action bounds。

- 边界要分清楚：galloping 中 open-loop component 显式设为 0；trotting 中才使用非零 gait reference。

- 如果想让机器人按某种风格 trot：

    - 给一个 open-loop trot reference；

    - 再让 $\pi(o)$ 在较小范围内补偿。

- 这个设计和现代 legged RL 里的 phase / gait prior 很像。

- 但注意它不是简单动作后处理。

- 因为 feedback policy 是在训练过程中学习如何叠加到 open-loop reference 上的。

- 如果部署时再临时把一个 reference 加到 policy 输出上，policy 没见过这种组合，就可能破坏稳定性。

## Trotting 的 open-loop reference：

- 作者想让 Minitaur 学 trotting。

- Trotting 的特点是对角腿同步运动。

- 作者没有靠 reward shaping 去手写“对角腿一起动”。

- 它直接给 open-loop component 两条正弦曲线：

$$
\bar{s}(t)=0.3\sin(4\pi t)
$$

$$
\bar{e}(t)=0.35\sin(4\pi t)+2
$$

- 这里：

    - $\bar{s}(t)$ 是 swing reference；

    - $\bar{e}(t)$ 是 extension reference；

    - 一个对角腿 pair 用同相曲线；

    - 另一个对角腿 pair 相位差 180 度。

- 关键点：这个 open-loop controller 自己并不能让真实 Minitaur 向前走。

- 论文说 open-loop signal 单独执行时，真实 Minitaur 会立刻失去平衡，坐到后腿上。

- 所以 RL 学的不是“跟随曲线”这么简单。

- 它学的是：

    - 在 reference 附近保持 gait style；

    - 同时调整每一步的 leg pose；

    - 让机器人保持平衡并产生前进速度。

- trotting feedback bounds 被设得很小：

$$
\pi(o)\in[-0.25,0.25]\ \text{rad}
$$

- swing 和 extension 两个自由度都是这个范围。

- 这相当于告诉 policy：

    - 你可以纠偏；

    - 但不要完全改掉用户指定的 trot style。

## Galloping 的从零学习：

- galloping 实验里，作者让系统从零学：

$$
\bar{a}(t)=0
$$

- feedback component 的输出范围更大。

- swing bounds：

$$
s\in[-0.5,0.5]\ \text{rad}
$$

- extension bounds：

$$
e\in\left[\frac{\pi}{2}-0.5,\frac{\pi}{2}+0.5\right]\ \text{rad}
$$

- 一开始，如果使用 baseline simulation，也就是没有 actuator model 和 latency，系统没有学出敏捷 gait。

- 它在仿真中学到的是比较慢的 walking gait。

- 而且真实 Minitaur 会因为 reality gap 直接摔倒。

- 加入 improved simulation 后，galloping gait 自动出现。

- 作者报告的速度是：

| Task | Simulation Speed | Real Speed |
|---|---:|---:|
| Galloping | 1.34 m/s | 1.18 m/s |

- 论文还把速度换算成 body lengths per second：

    - simulation：2.48 body lengths/s；

    - real：2.18 body lengths/s。

- 作者重复不同 hyperparameters 和 random seeds 后，发现多数解会收敛到 galloping。

- 也出现少量其他 gait，包括 trotting、pacing，以及一些不常见四足动物 gait。

- 这里是推断：这说明 reward 只给了“向前快跑 + 省能耗”，gait pattern 是系统在 action bounds、robot morphology、contact dynamics 和 energy penalty 下自己找到的局部最优行为。

## Leg-space 到 Motor-space 映射的工程意义：

- 我们再单独看一下 leg-space mapping，因为这对工程实现非常重要。

- policy 输出的不是直接 torque。

- policy 输出的是 leg pose target。

- 然后 leg pose target 通过固定映射转成 motor target。

- 这形成了一个中间 action manifold。

- 它的作用类似：

    - 给 RL 一个更容易探索的动作坐标；

    - 把明显无效或危险的 motor combination 排除掉；

    - 让 open-loop gait reference 能用更直观的 swing / extension 表达。

- 如果我们把这件事映射到 Go2：

    - Go2 不是 Minitaur 的五杆/并联腿结构；

    - 每条腿通常是 hip / thigh / calf 三个关节；

    - 所以不能照搬 $\theta_1=e+s,\theta_2=e-s$。

- 但抽象思想仍然有用：

    - 可以让 policy 输出 joint position target offset；

    - 可以按不同关节设置 action scale；

    - hip 的 lateral swing scale 可以小一些；

    - thigh / calf 可以保留更大的 sagittal motion；

    - action bounds 和 default offset 要和 robot morphology 绑定。

- 代码观察：`legged_wbc_mjlab` 当前 Go2 velocity task 中，`JointPositionActionCfg` 使用 `use_default_offset=True`，并按 hip / thigh / calf 设置不同 scale，其中 hip 乘了 `hip_reduction`。

- 这里是推断：这和论文里的 leg-space 思想是同一类工程约束，都是在 policy action 和真实 joint target 之间加一个可控、可解释、可限幅的 action interface。

## Actuator Model：

- 这篇论文最硬的工程部分之一，就是 actuator model。

- 作者指出，Bullet 默认 position control 和真实 PD servo 不完全一样。

- Bullet 的默认 position control 会在每个 motor 上构造一个约束，让当前 time step 末端满足：

$$
e_{n+1}=k_p(\bar{q}-q_{n+1})+k_d(\dot{\bar{q}}-\dot{q}_{n+1})
$$

- 这里：

    - $\bar{q}$ 是 desired motor angle；

    - $\dot{\bar{q}}$ 是 desired motor velocity；

    - $q_{n+1}$ 是当前 step 结束时的 motor angle；

    - $\dot{q}_{n+1}$ 是当前 step 结束时的 motor velocity；

    - $k_p$ 和 $k_d$ 是 position / velocity gain。

- 看起来它很像 PD，但关键差别是：

    - Bullet 的约束使用 step end 的状态；

    - 真实 PD servo 使用当前状态计算控制输入；

    - 仿真约束可能让大 gain 下的 motor 看起来稳定；

    - 真实电机却会振荡。

- 所以作者没有直接依赖默认 position control。

- 他们建立了一个基于理想 DC motor 的模型：

$$
\tau=K_t I
$$

$$
I=\frac{V_{pwm}-V_{emf}}{R}
$$

$$
V_{emf}=K_t\dot{q}
$$

- 这里：

    - $\tau$ 是 motor torque；

    - $K_t$ 是 torque constant，也等价于 back EMF constant；

    - $I$ 是 armature current；

    - $V_{pwm}$ 是 PWM 调制后的供电电压；

    - $V_{emf}$ 是 back EMF voltage；

    - $R$ 是 armature resistance；

    - $\dot{q}$ 是 motor angular velocity。

- 如果只用这个线性模型，仍然不够。

- 作者观察到：真实 Minitaur 经常会 sink to its feet，或者无法抬脚；但同一个 controller 在仿真里工作正常。

- 原因是理想 DC motor 的 torque-current 线性关系在真实大电流下不成立。

- 真实 motor 会发生 torque saturation。

- 所以作者进一步使用 piece-wise linear function 表征 nonlinear torque-current relation。

- 在仿真中：

    - 先由 PWM 和 back EMF 算 current；

    - 再通过 piece-wise function 查 torque；

    - 最后把 torque 作用到仿真 motor 上。

- position control 下，PWM 由真实 PD servo 形式给出：

$$
V_{pwm}=V\left(k_p(\bar{q}-q_n)+k_d(\dot{\bar{q}}-\dot{q}_n)\right)
$$

- 这里 $V$ 是 battery voltage。

- 注意公式使用的是当前 step 的 $q_n$ 和 $\dot{q}_n$，不是 step end 的 $q_{n+1}$。

- 作者还把 target velocity 设为：

$$
\dot{\bar{q}}=0
$$

- 这是根据 Ghost Robotics microcontroller 上的 PD 实现设置的。

- 作者做了一个验证实验：给 motor 一个 sine desired trajectory，比较真实 motor trajectory 和仿真 motor trajectory。

- 作者 claim：加入新 actuator model 后，仿真轨迹与真实测量更一致。

- 未执行/未验证：这份笔记没有复现实验，也没有拿到 Minitaur 的电机参数、PWM 曲线或 piece-wise torque-current 表。

## Actuator Model 给 MuJoCo 的启发：

- MuJoCo 本身可以建 actuator、gear、ctrlrange、forcerange、joint damping、armature、frictionloss。

- 但这篇论文提醒我们：只写一个 position actuator 不一定够。

- 为什么？

    - policy 输出的是 joint target；

    - joint target 到真实 torque 要经过 PD、控制周期、低层限幅、电机电流、电压、back EMF、热保护、通信延迟；

    - 仿真如果省略这些，会让 policy 学到硬件上不存在的控制能力。

- 对 `legged_wbc_mjlab` 来说，当前最现实的做法不是马上复现 Minitaur 的 DC motor model。

- 更实际的分阶段路径是：

    - 先核对 MJCF actuator 的 `ctrlrange`、`forcerange`、gear sign、joint axis 和 default pose；

    - 再确认 action scale 对应真实 joint target 幅度；

    - 然后加入 delay / action hold / encoder bias / motor strength randomization；

    - 最后如果有真实 log，再拟合更准确的 actuator response。

- 代码观察：当前 `Go2Cfg.control` 中已经有 `stiffness`、`damping`、`effort_limit`、`armature`、`friction`、`delay_min_lag`、`delay_max_lag`、`delay_hold_prob` 和 `delay_update_period` 这些入口。

- 这说明项目里已经有一些 actuator / delay 建模的配置边界。

- 未执行/未验证：我没有运行当前项目来确认这些字段是否全部被 MuJoCo actuator 或 mjlab runtime 实际消费，也没有验证它们在真实 Go2 上的单位和效果。

## Latency Modeling：

- latency 是这篇论文里的另一个关键点。

- 作者定义的 latency，是 motor command 被发送、导致机器人状态变化，然后 sensor measurement 把这个变化报告给 controller 之间的时间延迟。

- 在 Bullet 默认设置里：

    - motor command 立即生效；

    - sensor state 立即返回；

    - controller 看到的是无延迟反馈。

- 这会人为扩大 feedback controller 的 stability region。

- 也就是说：

    - 仿真里 policy 觉得自己可以很快补偿姿态误差；

    - 真实硬件上反馈慢了十几毫秒；

    - 同样的补偿可能已经过时；

    - 于是动作开始振荡、发散，最后摔倒。

- 作者的 latency model 很朴素，但很有用。

- 它保存 observation history：

$$
\{(t_i,O_i)\}_{i=0,1,\ldots,n-1},\qquad t_i=i\Delta t
$$

- 当前 step $n$ 需要 observation 时，不直接给 $O_n$。

- 而是查找两个相邻历史观测 $O_i$ 和 $O_{i+1}$，满足：

$$
t_i\le n\Delta t-t_{latency}\le t_{i+1}
$$

- 然后线性插值得到延迟后的 observation。

- 这样 policy 训练时看到的就是“过去某个时间点的传感器状态”。

- 作者还测了真实系统上的 latency。

- 方法是：

    - 发送持续一个 time step 的 PWM spike；

    - 让 motor 产生小幅运动；

    - 测量 spike 发出到 motor movement 被 sensor report 之间的时间。

- 作者报告有两种 latency：

    - microcontroller 上 PD servo latency 较低，约 3 ms；

    - TX2 上 locomotion controller latency 更高，通常 15-19 ms。

- dynamics randomization 中，latency 随机范围设为：

$$
t_{latency}\in[0,40]\ \text{ms}
$$

- control step 随机范围设为：

$$
\Delta t_{control}\in[3,20]\ \text{ms}
$$

- 这里很关键：

    - 论文不只建模固定延迟；

    - 也承认真实非实时系统的控制周期会波动；

    - 所以把 control step 也随机化。

## Latency 给 rsl_rl rollout 的启发：

- 对现代 `rsl_rl` / vectorized env 来说，latency 不能只在部署代码里加。

- 如果训练 rollout 里没有 latency，policy 仍然会学到无延迟反馈。

- 更合理的是在环境层加：

```python
# conceptual config, not executable training code
control_delay = {
    "action_lag_steps": (0, 4),
    "observation_lag_steps": (0, 4),
    "hold_probability": 0.5,
    "resample_every_steps": 10,
}
```

- 这段不是让当前项目直接照抄。

- 它表达的是接口：

    - action 可能不是本 step 立刻生效；

    - observation 可能来自历史 buffer；

    - delay 可以在 episode reset 或 interval 中重采样；

    - delay 的单位必须和 simulation timestep / decimation 对齐。

- 代码观察：`legged_wbc_mjlab` 当前 Go2 配置里有 `delay_min_lag=0`、`delay_max_lag=4`、`delay_hold_prob=0.5`、`delay_update_period=10`。

- 如果 MuJoCo timestep 是 0.005 s，decimation 是 4，那么 policy control step 约为：

$$
0.005\times4=0.02\ \text{s}=20\ \text{ms}
$$

- 这里是推断：如果 lag step 作用在 control step 级别，那么 4 step 可能对应约 80 ms；如果作用在 sim step 级别，则是 20 ms。这个必须看 mjlab actuator delay 的实际实现，不能只看配置名。

- 未执行/未验证：我没有追踪 mjlab 内部 delay 实现，也没有验证这些 delay 参数是在 action path、actuator path 还是 sensor path 生效。

## System Identification：

- 作者把缩小 reality gap 分成两类：

    - improve simulation fidelity；

    - learn robust controllers。

- system identification 属于第一类。

- 他们首先建立更准确的 URDF。

- 做法包括：

    - 拆解一台 Minitaur；

    - 测量每个 link 的尺寸；

    - 称重；

    - 找每个 link 的 center of mass；

    - 把这些信息写进 URDF。

- inertia 比较难直接测。

- 作者用 shape 和 mass 估计每个 link 的 inertia，并假设 density uniform。

- 作者还设计实验测 motor friction。

- 这里有一个很重要的工程顺序：

    - 先尽量把确定的 physical parameters 测准；

    - 对不确定、会变化或难测的参数再 randomize；

    - 不要把所有问题都交给 domain randomization。

- 为什么？

    - randomization 范围太窄，真实系统可能出界；

    - randomization 范围太宽，policy 会变保守，peak performance 下降；

    - 如果基础模型错得太远，perturbation 和 randomization 也救不回来。

- 作者在 ablation 中明确看到：baseline simulation 加 random perturbation 仍然不能让 policy 真实部署稳定；只有 improved simulation + perturbation 才显著缩小 gap。

## Dynamics Randomization：

- 作者在每个 episode 开始时随机采样一组物理参数。

- 采样方式是 uniform sampling。

- 论文 Table I 给出的 randomized parameters 和范围如下：

| Parameter | Lower Bound | Upper Bound |
|---|---:|---:|
| mass | 80% | 120% |
| motor friction | 0 Nm | 0.05 Nm |
| inertia | 50% | 150% |
| motor strength | 80% | 120% |
| control step | 3 ms | 20 ms |
| latency | 0 ms | 40 ms |
| battery voltage | 14.0 V | 16.8 V |
| contact friction | 0.5 | 1.25 |
| IMU bias | -0.05 rad | 0.05 rad |
| IMU noise std | 0 rad | 0.05 rad |

- 为什么这些范围不是随便设？

    - mass 和 motor friction 已经通过 system identification 测过，所以范围保守；

    - inertia 是基于 uniform density 假设估计的，不确定性更大，所以范围更宽；

    - motor strength 会随 wear and tear 变化；

    - control step 和 latency 会因非实时系统波动；

    - battery voltage 会随电量变化；

    - contact friction 难以准确辨识，所以只随机 lateral friction；

    - IMU 会有 bias 和 noise。

- contact friction 的范围 0.5 到 1.25，作者解释为 Minitaur 橡胶脚和不同 carpet floor 之间典型摩擦范围。

- 这里有个很重要的工程启发：

    - randomization 应该来自“我们知道哪些东西会变、哪些东西没测准”；

    - 而不是为了看起来 robust，把所有参数随便乘一个大范围。

- 这里是推断：对 Go2 / MuJoCo 项目来说，最好把 randomization 分成三类：

    - 已测但有小误差：mass、COM、joint damping；

    - 真实运行会漂移：motor strength、电压、摩擦、延迟；

    - 仿真器接触模型不可靠：contact friction、restitution、solver iteration、foot compliance proxy。

## Random Perturbation：

- 除了 dynamics randomization，作者还加入 random perturbation。

- 训练时，每隔 200 个 simulation steps 给 base 一个扰动力。

- 论文说 200 simulation steps 约等于 1.2 s。

- 扰动力持续 10 steps，约 0.06 s。

- force direction 随机。

- force magnitude 范围是：

$$
F\in[130,220]\ \text{N}
$$

- 作用位置是 simulated Minitaur 的 base。

- 这个扰动的作用不是模拟某一个具体物理参数。

- 它更像是让机器人练习“被打偏以后怎么恢复”。

- 作者还指出，物理参数不确定性可以看成额外 force / torque 对系统的影响，所以 randomization 和 perturbation 在鲁棒性上有类似作用。

- 但二者并不完全等价。

- dynamics randomization 改的是环境动力学本身。

- perturbation 改的是 rollout 中某些瞬间的外部扰动。

- 对足式机器人来说，两者最好都可以记录：

    - parameter randomization 看长期动力学适应；

    - push perturbation 看短时恢复和姿态稳定。

- 代码观察：当前 `legged_wbc_mjlab` Go2 velocity env 里有 `push_robot` interval event，通过设置 base velocity 的方式施加扰动，interval 是 5.0-6.0 s。

- 这和论文的 perturbation 思路相近，但实现形式不同：论文写的是 external perturbation force，当前项目代码观察到的是 velocity perturbation event。

- 未执行/未验证：我没有运行环境确认这个 event 在 MuJoCo step 中的实际效果，也没有比较它和真实外力冲击的等价性。

## Compact Observation Space：

- 作者把 compact observation space 作为缩小 reality gap 的第三个 robust controller 方法。

- 这点容易被忽视。

- 很多时候我们会本能地给 policy 更多信息。

- 但作者的实验显示：大 observation space 在仿真里更容易拿高分，真实部署反而更差。

- 为什么？

    - 观测维度越高，训练数据在 observation space 中越稀疏；

    - 仿真和真实之间只要每个维度有一点 mismatch，组合起来就可能变成很大的 distribution shift；

    - policy 会利用仿真中稳定、真实中噪声大的细节；

    - 真实机器人遇到相似 observation 的概率变低。

- 小 observation space 的好处是：

    - 训练和测试分布更容易重叠；

    - policy 被迫依赖最可靠的姿态反馈；

    - 不容易过拟合 motor angle 的仿真细节。

- 但小 observation 也有代价。

- 它可能限制 peak performance。

- 也可能让复杂地形、速度命令、转向任务不够可控。

- 所以这里不是“观测越少越好”。

- 更准确的结论是：

    - actor observation 应该只包含部署可靠且任务必要的信息；

    - critic 可以看更多 privileged information；

    - 如果 actor 需要历史，就让历史经过清晰的 estimator 或 recurrent module，而不是无脑堆一堆高噪声字段。

## 实验一：Galloping 从零涌现：

- galloping 实验是论文的亮点。

- 目标 reward 没有显式写 gallop。

- 用户也没有给 open-loop gait。

- 只设置：

    - $\bar{a}(t)=0$；

    - feedback action bounds 足够大；

    - reward 鼓励向前速度和低能耗。

- baseline simulation 下没有学出敏捷 gait，并且真实部署失败。

- improved simulation 后，galloping 自动出现。

- 作者报告：

    - simulation speed roughly 1.34 m/s；

    - real speed roughly 1.18 m/s；

    - learned gait 可以直接部署；

    - 多数 random seeds / hyperparameters 会收敛到 galloping。

- 这里的关键点是：gait emergence 依赖仿真 fidelity。

- 如果 actuator model 过于 overdamped，系统可能学不出敏捷运动。

- 如果 latency 不存在，学出的敏捷 feedback 可能真实不稳定。

- 所以“能否 emergence”不是只由 reward 决定，也由 simulator interface 决定。

## 实验二：Trotting 由 reference 控制风格：

- trotting 实验展示的是另一种使用方式。

- 用户想指定中速对角 gait。

- 但不想手写完整 balance controller。

- 作者给 open-loop trot signal，再用 RL 学 feedback residual。

- 训练中使用 improved simulator 和 random perturbations。

- 一开始使用较大 observation space 时，真实部署结果 mixed：

    - 有些 policy 可以 transfer；

    - 有些 policy 不行。

- 后来作者把 observation space 降到 4D，只保留 IMU roll / pitch / angular velocities。

- 结果 simulation 和 real 都出现稳定、可比的 trotting。

- 作者报告 trotting 速度：

| Task | Simulation Speed | Real Speed |
|---|---:|---:|
| Trotting | 0.50 m/s | 0.60 m/s |

- 真实速度略高于仿真速度，这不一定说明真实更优。

- 这里可能受真实地面摩擦、模型误差、控制周期、策略随机性等影响。

- 未执行/未验证：没有原始 log，不能进一步解释为什么 real speed 高于 sim speed。

## 和手工 gait 对比：

- 作者把 learned gait 和 Ghost Robotics 专家手工 gait 对比。

- Table III 给出的真实机器人结果是：

| Gait | Speed | Avg. Mechanical Power |
|---|---:|---:|
| Trotting handcrafted | 0.56 m/s | 92.72 W |
| Trotting learned | 0.60 m/s | 71.78 W |
| Galloping handcrafted | 1.21 m/s | 290.00 W |
| Galloping learned | 1.18 m/s | 188.79 W |

- 作者 claim：learned gait 在速度接近手工 gait 的情况下，机械功率显著更低。

- 具体下降幅度：

    - galloping 降低约 35%；

    - trotting 降低约 23%。

- 这说明 reward 中的 energy penalty 不只是装饰。

- 但也要注意：

    - 表格只覆盖 Minitaur 平地 gait；

    - 不说明 learned gait 在所有速度、所有地面、所有电池状态下都更省电；

    - 不说明安全裕度一定更高。

- 工程上我们应该把它理解成：

    - learned policy 可以找到手工调参不容易找到的 energy-speed trade-off；

    - 但真实部署前仍然要看电流、温度、关节限位、动作频谱和跌倒恢复。

## Reality Gap 的度量：

- 作者没有只用 success rate。

- success rate 是二值指标：

    - 一个 controller 能不能在真实机器人上 balance 整个 episode；

    - episode 是 1000 steps，大约 6 秒。

- 但 locomotion 不只是“不摔”。

- 还要看：

    - 跑多快；

    - 能耗多少；

    - reward 是否稳定；

    - 仿真和真实的 performance gap 多大。

- 所以作者使用 continuous measure：

    - 在仿真和真实实验中都计算 expected return；

    - 两者差值作为 reality gap 的度量。

- 这很实用。

- 对当前项目来说，评估 sim-to-real 或 sim-to-sim gap 时，不应该只看 episode 是否 terminate。

- 至少要记录：

    - episode return；

    - tracking reward；

    - energy / torque proxy；

    - action rate；

    - fall rate；

    - speed error；

    - foot slip；

    - joint limit violation；

    - sim/real 或 sim/sim 同一 command 下的差异。

## Ablation：Improved Simulation 是否必要：

- 作者用 trotting 做 reality gap ablation。

- 每组训练 100 个 controllers，使用不同 hyperparameters 和 random seeds。

- 然后选 simulation return 最高的 3 个 controller 部署到真实 Minitaur。

- 每个 controller 跑 3 次。

- 最后报告 9 次真实运行的平均 expected return。

- 第一组：baseline simulation。

    - 没有 actuator model；

    - 没有 latency handling。

- 第二组：baseline simulation + random perturbations。

- 第三组：improved simulation + random perturbations。

- 结果是：

    - 三组 top controllers 在仿真里都表现不错；

    - 第一、二组到真实机器人上表现差，sim-real gap 很大；

    - 第三组在仿真和真实中表现可比。

- 论文明确说：如果 model discrepancy 太大，即使训练时加了 random perturbations，也不能克服 reality gap。

- 这句话非常重要。

- 很多工程项目会把 domain randomization 当成万能补丁。

- 但这篇论文给出的结论更冷静：

    - randomization 可以提高 robustness；

    - 但不能替代基本正确的 actuator、latency 和系统参数；

    - 仿真闭环错得太远，policy 只会学到错误世界里的鲁棒性。

- 作者还指出，accurate actuator model 和 latency simulation 都重要。

- 没有其中任何一个，learned controllers 都不能在真实机器人上工作。

## Ablation：Randomization 的收益和代价：

- 作者把 dynamics randomization 和 perturbation 的结果合并称为 randomization，因为二者在鲁棒性上有类似效果。

- 他们用不同测试环境评估 controller。

- 方法是：

    - 对 Table I 中每个参数，在范围内均匀采样 10 个值；

    - 每次只改变一个参数，其他参数保持不变；

    - 计算 controller 在这些测试环境中的 return。

- 以 body inertia 为例：

    - 测试 inertia 从 default 的 50% 到 150%；

    - randomized controller 在不同 inertia 下 performance 更稳定；

    - non-randomized controller 如果实际 inertia 偏离训练值，performance 会明显下降；

    - 但 non-randomized controller 在 default 附近 peak performance 更高。

- 这就是 robustness vs optimality trade-off。

- randomization 带来的不是纯收益。

- 它让 controller 变保守：

    - mean return 可能下降；

    - standard deviation 下降；

    - 不同环境中的表现更稳定。

- 作者的原话意思可以概括为：randomization is not a free meal。

- 所以工程上不要一上来把 randomization 范围拉满。

- 更好的方式是：

    - 先用固定环境跑通最小任务；

    - 然后只打开一个 randomization 因子；

    - 看 reward、fall、energy、action rate 怎么变；

    - 再逐步扩展范围和组合。

## Ablation：Observation Space 的影响：

- 最后作者比较 small observation 和 large observation。

- small observation：4D IMU 信息。

- large observation：12D，IMU + motor angles。

- 实验结论是：

    - 仿真中 large observation return 更高；

    - 真实机器人上 large observation 表现更差，reality gap 更大；

    - small observation + randomization 组合最好；

    - top three controllers 的九次真实运行都能 trot 超过 3 米并保持整个 episode 平衡。

- 这里很值得记住。

- 仿真 reward 很容易把我们带偏。

- 如果一个 observation 在仿真里很干净、真实里很吵，它就可能成为 policy 的“作弊入口”。

- 尤其是 motor velocity、contact force、height scan、yaw 等量：

    - 仿真里可能精确；

    - 真实里可能延迟、漂移、跳变、坐标不一致；

    - actor 一旦依赖这些量，迁移风险就会上升。

## 方法伪代码：训练闭环怎么理解：

- 下面不是可执行训练代码，只是把论文逻辑映射成工程接口。

```python
# conceptual only: paper-level control/training flow
for episode in range(num_episodes):
    params = sample_physical_parameters(
        mass=(0.8, 1.2),
        inertia=(0.5, 1.5),
        motor_strength=(0.8, 1.2),
        latency_ms=(0, 40),
        control_step_ms=(3, 20),
        battery_voltage=(14.0, 16.8),
        contact_friction=(0.5, 1.25),
        imu_bias_rad=(-0.05, 0.05),
        imu_noise_std_rad=(0.0, 0.05),
    )

    env.reset(params)
    obs_history = ObservationDelayBuffer()

    for step in range(max_steps):
        delayed_obs = obs_history.query(now=step, latency=params.latency)
        open_loop = gait_reference(t=env.time)       # 0 for galloping; nonzero only for trotting reference
        residual = policy(delayed_obs)               # PPO feedback policy
        leg_action = clip(open_loop + residual)
        motor_targets = leg_space_to_motor_space(leg_action)
        env.step(motor_targets)

        if step % 200 == 0:
            env.apply_random_base_perturbation()
```

- 这个伪代码里真正要看的是接口，不是函数名。

- 训练闭环中必须同时出现：

    - parameter randomization；

    - observation delay；

    - open-loop reference；

    - feedback residual；

    - action bounds；

    - actuator response；

    - perturbation。

- 少一个，最终学出的 policy 就可能在某个真实接口上断掉。

## 论文配置块：可以怎么落到项目：

- 如果把论文思想迁移到当前项目，一个配置块可以先长这样。

- 注意：这是概念配置，不是说当前代码已经支持所有字段。

```python
sim_to_real_profile = {
    "actor_observation_policy": {
        "prefer": ["imu_ang_vel", "projected_gravity", "commands", "joint_pos", "last_action"],
        "be_careful": ["joint_vel", "yaw", "contact_force", "height_scan"],
        "rule": "actor only gets signals that are deployable and calibrated",
    },
    "action_interface": {
        "type": "joint_position_target_with_default_offset",
        "scale_by_joint_group": True,
        "clip_before_actuator": True,
    },
    "actuator_and_delay": {
        "pd_gain_randomization": "after base task works",
        "motor_strength_randomization": (0.8, 1.2),
        "action_delay_steps": (0, 4),
        "observation_delay_steps": "needs implementation check",
    },
    "domain_randomization": {
        "friction": (0.5, 1.25),
        "mass_scale": (0.8, 1.2),
        "base_com_offset_m": (-0.05, 0.05),
        "imu_bias_rad": (-0.05, 0.05),
    },
    "evaluation": {
        "do_not_rank_by_sim_reward_only": True,
        "compare_gap": ["return", "speed", "energy", "fall_rate", "action_rate"],
    },
}
```

- 这里的关键点是“先定义接口，再打开复杂度”。

- 不要一上来同时调 reward、terrain、actuator、latency、randomization、network size 和 PPO 参数。

## 对 `legged_wbc_mjlab` 的代码观察：

- 代码观察：当前项目中 Go2 velocity task 已经是 manager-based 环境配置。

- actor observation 包含：

    - base angular velocity；

    - projected gravity；

    - command；

    - phase；

    - joint position；

    - joint velocity；

    - last action。

- critic observation 在 actor terms 基础上增加：

    - base linear velocity；

    - height scan；

    - foot height；

    - foot air time；

    - foot contact；

    - foot contact forces。

- 这个 actor/critic 分工和论文里的 compact deployable observation 思想一致。

- 不过当前 actor 仍包含 joint velocity。

- 这里是推断：如果后续真实部署出现高频抖动或 sim-to-real gap，joint velocity 的噪声、滤波和延迟应优先检查。

- 代码观察：当前 action 使用 `JointPositionActionCfg`，entity 是 `robot`，actuator 匹配 `.*`，并使用 default offset。

- 这说明项目已经不是 torque policy，而是 joint position target policy。

- 这和论文选择 position control 的动机相近：

    - 学习更容易；

    - 部署更安全；

    - action bounds 更直观。

- 代码观察：当前 Go2 task 里已经有以下 event：

    - `push_robot`；

    - `foot_friction`；

    - `encoder_bias`；

    - `base_com`。

- 这对应论文里的 perturbation、contact friction randomization、sensor bias、COM uncertainty。

- 代码观察：当前 PPO 配置使用 hidden dims `(512, 256, 128)`，`clip_param=0.2`，`gamma=0.99`，`lam=0.95`，`desired_kl=0.01`，`num_steps_per_env=24`。

- 这比论文中的两层较小网络更现代、更大。

- 这里是推断：网络更大不一定有问题，但更应该配合 observation normalization、domain randomization 和 sim-to-real evaluation，否则更容易吃到仿真细节。

- 未执行/未验证：我没有运行 `scripts/train.py`，没有检查 observation tensor 维度，也没有验证 event 是否全部被当前 `mjlab==1.6.0` 正确调用。

## 对 MuJoCo 项目的具体启发：

### 1. 不要把 actuator 当成透明层

- 论文的 actuator model 说明：policy action 和真实 torque 之间的层很厚。

- MuJoCo 中如果只是写：

```xml
<position joint="..." kp="..." />
```

- 这不一定等价于真实电机闭环。

- 需要继续问：

    - PD gain 是真实低层 gain 还是训练用 gain；

    - actuator force limit 是否匹配真实 torque limit；

    - gear sign 是否匹配；

    - ctrlrange 是否对应 joint target，而不是 normalized action；

    - joint damping / armature / frictionloss 是否合理；

    - 电机速度越高时，是否应该降低可用 torque。

- 如果没有真实 log，至少要在笔记里标注“未验证”。

### 2. 延迟要和 decimation 一起看

- 当前 Go2 配置里 MuJoCo timestep 是 0.005 s，decimation 是 4。

- 这意味着 policy step 大概是 20 ms。

- 如果 delay 按 physics step 算，1 lag 是 5 ms。

- 如果 delay 按 policy step 算，1 lag 是 20 ms。

- 这个差别非常大。

- 所以 delay 配置必须明确单位。

- 建议后续文档或代码注释明确写：

```python
delay_contract = {
    "lag_index_unit": "physics_step or policy_step",  # must be explicit
    "physics_timestep_s": 0.005,
    "decimation": 4,
    "policy_dt_s": 0.020,
}
```

### 3. randomization 要分层开启

- 先从最简单的开始。

- 第一阶段只开：

    - friction；

    - small encoder bias；

    - mild push。

- 第二阶段再开：

    - motor strength；

    - COM offset；

    - joint damping / armature perturbation。

- 第三阶段再考虑：

    - latency randomization；

    - observation dropout / noise；

    - terrain randomization；

    - learned actuator model。

- 每加一层，都要比较固定 seed 下的 sim return、fall、action rate 和能耗 proxy。

### 4. observation 不要跟着 critic 一起膨胀

- critic 看更多信息是训练加速手段。

- actor 看更多信息是部署风险。

- 当前项目已经有 actor/critic observation group，这是好事。

- 后续要保持这个边界：

    - height scan 如果真实没有可靠感知，优先留在 critic；

    - contact force 如果真实传感不可用，优先留在 critic；

    - base lin vel 如果真实估计不稳定，要么通过 estimator，要么只在 critic；

    - actor observation 的每个字段都要能解释真实来源、频率、延迟、滤波和单位。

## 对 `rsl_rl` 的具体启发：

- `rsl_rl` 的核心接口是 vectorized environment。

- 它要求 step 返回：

    - observation TensorDict；

    - reward tensor；

    - done tensor；

    - extras dict。

- extras 里通常会包含 `time_outs`，用于区分 time limit 和真正 terminal。

- 这和论文中的 episode timeout / fell over termination 有对应关系。

- 如果做 sim-to-real 评估，建议 logging 不只记录总 reward。

- 至少把以下项放进 extras 或 logger：

    - tracking reward；

    - energy penalty；

    - action rate penalty；

    - foot slip；

    - termination reason；

    - sampled domain randomization parameters；

    - delay parameters；

    - mean / max torque proxy；

    - command distribution。

- 为什么要记录 sampled randomization parameters？

    - 否则 policy 摔倒时不知道是 friction 太低、COM 太偏、delay 太大，还是 reward 本身有问题；

    - 也无法复现某个失败 episode。

- 对 PPO 本身，论文没有提出新算法。

- 所以在 `rsl_rl` 中落地时，优先改环境接口和日志，而不是先改 PPO。

- 这里是推断：如果要实现论文里的 open-loop + residual policy，最小侵入方式可能是在 action manager 或 env step 中构造 reference，然后让 actor 输出 residual；不要直接改 PPO surrogate objective。

## Open-loop + Residual 在当前项目中怎么做：

- 如果想借鉴论文的 guided gait 思路，可以把 action 拆成：

$$
a_t=a^{ref}_t+\Delta a_t
$$

- 这里：

    - $a^{ref}_t$ 是 phase-based reference；

    - $\Delta a_t$ 是 policy residual；

    - 最终 action 再 clip 到 joint target range。

- 对 Go2 来说，reference 可以不是 leg-space $s/e$，而是 joint target offset。

- 概念配置可以是：

```python
gait_reference = {
    "period_s": 0.6,
    "phase_offsets": {
        "FL": 0.0,
        "RR": 0.0,
        "FR": 0.5,
        "RL": 0.5,
    },
    "joint_targets": {
        "hip": "small lateral sinusoid or zero",
        "thigh": "sagittal swing sinusoid",
        "calf": "extension sinusoid",
    },
    "residual_limit_rad": 0.25,
}
```

- 这个配置仍然不是可执行代码。

- 它表达的是训练约束：

    - policy 学 residual；

    - reference 控制 gait style；

    - residual bounds 控制 policy 不能完全破坏 style。

- 注意：如果当前任务已经有 `phase` observation 和 `feet_gait` reward，那么 open-loop reference 可能和已有 gait reward 发生重复约束。

- 工程上要先决定：

    - 用 reward 鼓励 gait；

    - 还是用 action reference 指定 gait；

    - 或者两者都用但权重要很小心。

## 论文和后续工作的关系：

- 这篇论文可以看成早期 sim-to-real legged RL 的工程基线。

- 后续很多工作是在它的某个方向上继续加深。

### 和 RMA / DreamWaQ / HIMLoco 的关系

- 这篇论文没有用 adaptation module。

- 它主要靠：

    - 更真实的仿真；

    - randomization；

    - compact observation。

- RMA / DreamWaQ / HIMLoco 更进一步，开始显式处理隐藏状态恢复：

    - friction、mass、terrain、disturbance 不直接给 actor；

    - 用 history estimator / latent 去恢复；

    - 训练时 critic 或 teacher 可以看 privileged information。

- 但底层问题没有变：

    - actuator 是否真实；

    - delay 是否真实；

    - observation 是否可部署；

    - randomization 是否合理。

### 和 AMP_for_hardware 的关系

- AMP_for_hardware 关注的是 reward engineering 和 motion style。

- 它用 discriminator 从参考动作中提供 style reward。

- 本文则更早、更底层：

    - 不用 motion prior；

    - reward 很简单；

    - 重点是 sim-to-real interface。

- 两者可以互补。

- 这里是推断：如果在 Go2 项目中加入 AMP style prior，也仍然不能跳过本文强调的 actuator / delay / observation / randomization 问题。

### 和 Deep-WBC 的关系

- Deep-WBC 处理的是腿臂统一控制。

- 本文处理的是纯 quadruped locomotion。

- 但它们共享一个工程原则：

    - action interface 要可部署；

    - 仿真里看到的闭环要接近真实；

    - 不要只用仿真 reward 评价策略。

- Deep-WBC 里 manipulation / locomotion reward 需要 credit assignment。

- 本文的 reward 更简单，但在 actuator 和 delay 上更强调系统辨识。

## 关键图表理解：

### Fig. 1：仿真和真实 gallop

- Fig. 1 展示 simulated Minitaur 和 real Minitaur 都学会 gallop。

- 这张图的作用不是证明所有条件都 robust。

- 它说明论文关注的是真实部署，而不是只在 simulator 中展示步态。

### Fig. 2：硬件结构

- Fig. 2 展示 TX2 + STM32 的硬件架构。

- 关键点是 neural policy 不在 microcontroller 上跑。

- microcontroller 负责传感器、执行器和简单控制。

- TX2 负责 neural network inference。

- 这就引入 UART 通信和非实时系统导致的 latency / variable control frequency。

### Fig. 3：leg space 和 motor space

- Fig. 3 解释 swing / extension 与两个电机角之间的关系。

- 它对应 action space 的结构化选择。

### Fig. 4：actuator model 验证

- Fig. 4 比较 sine trajectory 下真实和仿真 motor angle。

- 作用是支撑作者关于 improved actuator model 更贴近真实 motor response 的 claim；Fig. 4 本身比较的是新模型下的 simulated trajectory 与真实 ground truth，不是与默认 Bullet position control 的直接定量对比。

- 未执行/未验证：这里只记录作者 claim，没有复现图中曲线。

### Fig. 5：learning curves

- Fig. 5 展示 trotting 和 galloping 的学习曲线。

- 重点是两个任务都在几百万 simulation steps 内收敛。

### Fig. 6：simulation fidelity ablation

- Fig. 6 比较 baseline simulation、baseline + perturbation、improved simulation + perturbation。

- 核心结论：只加 perturbation 不够，improved simulation 是必要条件。

### Fig. 7 / Fig. 8：randomization trade-off

- Fig. 7 用 body inertia 变化展示 randomization 的鲁棒性。

- Fig. 8 聚合不同测试环境，展示 randomized controller mean return 更低但 variance 更小。

### Fig. 9：observation space ablation

- Fig. 9 展示 small observation 和 large observation 在 sim/real 上的反差。

- 核心结论：large observation 仿真更好，真实更差；small observation + randomization 最稳。

## 局限：

- 论文自己也承认，它的环境和任务相对简单。

- 主要局限包括：

    - flat ground；

    - 只最大化前进速度；

    - 不处理动态速度命令；

    - 不处理转向；

    - 不处理复杂地形结构；

    - 没有视觉输入；

    - 没有长期户外测试；

    - 没有现代大规模并行 GPU simulator；

    - 没有历史 estimator 或 privileged adaptation；

    - 没有 learned actuator network。

- 论文最后给的 future work 也很直接：

    - 学习能动态改变速度和方向的 locomotion policy；

    - 结合视觉处理复杂地形结构。

- 对我们来说，这些局限不是缺点，而是定位。

- 它解决的是早期 sim-to-real locomotion 的基础问题：

    - 仿真闭环怎么接近真实；

    - RL policy 怎么在硬件上不立刻失败；

    - 用户怎么在 learned controller 中保留 gait controllability。

## 常见误读：

### 误读一：domain randomization 能替代系统辨识

- 不对。

- 论文 ablation 明确说明，baseline simulation + perturbation 仍然不能 transfer。

- 如果 actuator / latency 错得太多，randomization 不能救。

### 误读二：observation 越多越好

- 不对。

- large observation 在仿真更好，但真实更差。

- actor observation 的可部署性比信息量更重要。

### 误读三：open-loop reference 已经能 trot

- 不对。

- open-loop signal 单独执行会失稳。

- RL feedback component 才负责 balance 和推进。

### 误读四：这篇论文证明简单 reward 足够解决所有 locomotion

- 不对。

- 它的任务是 flat ground forward running。

- 对 rough terrain、velocity command、turning、stairs、manipulation，reward 和 observation 都要重新设计。

### 误读五：position control 总是比 torque control 好

- 论文选择 position control 是为了安全和学习容易。

- 但这不代表 torque control 没价值。

- 关键是 action interface 要与真实低层控制和仿真 actuator 一致。

## 给当前项目的最小落地路线：

- 如果我们想把这篇论文的经验真正用到 `legged_wbc_mjlab`，不要一上来复现所有内容。

- 更稳的顺序是：

### Step 1：固定 action contract

- 明确 actor 输出是 normalized action 还是 joint target offset。

- 明确每个 joint group 的 scale。

- 明确 final joint target 的 clip 范围。

- 明确 default offset 来自哪里。

- 验证 action index 和 joint name 顺序。

### Step 2：固定 observation contract

- actor observation 只放真实可部署信号。

- critic observation 可以放 privileged / height / contact 信息。

- 每个 observation term 记录：

    - shape；

    - unit；

    - frame；

    - noise；

    - latency；

    - normalization。

### Step 3：只用轻量 randomization 跑通基础任务

- friction 小范围。

- encoder bias 小范围。

- mild push。

- 不要一开始就把 delay、terrain、mass、COM、motor strength 全拉满。

### Step 4：加入 delay 并做消融

- 固定其它参数。

- 比较 no delay / fixed delay / randomized delay。

- 看 reward、fall、action rate、joint acceleration 是否变化。

### Step 5：再加 actuator uncertainty

- motor strength。

- stiffness / damping perturbation。

- effort limit check。

- 如果有真实 log，再考虑拟合 actuator model。

### Step 6：建立 sim-to-sim gap benchmark

- 在没有真实机器时，至少做多个 MuJoCo parameter variants。

- 用 expected return 差异衡量 gap。

- 不要只看训练环境中的 reward。

## 一个建议的记录表：

- 下面这个表适合放到实验日志里。

| Run | Obs Set | Delay | Actuator Rand | Friction Rand | Push | Sim Return | Fall Rate | Energy Proxy | Action Rate | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---|
| base | actor_v1 | off | off | off | off | 未运行 | 未运行 | 未运行 | 未运行 | fixed env |
| rand_friction | actor_v1 | off | off | on | off | 未运行 | 未运行 | 未运行 | 未运行 | one factor |
| push | actor_v1 | off | off | on | on | 未运行 | 未运行 | 未运行 | 未运行 | recovery |
| delay | actor_v1 | on | off | on | on | 未运行 | 未运行 | 未运行 | 未运行 | latency ablation |
| motor | actor_v1 | on | on | on | on | 未运行 | 未运行 | 未运行 | 未运行 | actuator uncertainty |

- 这里所有“未运行”都不能提前写成成功。

- 未执行/未验证：这只是建议记录格式，当前没有运行任何实验。

## 对安全部署的提醒：

- 论文展示了真实 Minitaur 部署，但这不等于我们可以直接把类似策略上 Go2。

- 真实部署前至少要有：

    - emergency stop；

    - joint position / velocity / torque / current limits；

    - action clipping；

    - NaN / Inf guard；

    - observation freshness check；

    - communication timeout fallback；

    - low-level safe posture；

    - staged speed ramp；

    - tether 或保护架；

    - 人员隔离。

- 这篇论文没有替我们验证这些内容。

- 所以任何“直接部署”的说法，都必须绑定具体硬件、限幅、急停、日志和分阶段测试。

## 我的理解：为什么这篇论文现在还值得看：

- 现在看这篇 2018 年论文，很多技术细节已经不新了。

- PPO、domain randomization、position target、sim-to-real 都已经是 legged RL 常规词。

- 但它仍然值得看，因为它讲清楚了一个非常朴素的工程事实：

    - policy 不是在抽象 MDP 里运行；

    - policy 是在一条真实控制链路里运行；

    - 这条链路包括 sensor、通信、推理、action scaling、PD、actuator、电机饱和、接触和环境扰动。

- 如果训练闭环少了某个关键环节，policy 可能会在仿真里变强，但在真实硬件上更脆。

- 所以这篇论文最值得记住的不是某个公式。

- 而是这个顺序：

    - 先把系统辨识和 actuator/latency 做到足够可信；

    - 再用 randomization 覆盖剩余不确定性；

    - 再用 compact observation 降低过拟合；

    - 最后用真实部署或至少 sim-to-sim ablation 检查 gap。

- 对 `legged_wbc_mjlab` 来说，这个顺序比单纯调 reward 更重要。

## 未验证项汇总：

- 未运行论文官方 PyBullet 环境。

- 未训练 galloping 或 trotting policy。

- 未复现实验曲线、速度、功耗或 ablation 图。

- 未检查 Minitaur 原始 URDF、actuator piece-wise torque-current function 或真实 latency log。

- 未运行 `legged_wbc_mjlab` 当前训练脚本。

- 未验证当前项目 delay 参数实际作用在 action path、actuator path 还是 observation path。

- 未验证当前 MuJoCo actuator 与真实 Go2 执行器之间的等价关系。

- 未验证任何真实机器人部署安全条件。

## 局部结论：

- 这篇论文可以用一句工程话收束：**sim-to-real 不是在训练结束后再修补，而是从 observation、action、actuator、latency、randomization、reward 和 evaluation 一开始就要按真实部署闭环来设计。**

- 如果当前项目后续要做 Go2 的稳定 velocity policy，最优先的不是追求更复杂网络，而是先把 action/observation/actuator/delay/randomization 的接口契约钉死。

- 如果这些接口不清楚，PPO reward 再高也可能只是仿真里的局部胜利。
