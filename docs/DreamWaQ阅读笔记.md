# DreamWaQ阅读笔记

# 前言：

- 这篇论文的核心目标，是让四足机器人在**只依赖本体感知 proprioception** 的情况下，也能走过楼梯、草地、泥地、斜坡、碎石路和户外山路这类复杂地形。

- 论文标题里的 DreamWaQ 全称可以理解为 **Dream Walking for Quadrupedal Robots**。这里的 “Dream” 不是视觉想象，而是指策略通过一段历史本体观测，在内部隐式想象或者推断地形属性，比如高度变化、摩擦、恢复系数、外力扰动和障碍物。

- 传统 locomotion 里经常有两种极端做法：

    - 一种是给 robot 加相机、LiDAR、height map，让策略或者 planner 明确看到地形；

    - 另一种是完全只给 IMU、关节角、关节速度和动作历史，让 policy 自己硬学。

- 第一种方案训练更容易，但真实部署依赖复杂感知链路。相机受光照、天气影响，LiDAR 能分割几何结构，却不一定知道雪地、草地、泥地到底软不软、滑不滑。第二种方案部署简单，但把“控制”和“隐藏状态恢复”两个难题全塞给 actor，容易学不出来或者泛化差。

- DreamWaQ 的思路是折中：

    - 训练时用 asymmetric actor-critic，让 critic 看 privileged observation；

    - 部署时 actor 只用 proprioception；

    - 在 actor 前面放一个 **Context-Aided Estimator Network, CENet**，从历史观测中显式估计 body velocity，同时隐式压出一个 16 维 context latent；

    - 通过 beta-VAE 的 next-observation reconstruction，让这个 latent 不只是随便漂，而是必须解释系统的前后动力学。

- 这篇论文和我之前看的 HIMLoco 非常接近：它们都在解决 POMDP 下的隐藏状态恢复问题，都把速度作为一个明确的物理锚点，再用 latent 吸收那些难以命名的地形/扰动因素。区别在于 DreamWaQ 用的是 CENet + beta-VAE + AdaBoot，而 HIMLoco 后来更偏向 contrastive/prototype/Sinkhorn 那条路线。

- 未执行/未验证：这份笔记只基于 `/home/kk/下载/2301.10602v2.pdf` 的阅读整理，以及本地样例笔记 `/home/kk/飞书文档/论文学习部分.md`、`/home/kk/飞书文档/AMP_for_hardware学习记录.md`、`/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`、`/home/kk/legged_wbc_mjlab/docs/BFM-Zero阅读笔记.md` 的风格参考；没有跑作者代码、没有复现实验，也没有验证真实 Unitree A1 部署。

# 论文核心信息：

- **论文标题：** DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning

- **作者：** I Made Aswin Nahrendra, Byeongho Yu, Hyun Myung

- **机构：** KAIST, School of Electrical Engineering

- **会议/版本：** ICRA 2023，arXiv:2301.10602v2

- **机器人平台：** Unitree A1

- **训练仿真器：** Isaac Gym

- **真实部署计算平台：** Intel NUC onboard computer

- **控制方式：** policy 输出 12 维期望关节角偏移，底层 PD controller 跟踪 desired joint angle

- **策略部署输入：** 只使用 proprioception，包括 IMU、关节状态、速度命令、历史动作等，不依赖相机、LiDAR 或真实 height map

- **论文类型：** quadrupedal locomotion / proprioceptive RL / asymmetric actor-critic / representation learning / zero-shot sim-to-real

# 研究问题：

- 论文要回答的问题是：

    - **四足机器人能不能不依赖外部感知，只靠自身传感器和历史响应，在真实户外复杂地形上长距离稳定行走？**

- 这里的“只靠自身传感器”非常关键。因为真实机器人部署时，外部感知并不总是可靠：

    - 相机可能被光照、雨雪、逆光、夜晚影响；

    - LiDAR 能看到几何表面，但不一定知道地面材质的物理属性；

    - 高度图本身也有延迟、噪声、遮挡和坐标配准误差；

    - 有些地形属性，比如泥地软硬、草丛阻力、楼梯湿滑，必须真正踩上去以后才能知道。

- 这就引出一个很实际的 POMDP 问题：

    - robot 当前观测到的是 $o_t$，不是完整状态 $s_t$；

    - 真正影响控制的 friction、terrain height、contact condition、external disturbance 不一定直接可见；

    - 这些隐藏量会通过一段时间内的机体响应暴露出来，比如速度变化、脚滑、机身倾斜、关节负载、动作后续结果。

- 所以 DreamWaQ 不再问：

    - “我能不能直接把地形高度图给 actor？”

- 而是换成：

    - **“我能不能让 actor 从历史 proprioception 里恢复一个足够用于控制的内部地形表征？”**

- 这个转向很重要。它把问题从“传感器工程”变成了“表征学习 + 鲁棒控制”的问题。

# 研究背景：

- 在 DreamWaQ 之前，学习式四足 locomotion 大致有几条路线。

## 第一种：显式感知路线

- 这类方法会给机器人使用相机、深度图、LiDAR 或 height map。

- 好处很明显：

    - policy 或 planner 能提前看到楼梯、台阶、坑洼；

    - 训练时可以直接把地形信息喂给网络；

    - 对复杂几何结构，尤其是高台阶和障碍跨越，会更有前瞻性。

- 但它的问题也很现实：

    - 真实感知链路很复杂，包含标定、时间同步、点云分割、地形重建和坐标变换；

    - 视觉看到的是表面，不一定知道物理属性；

    - 机器人脚底真正遇到的摩擦和变形，经常要接触以后才知道。

- 举个直观例子：

    - 雪地在相机里可能看起来是完整平面，但踩上去会陷；

    - 高草在视觉里可能像障碍，但小型四足机器人实际能穿过去；

    - 湿楼梯在几何上和平常楼梯差不多，但动力学上完全不一样。

- 所以显式感知不是错，而是它解决的是“提前看见几何”，不一定解决“真实接触属性”。

## 第二种：Teacher-Student / RMA 路线

- RMA 这类方法通常先训练一个拥有 privileged information 的 teacher，再把它蒸馏成只看可部署观测的 student。

- 它的核心直觉是：

    - teacher 在仿真里知道 friction、mass、terrain 等隐藏信息；

    - student 真实部署时不知道这些信息，所以从历史观测中估计一个 adaptation latent；

    - actor 再根据当前观测和 adaptation latent 输出动作。

- 这条路线很强，但有两个问题：

    - **两阶段训练效率低。** 先 teacher，再 student，训练链路更长；

    - **student 上限受 teacher 约束。** 如果 teacher 在早期探索中学过一些失败状态，student 用 behavior cloning 只拿到 teacher 的好动作监督，未必真正探索过所有失败边界。

- DreamWaQ 认为，与其先训练一个 teacher 再蒸馏，不如在一个 actor-critic 框架里完成：

    - actor 只看可部署观测；

    - critic 在训练时看 privileged observation；

    - CENet 同步学习 body velocity 和 context latent；

    - PPO 训练过程本身推动 actor 学会利用这个 latent。

## 第三种：只用 proprioception 的端到端路线

- 这条路线最贴近真实部署：不依赖外部传感器，只看 IMU、关节角、关节速度、历史动作。

- 但难点是：如果直接把历史观测拼起来交给一个大 actor，actor 要同时做两件事：

    - 压缩历史信息，恢复隐藏状态；

    - 输出稳定动作，解决接触控制。

- 这很容易导致训练混乱。因为 locomotion 本来就高维、强接触、非线性，再叠加部分可观测性，policy 会把“我要怎么迈腿”和“我现在到底踩在什么地上”混在一起学。

- DreamWaQ 的工程判断是：

    - actor 不应该自己背所有历史信息压缩压力；

    - 应该有一个 estimator 专门负责从历史里提炼 body state + environment context；

    - actor 只需要拿这个紧凑信息做控制。

# 核心直觉：

- DreamWaQ 可以被理解成一句话：

    - **让 critic 在训练时看见真实地形，让 actor 在部署时只通过自己的“脚感历史”去想象地形。**

- 这里的“脚感历史”不是文学表达，而是实际的 temporal partial observations：

    - 过去几步的 IMU 角速度；

    - body frame 下的 gravity vector；

    - 用户给的速度命令；

    - 关节角、关节角速度；

    - 上一步动作。

- 如果 robot 一脚踩在硬地上，动作和机体响应之间的关系是一种模式；如果踩在泥地、草地、湿楼梯、碎石上，响应会变成另一种模式。CENet 就是在学：

    - “我刚才这样发力，身体却这样动，说明地面大概是什么情况。”

- 这和 HIMLoco 笔记里的“隐藏状态恢复”是一条线：

    - 单步 observation 看不到隐藏物理条件；

    - 但隐藏物理条件会改变系统响应；

    - 所以我们从历史响应中恢复一个内部状态。

- DreamWaQ 的关键不是发明一个更复杂的 reward，而是让 representation 更适合控制：

    - 显式估计 velocity，保证控制的物理锚点；

    - 隐式学习 context latent，吸收地形和扰动；

    - 用 privileged critic 稳住训练；

    - 用 AdaBoot 避免 estimator 早期噪声拖坏 policy。

# 问题建模：

- 论文把 locomotion 建模为 infinite-horizon POMDP：

$$
M=(S,O,A,d_0,p,r,\gamma)
$$

- 这里：

    - $S$ 是完整状态空间；

    - $O$ 是部分观测空间；

    - $A$ 是动作空间；

    - $d_0(s_0)$ 是初始状态分布；

    - $p(s_{t+1}\mid s_t,a_t)$ 是状态转移；

    - $r:S\times A\rightarrow R$ 是奖励函数；

    - $\gamma\in[0,1)$ 是折扣因子。

- policy 实际部署时拿不到完整状态 $s_t$，只能看到 partial observation $o_t$。所以论文引入历史观测：

$$
o_t^H=[o_t,o_{t-1},\ldots,o_{t-H}]^T
$$

- 论文里使用 $H=5$。这意味着 actor 不是只看当前一步，而是看过去 5 个 measurement 的历史上下文。

- 单步 proprioceptive observation 定义为：

$$
o_t=[\omega_t, g_t, c_t, \theta_t, \dot{\theta}_t, a_{t-1}]^T
$$

- 这里：

    - $\omega_t$ 是机体角速度；

    - $g_t$ 是 body frame 下的 gravity vector；

    - $c_t$ 是速度命令；

    - $\theta_t$ 是关节角；

    - $\dot{\theta}_t$ 是关节角速度；

    - $a_{t-1}$ 是上一时刻动作。

- 训练中的 privileged state 写成：

$$
s_t=[o_t,v_t,d_t,h_t]^T
$$

- 其中：

    - $v_t$ 是 body velocity；

    - $d_t$ 是仿真中随机施加在 robot body 上的 disturbance force；

    - $h_t$ 是 robot 周围的 height map scan。

- 注意这个信息边界：

    - critic 可以看 $s_t$；

    - actor 不能看 $d_t$ 和 $h_t$；

    - actor 只能通过 $o_t^H$ 间接推断这些东西。

- 这就是 asymmetric actor-critic 的核心价值：训练时用上帝视角降低 value learning 难度，部署时不把上帝视角泄漏给 actor。

# 方法主线：

- DreamWaQ 的方法可以拆成四个模块：

    - **Implicit Terrain Imagination：** 用 asymmetric actor-critic 让 actor 从 temporal proprioception 中隐式推断 privileged observation；

    - **Policy Network：** 输入当前 proprioception、估计 body velocity 和 context latent，输出 12 维 action；

    - **CENet：** 从历史观测中同时估计 $v_t$ 和 $z_t$，并用 beta-VAE 重构下一步观测；

    - **AdaBoot：** 根据训练 reward 的 coefficient of variation 自适应决定是否 bootstrapping estimator 输出。

- 整体信息流可以写成：

$$
o_t^H \xrightarrow{CENet} (\tilde{v}_t,z_t)
$$

$$
(o_t,\tilde{v}_t,z_t) \xrightarrow{Actor} a_t
$$

$$
(o_t,v_t,d_t,h_t) \xrightarrow{Critic} V(s_t)
$$

- 真实部署时只有前两条链路存在。critic 是训练期的老师，不上真实机器人。

# Implicit Terrain Imagination 到底在干什么：

- 论文里的 “implicit terrain imagination” 容易被误解成“网络生成了一张地形图”。实际上不是。

- DreamWaQ 没有让 actor 显式输出 height map，也没有让 CENet 回归 friction、restitution、stair label 或 terrain class。

- 它做的是：

    - 训练时 critic 看得到 height map 和 disturbance；

    - actor 只看历史 proprioception；

    - 在 PPO 的 actor-critic 互动中，actor 被迫学会利用历史信号去预测那些会影响 value 的隐藏因素。

- 可以把它理解成一种“通过价值函数施压的隐式表征学习”：

    - 如果 actor 不知道地形，动作就容易导致低 reward；

    - critic 因为看到了 privileged state，能更准确地区分“这个动作在当前地形下好不好”；

    - policy gradient 会把这种差异反向传给 actor；

    - actor 最后学到的不是可解释地形图，而是一个足够控制用的 terrain-aware behavior。

- 这和 Teacher-Student 最大的区别是：

    - Teacher-Student 是先有 teacher，再让 student 模仿 teacher；

    - DreamWaQ 是 actor 和 critic 同时训练，actor 在探索过程中直接经历成功和失败。

- 论文认为这样更高效，也更鲁棒。因为 student imitation 只能模仿 teacher 给出的好动作，而 actor-critic 里的 policy 可以在训练中探索失败轨迹，从而学到哪些动作在不确定地形上危险。

# Policy Network：

- policy 写成：

$$
\pi_\phi(a_t\mid o_t,v_t,z_t)
$$

- 这里输入有三部分：

    - 当前本体观测 $o_t$；

    - CENet 显式估计的 body velocity $v_t$；

    - CENet 隐式估计的 context latent $z_t$。

- 论文图里 policy MLP 是 $512\times256\times128\times12$。

- 动作空间是 12 维：

$$
a_t\in\mathbb{R}^{12}
$$

- 对 Unitree A1 来说，12 维对应 12 个关节的目标角度偏移。

- 论文没有直接让 policy 输出 torque，而是输出相对于 stand still pose 的关节角偏移：

$$
\theta_{des}=\theta_{stand}+a_t
$$

- 然后用每个关节的 PD controller 跟踪 $\theta_{des}$。

- 这个选择很符合 legged RL 的工程习惯：

    - 直接输出 torque 学习难度大，对 sim-to-real 的动力学误差敏感；

    - 输出 PD target 更稳定，也更容易接入 Unitree A1 的低层控制接口；

    - action rate、smoothness、joint power 等惩罚可以直接约束动作质量。

- 所以 DreamWaQ 的 actor 更像“高层关节目标生成器”，不是完整替代底层 motor control。

# Value Network：

- value network 输出：

$$
V(s_t)
$$

- 它的输入是 privileged observation：

$$
s_t=[o_t,v_t,d_t,h_t]^T
$$

- 图中 value MLP 是 $512\times256\times128\times1$。

- 这里的设计非常标准，但很关键：

    - critic 不需要可部署；

    - critic 的任务是给训练提供低方差、信息充分的 value estimate；

    - 所以它可以直接看仿真中的 disturbance 和 height map。

- 注意：这不是信息泄漏，只要 actor 推理时没有用这些 privileged variables，就不会造成真实部署不可用。

- 但它也带来一个边界：

    - 如果 critic 训练时过度依赖某些仿真特有变量，而 actor 从 proprioception 中根本推不出来，训练可能会给 actor 很难利用的梯度；

    - DreamWaQ 通过 CENet、history input 和 domain randomization 缓解这个问题，但不能从理论上保证所有 hidden variable 都能被恢复。

# CENet 的设计：

- CENet 是这篇论文最重要的模块。

- 它的输入是 temporal partial observations $o_t^H$，输出两类东西：

    - 3 维 body linear velocity estimation $\tilde{v}_t$；

    - 16 维 context vector $z_t$。

- 图中 CENet 共享 encoder 的输出是 19 维：

$$
19=3+16
$$

- 这 19 维再被拆给两个用途：

    - velocity head 负责明确估计 body velocity；

    - auto-encoder head 负责从 latent 中重构下一步 partial observation。

- 这就不是单纯的 EstimatorNet。

- EstimatorNet 只问：

    - “现在 body velocity 是多少？”

- CENet 还问：

    - “为了预测下一步观测，我必须理解哪些隐藏上下文？”

- 这一步非常重要。因为如果只监督 velocity，网络可能只学到对速度有用的部分，而忽略 friction、terrain compliance、脚碰撞等更复杂信息。VAE reconstruction 会逼 latent 保留更多系统动力学线索。

## 为什么 velocity 要显式估计

- 对 locomotion 来说，速度是最核心的反馈量。

- 如果速度估计不准，机器人会出现几个很直接的问题：

    - 以为自己还没动，于是加大输出；

    - 以为自己已经跟上命令，于是提前收力；

    - 上楼梯或者下楼梯时误判机体运动趋势；

    - 被推或者脚滑后，不能及时恢复平衡。

- 传统上可以用状态估计器融合 IMU、运动学、触地信息来估速度，但在滑地、软地、碰撞时容易累积 drift。

- DreamWaQ 用 learned estimator 来估 $v_t$，并且让这个估计器和 policy 同步训练。这里的直觉和 HIMLoco 很像：

    - velocity 是有物理意义、可监督、跨平台相对通用的锚点；

    - latent 则负责吸收那些难命名的隐藏因素。

## 为什么 context 不直接监督成地形标签

- 很多地形属性在真实世界里没有干净标签。

- 比如：

    - 草地到底是“障碍”还是“可通行软地”？

    - 泥地摩擦系数是 0.3 还是 0.5？

    - 湿楼梯是高度问题还是摩擦问题？

    - 脚陷进地面，是 compliance、slope、friction 还是局部障碍共同造成？

- 如果强行给这些东西打标签，系统会被仿真器的参数定义绑定住。

- DreamWaQ 选择不解释 $z_t$ 的每一维，只要求它对控制有用、对下一步观测重构有用。

- 这就是隐式 latent 的好处：

    - 不要求可解释；

    - 不绑定某个地形分类器；

    - 可以把多种物理效应混在一个表征里。

- 但这也有风险：

    - latent 学到了什么不容易审计；

    - 一旦真实地形超出训练分布，latent 可能给 actor 一个错误但自信的上下文；

    - 所以 domain randomization 和真实测试非常重要。

# CENet Loss 怎么组合：

- CENet 使用 hybrid loss：

$$
L_{CE}=L_{est}+L_{VAE}
$$

- 第一部分是速度估计损失：

$$
L_{est}=MSE(\tilde{v}_t,v_t)
$$

- 这里 $\tilde{v}_t$ 是 CENet 的速度估计，$v_t$ 是仿真器提供的真实 body velocity。

- 第二部分是 beta-VAE loss：

$$
L_{VAE}=MSE(\tilde{o}_{t+1},o_{t+1})+\beta D_{KL}(q(z_t\mid o_t^H)\Vert p(z_t))
$$

- 这里：

    - $\tilde{o}_{t+1}$ 是重构出来的下一步 observation；

    - $o_{t+1}$ 是真实下一步 observation；

    - $q(z_t\mid o_t^H)$ 是根据历史观测得到的 posterior；

    - $p(z_t)$ 是 prior distribution；

    - 论文选择 standard normal distribution 作为 prior，因为 observations 被归一化到 zero mean 和 unit variance。

- 这两个 loss 的分工很清楚：

    - $L_{est}$ 把 latent 系统钉在 body velocity 这个物理量上；

    - reconstruction loss 让 latent 解释前后动力学；

    - KL loss 防止 latent 空间发散，让 $z_t$ 保持可采样、可泛化的分布结构。

- 一个小例子：

    - 假设机器人给了向前走的命令，关节也输出了正常摆腿；

    - 但下一步 body velocity 明显比预期小，同时关节负载变大；

    - CENet 如果只看当前一帧，可能不知道为什么；

    - 但从 $o_t^H$ 看，它能发现“连续几步发力以后，身体响应都偏慢”；

    - latent 就有理由编码“地面可能软/阻力大/脚被草缠住”这种上下文。

- 注意，这个 latent 不需要说出“草地”两个字。它只需要让 actor 输出更合适的步态，比如增加 joint power、抬高脚、放慢速度或调整身体姿态。

# AdaBoot：为什么 bootstrapping 不能一开始就全开

- 论文提到，训练 policy 时从 estimator bootstrapping 可以增强 sim-to-real robustness。

- 但问题是：训练早期 estimator 很不准。

- 如果一开始就强行让 policy 依赖 estimator 输出，会出现一个很危险的反馈环：

    - estimator 乱猜 velocity 和 latent；

    - actor 根据错误估计输出动作；

    - robot 走得更差，采到更混乱的数据；

    - estimator 又在混乱数据上继续学。

- 这就像让一个新手领航员一上车就给赛车手报复杂山路。地图还没看懂，报得越积极越危险。

- DreamWaQ 的做法是 **adaptive bootstrapping**，根据 domain-randomized agents 的 episodic rewards 稳定程度决定 bootstrapping probability。

- 定义：

$$
p_{boot}=1-\tanh(CV(R))
$$

- 其中：

    - $p_{boot}\in[0,1]$ 是 bootstrapping probability；

    - $R$ 是 $m\times1$ 的 episodic rewards 向量；

    - $CV(R)$ 是 coefficient of variation，也就是标准差除以均值；

    - $\tanh(\cdot)$ 用来把 CV 平滑压到 1 附近。

- 直觉很简单：

    - 如果不同随机环境里的 reward 差异很大，说明 policy 还没稳定，$CV(R)$ 大，$p_{boot}$ 小；

    - 如果不同环境里的 reward 都差不多，说明 policy 已经比较稳定，$CV(R)$ 小，$p_{boot}$ 大；

    - 所以 bootstrapping 会随着训练成熟逐步打开。

- 一个极简数值例子：

    - 如果 $CV(R)=1.0$，则 $\tanh(1.0)\approx0.76$，$p_{boot}\approx0.24$；

    - 如果 $CV(R)=0.1$，则 $\tanh(0.1)\approx0.10$，$p_{boot}\approx0.90$。

- 这意味着 AdaBoot 不是“固定相信 estimator”，而是先看训练是否足够稳定，再决定相信多少。

- 我觉得这点非常工程化：

    - early training 阶段先保护 policy，不让 estimator 噪声放大；

    - later training 阶段再让 policy 适应 estimator 误差，增强部署鲁棒性。

# Reward Function：

- 论文强调 reward function 基本沿用之前 locomotion 工作的常见设计，目的不是靠 reward tuning 取胜，而是突出 DreamWaQ 模块本身的作用。

- 总奖励写成：

$$
r_t(s_t,a_t)=\sum_i r_iw_i
$$

- 主要 reward elements 包括：

| Reward | 作用 | Weight |
| --- | --- | ---: |
| Linear velocity tracking | 跟踪 xy 平面速度命令 | 1.0 |
| Angular velocity tracking | 跟踪 yaw rate 命令 | 0.5 |
| Linear velocity z | 惩罚竖直方向速度 | -2.0 |
| Angular velocity xy | 惩罚 roll/pitch 方向角速度 | -0.05 |
| Orientation | 惩罚 body 姿态偏离 | -0.2 |
| Joint accelerations | 惩罚关节加速度 | -2.5e-7 |
| Joint power | 惩罚关节功率 | -2e-5 |
| Body height | 保持期望 body height | -1.0 |
| Foot clearance | 约束脚抬高和摆动质量 | -0.01 |
| Action rate | 惩罚动作变化太快 | -0.01 |
| Smoothness | 惩罚二阶动作变化 | -0.01 |
| Power distribution | 惩罚电机功率使用不均 | -1e-5 |

- 这里最值得注意的是 **Power distribution reward**。

- 普通 joint power penalty 只会压低总能耗，但不会关心某一个电机是不是一直特别累。

- 真实机器人长时间走户外山路时，这个差别很重要：

    - 如果总功率不高，但某几个前腿电机一直承担主要负载，它们会更快过热；

    - 一旦电机进入 overheat protection mode，策略再强也没用；

    - 所以论文加入 motor power variance penalty，让各电机负载更均衡。

- 这点和真实 Course B 上山实验是对上的。论文里提到夏天爬山时电机会快速升温，因此给 robot 较低速度命令来降低所需 torque。

# Curriculum Learning 和 Domain Randomization：

- DreamWaQ 使用 game-inspired curriculum，让 robot 逐步从简单地形过渡到困难地形。

- 地形包括：

    - smooth terrain；

    - rough terrain；

    - discretized terrain；

    - stair terrain。

- 地形坡度有 10 个 level，范围是 $[0^\circ,22^\circ]$。

- 论文还提到，针对低速 locomotion 使用 grid-adaptive curriculum 可以让转向更稳定，并减少 foot tripping。

- Domain randomization 的范围如下：

| Parameter | Randomization range | Unit |
| --- | ---: | --- |
| Payload | [-1, 2] | kg |
| $K_p$ factor | [0.9, 1.1] | Nm/rad |
| $K_d$ factor | [0.9, 1.1] | Nms/rad |
| Motor strength factor | [0.9, 1.1] | Nm |
| Center of mass shift | [-50, 50] | mm |
| Friction coefficient | [0.2, 1.25] | - |
| System delay | [0.0, 15.0] | ms |

- 这几个随机化基本覆盖了真实部署里最容易出问题的差异：

    - 负载变化；

    - PD 增益误差；

    - 电机输出能力偏差；

    - 质心偏移；

    - 摩擦变化；

    - 控制延迟。

- 注意：DreamWaQ 虽然强调“只用 proprioception”，但它不是裸奔到真实世界。它仍然依赖 curriculum、domain randomization、action smoothness、功率惩罚和 PD target 这些非常传统但必要的 sim-to-real trick。

# 训练设置：

- 仿真环境使用 Isaac Gym。

- 论文基于 Rudin 等人的 massively parallel legged RL 实现，同步训练 policy、value network 和 CENet。

- 关键训练参数：

    - 训练 1,000 iterations；

    - 并行环境数量 4,096；

    - PPO clipping range 为 0.2；

    - GAE factor 为 0.95；

    - discount factor 为 0.99；

    - Adam optimizer；

    - learning rate 为 $10^{-3}$；

    - hidden layers 使用 ELU activation。

- 训练硬件：

    - Intel Core i7-8700 CPU；

    - 32 GB RAM；

    - NVIDIA RTX 3060Ti GPU。

- 论文说 DreamWaQ 大约 1 小时可以生成相当于真实世界 46 天训练的数据量。

- 这个数字的工程意义很明显：

    - 真实 robot 不可能在野外摔 46 天来学；

    - massively parallel simulation 是 locomotion RL 能成立的重要前提；

    - 但最终能不能上真实硬件，还要看 randomization 和 representation 是否足够覆盖现实差异。

# 对比方法：

- 论文比较了 5 类方法，而且所有方法都只允许使用 proprioception：

## Baseline

- 这是没有 adaptation mechanism 的 policy。

- 它只能靠当前/历史本体观测直接输出动作，没有额外环境 latent 或 state estimator。

- 这个 baseline 的意义是：验证“只加大 PPO 和 DR”够不够。

## AdaptationNet

- 这是类似 RMA 的 student-teacher adaptation 路线。

- 它用 implicit environmental factor encoder 学一个环境 latent，policy 包含 1D CNN 和 MLP。

- 这个 baseline 的意义是：比较 DreamWaQ 和两阶段/蒸馏式 adaptation 的差异。

## EstimatorNet

- 这是 concurrent training 的 state estimator 路线。

- 它显式估计 body state，但没有 context estimation。

- 这个 baseline 很关键，因为它能回答：

    - **只估速度够不够？还是必须同时学一个环境 context latent？**

## DreamWaQ w/o AdaBoot

- 去掉 adaptive bootstrapping，只保留 CENet 和 asymmetric actor-critic。

- 这个 ablation 用来判断 CENet 本身的价值。

## DreamWaQ w/ AdaBoot

- 完整方法。

- 这个版本用 reward CV 自适应调整 bootstrapping probability。

# 仿真实验结果：

- Fig. 3 显示 DreamWaQ 的学习曲线持续优于其他 proprioception-only baseline。

- 论文还拿 oracle policy 做参考。oracle 可以直接访问周围 terrain height map，相当于看到了 actor 部署时看不到的外部地形信息。

- 结果很有意思：

    - DreamWaQ 虽然没有 exteroception，但表现几乎接近 oracle policy；

    - EstimatorNet 初期 reward 可能比 AdaptationNet 高，但训练后期遇到更难地形后性能下降；

    - DreamWaQ 的曲线更稳定，说明 context latent 比单纯 velocity estimator 更能应对复杂地形。

- 我自己的理解是：

    - EstimatorNet 像只给 actor 一个速度表；

    - DreamWaQ 像同时给了速度表和一份“脚感上下文”；

    - 当地形简单时，速度表就够；

    - 当脚被楼梯绊、地面变软、摩擦突变时，速度之外的 context 才是区分成功和摔倒的关键。

# Command Tracking：

- 论文在 Gazebo 中评估 command tracking，因为 Gazebo 可以提供准确 ground truth。

- 测试设置：

    - robot 接收随机速度命令；

    - 命令每 10 秒从 $[-1.0,1.0]$ 均匀采样一次；

    - 每个 controller 用相同随机种子生成命令，保证公平；

    - 每个 controller 跑 5 次不同随机种子，验证 repeatability；

    - 使用 absolute tracking error, ATE 作为指标。

- Fig. 4 的结论是：

    - DreamWaQ 在 forward velocity error、lateral velocity error、yaw rate error 上都优于 baseline；

    - paired t-test 显示改进显著，图中标注 $p<10^{-4}$；

    - AdaBoot 对 DreamWaQ 还有进一步提升。

- 这里的重点不是“速度跟踪小一点”这么简单，而是：

    - command tracking 是 locomotion policy 的基本盘；

    - 如果为了复杂地形鲁棒性牺牲了基础 tracking，那真实部署也很难用；

    - DreamWaQ 的结果说明 CENet/AdaBoot 没有把基础控制性能弄坏。

# Estimation Comparison：

- Fig. 5 比较了 CENet 和 EstimatorNet 的 body velocity squared estimation error。

- 测试场景是 stairs environment。

- 结果是：

    - 正常平地行走时，CENet 已经有较小估计误差；

    - 当 robot 在楼梯上发生 foot stumble 时，EstimatorNet 的速度估计明显失败；

    - CENet 在 stumble 发生时仍能保持更准确的 body velocity estimation。

- 论文给出的解释有两点：

    - beta-VAE 的 forward-backward dynamics learning 提供了更准确的全地形估计；

    - encoder 同时预测 terrain properties，因此 context 能帮助 explicit velocity estimation。

- 我觉得这里是 DreamWaQ 最有说服力的实验证据之一。

- 因为它说明：

    - latent 不是一个可有可无的附属变量；

    - 当系统遭遇接触异常时，context latent 反过来会帮助速度估计；

    - 速度估计和地形表征不是两个完全独立任务，而是互相增强。

- 这和 CENet 的名字 “Context-Aided Estimator” 是一致的：context 不只是给 actor 用，也在帮助 estimator 自己变准。

# Robustness Analysis：

- 论文在仿真中做了随机 push 测试。

- 设置是：

    - 每隔 1 秒给 robot 施加一次随机方向 push；

    - push velocity 在 body frame 的 x、y、z 方向随机采样；

    - 不断增加/测试直到 robot 摔倒；

    - 记录 maximum push speed 和 30 分钟 random walk survival rate。

- Table III 的结果如下：

| Algorithm | Max. push (m/s) | Survival rate |
| --- | ---: | ---: |
| Baseline | 0.511 ± 0.053 | 20.51 ± 6.44% |
| AdaptationNet | 0.714 ± 0.096 | 82.37 ± 2.49% |
| EstimatorNet | 0.871 ± 0.124 | 80.92 ± 5.73% |
| DreamWaQ w/o AdaBoot | 1.015 ± 0.121 | 90.71 ± 1.25% |
| DreamWaQ w/ AdaBoot | 1.121 ± 0.164 | 95.23 ± 1.61% |

- 这个表非常直观：

    - Baseline survival rate 只有大约 20%；

    - AdaptationNet 和 EstimatorNet 都明显提升；

    - DreamWaQ w/o AdaBoot 已经超过 90%；

    - AdaBoot 进一步把 survival rate 提升到约 95%，max push 也最高。

- 这说明完整 DreamWaQ 的鲁棒性来自两个层面：

    - CENet 提供更准确的 velocity/context estimation；

    - AdaBoot 让 policy 在训练后期主动适应 estimator 误差，提高抗扰动能力。

- 论文还观察到，各方法大多是在 command vector 突然变化时摔倒。这个现象很合理：

    - 突然变向或者刹车要求 robot 快速改变动量；

    - 如果地面摩擦、脚底接触或者 body velocity 估错，很容易失稳；

    - CENet 对隐藏状态估计越准，policy 越能提前输出保守但稳定的恢复动作。

# 真实机器人部署：

- 真实实验使用 Unitree A1。

- 部署设置：

    - policy 和 CENet 在 onboard Intel NUC 上同步运行；

    - inference frequency 是 50 Hz；

    - PD controller 运行在 200 Hz；

    - PD gains 为 $K_p=28$、$K_d=0.7$；

    - 额外 onboard PC 和电池带来约 500 g payload；

    - 通过 PyBind interface 发送关节角命令。

- 这里有两个工程点：

    - 50 Hz 的 policy frequency 对 Unitree A1 这种平台是比较现实的，不是只能在仿真里跑的超高频神经网络；

    - PD 200 Hz 跟踪目标角，把神经网络控制和底层电机控制分开，减少部署风险。

- Fig. 6 展示了 robot 在 foot stumble 和 foot slip 下的 reflex：

    - 下楼梯时，robot 会让身体更靠近地面，并把前脚保持在身体前方更远的位置，用来快速寻找稳定 foothold；

    - 上楼梯时，robot 显著增加 footsteps，让脚能越过台阶并找到稳定落脚点；

    - 打滑时，robot 会先适应 irregular foothold，再恢复正常步态。

- 这类结果虽然偏定性，但很重要。因为复杂地形上真正难的不是平均速度跟踪，而是遇到意外接触时能不能即时改步态。

# Long-Distance Walk：

- 论文做了两个真实户外长距离测试。

## Course A

- Course A 是 KAIST 校园内的 yard，包含大量自然非结构化地形。

- 总长度约 430 m。

- 地形包括：

    - slopes；

    - deformable terrain；

    - thick vegetation；

    - stairs；

    - wet terrain after rainfall；

    - muddy slopes。

- 论文提到 thick vegetation 会缠住 robot legs，但 DreamWaQ 会通过增加 joint power 调整速度来脱困。

- 更困难的是湿楼梯和泥坡：

    - 下湿楼梯会打滑；

    - 泥地会让脚踩得更深；

    - 这些因素都不是简单 height map 能完全解释的。

- DreamWaQ 在干燥和雨后湿地条件下都完成了行走。

## Course B

- Course B 是校园内 hiking track。

- 总长度约 465 m，elevation gain 最高约 22 m。

- 地形包括：

    - asphalt；

    - gravel；

    - slopes；

    - uphill hiking terrain。

- 论文提到实验在夏天进行，motor 很容易升温，所以给 robot 较慢速度命令来降低 torque。

- 最终 DreamWaQ 控制的 Unitree A1 完成了 465 m 轨迹，并在 10 分钟内到达山顶。

- 这个实验的意义不是“走了 465 m”这个数字本身，而是：

    - 小型 Unitree A1 比 ANYmal 这类大型平台更容易受电机热、腿长、负载和地形高度影响；

    - 长距离测试会暴露短 demo 看不到的问题，比如电机过热、累积 drift、连续地形切换、草地阻力；

    - DreamWaQ 至少证明了 proprioception-only policy 可以在真实户外复杂路线中持续工作一段时间。

# 论文的核心贡献：

- **第一点：提出了 proprioception-only 的隐式地形想象框架。** 通过 asymmetric actor-critic，让 actor 在不看 height map 的情况下学会从历史本体观测中推断地形和扰动。

- **第二点：提出了 CENet。** CENet 用共享 encoder 同时做 body velocity estimation 和 context latent learning，再用 beta-VAE reconstruction 约束 latent 解释前后动力学。

- **第三点：提出了 AdaBoot。** 用 episodic reward 的 coefficient of variation 自适应控制 bootstrapping probability，避免训练早期 estimator 噪声伤害 policy，又让训练后期 policy 适应 estimator 误差。

- **第四点：做了比较完整的真实户外验证。** Unitree A1 在 yard、stairs、mud、vegetation、hiking track 等环境中完成长距离行走，展示了 zero-shot sim-to-real 的实际可用性。

- **第五点：强调了长时间硬件运行问题。** Power distribution reward 和 uphill motor overheating 讨论说明作者不是只追求仿真 reward，而是在关注真实电机负载和持续运行。

# 我觉得最值得注意的点：

## 1. DreamWaQ 的关键不是“没有视觉”，而是“把接触后的响应当成感知”

- 很多人看到这篇论文会简单总结成：不用相机也能走复杂地形。

- 但更准确地说，它不是不用感知，而是换了一种感知方式：

    - 视觉是接触前感知；

    - proprioception history 是接触后感知；

    - DreamWaQ 主要依赖后者。

- 这两种感知解决的问题不完全一样。

- 对高台阶、悬崖、坑洞这类必须提前规划的问题，视觉仍然重要。

- 但对湿滑、松软、草地、泥地、轻微碰撞这类只有踩上去才知道的属性，proprioception history 反而更直接。

- 所以 DreamWaQ 的价值不是证明视觉没用，而是证明：**即使没有视觉，只要历史响应建模得好，也能覆盖很大一部分真实地形适应问题。**

## 2. Velocity + latent 是一个很漂亮的切割

- 如果全部隐式放进 latent，网络可能学得很虚，actor 不知道该相信什么。

- 如果全部显式估计成 friction、height、disturbance，又会绑定仿真标签，真实世界很难定义。

- DreamWaQ 的切割是：

    - velocity 显式估计；

    - terrain/context 隐式编码。

- 这和 HIMLoco 的设计非常接近，也是我觉得 legged locomotion 里很实用的一类结构。

- 速度是物理锚点，latent 是复杂性的缓冲区。

## 3. CENet 的 reconstruction 目标解决了 latent 漂移问题

- 只做 velocity MSE 时，latent 很容易没有约束。

- 这和 HIMLoco 里“后 16 维 latent 如果没有 representation objective 就会乱填”的问题很像。

- DreamWaQ 用 beta-VAE reconstruction 来约束 latent：

    - 你不能随便输出一个看似有用但没有结构的 $z_t$；

    - 你必须能用它帮助重构下一步 observation；

    - 所以 latent 被迫携带系统动力学信息。

- HIMLoco 后来用的是 prototype / contrastive / Sinkhorn；DreamWaQ 用的是 VAE。这两者解决的是同一个大问题：**怎么让隐式 latent 真的有用，而不是一串附属噪声。**

## 4. AdaBoot 是典型的训练期工程补丁

- AdaBoot 不是一个宏大的理论创新，但非常实用。

- 它承认 estimator 在训练早期是不可靠的，所以不要强行让 policy 过早依赖它。

- 这个思路可以迁移到很多模块：

    - 早期别相信还没训练好的 terrain encoder；

    - 早期别把 noisy teacher latent 权重拉满；

    - 早期别让不稳定的 auxiliary objective 主导 policy；

    - 等主任务 reward 稳定以后，再逐渐增加这些模块的影响。

- 也就是说，不要一上来全拉满。先让 policy 学会基本站稳和走，再逐步让它适应估计器误差和复杂地形。

## 5. 论文的真实实验比仿真曲线更有信息量

- 仿真里 DreamWaQ 接近 oracle policy 很漂亮，但真实户外长距离测试更能说明问题。

- 因为真实测试会同时出现：

    - motor heating；

    - payload；

    - wet stairs；

    - mud；

    - thick vegetation；

    - long-duration drift；

    - terrain transition。

- 这些因素单独看都不一定致命，但叠在一起就很容易把一个只会短 demo 的 controller 打崩。

- DreamWaQ 能完成 Course A/B，说明它不只是仿真 reward 做得高，而是有一定工程鲁棒性。

# 局限和风险：

- **第一点：它仍然是接触后适应，不是接触前规划。** 论文自己也承认，DreamWaQ 的 adaptation mechanism 必须先让腿碰到障碍，才能通过响应推断地形。这对普通楼梯、草地、泥地可以，但对高台阶、悬崖、深坑这类需要提前规划的结构不够。

- **第二点：复杂地形上仍然可能需要 exteroception。** 作者在 conclusion 里明确说，高层楼梯等更复杂结构未来需要把 exteroception 集成进 locomotion system，以便在接触前做 gait planning。

- **第三点：latent 不可解释。** CENet 的 16 维 context vector 对控制有用，但我们很难知道每一维对应什么。真实部署失败时，调试难度会比显式地形估计更高。

- **第四点：真实实验量化仍有限。** Course A/B 很有说服力，但真实世界没有像仿真表格那样对所有 baseline 做同路线、同天气、同电池状态的大规模量化对比。

- **第五点：A1 平台硬件限制明显。** 论文提到上坡时电机升温，需要降低速度命令。这说明 policy 鲁棒并不等于硬件无限制，motor thermal、torque limit 和 battery 仍然是部署边界。

- **第六点：CENet 的泛化依赖训练分布。** 如果真实地形的接触响应超出 randomization/curriculum 范围，latent 可能无法正确编码，actor 也就没有可靠依据。

# 和我之前笔记里几个方法的关系：

## 和 HIMLoco 的关系

- DreamWaQ 和 HIMLoco 非常适合放在一起理解。

- 两者都承认 locomotion 是 POMDP：

    - 当前 observation 不够；

    - 必须从历史里恢复 hidden state；

    - actor 需要 velocity + latent，而不是裸历史输入。

- 两者的共同结构是：

    - estimator 从 history 中提取信息；

    - actor 使用 current obs + estimated velocity + latent；

    - critic 在训练时可以使用 privileged observation。

- 区别在于 representation objective：

    - DreamWaQ 用 beta-VAE reconstruction 学 context latent；

    - HIMLoco 用 contrastive learning、prototype、Swap Loss、Sinkhorn 防止 latent collapse；

    - DreamWaQ 的 CENet 更像“预测下一步响应”；

    - HIMLoco 的 HIMEstimator 更像“通过跨时空一致性把脚感归类到稳定原型”。

- 如果按工程演化看，DreamWaQ 可以理解成 HIMLoco 这类方法之前非常重要的一步：它证明 velocity + context latent 的框架在真实 A1 上能跑长距离。

## 和 AMP_for_hardware 的关系

- AMP_for_hardware 解决的问题是：不要手工写复杂 reward，让 discriminator 从动物 motion 里提供 style reward。

- DreamWaQ 解决的问题不一样：

    - 它仍然使用比较传统的 locomotion reward；

    - 它的重点不是动作风格自然，而是复杂地形下的 hidden context recovery；

    - 它不使用 motion prior 或 discriminator。

- 但两者有一个共同的工程目标：

    - 仿真里学出来的动作必须真实硬件能承受。

- AMP 用 discriminator 把动作拉回自然运动分布；DreamWaQ 用 action smoothness、power penalty、power distribution、PD target 和 domain randomization 把动作拉回硬件可执行范围。

- 换句话说：

    - AMP 是“动作风格先验”；

    - DreamWaQ 是“地形上下文估计”；

    - 两者都在补纯 RL 容易学出仿真投机动作的问题。

## 和 CTS-MoE 的关系

- CTS-MoE 的重点是多地形、多任务下的专家分工和 value interference。

- 它用 MoE actor、router、multi-critic 让不同地形/任务有不同专家或不同 value head。

- DreamWaQ 没有显式专家，也没有地形分类器。

- 它更像单一 actor + 单一 context latent：

    - 不问“现在是哪类地形”；

    - 不切换到某个专家；

    - 只让 CENet 把历史响应压成一个连续 context，然后 actor 自己调整步态。

- 所以两者的控制哲学不一样：

    - CTS-MoE 是“多个专家软组合”；

    - DreamWaQ 是“单策略隐式适应”。

- 如果未来要结合，我觉得可以有两种方向：

    - 用 DreamWaQ 的 CENet latent 作为 MoE router 的输入，让专家分配基于 proprioceptive hidden context；

    - 或者在 DreamWaQ actor 内部加入轻量 MoE，但保持不需要显式地形标签。

## 和 BFM-Zero 的关系

- BFM-Zero 更关注“任务接口”：motion、goal、reward 都能变成 latent prompt。

- DreamWaQ 更关注“环境接口”：如何从 proprioception history 中恢复地形和扰动 context。

- 两者的 latent 语义不同：

    - BFM-Zero 的 $z$ 是任务/行为 prompt；

    - DreamWaQ 的 $z_t$ 是环境/上下文 latent；

    - 一个回答“我要做什么”；

    - 一个回答“我现在踩在什么隐含物理条件里”。

- 如果把它们放在同一个系统里，合理分工可能是：

    - 上层 task latent 决定目标行为；

    - 下层 context latent 决定当前地形适应；

    - WBC/safety layer 再处理接触约束和力矩保护。

# 对工程实现的启发：

- 如果要在自己的 legged_wbc_mjlab 里借鉴 DreamWaQ，不建议一上来完整复现论文。可以先做最小闭环。

## 1. 先复现 estimator 输入输出边界

- 最小接口可以设成：

    - 输入：最近 $H$ 步 proprioception history；

    - 输出：$\tilde{v}_t\in\mathbb{R}^3$ 和 $z_t\in\mathbb{R}^{16}$；

    - actor 输入：current obs + $\tilde{v}_t$ + $z_t$；

    - critic 输入：privileged obs。

- 先不要急着加复杂地形。可以从 flat + friction randomization 开始，验证 velocity estimation 是否稳定。

## 2. 再加 next-observation reconstruction

- 如果只训练 velocity MSE，很容易变成 EstimatorNet。

- DreamWaQ 真正有区别的是 VAE reconstruction。

- 实现检查点：

    - $o_t^H$ 的 shape 要固定；

    - next observation $o_{t+1}$ 要和 decoder 输出维度一致；

    - velocity target 要从 privileged state 中取真实 body linear velocity；

    - KL term 的权重要小心，不要一开始把 latent 压成无信息噪声。

## 3. PPO 和 estimator 更新要保持同步

- DreamWaQ 的思路不是先离线训练 estimator，再冻结给 actor 用。

- 更合理的流程是：

    - rollout 时使用当前 estimator/policy；

    - update 时同时优化 PPO loss 和 CENet loss；

    - 让 estimator 跟着 actor 当前采到的数据分布变化。

- 这和 HIMLoco 笔记里“estimator 和 PPO 放在同一个 mini-batch 循环里”的直觉一致。

## 4. AdaBoot 可以作为稳定训练的开关

- 如果训练早期 policy 很不稳定，先不要强制 actor 依赖 estimator 输出。

- 可以记录每个 domain-randomized env 的 episodic reward，计算 $CV(R)$，再得到 $p_{boot}$。

- 实现上可以把 bootstrapping 设计成一个概率开关或者混合系数：

    - policy 稳定前，更多使用 ground-truth privileged velocity 或较少注入 estimator noise；

    - policy 稳定后，提高 estimator 输出参与比例。

- 未验证：具体在当前项目里应该怎么接，需要看现有 rollout storage、actor_critic 和 PPO update 的代码结构。

## 5. 硬件友好 reward 不要删

- DreamWaQ 的 power distribution reward 很值得保留。

- 真实机器人上长时间运行时，最先出问题的经常不是平均 reward，而是：

    - 某个电机过热；

    - 某条腿长期承担更大 power；

    - action 抖动导致机械冲击；

    - 低层 PD target 太激进。

- 所以即使我们主要研究 representation，也不能把 action rate、smoothness、joint power、power distribution 这些项当成装饰。

## 6. 真实部署前先做分层验证

- 建议的渐进路线：

    - 先在平地验证 velocity tracking；

    - 再加入 friction randomization，看 CENet latent 是否改变；

    - 再加入楼梯/rough terrain；

    - 再做 push robustness；

    - 最后才做真实机器人长距离路线。

- 每一步只改变一个因素。不要一上来把楼梯、泥地、延迟、负载、视觉缺失、真实硬件全部同时拉满，否则失败了也不知道是哪一环的问题。

# 如果要写成代码，大概有哪些模块：

- 这里不写具体代码，只整理接口，方便后续看 legged_wbc_mjlab 时对照。

## CENet module

- 输入：

$$
obs\_history \in \mathbb{R}^{N\times(H\cdot D_{obs})}
$$

- 输出：

$$
\tilde{v}_t\in\mathbb{R}^{N\times3},\quad z_t\in\mathbb{R}^{N\times16}
$$

- loss：

$$
L_{CE}=MSE(\tilde{v}_t,v_t)+MSE(\tilde{o}_{t+1},o_{t+1})+\beta D_{KL}(q(z_t\mid o_t^H)\Vert p(z_t))
$$

- 注意 shape：

    - encoder 输出至少要能拆成 velocity 和 latent；

    - decoder 输出必须和 next partial observation 对齐；

    - 如果 observation 里包含 command 或 previous action，要明确 next observation 的定义是否包含这些项。

## Actor-Critic module

- actor 输入：

$$
[o_t,\tilde{v}_t,z_t]
$$

- actor 输出：

$$
a_t\in\mathbb{R}^{12}
$$

- desired joint angle：

$$
\theta_{des}=\theta_{stand}+a_t
$$

- critic 输入：

$$
s_t=[o_t,v_t,d_t,h_t]
$$

- critic 输出：

$$
V(s_t)
$$

## Rollout storage

- 除了普通 PPO 需要的 obs、critic_obs、actions、rewards、dones、values、log_probs，还需要保留 CENet 训练所需信息：

    - obs_history；

    - next partial observation；

    - body velocity target；

    - episode reward statistics，用于 AdaBoot 的 $CV(R)$。

## Training loop

- 一个合理的 update 顺序是：

    - rollout 收集当前 policy 的数据；

    - 计算 PPO returns/advantages；

    - 用 mini-batch 更新 CENet loss；

    - 用 actor/critic 更新 PPO loss；

    - 根据 episodic reward CV 更新 $p_{boot}$；

    - 记录 velocity estimation error、reconstruction loss、KL loss、survival/tracking metrics。

- 未验证：这是根据论文结构和常见 legged_gym/rsl_rl 流程做的工程映射，具体落地还要以当前项目代码为准。

# 论文最重要的边界：

- DreamWaQ 的结论不是“视觉没必要”。

- 更准确的结论是：

    - 对大量中低高度、接触后可适应的复杂地形，只用 proprioception history 也能学出很强鲁棒性；

    - 但对需要提前规划的高障碍和复杂结构，proprioception-only 仍然不够；

    - 未来更合理的系统可能是 exteroception 做前瞻规划，proprioception latent 做接触后自适应。

- 这个边界对工程实现很重要。不要把 DreamWaQ 当成万能控制器，也不要因为它没有视觉就忽略外部感知。它真正提供的是一个底层鲁棒适应模块。

# 一句话总结：

- **DreamWaQ 的价值在于，它把四足机器人复杂地形适应从“必须提前看见地形”，推进到“可以通过历史本体响应隐式恢复地形上下文”，用 CENet 的 velocity + latent 表征和 asymmetric actor-critic，让 Unitree A1 在没有外部感知的情况下完成真实户外长距离鲁棒行走。**

# 我的笔记：

- 这篇论文我觉得最核心的是两个字：**脚感**。

- 四足机器人走复杂地形时，很多事情不是眼睛先知道的，而是脚踩上去以后身体才知道的。DreamWaQ 把这种“踩上去以后的身体响应”变成了一个可训练的 context latent。

- 它没有强行让网络说“这是泥地”“这是草地”“摩擦系数是 0.4”。它只要求网络回答一个更工程的问题：**根据过去几步的传感器响应，现在应该怎样走才稳？**

- 从这个角度看，CENet 的 velocity + latent 切割非常好：velocity 保证控制方向不飘，latent 吸收地形和扰动的不确定性。AdaBoot 则是在训练过程中保护这个系统，不让早期不靠谱的 estimator 过早影响 policy。

- 对我们自己的 legged 项目来说，我觉得最值得学的不是某一个 reward 权重，而是这三个思想：

    - actor 部署时只吃真实可得信息，critic 训练时可以吃 privileged information；

    - estimator 不只估速度，还要通过某种 representation objective 让 latent 真的包含环境上下文；

    - 真实硬件长时间运行要关注 power distribution、延迟、负载和电机过热，而不是只看仿真 reward 曲线。

- 后续如果要把 DreamWaQ 和 WBC/MJLab 结合，我觉得更现实的路线是：先让 RL policy 提供 robust joint target 或 gait intent，再让 WBC/safety layer 处理接触约束、力矩边界和硬件保护。这样既能利用 DreamWaQ 的隐式地形适应，也不会把所有真实硬件风险都交给一个黑盒 actor。
