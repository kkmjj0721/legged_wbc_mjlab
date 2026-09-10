# Learning Agile and Dynamic Motor Skills 阅读笔记

## 前言：

- 这篇论文的核心目标，是让 **ANYmal 这类真实中大型四足机器人**，不再完全依赖人工设计的模块化 gait controller 或在线轨迹优化，而是通过仿真中的强化学习直接学出可部署到真实硬件上的动态 motor skill。

- 这里要注意，论文并不是简单说“RL 可以训练四足机器人走路”。在 2019 年这个时间点，RL locomotion 在仿真里已经能做很多漂亮动作，但真正上真实四足硬件会遇到一个很硬的问题：**仿真和真实之间的 reality gap，尤其是执行器、延迟、低层控制器、SEA 弹性结构和未建模阻尼带来的差异。**

- 所以这篇论文真正想解决的问题是：

    - 能不能在仿真里训练复杂动态动作；

    - 又不要求我们把真实执行器的每一个物理细节都手写成解析模型；

    - 同时还能把训练好的 policy 直接搬到真实 ANYmal 上，完成速度跟踪、高速奔跑和跌倒恢复。

- 作者给出的答案是一个很工程化的混合路线：

    - **刚体和接触动力学** 用快速的物理仿真器来算；

    - **执行器和低层软件链路** 用真实机器人数据训练出来的 actuator network 来近似；

    - **模型误差和硬件差异** 用 domain randomization、噪声注入和随机初始状态来覆盖；

    - **动作策略** 用 TRPO 在仿真里训练，输出低阻抗 joint position target，再通过真实机器人上的低层 PD / actuator 模块执行。

- 这篇论文和我之前看的 DreamWaQ、HIMLoco、AMP_for_hardware、Deep-WBC 都有关系，但它站在更早、更基础的位置：它不是在解决“policy 如何恢复地形 latent”，也不是在解决“机械臂和腿怎么统一”，而是在回答一个更底层的问题：**如果执行器模型不靠谱，RL policy 根本过不了 sim-to-real 这一关。**

- 未执行/未验证：这份笔记只基于 `/home/kk/下载/ScienceRobotics_2019_arxiv(3).pdf` 的 PDF 文本抽取和阅读整理，并参考 `/home/kk/飞书文档/论文学习部分.md`、`/home/kk/飞书文档/AMP_for_hardware学习记录.md`、`/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md` 以及项目内已有阅读笔记的写法；没有运行作者代码、没有复现实验，也没有验证真实 ANYmal 部署效果。

## 核心信息：

- **论文标题：** Learning Agile and Dynamic Motor Skills for Legged Robots

- **标题翻译：** 为足式机器人学习敏捷且动态的运动技能

- **作者：** Jemin Hwangbo, Joonho Lee, Alexey Dosovitskiy, Dario Bellicoso, Vassilios Tsounis, Vladlen Koltun, Marco Hutter

- **机构：** ETH Zurich Robotic Systems Lab，Intel Intelligent Systems Lab

- **期刊：** Science Robotics, 2019, 4(26)

- **DOI：** 10.1126/scirobotics.aau5872

- **论文版本：** PDF 中显示 compiled January 24, 2019；ETH Research Collection 页面显示 publication date 为 2019-01-16

- **机器人平台：** ANYmal，约 32 kg，四条腿，每条腿 3 个 actuated DoF，共 12 个 SEA actuator

- **核心方法：** fast rigid-body simulator + learned actuator network + dynamics randomization + TRPO policy training + direct real deployment

- **策略输出：** 12 维 joint position target，不直接输出 torque

- **真实部署频率：** command-conditioned / high-speed locomotion 使用 200 Hz；fall recovery 使用 100 Hz

- **训练算法：** TRPO，论文称使用默认参数和自定义快速实现

- **论文类型：** quadrupedal locomotion / sim-to-real / learned actuator model / reinforcement learning / dynamic recovery

## 原文摘要翻译：

- 足式机器人是机器人领域里最难的一类系统之一。动物那种敏捷、动态的运动能力，很难靠人工设计的传统控制方法完整复现。

- 强化学习是一个很有吸引力的替代方案，因为它可以减少人工规则设计，让控制策略通过试错自动演化。但在这篇论文之前，足式机器人 RL 大多停留在仿真里，真正部署到真实动态平衡机器人上的例子很少，而且通常任务比较简单。

- 这篇论文提出了一种在仿真中训练神经网络 policy、再迁移到真实先进四足机器人上的方法。作者把这个方法应用到 ANYmal 上，使机器人能够更准确、更节能地跟踪高层 body velocity command，以更高速度奔跑，并且能从复杂倒地构型中恢复。

- 换成我自己的理解就是：作者不是只训练了一个会走路的 policy，而是搭了一套足够接近真实硬件的训练闭环，让 policy 在仿真里提前学会真实执行器会带来的延迟、带宽限制和力矩响应。

## 创新点：

- **learned actuator network：** 不再强行手写 SEA actuator 的完整解析模型，而是用真实机器人采集的 position error、joint velocity、torque 数据，监督学习 command-to-torque 映射。

- **hybrid simulation：** 刚体和硬接触仍然用快速物理仿真，执行器和低层软件链路用神经网络补齐，把“物理可解释”和“数据驱动拟合”拆到各自更适合的位置。

- **task-level RL pipeline：** 同一套仿真和训练框架可以训练速度跟踪、高速奔跑、跌倒恢复等不同 motor skills，只需要改变 task description、cost function、initial state distribution 和 randomization。

- **direct hardware deployment：** policy 不是只在仿真里好看，而是直接部署到真实 ANYmal onboard computer 上，以 200 Hz 或 100 Hz 控制频率运行。

- **actuator ablation 证明必要性：** ideal actuator model 和 hand-tuned analytical actuator model 训练出来的 policy 都无法在真实机器人上迈出一步，这直接说明 actuator modeling 是这篇方法成立的关键。

## 研究问题：

- 论文要回答的问题可以拆成三层。

- 第一层是 **控制问题**：

    - ANYmal 这种 12 DoF、带 SEA 的中型四足机器人，能不能不依赖手工 gait pattern，而是让一个神经网络 policy 根据当前状态和历史状态直接输出关节目标？

    - 这里的难点不是动作维度有多高，而是四足 locomotion 是非光滑接触系统。脚什么时候接触、哪里接触、接触力怎么变化，都不是一个固定时序可以完全预设的。

- 第二层是 **仿真问题**：

    - 真实机器人训练太慢、太贵、也太危险，所以必须主要在 simulation 里训练；

    - 但如果 simulation 里的 actuator 太理想，policy 学出来的动作一上真实硬件就会抖、摔、或者完全无法迈步；

    - 特别是 ANYmal 的 SEA actuator 不是 direct-drive motor，内部有 transmission、弹性元件、低层控制器、延迟和不可直接观测的 internal state。

- 第三层是 **部署问题**：

    - 训练出来的 policy 能不能直接部署到 onboard computer 上；

    - 推理时间能不能足够低；

    - 不同任务是否可以共享同一个软件框架，而不是每个 maneuver 都重新手写一套控制器。

- 所以这篇论文的主线不是：

    - “我们训练了一个更大的神经网络。”

- 而是：

    - **把容易建模的部分交给解析物理，把难以建模的执行器链路交给数据驱动模型，再用 RL 在这个 hybrid simulator 里学习 policy。**

## 数据与任务定义：

- 这篇论文的数据主要分两类。

- 第一类是 **actuator network 的监督学习数据**：

    - 来自真实 ANYmal；

    - 采集 joint position error、joint velocity 和 measured torque；

    - foot trajectory 用正弦轨迹激励，通过 IK 转成 joint position；

    - amplitude 为 5 到 10 cm，frequency 为 1 到 25 Hz；

    - 采样频率 400 Hz；

    - 总样本超过一百万；

    - 90% 用于训练，10% 用于 validation。

- 第二类是 **policy training 的仿真交互数据**：

    - rigid-body simulator 推进机器人状态；

    - actuator net 在仿真 loop 里输出 joint torque；

    - policy 根据 observation/history 输出 joint position target；

    - TRPO 用 rollout 数据更新策略。

- 任务定义主要有三类：

    - **command-conditioned locomotion：** 跟踪 forward velocity、lateral velocity、turning rate 三维 command；

    - **high-speed locomotion：** 更聚焦 forward speed，目标是接近 ANYmal 硬件速度极限；

    - **recovery from a fall：** 从随机倒地构型中动态翻回 upright configuration。

- 注意：这篇论文的任务不是复杂视觉导航，也不是粗糙地形 foothold planning。它主要证明的是在真实四足硬件上，基于混合仿真的 RL policy 能完成动态运动技能。

## 研究背景：

### 第一种：模块化控制路线

- 传统四足机器人最常见的是 modular controller design。

- 这类控制器会把问题拆成几个模块：

    - 上层给 desired body velocity 或 foothold；

    - 中间层根据 template dynamics 或 heuristic 生成足端轨迹；

    - 低层 PID / whole-body controller 跟踪这些 reference。

- 这个路线的优点很明显：每个模块都输出物理量，比如 body height、foot trajectory、joint target，工程师可以逐个调参，也可以把安全边界塞进某些模块里。

- 但它的问题也很明显：

    - **模型近似会限制动作范围。** 如果 template dynamics 只在小范围内有效，那机器人就容易被限制在慢加速、躯干保持直立、腿部速度有限的状态。

    - **调参成本非常高。** 跑步、爬坡、恢复站立可能需要完全不同的控制结构，每一个新 maneuver 都要重新设计和调参。

    - **模块之间不一定互相适应。** 上层规划出来的东西，低层未必能跟；低层执行器的限制，上层也未必知道。

- 也就是说，模块化控制不是不强，而是它把很多“身体协调”的问题提前写死了。对于常见 gait 很好用，但对于动态翻身、高速接近硬件极限的动作，会变得非常吃人工经验。

### 第二种：trajectory optimization 路线

- trajectory optimization 的思路更自动一些：用刚体动力学和数值优化，直接算一条机器人应该走的轨迹，然后再用 tracker 去跟踪。

- 这条路线的问题主要有三个：

    - **接触点太难处理。** 四足机器人走路时 contact sequence 不是固定的；如果预先指定接触时序，问题会简单很多，但也限制了动作发现能力。

    - **在线优化太贵。** 如果每一步都要在机器人上求一个复杂非线性优化，很容易需要外部高性能计算机，或者降低模型精度。

    - **planner 和 tracker 仍然割裂。** planner 可能生成理论上很漂亮的动作，但 tracker 在真实执行器、延迟和接触扰动下不一定能跟上。

- 所以 trajectory optimization 的瓶颈不是“数学不优雅”，而是“真实硬件的非光滑接触和执行器限制太复杂”。

### 第三种：直接在真实机器人上 RL

- RL 的吸引力在于，它可以从 sensor reading 到 low-level control signal 做端到端优化，不需要我们提前指定 gait pattern 或 contact schedule。

- 但直接在真实四足机器人上训练几乎不可接受：

    - 学复杂动作可能需要几周甚至几个月真实时间；

    - 训练早期 policy 会随机乱动，容易摔倒或损坏硬件；

    - 动态平衡系统一旦失稳，恢复和安全保护都很麻烦。

- 所以一个更现实的问题出现了：

    - **我们能不能在仿真里完成主要探索，然后把 policy 直接迁移到真实机器人？**

- 这就是 sim-to-real 的核心。

### 第四种：只做高保真仿真也不够

- sim-to-real 通常有两条路：

    - 第一条是把 simulation 做得更准，也就是 system identification；

    - 第二条是承认 simulation 不完美，通过 randomization、noise、disturbance 等方法让 policy 对差异更鲁棒。

- 这篇论文的关键判断是：**两条路都要做，但不能把所有精力都放在解析 actuator model 上。**

- 对 direct-drive actuator，解析建模可能还比较可行；但对 ANYmal 的 SEA actuator，里面有电机、齿轮传动、弹性元件、低层 PD/PID/FOC 链路、通信延迟、机械响应时间等。想把这一整条链路靠 datasheet 和手工参数拟合到足够准确，成本很高，而且很容易漏掉 amplitude-dependent delay 这类细节。

- 所以作者的思路是：

    - 刚体运动和硬接触，尽量用物理模型；

    - 执行器从 command 到 torque 的复杂映射，用真实数据训练一个小网络；

    - 剩下不可避免的误差，用 randomization 和 noise 让 policy 不要卡在单一模型上。

## 核心直觉：

- 我们可以把这篇论文的方法理解成四步。

- **第一步：先确定机器人“身体大结构”大概是什么。**

    - link inertia、center of mass、joint positions 这些东西来自 CAD 和物理测量；

    - 它们不是完美的，所以用一组随机化模型表示不确定性。

- **第二步：不要手写执行器全部细节，直接学一个 actuator net。**

    - policy 输出的是 joint position target；

    - 真实执行器最终产生的是 joint torque；

    - 这中间有低层控制器、通信延迟、机械响应、弹性形变和不可观测 internal state；

    - actuator net 的任务就是根据最近的 position error 和 joint velocity history，预测当前会产生的 torque。

- **第三步：在 hybrid simulation 里训练 policy。**

    - rigid-body simulator 负责接触和刚体动力学；

    - actuator net 负责把 position target 映射成 torque；

    - TRPO 负责更新 policy。

- **第四步：直接把 policy 参数放到真实机器人上跑。**

    - 真实部署时并不跑复杂在线优化；

    - onboard 只做一个小 MLP 的前向推理；

    - 论文里单线程推理小于 $25\mu s$。

- 如果用一个简单比喻：

    - 传统建模像是想把整台机器人从骨骼到神经末梢都手写成方程；

    - 这篇论文的做法像是先承认：骨架运动可以用力学，肌肉和神经延迟很难写准，那就让真实数据告诉我们“给这个关节目标后，实际大概会产生什么力矩”。

- 关键点：**actuator net 不是 policy，它是 simulator 的一部分。**

- 这点很重要。它不是部署时替代真实执行器，也不是给 policy 增加一个额外控制层，而是在训练时让仿真环境更像真实硬件。policy 学到的是在这种更真实的执行器响应下如何走路、跑步和翻身。

## 方法主线：

- 整个方法可以拆成四个模块。

- **Hybrid simulator：** 用快速 rigid-body contact solver 处理刚体和硬接触，用 actuator network 处理执行器和低层控制链路。

- **Actuator network：** 从真实 ANYmal 上采集 position error、joint velocity、torque 数据，训练一个小 MLP 预测 joint torque。

- **Policy network：** 输入当前 observation 和 sparse joint history，输出 joint position target。

- **Training tricks：** TRPO、domain randomization、observation noise、curriculum cost shaping、随机初始状态和任务特定 cost function。

- 论文 Fig. 1 的流程可以简化成：

    1. identify physical parameters and uncertainties；

    2. train actuator net；

    3. train control policy in simulation；

    4. deploy directly on physical system。

- 论文 Fig. 5 的仿真闭环可以理解成：

    1. policy 根据 observation/history 输出 joint position targets；

    2. actuator net 根据 joint position error 和 joint velocity history 输出 torques；

    3. rigid-body simulator 用 torques 推进一步状态；

    4. 新的 joint states 进入 history buffer；

    5. 下一步继续。

- 这里最值得记住的是：**policy 训练时看到的是“经过真实执行器数据校正过的仿真世界”，而不是一个理想力矩源世界。**

## 图表阅读：

> [!figure] Fig. 1：Creating a control policy
> 建议位置：放在“方法主线”之后。
> 放置原因：这张图最适合帮助我们记住整条 pipeline：物理参数识别、actuator net、policy training、real deployment。
> 当前状态：未插入原图，只在笔记中保留图表索引和解释。

> [!figure] Fig. 2：command-conditioned locomotion quantitative evaluation
> 建议位置：放在“Command-conditioned locomotion”和“Forward running comparison”附近。
> 放置原因：这张图同时展示 gait pattern、速度跟踪误差、power、torque，是证明 learned controller 优于 prior controller 的核心结果图。
> 当前状态：未插入原图，只在笔记中保留图表索引和解释。

> [!figure] Fig. 3：high-speed locomotion evaluation
> 建议位置：放在“High-speed locomotion”实验段。
> 放置原因：这张图对应 1.5 m/s 真实高速奔跑、joint velocity、torque 和 gait pattern，能说明 policy 实际用到了硬件极限。
> 当前状态：未插入原图，只在笔记中保留图表索引和解释。

> [!figure] Fig. 4：recovery controller on real robot
> 建议位置：放在“Recovery from a fall”实验段。
> 放置原因：这张图展示真实 ANYmal 从随机倒地构型动态翻身，是论文最直观的真实硬件结果。
> 当前状态：未插入原图，只在笔记中保留图表索引和解释。

> [!figure] Fig. 5：training control policies in simulation
> 建议位置：放在“Actuator Network”或“Policy Network 和 Observation”之前。
> 放置原因：这张图解释了 policy、joint history、actuator net、rigid-body simulator 如何组成训练闭环。
> 当前状态：未插入原图，只在笔记中保留图表索引和解释。

> [!figure] Fig. 6：validation of the learned actuator model
> 建议位置：放在“网络结构和误差”小节。
> 放置原因：这张图是 actuator net 必要性的关键证据，展示 learned model 比 ideal actuator model 更接近真实 torque。
> 当前状态：未插入原图，只在笔记中保留图表索引和解释。

## Rigid-body dynamics 和 domain randomization：

- 论文没有把全部 dynamics 都交给神经网络。

- 对于刚体 link、关节和接触，作者使用已有的快速 contact solver。这个 solver 使用 hard contact model，并且尊重 Coulomb friction cone constraint。

- 为什么这部分不直接学？

    - 因为刚体动力学和接触虽然复杂，但仍然有明确的物理结构；

    - 如果完全用黑盒网络替代，样本需求会更大，外推会更危险；

    - 用解析模型保留物理结构，再对不确定部分做随机化，是更稳的工程选择。

- 论文里提到，单独 rigid-body contact solver 在普通桌面机上可以达到约 $900,000$ timesteps/s；完整 hybrid simulator 加上 actuator nets 后约 $500,000$ timesteps/s，相当于大约 1000 倍 real time。

- 为了处理 CAD 和真实机器人的差异，作者对模型做了随机化。

- 随机化的重点是 inertial properties：

    - 作者认为 link inertial properties 来自 CAD，但由于线缆、电子元件等未建模部分，可能有约 20% 的估计误差；

    - 训练时使用 30 个不同 ANYmal model；

    - center of mass position 加噪声 $U(-2,2)$ cm；

    - link mass 加噪声 $U(-15,15)$%；

    - joint position 加噪声 $U(-2,2)$ cm。

- 这个地方的直觉和后来的 DreamWaQ / RMA / HIMLoco 不太一样。

    - 后来的方法经常让 policy 或 adaptation module 去估计隐藏环境参数；

    - 这篇论文更像是通过随机化把 policy 训练成“不要只适应某一个仿真模型”；

    - 它没有显式学习一个 terrain latent 或 dynamics latent，而是让 policy 的行为本身对这些差异不敏感。

## Actuator Network：

### 为什么 actuator 是这篇论文的关键

- 对四足机器人来说，执行器不是一个简单的“输入 torque，马上输出 torque”的理想模块。

- ANYmal 使用的是 SEA，也就是 series elastic actuator。一个简化链路是：

    - policy 输出 joint position target；

    - joint-level PD controller 把 position command 转成 desired torque；

    - torque command 再进入 PID / current controller；

    - current command 通过 FOC 变成 phase voltage；

    - motor 和 transmission 输出到 elastic element；

    - 弹性元件 deflection 最后形成 joint torque。

- 这条链路里有很多不可直接观测的 internal state。

- 如果我们在仿真里假设执行器没有延迟、无限带宽、可以立即输出任意 torque，那么 policy 会学到一种真实硬件根本实现不了的动作。

- 论文的 ablation 很直接：

    - 用 ideal actuator model 训练的 policy，上真实机器人一步都走不了；

    - 用 hand-tuned analytical actuator model 训练的 policy，也一步都走不了；

    - 作者还观察到 limb violent shaking，原因很可能是各种 delay 没建模准。

- 这说明 actuator model 不是一个“锦上添花”的仿真细节，而是 sim-to-real 是否成立的入口。

### actuator net 的输入输出

- actuator net 要学的是：

    - 输入：最近一小段时间的 joint position error 和 joint velocity；

    - 输出：当前 joint torque。

- position error 在论文里定义为：

$$
e_t = \phi_t^\* - \phi_t
$$

- 其中：

    - $\phi_t^\*$ 是 commanded joint position target；

    - $\phi_t$ 是当前实际 joint position；

    - $\dot{\phi}_t$ 是 joint velocity。

- actuator 的 internal state 不能直接观测，所以作者用 history 来补。

- 具体 history 使用：

    - 当前时刻；

    - $t-0.01s$；

    - $t-0.02s$。

- 这里和 HIMLoco / DreamWaQ 里“用历史恢复隐藏状态”的思想很像，只是恢复对象不同。

    - HIMLoco/DreamWaQ 恢复的是环境、地形、速度或系统响应 latent；

    - 这篇论文的 actuator net 恢复的是执行器内部状态对 torque 输出的影响；

    - 本质上都是：**单帧观测不够，就用 history 补不可观测状态。**

### 为什么 history 不能太短也不能太长

- 论文里有一个很实用的工程判断：history 的长度要比通信延迟和机械响应时间的总和更长。

- 如果 history 太短：

    - actuator 内部的延迟和滞后还没暴露出来；

    - 网络看不到“前一个 command 对现在 torque 的影响”；

    - 高频动作尤其容易建模失败。

- 如果 history 太长：

    - 输入维度增加；

    - 更容易过拟合；

    - 推理成本也更高。

- 所以作者没有把 history 当成越长越好，而是用 validation error 去调。

- 作者还提到，如果输入太 sparse，可能捕捉不到高频 dynamics；这个问题会被 policy smoothness cost 部分缓解，因为 policy 不会产生太突兀的 torque/output change。

### 数据采集方式

- actuator net 的数据来自真实 ANYmal。

- 作者使用一个简单参数化控制器生成正弦形式的 foot trajectory，再通过 inverse kinematics 得到 joint positions。

- 数据采集时有几个细节：

    - foot 会不断接触和离开地面，让轨迹更像 locomotion controller 真实会遇到的情况；

    - foot trajectory amplitude 在 5 到 10 cm 之间变化；

    - frequency 在 1 到 25 Hz 之间变化；

    - 人为扰动机器人，增加数据丰富性；

    - 数据采样频率为 400 Hz；

    - 总共采集超过一百万个样本；

    - 数据采集时间小于 4 分钟，因为 12 个相同 actuator 可以并行采集；

    - 约 90% 用于训练，剩余用于 validation。

- 这里有一个很关键的观察：作者发现 excitation 必须覆盖足够宽的 frequency spectrum。

- 如果采集数据只覆盖很窄的频率范围，actuator net 即使在训练数据上看起来还可以，也可能导致训练中的 policy 出现 unnatural oscillation。

- 这对工程很重要：**actuator data collection 不是随便动几下关节就够了，它要覆盖 policy 未来可能用到的频率和接触状态。**

### 网络结构和误差

- actuator network 是一个小 MLP：

    - 3 个 hidden layers；

    - 每层 32 units；

    - 激活函数选择 softsign。

- 作者比较了 tanh 和 softsign：

    - 两者 validation RMS error 都大约在 0.7 到 0.8 Nm；

    - 但 softsign 推理更快；

    - 12 个 joints 全部评估时，softsign 约 $12.2\mu s$，tanh 约 $31.6\mu s$。

- actuator net 的误差表现：

| 数据 / 模型 | 平均误差 |
| --- | ---: |
| actuator net validation data | 0.740 Nm |
| ideal actuator model validation data | 3.55 Nm |
| actuator net policy-generated test data | 0.966 Nm |
| ideal actuator model policy-generated test data | 5.74 Nm |

- 这个结果说明两件事：

    - actuator net 对真实 torque 的预测明显比 ideal model 准；

    - policy-generated data 上误差更高，说明训练数据和最终 policy 产生的数据仍然存在 distribution shift，但误差仍远小于 ideal model。

- 注意：这并不意味着 actuator net 已经“完美理解执行器”。它只是把最影响 sim-to-real 的 actuator/software dynamics 压到一个足够可用的监督学习模型里。

## Policy Network 和 Observation：

### policy 的角色

- policy 是一个 MLP，输入当前 observation 和 joint state history，输出 joint position targets。

- 论文里 policy network 使用：

    - 两个 hidden layers；

    - hidden units 分别为 256 和 128；

    - tanh nonlinearity。

- 为什么用 tanh？

    - 作者发现 activation function 对真实部署影响很大；

    - ReLU 这类 unbounded activation 在训练未访问状态下可能输出很大 action；

    - bounded activation function 会在扰动下产生更不激进的 trajectory。

- 这点很像硬件部署里的一个常识：仿真里两个网络都能收敛，不代表真实机器人上同样安全。**bounded output behavior 本身就是一个 safety prior。**

### observation 定义

- 论文把时刻 $t=t_k$ 的 observation 写成：

$$
o_k = \langle \phi_g, r_z, v, \omega, \phi, \dot{\phi}, \Theta, a_{k-1}, C \rangle
$$

- 其中：

    - $\phi_g$：IMU frame 下的 gravity direction unit vector；

    - $r_z$：base height；

    - $v$：base linear velocity；

    - $\omega$：base angular velocity；

    - $\phi$：joint positions；

    - $\dot{\phi}$：joint velocities；

    - $\Theta$：sparsely sampled joint state history；

    - $a_{k-1}$：previous action；

    - $C$：command。

- 注意这里的 orientation 表达不是完整 quaternion。

- 作者认为 IMU 只能可靠观测 orientation 中的两个自由度，这可以和 $S^2$ 上的 unit vector 对应，也就是“重力方向在 IMU 坐标系下的表示”。

- 这和很多 legged_gym / IsaacLab 里的 `projected_gravity` 很接近：不用直接给全局 yaw，而是给 roll/pitch 相关的重力方向。

### height 为什么有边界

- base height 不是直接可观测量，作者用 leg kinematics 和 1D Kalman filter 做了简单估计，并假设地面是平的。

- 但是 recovery from a fall 时，机器人不一定站在脚上，这个 height estimator 就不适用了。

- 所以 fall recovery 训练里移除了 height observation。

- 这个细节很重要：**observation 不是越多越好，而是部署时真正可靠、任务状态下真正可用的量才应该给 actor。**

### joint history 的作用

- 论文特别强调 joint state history 对 locomotion policy 很重要。

- 作者的解释是：history 可能让 policy 隐式检测 contact。

- 这很合理。因为不用 foot force sensor 时，policy 仍然可以从这些信号里推断触地状态：

    - joint velocity 突然变化；

    - action 发出后关节没有按预期移动；

    - position error 变大；

    - base velocity / body twist 响应变化。

- 这和 DreamWaQ 的“脚感”直觉非常接近，只是这篇论文没有把它显式命名成 terrain imagination 或 context latent。

## Action Space：为什么输出 joint position target

- 这篇论文没有让 policy 直接输出 torque，而是输出 low-impedance joint position command。

- 真实执行时，position reference 通过固定低层 gain 转成 torque：

$$
\tau = k_p(\phi^\* - \phi) + k_d(0 - \dot{\phi})
$$

- 论文使用的 gain 是：

    - $k_p=50\,Nm/rad$；

    - $k_d=0.1\,Nm/rad/s$；

    - target velocity 为 0。

- 为什么不用 torque action？

    - torque action 在理论上更直接，但训练早期很容易随机产生大量摔倒轨迹；

    - position target 一开始更像一个 standing controller，训练更稳定；

    - position action 和 torque action 虽然存在某种映射关系，但它们的 smoothness 不一样，优化难度也不一样。

- 论文里还有一个容易误解的点：作者说他们的 position policy 和传统 time-indexed position controller 不一样。

- 传统 position controller 的问题是：

    - 上层给一条按时间索引的 reference trajectory；

    - 低层假设这条轨迹会被高精度跟踪；

    - 一旦真实执行器跟不上，整个 trajectory tracking 会崩。

- 这篇论文的 policy 是 **state-indexed**。

- 它不是按固定时间播放关节轨迹，而是每一步根据当前状态和历史状态重新输出 joint position target。

- 所以 policy 会学会预期 position error 的存在，甚至利用 position error 来产生加速度和 interaction force。

- 这里是这篇论文非常值得学的地方：**joint position target 不等于老式僵硬轨迹跟踪；如果 target 是由 closed-loop policy 根据状态实时生成，它仍然可以表现出很强的动态性。**

## Reinforcement Learning 目标：

- 论文把控制问题建成离散时间 RL。

- 每一步：

    - agent 得到 observation $o_t \in O$；

    - 执行动作 $a_t \in A$；

    - 得到标量 reward $r_t \in R$。

- policy 不只看单帧，而是看 observation history：

$$
O_t = \langle o_t, o_{t-1}, \ldots, o_{t-h}\rangle
$$

- 随机策略：

$$
\pi(a_t|O_t)
$$

- 优化目标：

$$
\pi^\* = \arg\max_\pi E_{\tau(\pi)}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

- 这里：

    - $\gamma \in (0,1)$ 是 discount factor；

    - $\tau(\pi)$ 是由 policy 和 environment dynamics 一起决定的 trajectory distribution；

    - action 在这个论文里就是 position command；

    - reward / cost 用来诱导不同任务行为。

- 作者选择 TRPO，而不是 PPO。

- 在今天看，PPO 更常见，但这篇论文的重点不是 TRPO 本身，而是：**当 simulator 足够快、actuator model 足够接近真实、reward/curriculum 设计合理时，on-policy policy gradient 也能在几个小时内训练出真实可用的动态技能。**

## Curriculum Learning：为什么不能一开始就全惩罚

- 论文里最有工程味的训练技巧是 curriculum cost shaping。

- 问题是：locomotion reward 里通常会有很多 regularization：

    - torque penalty；

    - joint velocity penalty；

    - slip penalty；

    - smoothness penalty；

    - orientation penalty；

    - foot clearance 等。

- 如果这些 penalty 一开始就很强，会发生一个常见问题：**站着不动会变成一个很好的局部最优。**

- 因为机器人一动就有 torque、joint speed、slip、姿态变化等 penalty；而站着虽然没有完成速度跟踪，但惩罚可能更小。

- 作者的做法是用 curriculum factor $k_c$ 调节 cost term 和 disturbance：

$$
k_{c,j+1} \leftarrow (k_{c,j})^{k_d}
$$

- 其中：

    - $k_c=k_0\in(0,1)$ 表示 curriculum 开始；

    - $k_c=1$ 表示最终难度；

    - $k_d\in(0,1)$ 控制逼近最终难度的速度；

    - 论文使用 $k_0=0.3$，$k_d=0.997$。

- 因为 $0<k_c<1$ 且指数 $k_d<1$，所以 $k_c^{k_d}$ 会比 $k_c$ 更接近 1。

- 关键点：

    - 和任务目标直接相关的 cost 不乘这个 factor；

    - 例如 command-conditioned / high-speed locomotion 里的 base velocity error cost 不乘；

    - recovery task 里的 base orientation cost 不乘；

    - 其他 constraint / regularization cost 逐步增强。

- 这就相当于训练早期先告诉机器人：

    - **先别急着优雅，先学会往目标方向动。**

- 等它已经会完成主要任务后，再逐渐要求：

    - torque 更小；

    - 关节速度更合理；

    - 脚不要滑；

    - 动作更平滑。

- 这个思路和 AMP_for_hardware 里“不要只靠复杂 reward 手工硬调”有点相反但也互补。

    - AMP 是用运动先验减少手写 reward；

    - 这篇论文仍然使用手写 cost，但通过 curriculum 让手写 cost 不要在训练早期把探索压死。

## Cost Function：

### bounded kernel

- 论文附录里使用了一个 logistic kernel：

$$
K(x) = -\frac{1}{e^x+2+e^{-x}}
$$

- 它的值域是 $[-0.25,0)$。

- 当误差 $x=0$ 时，分母是 4，所以 $K(0)=-0.25$；当误差变大时，$K(x)$ 接近 0。

- 论文把 reward 和 cost 的符号有时交替使用。我们可以把它理解成：

    - tracking 越好，得到越大的“负 cost / reward”；

    - tracking 很差时，惩罚不会无限变大。

- 为什么不用简单欧氏距离？

    - 如果训练早期 tracking error 很大，欧氏距离会给非常大的 cost；

    - 这可能让“早点摔倒终止”反而看起来更划算；

    - bounded kernel 把 tracking error 的坏处限制住，让 termination 不会变成投机策略。

- 这个思想非常实用。RL 训练早期不要让错误动作产生过大的惩罚，否则 policy 可能学会“少动少错”或者“直接终止”。

### command-conditioned / high-speed locomotion cost

- locomotion 和 high-speed 的 cost 项主要包括：

| Cost 项 | 系数 / 形式 | 作用 |
| --- | --- | --- |
| base angular velocity tracking | $c_\omega=-6\Delta t$ | 跟踪目标 yaw / angular velocity |
| base linear velocity tracking | $c_{v1}=-10\Delta t, c_{v2}=-4\Delta t$ | 跟踪目标 base linear velocity |
| torque cost | $k_c c_\tau\lVert\tau\rVert^2, c_\tau=0.005\Delta t$ | 抑制能耗和过大力矩 |
| joint speed cost | $k_c c_{js}\lVert\dot\phi_i\rVert^2, c_{js}=0.03\Delta t$ | 抑制关节高速抖动 |
| foot clearance cost | $k_c c_f(\hat p_{f,i,z}-p_{f,i,z})^2\lVert v_{ft,i}\rVert, \hat p=0.07m$ | 摆动脚有速度时鼓励抬脚 |
| foot slip cost | $k_c c_{fv}\lVert v_{ft,i}\rVert, c_{fv}=2.0\Delta t$ | 接触时抑制切向滑动 |
| orientation cost | $k_c c_o\lVert[0,0,-1]^T-\phi_g\rVert, c_o=0.4\Delta t$ | 保持机身姿态 |
| smoothness cost | $k_c c_s\lVert\tau_{t-1}-\tau_t\rVert^2, c_s=0.5\Delta t$ | 抑制 torque 突变 |

- 注意 foot clearance cost 不是无条件让脚越高越好，而是和 tangential foot velocity 相乘。

- 直觉是：

    - 脚在摆动时才需要 clearance；

    - 脚不动或触地时，不应该因为 clearance 目标把脚乱抬。

### recovery from a fall cost

- fall recovery 的 cost 更像是“从任意奇怪姿态回到可站立构型，同时别把自己撞坏”。

| Cost 项 | 系数 / 条件 | 作用 |
| --- | --- | --- |
| torque cost | $c_\tau=0.0005\Delta t$ | 控制力矩成本 |
| joint speed cost | $c_{js}=0.2\Delta t$，仅当 $\lvert\dot\phi_i\rvert>8rad/s$ | 只惩罚过高关节速度 |
| joint acceleration cost | $c_{ja}=0.0000005\Delta t$ | 抑制关节加速度突变 |
| HAA posture cost | $c_{HAA}=6.0\Delta t$，当 $\lvert\phi_{roll}\rvert<0.25\pi$ | 接近正立时收敛到目标髋外展构型 |
| HFE posture cost | $c_{HFE}=7.0\Delta t$，目标约 $\pm0.5\pi$ | 接近正立时引导髋屈伸 |
| KFE posture cost | $c_{KFE}=7.0\Delta t$，目标约 $\mp2.45rad$ | 接近正立时引导膝关节 |
| contact slip cost | $c_{cv}=6.0\Delta t$ | 减少接触点滑动 |
| body contact impulse cost | $c_{cimp}=6.0\Delta t$ | 减少非脚部身体碰撞冲击 |
| internal contact cost | $c_{cint}=6.0\Delta t$ | 惩罚自碰撞 |
| orientation cost | $c_o=6.0\Delta t$ | 让机器人最终回到正确朝向 |
| smoothness cost | $c_s=0.0025\Delta t$ | 动作平滑 |

- recovery cost 里有一个有意思的条件：某些 joint posture cost 只在 body roll 接近正立时启用。

- 这很合理：

    - 当机器人还倒在地上时，强行要求每条腿立刻回到正常站姿可能会阻碍翻身；

    - 先用 orientation 和 contact/collision 相关 cost 把身体翻回来；

    - 接近正立后，再让各关节回到可站立构型。

- 所以 recovery 不是简单 replay 一条 stand-up trajectory，而是学一个能利用接触和动量的 closed-loop policy。

## Command 和 Initial State Sampling：

### command-conditioned locomotion

- command-conditioned locomotion 使用三维 command：

    - forward velocity；

    - lateral velocity；

    - turning rate / yaw rate。

- 训练分布：

| Command | Min | Max |
| --- | ---: | ---: |
| forward velocity | -1.0 m/s | 1.0 m/s |
| lateral velocity | -0.4 m/s | 0.4 m/s |
| turning rate | -1.2 rad/s | 1.2 rad/s |

- 这个范围是为了匹配当时已有 controller 的能力范围。

### high-speed locomotion

- high-speed 任务更聚焦 forward speed，因此 lateral / yaw 范围更窄。

| Command | Min | Max |
| --- | ---: | ---: |
| forward velocity | -1.6 m/s | 1.6 m/s |
| lateral velocity | -0.2 m/s | 0.2 m/s |
| turning rate | -0.3 rad/s | 0.3 rad/s |

- 这其实是一个很常见的训练取舍：如果目标是冲最高速度，就不要同时要求 policy 在大横向速度和大 yaw rate 下都表现完美，否则探索空间会被拉得太大。

### random command evaluation

- 真实实验区域有限，如果随机 command 无限制采样，机器人可能会被指令带出实验场地。

- 所以作者的 random command evaluation 不是完全无约束采样，而是：

    1. 从 Table S1 的分布里采样 command；

    2. 假设机器人完美跟踪该 command，模拟它的位置轨迹；

    3. 如果轨迹越过实验场地边界，就拒绝该 command 并重新采样；

    4. 直到得到足够长度的 command sequence。

- 这个细节虽然小，但很工程：真实机器人实验不是只考虑算法，还要考虑场地边界、人员安全和测试可重复性。

### 初始状态随机化

- locomotion / high-speed 的初始状态来自两类：

    - previous trajectory；

    - 一个随机分布。

- 随机分布大致是：

| 状态项 | Mean | Std |
| --- | --- | --- |
| base position | $[0,0,0.55]^T$ | 1.5 cm |
| base orientation | $[1,0,0,0]^T$ | about random axis 的 0.06 rad |
| joint position | nominal standing vector | 0.25 rad |
| base linear velocity | $0_3$ | 0.012 m/s |
| base angular velocity | $0_3$ | 0.4 rad/s |
| joint velocity | $0_{12}$ | 2 rad/s |

- 这里的 previous trajectory 初始化很重要。

- 如果每个 episode 都从标准站姿开始，policy 只会熟悉很窄的状态分布。把 previous trajectory 的状态作为初始化，可以让训练数据包含更复杂的 transition，也让 policy 更鲁棒。

## Noise、Velocity 和输入预处理：

- 论文里有一个非常值得记的结果：**去掉 velocity observation 会导致训练完全失败。**

- 理论上，policy 可以从 position 的有限差分里推断 velocity。

- 但实践上，神经网络训练是非凸优化，输入预处理非常重要。让网络自己从 noisy position history 里“顺便学出速度”，可能会把表示学习和控制学习两个难题叠在一起。

- 这和 DreamWaQ / HIMLoco 里显式估计 velocity 的思路是相通的：

    - velocity 是 locomotion 里非常核心的物理量；

    - 如果不给或者不给好，policy 很容易失去控制方向；

    - 隐式 latent 可以吸收很多东西，但 velocity 这种强物理锚点最好不要完全扔给黑盒。

- 作者还考虑了真实机器人里 velocity measurement 的噪声。

- 因为 joint velocity 不是直接测出来的，而是由 position signal 数值微分得到，所以噪声很强。

- 训练时注入：

    - joint velocity noise：$U(-0.5,0.5)$ rad/s；

    - base linear velocity noise：$U(-0.08,0.08)$ m/s；

    - base angular velocity noise：$U(-0.16,0.16)$，原文单位写法略容易混淆，按物理量应理解为角速度噪声；

    - 其他 observation 不加噪声。

- 这个策略不是为了让输入更难，而是为了让训练时 policy 已经习惯真实部署时会看到的不干净 velocity。

## 训练设置：

- 论文里的几个训练数字很有参考价值。

### TRPO 和样本量

- 作者使用 TRPO，并称学习 session 里使用默认参数。

- 借助快速仿真和自定义 TRPO 实现，论文可以在约 4 小时内生成和处理约 2.5 亿个 state transitions。

- 这个数字说明：

    - 这篇论文不是靠超大 GPU 集群堆出来的；

    - 核心能力来自 simulator 和 actuator model 足够快；

    - RL 样本量仍然非常大，所以仿真速度是硬前提。

### discount factor

- 不同任务使用不同 $\gamma$。

| 任务 | $\gamma$ | half-life |
| --- | ---: | ---: |
| command-conditioned locomotion | 0.9988 | 5.77 s |
| high-speed locomotion | 0.9988 | 5.77 s |
| recovery from a fall | 0.993 | 4.93 s |

- 作者提到，更高的 discount factor 会让 standing posture 更自然，因为它更重视长期 torque 和运动代价；但太高会让收敛变慢，所以需要按任务调。

### 训练耗时

| 任务 | simulated time | wall-clock computation |
| --- | ---: | ---: |
| command-conditioned / high-speed locomotion | 9 simulated days | about 4 hours |
| recovery from a fall | 79 simulated days | about 11 hours |

- recovery 更慢很正常，因为它的初始状态更复杂，接触更多，碰撞风险更高，而且需要探索用动量把身体翻回来的动作。

### episode 和 termination

- locomotion / high-speed trajectory 每条最长 6 秒。

- 终止条件主要有两个：

    - violating joint limits；

    - base hitting the ground。

- 终止时 agent 得到 cost 1 并重新初始化。

- 作者说 termination cost 本身没有调，因为最终性能只取决于 cost 系数之间的比例；其他 cost terms 会被调到和这个 terminal value 配合。

### recovery initialization

- fall recovery 不能简单随机采样姿态，否则很容易出现不真实的 interpenetration。

- 作者使用的方法是：

    - 从 1.0 m 高度把 ANYmal 以随机 orientation 和 joint position 丢下；

    - 仿真 1.2 s；

    - 使用这个状态作为 recovery training 的初始状态。

- 这比“手写几个摔倒姿势”更合理，因为它产生的接触状态更接近真实倒地后的动力学结果。

## 实验结果：

### Command-conditioned locomotion

- 这个任务是让 robot 跟踪高层 velocity command。

- command 包含：

    - forward velocity；

    - lateral velocity；

    - yaw rate。

- 实验里，作者用 joystick 给随机 command，还对机身施加多次外部推扰。

- 论文报告：

    - 视频中展示约 40 秒 robust command following；

    - 另有 5 分钟测试没有出现一次 failure；

    - 虽然训练 forward command 从 $U(-1,1)$ m/s 采样，但当 command 设置为 1.23 m/s 时，真实机器人可以可靠达到约 1.2 m/s forward velocity。

- random command quantitative evaluation：

    - 每 2 秒更新一次 command；

    - 测试 30 秒；

    - 总共 15 次随机 command transition，包括从零速度开始的 transition；

    - learned controller 平均 linear velocity error 为 0.143 m/s；

    - 平均 yaw rate error 为 0.174 rad/s。

- 与当时 ANYmal 上最好的 model-based controller 对比：

| 指标 | Learned controller | Model-based controller |
| --- | ---: | ---: |
| average linear velocity error | 0.143 m/s | 0.231 m/s |
| average yaw rate error | 0.174 rad/s | 0.278 rad/s |
| average torque magnitude | 8.23 Nm | 11.7 Nm |
| mechanical power | 78.1 W | 97.3 W |

- 同一 command profile 下，model-based controller 的 linear velocity tracking error 约高 95%，yaw rate error 约高 60%。

- 这组结果很重要，因为它不是只说 learned policy 能走，而是说它在同一真实机器人上比已经多年调过的 model-based controller 更准、更省力。

### Forward running comparison

- 作者又用 0.25、0.5、0.75、1.0 m/s 四个速度 step command 做了对比。

- 结果包括：

    - learned policy 在真实机器人上的平均速度误差为 2.2%；

    - 比 simulation 中高 1.1 个百分点左右；

    - 相比 dynamic lateral walk，速度误差降低约 1.5 到 2.5 倍；

    - 相比 flying trot，速度误差降低约 5 到 7 倍；

    - power efficiency 接近 dynamic lateral walk，并比 flying trot 高约 1.2 到 2.5 倍；

    - torque magnitude 比两个 prior gaits 低 23% 到 36%。

- 作者解释 torque 更低的一个原因是 learned controller 会采用更直的 knee posture，大约比 prior controller 直 10 到 15 度。

- 这个结果说明 RL policy 不只是“动起来”，还学到了一个人为 controller 不敢使用的姿态区域。传统 controller 如果把 knee 调得这么直，可能显著增加摔倒风险；但 learned policy 在训练中学会了如何在这个姿态下稳定交互。

### Actuator model ablation

- 这是我觉得全篇最有说服力的实验之一。

- 作者比较了三种训练仿真：

    - learned actuator network；

    - ideal actuator model；

    - hand-tuned analytical actuator model。

- ideal model 假设：

    - actuator 无限带宽；

    - 零延迟；

    - 可以立即产生任意 commanded torque。

- analytical model 使用真实 actuator controller code 和实验/CAD 识别的动态参数，还手工调了 latency、damping、friction 等参数。

- 但结果是：

    - ideal model 训练出的 policy 真实部署一步都走不了；

    - analytical model 训练出的 policy 也一步都走不了；

    - 真实动作表现为 limbs violent shaking；

    - 作者调 analytical model 超过一周仍然没有成功。

- 这说明对 SEA 这种复杂执行器来说，**“看起来更物理”的解析模型不一定比一个小型监督学习 actuator net 更适合 policy training。**

- 这里不是说物理建模没用，而是要把物理建模放在合适位置：

    - 刚体和接触用物理模型；

    - 执行器细节用数据驱动补；

    - 再用 randomization 处理剩余误差。

### High-speed locomotion

- high-speed 任务的目标不是全向速度跟踪，而是尽可能接近 ANYmal 硬件极限地往前跑。

- 当时 ANYmal 的速度记录是 1.2 m/s，由 flying trot gait 达到。

- 作者训练的 high-speed policy 在真实机器人上做了如下测试：

    - command velocity 慢慢增加到 1.6 m/s；

    - 机器人跑 10 m 后 command 降到 0；

    - simulation 中达到 1.58 m/s；

    - real system 中达到 1.5 m/s；

    - 速度值至少平均了 3 个 gait cycle。

- 这比之前 1.2 m/s 的记录提高了 25%。

- 更关键的是，policy 真实使用到了硬件极限：

    - maximum torque：40 Nm；

    - maximum joint velocity：12 rad/s。

- 传统 planner 难在这里，因为它不仅要规划动作，还要知道后面执行器是否能实现这个动作。很多模块化系统的上层规划模块并不知道低层执行限制，最后输出的 motion 在真实系统上不可实现。

- learned policy 的优势是：训练闭环里已经包含 actuator model 和 hardware limit，所以它学到的是“在这些约束下仍然能实现的动作”。

- 高速策略学出的 gait 也很有意思：

    - 接近 flying trot；

    - 但 flight phase 更长；

    - flight phase duration 还有不对称；

    - 作者说这不是自然界常见 gait，可能是该任务下多个近似最优解之一。

- 这个结果提醒我们：RL 不一定学出“看起来最像动物”的动作，它会学出 reward、硬件约束和 simulator 共同定义下的高性能动作。

### Recovery from a fall

- fall recovery 是这篇论文最展示动态性的实验。

- 传统恢复方法经常是 replay 一条手工调好的关节轨迹。

- 但这种方法有几个问题：

    - 需要人工调很多 posture；

    - 执行时间长；

    - 不利用真实动力学和动量；

    - 对复杂倒地姿态泛化差。

- ANYmal 的 recovery 更难，因为它的 collision model 有 41 个 collision bodies，倒地时会有很多 unspecified internal / external contact。

- 作者训练了 recovery policy，并在真实机器人上测试 9 个随机 configuration。

- 测试包括：

    - 几乎完全 upside-down 的姿态；

    - 自己压在腿上的复杂接触姿态；

    - 多种随机倒地构型。

- 论文报告所有测试中 ANYmal 都成功翻回 upright configuration。

- Fig. 4 展示的例子里，机器人在小于 3 秒内恢复。

- Discussion 里还有两个信息：

    - recovery task 第一次上真实硬件就成功了；

    - 后续通过放宽 joint velocity constraints，把成功率提高到 100%。

- 这里要注意边界：这不是说 recovery policy 天然安全。作者也说 recovery policy 开发用了约一周，主要是因为 high impacts、fast swing legs、fragile components collision 等 safety concerns 不容易写进 cost function。

## 真实部署和计算开销：

- 训练好 policy 后，作者把自定义 MLP implementation 和参数移植到 ANYmal onboard PC。

- 部署频率：

    - command-conditioned / high-speed locomotion：200 Hz；

    - fall recovery：100 Hz。

- 论文提到 recovery motion 训练时是 20 Hz，但部署提升到 100 Hz 后表现一致。这是因为 flip-up 行为的 joint velocity 大多低于 6 rad/s。

- 更动态的 locomotion 通常需要更高 control rate。

- 计算开销非常小：

    - policy inference 小于 $25\mu s$，单 CPU thread；

    - 对应约 0.1% onboard computational resources；

    - 100 Hz 评估时约使用单 CPU core 的 0.25%。

- 这和在线 trajectory optimization 形成对比：

    - 传统方法把计算压力留在运行时；

    - 这篇论文把主要计算压力转移到训练期；

    - 真实部署时只做小网络前向推理。

- 对真实机器人来说，这个差异很重要。onboard compute 省下来后，可以留给 state estimation、safety monitor、mapping、communication、logging 等其他模块。

## 鲁棒性观察：

- Discussion 里有一个很有价值的长期观察。

- 作者说所有 policy 在真实机器人上测试超过三个月，没有修改。

- 期间机器人被大量使用，也发生了硬件变化：

    - 不同机器人配置给总重量带来约 2.0 kg 变化；

    - 新 drive 的 spring stiffness 是原来的 3 倍。

- 这些 policy 仍然表现鲁棒。

- 这说明 learned actuator dynamics + stochastic modeling 没有只 overfit 到一台完全固定的机器人。

- 但这里也要谨慎理解：

    - 论文证明的是在 ANYmal 平台和这些变化范围内鲁棒；

    - 不是证明任意硬件改动都可以不重新训练；

    - 如果换成 coupled hydraulic actuator 或完全不同 transmission，独立 actuator net 的假设可能不成立。

## 关键结果：

- **速度跟踪更准。** random command 测试中，learned controller 的平均 linear velocity error 是 0.143 m/s，yaw rate error 是 0.174 rad/s；同一测试下 model-based controller 分别是 0.231 m/s 和 0.278 rad/s。

- **能耗和力矩更低。** learned controller 的平均 torque magnitude 是 8.23 Nm，对比 model-based controller 的 11.7 Nm；mechanical power 是 78.1 W，对比 97.3 W。

- **步态不是手工指定。** 在 1.0 m/s forward velocity command 下，policy 自己表现出 flying trot；低速下 flight phase 消失，变成 walking trot。

- **高速能力刷新 ANYmal 记录。** high-speed policy 在真实机器人上达到 1.5 m/s，相比此前 1.2 m/s 记录提高 25%，并实际用到 40 Nm 最大 torque 和 12 rad/s 最大 joint velocity。

- **跌倒恢复是真实硬件闭环。** recovery policy 在 9 个随机倒地 configuration 中都能把 ANYmal 翻回 upright configuration，示例动作小于 3 秒。

- **actuator model 是决定性因素。** ideal actuator 和 hand-tuned analytical actuator 两个 ablation 都无法让真实机器人迈出一步，而 learned actuator network 训练出的 policy 可以稳定部署。

- **计算开销很低。** policy inference 小于 $25\mu s$，部署时只占很少 onboard computation；主要计算成本被转移到了训练阶段。

## 论文的核心贡献：

- 我觉得这篇论文的贡献可以总结成四点。

### 1. 把 actuator model 放到了 sim-to-real 的中心

- 以前很多 sim-to-real 讨论会重点看 terrain randomization、mass randomization、sensor noise。

- 这篇论文非常明确地指出：对中大型四足机器人，执行器链路本身就是 reality gap 的核心来源。

- 如果 actuator model 不准，policy 在仿真里学到的 contact timing、swing speed、torque profile 和 body motion 都可能完全不适合真实硬件。

### 2. 提出 practical hybrid simulator

- 它没有全物理，也没有全神经。

- 它把系统拆成两部分：

    - 结构清晰、物理规律明确的 rigid-body dynamics；

    - 难建模、隐藏状态多的 actuator/software dynamics。

- 前者用解析仿真，后者用监督学习。

- 这个切割非常工程：不是为了理论统一，而是为了让仿真既快又足够真实。

### 3. 真实部署了多个 dynamic motor skills

- 论文不是只做一个平地速度跟踪。

- 它展示了：

    - command-conditioned locomotion；

    - high-speed locomotion；

    - recovery from a fall。

- 这三类任务覆盖了不同难点：

    - command following 需要泛化到随机速度和转向；

    - high-speed 需要接近硬件极限；

    - recovery 需要复杂接触和动量使用。

### 4. 证明小型 MLP 也能承载强动态策略

- policy 不是很大的网络。

- 真实部署也不是靠在线优化。

- 它说明在合适的仿真、动作空间和训练流程下，一个小 MLP 可以在真实四足机器人上表现出很强的动态能力。

- 这对工程启发很大：不要一上来就把问题归结为“网络要更大”。很多时候，**仿真闭环、执行器建模、输入边界、action parameterization 和 reward/curriculum 更决定成败。**

## 深度分析：

- 我觉得这篇论文最值得细读的地方，是它对 sim-to-real 问题做了一个很清晰的分解。

- 很多人说 reality gap，容易把它说成一个笼统问题：仿真不够像真实。所以常见反应是把 randomization 加大，或者把 simulator 换得更复杂。

- 但这篇论文没有这么粗暴。它先问：真实差异到底从哪里来？

- 对 ANYmal 来说，刚体 link 和接触虽然难，但它们有比较明确的物理结构；执行器链路反而更难，因为里面混着低层控制软件、不可观测内部状态、延迟、弹性、阻尼和带宽限制。

- 所以作者不是把所有东西都 randomize，也不是把所有东西都 neuralize，而是做了一个分工：

    - 物理结构明确的部分，用 fast rigid-body simulation；

    - 难以解析建模但能采数据的部分，用 actuator net；

    - 仍然无法完全确定的部分，用 stochastic modeling / randomization。

- 这个分工非常像工程里的“先找主误差源”。如果主误差源是执行器延迟，那盲目加 terrain randomization 没有用；如果主误差源是 torque response，不把 actuator loop 建进仿真，policy 学到的动作就会偏离真实因果关系。

- 另一个值得注意的点是，actuator net 并没有试图变成一个万能动力学模型。它只学一个很窄的接口：position error / velocity history 到 torque。

- 这个窄接口反而让它可靠：

    - 输入是硬件上容易记录的；

    - 输出是直接影响 rigid-body dynamics 的 torque；

    - 网络很小，推理很快；

    - 它被放在 simulation loop 里，而不是部署时增加一个新的风险模块。

- 所以这篇论文对我们自己的启发不是“所有机器人都要学 actuator net”，而是：**在做 sim-to-real 之前，先确定训练闭环和真实部署闭环之间哪个接口最不可信，然后优先把这个接口建准。**

- 对 legged_wbc_mjlab，如果当前还没有真实硬件 torque 数据，最安全的默认不是假装有 learned actuator，而是先显式记录这个缺口：用 delay / PD gain / torque limit / motor strength randomization 做保守近似，同时把未来真实日志采集接口预留出来。

## 我觉得最值得注意的点：

### 1. 这篇论文的关键不是 RL，而是 simulator 的接口设计

- 如果只看标题，我们可能会以为这是一篇“RL 训练四足机器人”的论文。

- 但真正读下来，它最核心的是 simulator 和 hardware 之间的接口设计。

- 作者没有要求 simulator 完美复刻真实世界，而是问：

    - 哪些部分可以可信地物理建模？

    - 哪些部分必须从数据里学？

    - 哪些误差应该用 randomization 覆盖？

    - policy 输出的 action 怎样才能既易学又容易上硬件？

- 这个问题比“换 PPO 还是 TRPO”更底层。

### 2. actuator net 是一种很早期的“系统响应建模”

- 在 HIMLoco / DreamWaQ 里，我们经常讲 history latent 是恢复环境隐藏状态。

- 这篇论文的 actuator net 也是在恢复隐藏状态，只不过对象是执行器内部状态。

- 内部 PID、current loop、elastic element、delay 都不可见，但它们会通过 position error / velocity history 影响 torque。

- 所以 actuator net 的本质是：

    - 不直接知道内部状态；

    - 但从过去的输入输出响应里预测当前动力学结果。

- 这就是一种非常工程化的 POMDP 处理方式。

### 3. Position target action 不是保守选择

- 很多时候我们会觉得 torque control 更“高级”，position control 更传统。

- 但这篇论文说明，action space 的好坏不能只看物理直觉，还要看优化 landscape 和真实部署。

- joint position target 有几个优势：

    - 初始 policy 更接近站立，不容易随机摔；

    - 输出经过低层 impedance 后更平滑；

    - 对真实执行器更友好；

    - policy 是 state-indexed，不是死板轨迹回放。

- 对我们做 legged RL 来说，这个结论非常实用。先把 joint target policy 跑稳定，再考虑 torque-level 或 WBC-level action，不要一上来追求最高自由度。

### 4. curriculum 解决的是 reward 早期形状问题

- reward/cost 不是写出来就结束了。

- 同一个 cost，在训练早期和训练后期的作用完全不同。

- 早期如果 regularization 太强，机器人会学会不动；后期如果 regularization 太弱，机器人会抖、耗能、滑脚或撞自己。

- curriculum factor 的价值就在这里：

    - 先让主要任务吸引 policy；

    - 再逐渐把动作质量要求加上去。

- 这比“所有 reward weight 从第一步就固定”更符合实际训练过程。

### 5. 真实实验里最强的证据是 actuator ablation

- 高速 1.5 m/s 和三秒内翻身都很亮眼。

- 但我觉得最能证明方法必要性的，是 ideal / analytical actuator model 训练出来的 policy 都一步走不了。

- 因为这直接排除了一个反驳：

    - “是不是只要普通 domain randomization 加一个好 reward 就够了？”

- 至少在 ANYmal 的 SEA 执行器上，答案是否定的。

## 局限和风险：

- **仍然需要人工设计 cost function。** 论文承认，每个任务仍需要设计 cost function 和 initial state distribution。locomotion policy 的任务设计大约需要两天，recovery policy 大约一周。

- **recovery 的安全成本很难写。** high impact、fast swing leg、自碰撞、脆弱部件碰撞等风险不容易完全编码进 cost。论文里的 recovery 成功不代表我们可以直接在新硬件上尝试类似动作。

- **每个行为仍然分开训练。** 作者说不同 controller 共享同一 code base，但只要换 network parameter set；这意味着它不是一个单一 multi-task policy。单个网络在一次训练中表现为 single-faceted behavior，跨任务泛化有限。

- **actuator independence assumption 有边界。** 对 ANYmal 的 12 个相同 SEA，独立建模每个 actuator 可行；但对共享液压 accumulator 或强耦合驱动系统，单关节独立 actuator net 可能不够。

- **需要真实硬件 torque logging 能力。** 如果一个平台没有可靠 torque measurement、日志系统和可重复激励流程，actuator net 训练会变难。

- **height estimator 假设平地。** locomotion 里的 base height 估计依赖 leg kinematics 和 flat terrain 假设；这篇论文没有证明复杂 rough terrain 上同样成立。

- **论文没有解决视觉 / 地形前瞻规划。** 它的 command-conditioned locomotion 主要是速度命令跟踪，不是主动看地形选 foothold。

- **真实部署结果绑定 ANYmal 平台。** 它证明的是这套方法能在 ANYmal 上有效，不能直接推出任意四足平台都可以零改动迁移。

## 和我之前笔记里几个方法的关系：

### 和 HIMLoco 的关系

- HIMLoco 关注的是 POMDP 下如何从本体历史中恢复 hidden state / velocity / latent。

- 这篇论文更早，重点不是把环境 latent 显式建出来，而是用 history 解决 actuator internal state 不可见的问题。

- 两者共同点是：

    - 单帧 observation 不够；

    - 历史响应携带隐藏状态信息；

    - 训练和部署都要尊重真实可观测边界。

- 区别是：

    - HIMLoco 的 estimator 是 actor 前面的表征模块；

    - Hwangbo 这篇的 actuator net 是 simulator 里的动力学模块；

    - 前者帮助 policy 决策，后者帮助 policy 在更真实的仿真环境中训练。

### 和 DreamWaQ 的关系

- DreamWaQ 的核心是只用 proprioception history 推断 terrain context。

- 这篇论文的 joint history 也可能隐式帮助 contact detection，但没有专门设计 CENet、VAE reconstruction 或 terrain imagination。

- 从工程演化看，可以这么理解：

    - 这篇论文先证明：只要仿真和执行器建模足够靠谱，history-based policy 可以上真实四足；

    - DreamWaQ 往后推进一步：部署时不依赖外部感知，通过 history 显式估计 velocity、隐式压缩地形上下文；

    - HIMLoco 再进一步强化 representation objective，防止 latent 随便漂。

### 和 AMP_for_hardware 的关系

- AMP_for_hardware 解决的是 reward design 太繁琐、动作不自然的问题。

- 这篇论文仍然使用较多手写 cost terms，但它解决的是另一个关键问题：执行器和仿真之间的差异。

- 两者都面向真实硬件 sim-to-real，但侧重点不同：

    - AMP 用 discriminator / motion prior 把动作风格拉向自然数据；

    - Hwangbo 用 actuator net / hybrid simulator 把训练环境拉向真实硬件。

- 如果放到一个工程系统里，它们可以互补：

    - actuator net 保证动作在硬件动力学上可实现；

    - AMP reward 保证动作形态自然、少手写 reward。

### 和 CTS-MoE 的关系

- CTS-MoE 关注多地形、多任务下的隐式专家组合。

- 这篇论文不是 MoE，也不是多任务统一 policy。

- 作者明确说不同 behavior 通过 swapping network parameter set 实现，也就是说每个任务还是单独训练一个 controller。

- 这正好暴露了后续工作要解决的问题：

    - 如果每个 maneuver 都要一个单独 policy，部署和扩展会变复杂；

    - 如果所有任务塞进一个普通网络，又会产生 task interference；

    - MoE / multi-critic 这类方法可以看作是在尝试把多个行为放回一个更统一的策略结构里。

### 和 Deep-WBC 的关系

- Deep-WBC 也使用 joint-space position target，让 policy 直接输出 arm 和 leg 的 joint target。

- 这篇论文可以看作更早的 legged-only 基础：

    - 证明 joint position target policy 可以真实硬件部署；

    - 证明低层执行器模型对 sim-to-real 很关键；

    - 证明 state-indexed policy 不等于简单轨迹回放。

- Deep-WBC 后来把这个思想扩展到 legged manipulator：不只是腿走路，还要让腿和机械臂统一协调。

## 对当前 legged_wbc_mjlab 的工程启发：

### 1. 先明确 action interface

- 如果当前项目要做 Go2 / WBC / MJLab 的 RL 控制，不建议一开始就把 action space 做得太复杂。

- 一个更稳的路线是：

    - 先使用 joint position target 或 joint position offset；

    - 明确 joint ordering；

    - 明确 PD gains；

    - 明确 action scale；

    - 再讨论 torque action、residual torque 或 WBC desired force。

- 原因很简单：action interface 一旦漂，reward、observation、policy checkpoint、sim-to-real 分析都会跟着漂。

### 2. 执行器延迟和低层控制不能假装不存在

- 如果 MJLab / MuJoCo 里只用理想 motor 或理想 PD，policy 可能会学到真实 Go2 无法执行的高频动作。

- 最小可做的工程版本可以是：

    - actuator delay randomization；

    - PD gain randomization；

    - torque limit / velocity limit；

    - action rate penalty；

    - joint velocity noise；

    - motor strength randomization。

- 如果后续有真实硬件日志，再考虑学习一个更像本文 actuator net 的模型。

- 未验证：当前笔记没有检查 `legged_wbc_mjlab` 代码中是否已有这些 event 或 actuator 模型，需要以项目代码为准。

### 3. history buffer 应该是核心接口

- 这篇论文里 actuator net 和 policy 都依赖 history。

- 对我们自己的项目来说，应该尽早固定：

    - history 包含哪些量；

    - 采样间隔是多少；

    - history 是按秒定义还是按 control step 定义；

    - 是否包含 previous action；

    - 是否包含 position error；

    - actor、critic、estimator 是否共享同一份 history。

- 不要等训练不稳定后再临时拼接历史，因为那时 observation shape、normalizer、checkpoint 都会变。

### 4. velocity 不要完全交给网络自己猜

- 论文里移除 velocity observation 会导致训练失败。

- 这对任何 legged 项目都很重要。

- 如果真实机器人上 base linear velocity 不可靠，可以选择：

    - 训练时显式估计 velocity；

    - 使用 history estimator；

    - 给 critic privileged velocity，但 actor 用估计值；

    - 加 velocity noise / bias randomization；

    - 在部署前做 velocity estimator 的独立验证。

- 不建议简单说“网络有历史，它会自己学”。这个假设在非凸训练里很脆弱。

### 5. reward 早期不要把探索压死

- 论文里的 curriculum factor 很适合搬到自己的训练流程里。

- 一个稳妥的训练节奏是：

    - 早期强调 command tracking / stand-up / reaching 主目标；

    - torque、smoothness、slip、joint speed 等项从较低权重开始；

    - policy 会完成任务后逐步增加动作质量约束；

    - 每次只改一个 reward 或 curriculum 参数，观察 failure mode。

- 不要一上来全拉满所有 penalty。这样很容易训练出一个“看起来很安全，但完全不动”的 policy。

### 6. bounded tracking cost 很值得保留

- 论文用 logistic kernel 避免大 tracking error 产生无限惩罚。

- 对当前项目也一样：

    - 训练早期错误很大是正常的；

    - 如果 L2 tracking penalty 太大，termination / standing 可能变成更优选择；

    - bounded reward/cost 可以让 policy 有机会从失败状态附近继续探索。

### 7. 真实部署前要做 actuator model ablation

- 如果要做 sim-to-real，建议至少比较：

    - ideal actuator；

    - delayed PD actuator；

    - randomized actuator；

    - learned actuator / residual actuator model（如果有真实数据）。

- 只有 policy 在这些模型变体下表现稳定，才更有理由相信它不是 overfit 到某个仿真执行器。

- 未验证：这只是基于论文的工程建议，不等于当前项目已经具备真实硬件部署安全性。

## 如果要写成代码，大概有哪些模块：

### ActuatorModel / ActuatorNet

- 作用：在 simulation step 里把 policy 输出的 joint position target 转成 joint torque。

- 最小输入：

    - current joint position error；

    - previous joint position error history；

    - current joint velocity；

    - previous joint velocity history。

- 输出：

    - 12 维 joint torque。

- 如果没有真实数据，可以先做 analytic delayed PD + randomization，不要直接伪造 learned actuator net 的效果。

### HistoryBuffer

- 作用：统一管理 policy、actuator model、estimator 需要的历史。

- 需要固定：

    - history length；

    - sample stride；

    - flatten order；

    - batch/env dimension；

    - reset 时如何清空或填充；

    - terminated/truncated 后如何处理。

### ObservationBuilder

- 建议包含：

    - projected gravity / gravity vector in body or IMU frame；

    - base linear/angular velocity 或 estimator 输出；

    - joint position / velocity；

    - previous action；

    - command；

    - joint state history。

- 注意：部署不可得的 privileged state 不能直接喂给 actor。

### CommandSampler

- locomotion command 至少包含：

    - forward velocity；

    - lateral velocity；

    - yaw rate。

- 如果有场地或仿真 workspace 限制，可以像论文 S2 一样做 rejection sampling，避免 command sequence 把机器人带出边界。

### RewardTerms

- 建议把 reward/cost 拆开 logging：

    - velocity tracking；

    - orientation；

    - torque；

    - joint speed；

    - foot clearance；

    - foot slip；

    - action / torque smoothness；

    - termination。

- 不要只看 total reward。很多训练失败都要靠单项 reward 曲线定位。

### CurriculumScheduler

- 用来控制：

    - regularization cost weight；

    - disturbance strength；

    - command range；

    - terrain difficulty；

    - initialization difficulty。

- 本文的 $k_c$ 是一个简单但有效的参考。

### DomainRandomization / Events

- 至少考虑：

    - link mass；

    - center of mass；

    - joint position offset；

    - friction；

    - motor strength；

    - latency；

    - observation noise；

    - initial state。

### DeploymentWrapper

- 真实部署要明确：

    - policy inference frequency；

    - observation timestamp；

    - state estimator latency；

    - action hold / interpolation；

    - safety clamp；

    - NaN / Inf handling；

    - emergency stop；

    - torque / velocity / position limit。

- 未验证：本文没有替当前项目做硬件 safety review，不能因为论文在 ANYmal 上成功就直接跳过这些检查。

## 论文最重要的边界：

- 这篇论文的结论不是“强化学习从此可以无脑替代模型控制”。

- 更准确的结论是：

    - 对 ANYmal 这种带复杂 SEA 执行器的真实四足机器人，只靠理想仿真或手调解析 actuator model 不够；

    - 把刚体/接触物理仿真、真实数据训练的 actuator network、domain randomization 和 RL policy training 结合起来，可以显著缩小 sim-to-real gap；

    - 在这个框架下，policy 可以真实部署完成速度跟踪、高速奔跑和动态跌倒恢复；

    - 但每个任务仍需要 cost function、initial state distribution 和安全相关设计。

- 它没有证明：

    - 一个 policy 能覆盖所有 locomotion / recovery / manipulation 任务；

    - 不需要人工 reward/cost 设计；

    - 不需要真实 actuator 数据；

    - 不同机器人平台可以直接使用同一 actuator net；

    - 复杂 rough terrain 和视觉前瞻规划已经解决；

    - recovery 动作可以在任意硬件上安全尝试。

- 对我们自己的项目来说，最该学的是这条工程主线：

    - **先把仿真闭环里最影响真实部署的接口建准，再谈 policy 网络和算法花样。**

## 一句话总结：

- **这篇 Science Robotics 2019 论文的价值在于，它把真实四足机器人 RL sim-to-real 的关键从“仿真里能不能学会走路”，推进到“仿真闭环能不能足够真实地包含执行器、延迟、硬件限制和接触响应”，并通过 learned actuator network + fast rigid-body simulation + domain randomization，让 ANYmal 在真实硬件上完成速度跟踪、高速奔跑和动态跌倒恢复。**

## 我的笔记：

- 这篇论文我觉得最核心的是两个字：**执行器**。

- 我们看 legged RL 时，很容易把注意力放在 policy、reward、PPO/TRPO、terrain randomization、网络结构上。但真实机器人最终不是在执行一个抽象 action，而是在执行一个经过低层控制器、通信链路、弹性元件、摩擦、延迟和限幅之后的物理响应。

- 如果这一层没建好，policy 在仿真里再聪明也没用。它学到的可能是“理想电机世界”的走路方式，而真实机器人根本做不到。

- 这篇论文的 actuator net 给了一个很朴素但很强的答案：

    - 不要试图手写所有不可见细节；

    - 让真实机器人自己提供 input-output 数据；

    - 用一个小网络把这条执行器链路塞回 simulator；

    - 再让 policy 在这个更接近真实的闭环里学习。

- 对当前 `legged_wbc_mjlab` 来说，我觉得最现实的学习方式不是马上复现完整 ANYmal actuator net，而是先把几个边界固定：

    - action 是 joint target 还是 torque；

    - PD gain 和 action scale 怎么定义；

    - observation history 怎么组织；

    - actuator delay / motor strength / velocity noise 怎么随机化；

    - reward early-stage curriculum 怎么避免 policy 学成站着不动。

- 先从最简单的开始。不要一上来就同时做 learned actuator、WBC、rough terrain、vision、multi-task 和 sim-to-real。先用一个干净的 velocity command policy 跑通：能站、能走、能跟踪，再逐步加 delay、noise、randomization、history、curriculum 和更复杂任务。

- 这篇论文最值得长期记住的一句话是：**sim-to-real 的问题不是“仿真像不像现实”这么笼统，而是训练闭环里的每一个接口，是否足够接近真实部署时 policy 会遇到的因果关系。**

## 引用：

- `/home/kk/下载/ScienceRobotics_2019_arxiv(3).pdf`

- `/home/kk/飞书文档/论文学习部分.md`

- `/home/kk/飞书文档/AMP_for_hardware学习记录.md`

- `/home/kk/飞书文档/Himloco学习记录/Himloco学习记录.md`

- `/home/kk/legged_wbc_mjlab/docs/BFM-Zero阅读笔记.md`

- `/home/kk/legged_wbc_mjlab/docs/DreamWaQ阅读笔记.md`

- `/home/kk/legged_wbc_mjlab/docs/Deep-WBC阅读笔记.md`
