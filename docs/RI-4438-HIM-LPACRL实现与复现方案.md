# RI-4438 HIM 接入 LP-ACRL：独立任务实现与复现方案

> 状态：设计与只读审查文档，尚未实现 LP-ACRL task，也未完成训练验证。
>
> 审查日期：2026-10-09。依据本地论文 v1、阅读笔记、当前工作区代码，以及当前安装的 mjlab 源码。工作区中已有用户修改，本文以当时磁盘内容为准，不将其改回 Git HEAD。
>
> 本文档已整合任务空间可配置化与代码来源检索的修订。交付范围仅为文档，不修改已有代码或填充目标目录。下面的模块、接口和运行命令均是后续实施规格，不能理解为现在已经可用。

## 1. 结论、目标与证据边界

建议把 LP-ACRL 实现为独立的任务采样层，放在 `src/tasks/locomotion/ri_4438_him_lpacrl` 内。保留 RI-4438 的 HIM 网络、动作、观测、奖励和训练随机化，改变每次 reset 时选择的地形与速度任务。

复现目标是：**复现有符号学习进展驱动的任务采样机制，验证它在 RI-4438 / mjlab / HIM 下是否改善学习。** 机器人、仿真器、策略信息和奖励均与论文不同，不声称完整复刻 ANYmal 系统，也不把作者的“1500 迭代达到 80%”作为本项目验收线。

用户已选择：task 层实现独立；共享基础框架及 HIM 算法；独立全任务评价；粗糙地形使用边界截断并单独统计。首版限定单 GPU。

任务数量、速度分箱、地形列表与各类等级、几何副本、课程温度及评价预算均由配置决定。3、18、600 只作为示例预设；任务表、地形映射、采样器、日志和 checkpoint 不依赖这些固定数字。一次运行内任务定义保持固定，LP 自动改变采样概率；修改定义后启动新实验或显式 warm-start，不在首版实现运行中自动扩张、拆分或合并任务。

### 1.1 论文明确规定什么

论文第 3 页式（5）—（7）给出：

\[
LP_j(t)=\bar R_j(t)-\bar R_{j-1}(t),\qquad
p_{j+1}(t)=\operatorname{softmax}(LP_j(t)/\beta),\quad\beta>0.
\]

`R` 是按任务统计的平均回合累计奖励。LP 保留负号，不取绝对值、不经过 ReLU。ALP 对照是对差分取绝对值后 softmax；本文方法和该 ALP 对照都没有 GMM。

| 项目 | 本地论文依据 | 本项目处理 |
| --- | --- | --- |
| 600 个任务 | 第 5–6 页，5 个纵向速度箱 × 6 个转向速度箱 × 5 类地形 × 4 等级 | 作为扩展任务集合，不作为首轮调试规模 |
| 速度 | `abs(vx)` 至 2.5 m/s、`abs(wz)` 至 3 rad/s，箱宽均 0.5；`abs(vy)` 至 0.5 m/s | 横向速度不增加任务轴；600 集合不增加平地类别 |
| 地形端点 | 第 8 页附录 C，台阶高 0.05–0.20 m、坡度 gradient 0–0.4 | 中间档与具体生成器参数属于工程选择 |
| 温度与评价调度 | 给出温度作用和相邻评价差分，未公开完整数值、频率及样本量 | 下文默认值全部标为项目约定 |
| EPTE-SP | 第 4 页式（8），`(epsilon*k_f + K-k_f)/K` | 原文未完整给出零命令归一化、裁剪等定义 |
| 成功判定 | 第 6 页，alive steps 严格大于 900，且 vx、wz 的 EPTE-SP 各小于 30% | 原文未明确 K，不能等同“存活超过 90%” |
| Figure 9 的奖励子集 | 用 LP-ACRL 在 3000 迭代的成功任务构成所有方法共用的比较子集 | 不能让每种方法各选自己的成功任务后比较 |

阅读笔记对上述主线判断正确。论文正文高度图 108 点与 Table I 的 275 维不一致，action-rate 等公式也有歧义，因此本迁移保留已有 HIM 接口和奖励，不按论文表格机械重建网络及奖励。

### 1.2 当前项目已确认的事实

- 目标目录目前有 12 个 Python 占位文件，全部为空，不存在可运行的 LP-ACRL 实现。
- `src/tasks/__init__.py` 自动发现任务包；新 task 可以自行注册，不必修改通用训练脚本。
- 基线并非完全独立：环境、奖励、命令父类、终止函数等仍引用 `mjlab.tasks.velocity`。
- 基线实际接线的地形课程是上游 `terrain_levels_vel`；不要把本地未接线的同名逻辑当成当前行为。
- 基线已删除坡面和随机粗糙 heightfield，加入 box grid 与多种台阶宽度；它不是论文的五类地形。
- 控制步长为 `0.005 s × 4 = 0.02 s`，20 秒对应约 1000 控制步。这是项目配置推导，不是论文的 K。
- 当前安装版对 `auto_reset=False` 有强约束：已结束的环境必须显式 reset，才允许下一次 step。
- registry 保存已经构建的环境配置，通用训练入口随后才应用 Tyro 覆盖；修改结构性配置后必须重新编译任务表和地形，不能等环境建好后在 runner 内补做。

### 1.3 官方项目与代码仓库核查

本次检索结论为：**找到官方项目页和第三方改编仓库，尚未核实到作者公开的 LP-ACRL 训练源码**。这不是“作者确定未开源”的断言。

| 来源 | 核实结果 | 对本方案的作用 |
| --- | --- | --- |
| [arXiv](https://arxiv.org/abs/2601.17428) | 页面列出 v1，与本地方法依据对应 | 论文公式与实验任务定义的来源 |
| [作者项目主页](https://sites.google.com/view/lp-acrl) | 已读取页面；有论文、方法介绍及视频，本次未发现训练代码链接 | 确认官方项目来源，不据此推断未公开参数 |
| [期刊入口](https://ieeexplore.ieee.org/document/11563605) | 作者主页 Paper 按钮指向此页；未取得可逐项比对的完整新版正文 | 不假定本地 v1 缺失的实现细节已得到补充 |
| [第三方 Go2 仓库](https://github.com/FIBO-Physical-AI/curriculum_training) | README 描述 Go2 / Isaac Lab / RSL-RL 的课程比较 | 可参考工程结构，不能当作作者 ANYmal 实验源码 |

第三方 README 描述了单向 8 个速度箱、箱中心命令、归一化速度课程评分，并加入均匀混合；其运行参数包括 `beta=0.05`、混合比例 `0.15`、阶段长度 `50`。这些属于该仓库的改编选择，不能据此填补论文原始超参数。本次完成 README 级核查，具体源码文件抓取未成功，未完成源码审计。[第三方课程说明](https://github.com/FIBO-Physical-AI/curriculum_training#curriculum-updates)

## 2. 独立边界、目录和公共接口

### 2.1 允许共享与必须复制的内容

| 内容 | 决策 |
| --- | --- |
| `mjlab.envs`、通用 `envs.mdp`、managers、scene、sensor、terrain、math | 允许直接使用 |
| `mjlab.tasks.registry` | 仅用于任务注册，允许 |
| 机器人模型、资产工厂 | 允许；获得配置后 `deepcopy` 再修改 |
| `rsl_rl` 的 HIM 模型、估计器、PPO、storage | 允许共享，本轮不改变算法损失 |
| 其他 `src.tasks.*` 的 MDP、配置、runner、评估 | 禁止导入或继承；需要的实现复制到目标目录 |
| `mjlab.tasks.velocity.*` 的命令、奖励、终止、任务配置 | 同样禁止；必要函数和类本地化 |
| `src.config.ri_4438.*` 的任务行为配置与继承链 | 复制使用到的配置值和类型，不继续继承旧配置 |

复制时保留来源和许可证信息，只复制实际使用的代码。禁止通配导入掩盖依赖，禁止通过动态类字符串、旧 runner 继承或 monkeypatch 绕过独立性检查。通用训练入口全局导入其他注册任务不等于新 task 复用它们的实现；审查重点是目标 task 的实现依赖图。

### 2.2 建议结构

以下均属于目标 task，不在根级共享模块新增 LP 逻辑：

```text
ri_4438_him_lpacrl/
  __init__.py
  ri_4438_him_lpacrl_env_cfg.py
  config/
    __init__.py                 # 注册 Flat / Rough
    env_cfgs.py                 # train / play / evaluation 工厂
    rl_cfg.py                  # 本地 HIM runner 配置
    experiment_cfg.py          # 原始实验配置、合并、校验与编译
    curriculum_cfg.py          # 任务空间、LP、评价配置类型
    presets.py                 # 示例任务定义，不是不同实现分支
    robot_task_cfg.py          # 复制的行为配置，不继承旧 task 配置
    numerics.py                # 本地数值契约
  mdp/
    task_catalog.py            # ID、参数箱、任务定义指纹
    curriculums.py              # 课程状态与 reset 时联合分配
    velocity_command.py        # 整回合固定、服从 task 的命令
    terrains.py                # 本地地形配置与 rough 等级映射
    observations.py / rewards.py / terminations.py
    domain_randomization.py / gait.py / numerics.py
    terminal_observations.py   # 本地 HIM terminal target recorder
  rl/
    train.py                   # 先编译配置再创建环境的本地入口
    runner.py                  # 评价钩子、checkpoint、导出
    wrapper.py                 # timeout 标志适配
  evaluation/
    evaluator.py               # 独立全任务评价
    metrics.py                 # 首试验统计与分任务汇总
    evaluate.py                # 独立 checkpoint 评价入口
```

注册名固定为 `Ri-4438-HIM-LPACRL-Flat`、`Ri-4438-HIM-LPACRL-Rough`。前者默认 `ri_flat_smoke`，后者默认 `ri_joint_smoke`，供通用入口加载默认配置。自定义任务空间通过本地入口选择预设或配置文件，不为每个任务数量创建新的环境类或注册名。旧方案的 `flat3/rough18/rough600` 仅是数量描述，本版不再把它们作为实现分支。

最小接口：

| 接口 | 约定 |
| --- | --- |
| `ExperimentCfg` | 唯一原始输入；包含 `TaskSpaceCfg`、几何、课程、评价、训练和本地 HIM 行为配置 |
| `compile_experiment(cfg)` | 解析全部覆盖后，一次性生成任务表、逻辑地形映射、生成参数、环境配置和评价协议；不生成真实仿真地形 |
| `TaskCatalog.encode/decode` | 语义任务键与当前表内连续整数 ID 双向查表；不使用固定四轴公式 |
| `TaskCatalog.num_tasks` | 只读派生值，等于有效任务记录数 |
| `TaskCatalog.fingerprint` | 包含有序任务表、箱边界、符号与任务内采样定义；不是只保存任务数量 |
| `TerrainMap` | 编译期记录语义到 row/column 的映射；环境构建后绑定真实 `terrain_origins`，再提供 origin 查询 |
| `CurriculumController.assign(env_ids)` | 只为 reset 环境联合分配任务与出生地形 |
| `CurriculumController.commit_evaluation(result, update_count)` | 验证全任务数据后原子提交均值、LP、概率和时钟 |
| `state_dict/load_state_dict` | 保存并校验课程版本、指纹、统计与采样随机状态 |
| `TaskEvaluator.evaluate(frozen_actor)` | 返回完整任务顺序下的回报、计数、长度和终止分类 |

保留 actor 单帧 47 维、6 帧历史共 282 维、critic 244 维、estimator 50 维及 12 维动作；运行时验证实际观测布局。保留 estimator velocity slice `(47, 50)`、target slices `((0, 6), (9, 47), (47, 50))`，以及原有 clipping、历史顺序和 ONNX 契约。

扫描尺寸和分辨率可配置，但首版要求生成的射线数量仍为 187，以保持上述 critic 布局；编译与启动检查不满足时拒绝启动。改变扫描点数应作为新的观测/网络契约另行实现，不能仍声称 critic 固定 244 维或直接恢复旧模型。参数可配置不代表绕过现有 HIM 接口约束。

## 3. 任务集合、地形与命令

### 3.1 预设与任务数量的唯一来源

| 预设 | 默认任务定义 | 数量与定位 |
| --- | --- | --- |
| `ri_flat_smoke` | plane，`abs(vx)` 三箱：0–0.5、0.5–1.0、1.0–1.5；vy=wz=0 | 3，最小接线验证 |
| `ri_joint_smoke` | rough、正/反台阶各 2 等级，配上述 3 个 vx 箱；vy=wz=0 | 18，联合地形采样验证 |
| `paper_flat_velocity` | plane，`abs(vx)` 为 0–4 m/s 的 8 箱；符号均匀，vy=wz=0 | 8，对齐论文第一组任务定义 |
| `paper_terrain_types` | 平地、rough、正/反台阶、正/反坡面各一个显式几何配置；三个命令分量在绝对值 0–1 内采样，不分箱 | 6，对齐论文第二组任务轴；几何仍是项目选择 |
| `paper_joint_space` | 5 个 vx 箱 × 6 个 wz 箱 × 5 类地形各 4 等级；vy 在任务内采样 | 600，对齐论文第三组离散结构 |
| `custom` | 显式配置的命令轴、地形等级和组合筛选 | 编译后计算，无固定数量 |

以上全部使用同一实现。所谓“对齐”只涉及任务定义，不能据此声称 RI-4438、HIM、奖励和几何均与原论文一致。预设的演进是独立实验或显式 warm-start，不是 LP 自动改变任务数量。

没有筛选且符号不单独分任务时：

\[
N_{\mathrm{tasks}}=
\left(\prod_{a\in\text{离散命令轴}}N_{\mathrm{bins},a}\right)
\left(\sum_{g\in\text{地形类别}}N_{\mathrm{levels},g}\right).
\]

各类等级数不要求相同；固定值或连续未分箱的命令轴贡献一个取值。符号若单独参与任务定义，则把相应符号枚举计入组合。筛选后统一以 `len(compiled_task_catalog)` 为唯一真值；`num_tasks` 是只读结果，不允许另填一个与分箱可能冲突的数量。

### 3.2 可配置字段与任务表编译

| 配置项 | 语义与约束 |
| --- | --- |
| vx、vy、wz | 各自选择固定值、连续范围、绝对值分箱或有符号分箱 |
| 分箱 | 支持 `edges` 或有名区间列表，二者互斥；允许非等宽和显式空隙，禁止重叠、非有限端点和零宽箱 |
| 符号 | 绝对值模式支持正向、负向、双向；默认任务内等概率双向，可配置正负概率；若 `sign_as_task_axis=true` 则分别统计；有符号区间禁止叠加此规则 |
| 地形与等级 | 每类有自己的具名等级和实际几何参数；flat 不为了凑矩形表复制伪等级 |
| include/exclude | 按地形、等级、速度箱和符号的名称声明匹配；不执行任意 Python 表达式 |
| 任务内采样 | 默认箱内均匀、命令整回合固定；连续未分箱轴的范围和分布也写入指纹 |
| 几何副本 | 首版使用全局统一正整数 `geometry.replicas`，默认 1；不增加任务数 |
| 运行参数 | 训练/评价 env 数、rollout 步数、评价间隔和试验数与任务数独立 |

固定命令使用固定值类型，不能用 `[0, 0]` 模拟分箱。分箱采样采用 `[low, high)`；对外部命令归类时只允许最高箱包含整个轴的最大端点，间隙中的命令报不属于任务集合。精确零值不生成“正零”和“负零”两个任务。

编译流程固定为：命令组合 × 每类地形自己的等级 → include → exclude → 有效记录表。include 为空表示接受全部候选；每条筛选规则必须匹配至少一个原始候选，空结果集、重复语义键或无效引用均报错。

有效表分配连续 ID `0..N-1`，同时保留稳定语义键，例如 `vx=v1/wz=w0/terrain=random_rough/level=high`。`encode` 按语义键查表，`decode` 按 ID 查表；被排除的组合不分配 ID。轴、箱、类别及等级的顺序在解析配置中显式保存，整数 ID 不被视为跨配置稳定身份。

每回合固定 `episode_task_id` 与命令。`paper_joint_space` 的 vx/wz 符号在任务内均匀抽取，vy 在 `[-0.5, 0.5]` 内连续采样；其他预设或 custom 按各自配置执行。禁止旧 heading、standing、强制直行和全范围定时重采样覆盖命令。command 构造时不能读取尚未初始化的 curriculum，应在首次 reset 时获取控制器。

### 3.3 配置示例与采样语义

以下 YAML 是待实现的项目层接口示例，不是当前可执行配置。`height_max_m`、`step_height_m` 等语义字段由编译器转为本地地形包装器参数，不能原样传给 mjlab 的所有生成器。

```yaml
task_space:
  command_axes:
    vx:
      mode: magnitude_bins
      edges: [0.0, 0.4, 0.8, 1.2]
      signs: both
      sign_as_task_axis: false
    wz:
      mode: magnitude_bins
      edges: [0.0, 0.5, 1.0]
      signs: both
      sign_as_task_axis: false
    vy:
      mode: continuous
      range: [-0.3, 0.3]
  terrains:
    flat:
      levels:
        base: {}
    random_rough:
      levels:
        low:  {height_max_m: 0.02}
        high: {height_max_m: 0.05}
    stairs_outward_up:
      levels:
        low:  {step_height_m: 0.05}
        high: {step_height_m: 0.10}
    stairs_outward_down:
      levels:
        low:  {step_height_m: 0.05}
        high: {step_height_m: 0.10}
  include: []
  exclude: []
geometry:
  replicas: 2
  patch_size_m: [8.0, 8.0]
  boundary_margin_m: 1.1
  seed: 0
curriculum:
  method: lp
  temperature: 1.0
  uniform_mix: 0.0
evaluation:
  interval_updates: 50
  trials_per_task: 8
  num_envs: 128
  horizon_s: 20.0
training:
  num_envs: 1024
  rollout_steps: 100
```

示例产生 `3×2×(1+2+2+2)=42` 个任务、`7×2=14` 个几何实例和 1024 个并行训练环境。将 vy 改为 3 个分箱后得到 126 个任务；增加几何副本则不改变任务数。多个速度任务共享同一几何实例，多个独立模拟环境也可以使用相同 origin，不能通过 `num_envs // num_tasks` 绑定任务或评价次数。

默认 Uniform 是有效记录表逐任务等概率 `1/N`，LP/ALP 的 softmax 也只在此表上归一化；排除任务不会被采样。先独立抽各轴再丢弃非法组合容易引入不明确的分布，统一先抽有效 task，再生成任务内参数和几何副本。

每类地形拥有更多有效记录时自然获得更多总概率。箱等概率不等于连续速度空间等密度，细分一个箱也会改变该区域先验质量和宏平均指标权重；跨分箱粒度比较使用额外固定的外部评价命令集合。若增加“先地形等概率”的策略，应单独命名为另一基线，不能仍当作同一个 Uniform。

### 3.4 地形参数与独立几何映射

下表是预设中的工程参数，custom 可覆盖，每个等级最终都展开为明确几何记录：

| 参数 | `ri_joint_smoke` | `paper_joint_space` |
| --- | --- | --- |
| 台阶高度 m | 0.05、0.10 | 0.05、0.10、0.15、0.20 |
| 坡度 gradient | 无 | 0、0.4/3、0.8/3、0.4 |
| rough 高度上界 m | 0.02、0.05 | 0.02、0.02+0.08/3、0.02+0.16/3、0.10 |

rough 使用本地包装的 `HfRandomUniformTerrainCfg`：按等级设置 `noise_range=(0, a_level)`、`noise_step=0.005`、`vertical_scale=0.005`、`horizontal_scale=0.1`、`scale_with_difficulty=False`。默认关闭 `downsampled_scale` 的额外粗采样。实际高度受量化及生成器的高度平移影响，保存输入参数和有效量化结果，测试使用相同离散规则，不断言 0.04667 m 精确出现。

不能直接使用默认 rough 的四行：默认 `scale_with_difficulty=False` 时只是不同随机地形，难度上界不变；直接改为 True 又会让第零级全平。

台阶采用 `step_width=0.30 m`、`platform_width=3.0 m`、`border_width=1.0 m`、`holes=False`。坡面中心平台宽 `2.0 m`、边框 `0.25 m`；rough 边框 `0.25 m`，不添加其 API 不支持的中心平台字段。地形块默认 `8×8 m`；坡面与 rough 的水平/垂直离散参数一并保存到实验配置。上述插值、粗糙生成语义、台阶宽度和平台尺寸都是项目约定，不是论文已公开的全部参数。

当前 mjlab 在 `curriculum=True` 时每个 sub-terrain 配置一列、忽略 `num_cols`，所有列共享一个 `num_rows`；难度范围为 `(0, 1)` 时向生成器传入 `row/max(num_rows-1, 1)` 的 difficulty。仅改行数会改变中间行的几何，不能用它隐式表达可配置语义等级。

本版改用 **每个 `(family, level)` 一列、几何副本一行** 的矩形布局。只为有效任务实际引用的几何组合生成列，`num_rows=geometry.replicas`。每列的本地 `SubTerrainCfg` 包装器固定该等级参数，不使用上游行 difficulty 改变高度、坡度或 rough 振幅。生成器难度范围固定 `(0, 1)`；因 `function` 不直接接收 row，包装器在副本数 R>1 时以 `round(difficulty*(R-1))` 恢复副本行号，R=1 时取零，并校验编号及编码一致性。几何 seed 从配置 seed、几何语义键和副本编号稳定派生，不依赖任务表遍历顺序。

本地 `TerrainMap` 在编译期保存 `(family, level, replica) → (row, column)` 和生成参数；环境构建完静态地形后调用 `bind(terrain.terrain_origins)` 校验形状并绑定实际 xyz。绑定完成后才允许 assign/reset 查询出生原点，不能在编译期伪造 rough 等地形的真实出生高度。纯 plane 使用独立的平地出生位置适配，无需假装存在 tile 数组。

因此底层 `terrain_levels` 是物理副本行号，`terrain_types` 是几何列号，都不能直接当作语义难度/类别。采样、命令判断、日志和评估统一通过本地映射读取语义。

训练先按课程抽 task，再在该任务的合法副本中等概率抽实例，分别保存 `episode_task_id` 和 `episode_geometry_id`。LP 按 task 聚合；增加副本不增加任务权重。不含随机性的台阶或坡面副本可能相同，预检须说明，不能把复制几何宣称为增加多样性。纯平地预设使用 plane；混合地形中的 flat 使用普通平地块，与其他块共用边界协议。

地形在环境构建时生成，不在每个 reset 重建。reset 同步写入 `terrain_levels`、`terrain_types`、`env_origins`，不能再调用旧升降级函数覆盖 LP 选择。训练与课程评价使用同一地形 seed 和几何定义；不同地形 seed 的泛化评价另列，不能混入一次 LP 前后差分。

正金字塔通常中心高，中心向外走是下行；反金字塔相反。注册和日志使用明确的几何名称，并记录与上/下行的对应关系。随机朝向、正负速度和转弯意味着不能保证整个回合持续上行或下行。

### 3.5 边界与有限地形的解释

plane 阶段没有 tile boundary。粗糙地形按所属 patch 的矩形边界截断，根位置安全余量默认修订为 **1.1 m**，记录 `boundary_truncation`。上一版的 0.5 m 不足以覆盖现有 `1.6×1.0 m` 扫描网格：任意 yaw 下最大水平投影半径约为 `sqrt(0.8²+0.5²)=0.9434 m`，还需留出 frame offset 和步间位移余量。

GPU 几何验收须逐步检查实际扫描点和足部范围，保存超出所属 patch 的计数；发现非终止状态仍有足部/扫描跨块时，当前几何配置不通过验收，不启动正式 LP 实验。按观测到的最大相对范围加至少 0.1 m 余量扩大 root 边界参数，再重跑该检查。不能声称一个未经验证的常数可严格保证所有姿态和速度下的隔离。边界参数属于协议指纹，修改后不可直接沿用旧回报基准。

patch 尺寸、边界余量、扫描网格、速度范围和时限均可配置；预检先验证可用区域为正、出生点合法，正式实验前再完成实际几何检查。这些参数改变后重新建立评价基准，不能通过偷偷除以回合长度来抵消边界影响。

发生跌倒与边界同时触发时保存两个原始标志，分类以物理失败优先。PPO 接收到的 timeout 应为 `truncated & ~terminated`。

边界回合的累计回报参与 LP，但它是含人工边界截断的操作性回报：策略更快出界可能使累计回报下降。因此每任务同时报告时长和边界率，不能把任何负 LP 都解释为退步。首版粗糙地形仅做有限地形块实验，不报告论文式任务成功率，也不把走出边界后的剩余时间伪装成摔倒惩罚。

例如，在 8 m 地形中从中心以 2.5 m/s 直行，根位置余量 0.5 m 时约 1.4 秒即到边界；若补齐剩余 20 秒为 100% 误差，正常高速行走也会被必然判失败。后续论文式 EPTE-SP 实验必须先建立足够大的独立评价地形及明确的误差定义，不能用首版结果对标论文 80% 成功率。

## 4. 回报统计、课程更新与独立评价

### 4.1 reset 事务

当前 mjlab 的自动 reset 顺序为：

```text
termination / reward compute（含最后一步）
  → recorder pre_reset
  → curriculum.compute
  → sim/scene reset、reset events
  → observation/action/reward/metrics/curriculum reset
  → command reset
  → episode length 清零
```

在本地课程 term 中先按旧 ID 读取旧回合信息，再分配新 ID 和地形；command reset 只负责生成新任务对应的命令。保留本地化 HIM terminal recorder，不能删除后让估计器学习 reset 后的新出生状态。

旧回报来自 reward manager 的逐项 `_episode_sums`，求和得到逐环境累计回报。它已包含权重和 dt，不再次乘 dt。`Episode_Reward/*` 日志经过跨 reset 环境平均并除以最大回合秒数，不能用于分任务 LP。依赖 `_episode_sums` 是明确的私有接口依赖，升级 mjlab 时必须重跑契约测试。

初始化、手工 reset、load 后 reset 的片段丢弃，不作为自然结束回合。训练完整回合统计只接受真实 done 以及已开始的 episode；课程 term 的逐回合 `reset` 不得清空全局历史和概率。训练脚本传入的随机初始回合长度在本地 runner 中关闭。

terminal recorder 内部可能调用 command compute，不能在它之后用瞬时命令重新推断旧 task；以整回合固定的 `episode_task_id` 为准。terminal reward 恰计一次，统计必须先于 reward buffer 清零。

### 4.2 更新协议及默认值

| 参数 | 工程默认 |
| --- | --- |
| `training.num_envs`、`training.rollout_steps` | 1024、100 |
| 训练时限、`evaluation.horizon_s` | 默认均为 20 s；本地环境时限参数同样可配置 |
| `curriculum.method` | lp；另支持 uniform、alp |
| `curriculum.temperature` | 1.0，必须有限且大于零 |
| `curriculum.uniform_mix` | 0.0；显式大于零时属于独立扩展实验 |
| `evaluation.interval_updates` | 50 次成功 PPO 更新 |
| `evaluation.trials_per_task` | 8 |
| `evaluation.num_envs` | 128 |
| `evaluation.max_consecutive_failures` | 3 |
| 正式训练种子 | 0、1、2 |
| EMA / 奖励归一化 | 原始 LP 组关闭，首版不自动添加 |

表中数值均为可覆盖的工程默认。整数计数须为正，时限须有限且大于零，温度须有限且大于零，混合比例须满足 `0 <= uniform_mix < 1`。训练、课程评价和结果报告使用同一份已解析协议，禁止各自保存一套互相矛盾的默认值。

首次训练或不含 LP 状态的显式 warm-start，在可选模型 load 完成后执行第零次评价，只建立基准并保持均匀概率；标准 resume 恢复历史均值、概率与评价时钟，不重复初评。之后每完成配置的 `evaluation.interval_updates` 次 PPO 更新评价一次，全部有效任务各获得配置的 `evaluation.trials_per_task` 个有效试验后，才计算差分与 softmax。

stable softmax 使用 float64 计算 `(LP - max(LP)) / beta`，最终采样仍使用可支持的浮点类型；检测有限性并记录下溢后的零概率。数学上的正概率不等于机器精度下绝不会为零。不静默增加 epsilon、截断 LP 或修改温度。ALP 先取绝对值再用同一公式；Uniform 也执行相同评价协议和预算。

只有用户显式配置 `uniform_mix=eta > 0` 才使用 `(1-eta)*p + eta/N`，并将结果标为 LP/ALP 加均匀混合的扩展，原始论文机制对照保持零混合。此处 N 是有效任务数；`method=uniform` 要求混合比例为零以避免无意义配置。单任务时概率恒为 1，归一化熵 `H/log(N)` 单独记为 0，不能除以零。

缺失试验、非有限回报或内部异常使整次评价失效；保留上次有效均值、概率、评价时间，记录原因，在下一次 PPO 更新边界重试。达到配置的连续失败次数上限（默认 3）则停止训练并报错，避免无声退化为长期固定课程。无效数据不填零，不部分提交。

所有温度和评价默认值都不是论文原参数。β=1.0 可能使分布极集中；正式结论先报告该默认实验及分布诊断，再单独记录温度敏感性实验，不能中途改 β 而不改变实验标识。

### 4.3 评价环境与试验表

独立评价环境使用同一动作、观测布局、奖励与地形目录，关闭学习概率更新。训练与评价时限默认相同，允许显式分别配置；评价以 `evaluation.horizon_s` 为准并纳入协议指纹，修改后重新建立基准。它不是旧 `play=True` 配置：旧播放逻辑会改变时限、地形布局、随机事件和课程，无法直接承担固定任务评价。

完整 deepcopy actor，包含 estimator 和 normalizer；使用 `.eval()`、inference mode 及确定性接口，不调用训练 `alg.act`。评价使用名义物理参数，关闭观测噪声、IMU 随机误差、传感器/执行器延迟、动力学随机化及外力事件；保留必要的确定性 reset 逻辑。该评价测量名义条件下的进展，随机化鲁棒性另行报告。

评价仍保留本地任务分配入口，由固定试验表决定 reset 时的 task/origin；“关闭学习课程”不等于删除整个分配 term。初始化期间命令和观测允许零初值，不访问尚未创建的课程对象。评价不用训练专用 terminal recorder 生成 estimator 监督，直接在手动 reset 前读取返回的终止状态。

每任务 M 组固定 `(task_key, trial_id)` 试验，M 来自 `evaluation.trials_per_task`，默认 8。用独立、固定 seed 的本地生成器预生成箱内速度、符号、几何副本和初始朝向，出生 xy 使用 patch 中心，初始关节为名义姿态、速度为零。`trial_id` 是重复试验编号，与几何 `replica_id` 分开。试验表保存后在所有阶段与方法中复用，记录各符号与副本的实际覆盖；M 小于副本数时不能声称覆盖所有几何实例。

相同种子加异步随机 reset 本身不能保证配对试验，因此不在每次结束后临时为新的有效试验抽随机量。若环境或 wrapper 构造已执行初始 reset，该次不计分；安装当前批次试验映射后再显式 reset 并刷新初始观测，才开始累计。

按配置的评价并行槽数 E 分固定批次运行，默认 E=128，每槽只统计首个试验。N 可以大于训练或评价环境数，不能因此漏任务或降低每任务试验数。`auto_reset=False` 允许在 reset 前取得终止状态，但当前 mjlab **不允许已结束环境一直不 reset 却继续 step**。正确流程为：

1. 每步仅对仍 active 的槽累计奖励、误差和步数。
2. 保存首次 done 的结果并永久清除此批次的 active 标志。
3. 在下一步前对所有 done 槽调用局部 reset；使用同一确定性任务，不启动第二个有效试验。
4. 已 inactive 槽后续仿真、reset 和奖励全部排除统计，但仍按 API 要求处理 done，防止数值失稳或 pending-reset 异常。
5. 所有有效槽结束后立即开始下一批；最后一批的填充槽从开始就 inactive。

局部 reset 后仅合并被 reset 槽的观测，不重采样或推进 active 槽历史。nominal 评价禁用随机事件，因此 inactive 槽的 reset 不改变 active 槽的随机扰动序列。实际仿真了的 inactive/填充槽步数也计入评价计算成本。

环境创建、wrapper 初始 reset 和每轮评价均包在 RNG 保存/恢复作用域内，覆盖 Python random、NumPy、Torch CPU 与 CUDA；地形与任务生成器使用独立实例。即使异常也必须恢复。评价结束不 reset 训练环境，不改变训练观测、storage 指针、normalizer 或采样随机状态。

全任务评价始终使用同一个冻结策略，批次之间不推进训练；不能跨多个 PPO 策略版本拼出一次全任务均值。首版明确实现 nominal 协议；若后续开放随机化评价参数，必须为每个 trial 预生成初始随机化和扰动序列或使用隔离的逐试验 RNG，并形成新的评价协议指纹，不能简单打开依赖全局 RNG 的事件开关。

### 4.4 指标与实验成本

每任务至少保存平均 return、样本数、return 标准差、平均长度、vx/vy/wz RMSE、物理失败率、边界截断率、时限结束率。误差基于 body-frame 实际速度与本回合固定命令，单位分别为 m/s 和 rad/s。终止步在 reset 前采集，累计回报不加入 PPO 的 value bootstrap。

训练过程另存目标概率、episode-start 数、completed-episode 数和 rollout 步数占比。它们不会天然相等：短回合失败任务会更频繁 reset，按回合采样不等于按交互步分配预算。

令 N 为任务数、M 为每任务试验数、E 为评价槽数，K 为评价时限对应的控制步上限，则：

\[
N_{\mathrm{trials}}=NM,\qquad
B=\left\lceil\frac{NM}{E}\right\rceil,\qquad
S_{\mathrm{eval,max}}=EKB.
\]

评价间隔 I 对应的训练步数为 `training.num_envs * training.rollout_steps * I`。预检打印以上数量、物理几何实例数与评价/训练槽步比，不能把任务数直接等同仿真地图或 GPU batch 规模。

例如 600×8=4800 个评价试验、128 槽需要 38 批；K=1000 时最多约 486.4 万槽步。每 50 次更新的训练量为 512 万步，评价约增加 **95% 的槽步上界**，不是实测墙钟开销，也不是评价占总时间的比例。显式把间隔改为 250 次时，该比值约为 19%。调度改变会改变 LP 的时间尺度，比较方法必须使用相同调度并重新声明实验配置。

不能因 N 增大而静默减少 M、漏任务或自动改为部分任务评价。显存不足可显式减小 E 分批执行，记录填充/失效槽的真实计算成本。实际记录训练步数、有效评价步数、全部评价槽步数与墙钟时间；跨任务定义比较还应使用第 3.3 节所述固定外部评价集合。

## 5. Runner、恢复和部署兼容

通用训练入口依次创建环境、wrapper、runner，再 load，最后 learn。初评必须在 learn 内且 load 后进行。共享 HIM runner 没有合适钩子时，将必要 learn 流程复制到目标本地，只增加更新后的课程评价钩子，保留原来的 estimator、数值检查、logger 和导出行为。

不能只写一个无人使用的 wrapper 文件：入口已经构造通用 `RslRlVecEnvWrapper`。本地 runner 构造时使用代理包装传入 env，仅当原 wrapper 的 extras 已含 `time_outs` 时，将它改为 `unwrapped.reset_time_outs & ~unwrapped.reset_terminated`；不为 `is_finite_horizon=True` 的配置额外添加 bootstrap 标志。代理其余接口并避免二次初始化/reset，然后把代理交给基础 runner。此处需集成测试确认视频 wrapper 路径也能取得 unwrapped 状态。

共享 HIMPPO 当前 timeout 奖励补偿使用 `gamma * V(s_t)`，不是 terminal `V(s_{t+1})`。本方案保留这个基线近似，只屏蔽真实失败的 bootstrap，不宣称已修复精确 terminal value；若后续修改，应作为本地独立算法变体，不能混进 LP 对照。

### 5.1 本地入口与配置编译时机

当前 registry 中的环境配置已在注册时构建，`load_env_cfg` 只做复制，通用入口之后才应用 Tyro 覆盖。若 TaskCatalog 和地形在注册工厂中提前派生，修改分箱后可能得到“新参数、旧地形”；runner 创建时环境已完成编译，修复已经太晚。

目标目录新增本地 `rl/train.py`，复制现有入口所需的最小单 GPU 启动流程，直接使用本地 runner。保留设备选择、seed、日志、checkpoint、video、wrapper、配置导出及 finally 关闭资源；去掉 tracking 和多 GPU 分支，不导入其他 task 的启动实现。

```text
解析 ExperimentCfg
  → 默认值 < preset < 配置文件 < 显式 CLI
  → 输入校验
  → 编译 TaskCatalog、逻辑 TerrainMap、环境配置与评价协议
  → 派生结果校验及指纹
  → --describe 时打印纯数据摘要并退出
  → 选择设备、建立日志
  → 创建环境并绑定真实 terrain_origins
  → 创建 wrapper 一次、本地 runner
  → load 或 warm-start
  → learn
  → finally 关闭训练与评价资源
```

配置合并约定：标量覆盖；列表整体替换，不按下标合并；`task_space.terrains` 一旦在覆盖文件中出现，就整体替换预设地形集合，因此需列全想保留的类别和等级。`command_axes` 内被提供的单轴定义整体替换该轴，避免切换 mode 后残留旧分箱字段；未提供的轴继承预设。其余映射递归覆盖。CLI 对当前有效模式内的标量/列表字段覆盖后统一重新校验；改变轴模式使用完整轴配置。未知字段、冲突的轴模式、非法筛选和空任务表在仿真前报错。编译产物不再接受改变任务/地形/奖励的第二轮覆盖。

`--describe` 不构造 env、仿真或 GPU tensor，只输出最终任务表摘要、每地形任务数、逻辑 row/column 映射及生成参数、评价预算、完整解析配置与指纹；实际 origin 标记为待绑定，不宣称已测得地形高度。训练启动后记录相同编译产物以及运行时绑定结果，保证预检与实际输入一致。模块顶层只声明类型和函数，CLI 解析与训练放在 `main()` 守卫内，避免自动任务发现触发训练。

通用 `scripts/train.py` 仍可运行注册默认预设，并修改并行度等非结构参数；本地课程初始化时验证编译任务表、实际地形配置/原点形状及奖励协议指纹。发现结构性 CLI 覆盖未经重新编译时，在首个 rollout 前报错并指向本地入口。无需修改通用入口或注册表私有数据。

### 5.2 Checkpoint 与配置兼容

新增 `infos['lpacrl']`，保存：状态版本；语义任务键、有序任务表及完整解析配置；任务/几何/奖励/评价协议指纹；当前概率、历史均值、LP、样本计数；`completed_policy_updates`、`last_evaluated_update`；task sampler RNG 与评价试验表标识。课程数组长度一律来自 N，不写死 600，也不按固定四轴 reshape；热力图按任务元数据聚合。

不能直接依赖旧 `current_learning_iteration % 50`：其从零开始的保存语义可能在恢复时重复编号。本地时钟只在成功 `alg.update()` 后加一，评价成功后提交时钟，再保存对应课程状态。

- 标准 resume：先检查任务、地形、奖励、评价协议及数值契约，再恢复课程状态；随后执行一次不计数的训练环境 reset，刷新命令、观测和历史，消除入口 wrapper 在 load 前按初始分布产生的状态；不重复第零次评价。
- 旧 HIM checkpoint：没有 LP 状态时作为明确的 warm-start，均匀概率、重建基准、课程时钟归零。不得假装恢复了不存在的课程历史。
- 新环境 reset 不继承旧模拟器的半截 episode；课程统计不把两段轨迹拼接。
- 不兼容的 task catalog 或协议默认拒绝 resume；要迁移只能显式 warm-start。
- 只保证保存状态的恢复一致性，不保证未保存模拟器状态的完整轨迹逐位复现。

相同 N 不代表兼容。箱边界、符号规则、未分箱命令范围、筛选、等级参数、几何 seed/副本、边界、奖励、时限、试验表和评价调度均属于需要校验的语义或协议。配置变化后不能直接用旧均值做下一次差分；拆箱不能把父箱均值复制给子箱，合箱也不能仅平均旧均值冒充新分布的基准。

首版不自动迁移 LP 历史。显式 warm-start 可复用数值/观测契约兼容的 HIM 权重，但均匀初始化 sampler，清空课程统计和时钟并重新全任务评价。

训练/评价环境数与输出目录单独作为运行元数据，允许在语义兼容时调整并重新创建 rollout storage；保持已有试验表，不因槽数改变重新抽试验。训练并行度改变会影响 PPO batch、交互量和固定更新间隔的训练预算，应记录为实验运行条件变化，不承诺相同学习轨迹或等预算比较。

初版拒绝多 GPU 配置并给出清晰错误。后续分布式版本需在统一 rollout 边界归约和广播课程状态，不能在各 rank 的异步 reset 内做 collective。

## 6. 实施顺序、运行方式与验收

### 6.1 实施顺序

1. 本地化配置、MDP、runner 与 terminal recorder，完成依赖和 HIM 契约检查。
2. 接入原始配置类型、本地入口、TaskCatalog/TerrainMap 编译与 `--describe`，用非等宽箱和不同等级数验证可配置性。
3. 用 `ri_flat_smoke` 和 Uniform 验证 reset 归因，再接入独立 evaluator、LP/ALP 更新、日志与 checkpoint。
4. 用 `ri_joint_smoke` 和 custom 验证实际几何、副本、边界和跨 task 归因。
5. 完成三种子 Uniform/LP/ALP 比较后，再使用论文相关预设扩展；不为更大 N 增加另一套硬编码实现。

受控比较只改变采样模式；网络、奖励、随机化、任务集合、训练交互预算和评价时刻保持一致。旧 RI-HIM 可作为系统参考，但因其地形和命令分布不同，不称为只改变课程的一项消融。

### 6.2 预定命令

以下命令仅描述实施后的预期入口；当前空 task 尚未注册，不能运行这些命令验证 LP-ACRL：

```bash
.venv/bin/python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.train --preset ri_flat_smoke --describe
.venv/bin/python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.train --config /path/to/experiment.yaml --describe
.venv/bin/python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.train --config /path/to/experiment.yaml
.venv/bin/python -m scripts.train Ri-4438-HIM-LPACRL-Flat --help
.venv/bin/python -m scripts.train Ri-4438-HIM-LPACRL-Rough --help
```

本地入口的 `--preset`、`--config`、`--describe` 是本方案约定的待实现顶层接口；其余 Tyro 嵌套参数名称以实现后的 `--help` 为准。课程编译产物放入本地环境的 term 参数，runner 从控制器读取，避免 environment 与 agent 各存一份不同的任务定义。

独立 evaluator 提供 `--checkpoint`、`--config`、`--output-dir` 参数；默认使用 checkpoint 保存的评价协议，显式外部评价配置作为单独报告且不修改训练课程。导出逐任务 JSON/CSV、语义任务表与汇总，不能因只读取整数 ID 而误配新旧任务。

### 6.3 必须通过的测试

| 层级 | 场景与验收 |
| --- | --- |
| 依赖静态检查 | import、动态字符串、callable 来源、MRO 均不指向其他 task 实现；允许 registry |
| TaskCatalog | 非等宽箱、可选 vy、符号模式、各类不同等级数和筛选后的 ID 双向映射；端点/间隙归属正确；语义变化改变指纹 |
| LP 数学 | 首轮只建基准；负 LP 保留；等 LP 均匀；平移不变性；非法温度与非有限数据拒绝；无效评价不提交 |
| reset | 末步回报恰一次；旧 ID 先统计、新 ID 后分配；非 reset 槽原点和命令不变；人工 reset 不污染统计 |
| 配置 | preset/文件/CLI 合并优先级与替换语义正确；修改输入后任务表/地形同时重编译；预检不创建 GPU 环境；train/play/eval 不互相污染 |
| evaluator | 首个 done 后不再累计；结束槽必须 reset 才再 step；局部 reset 不改变 active 槽历史；不足一批不多记试验 |
| RNG/状态隔离 | 评价前后训练权重、normalizer、storage、观测历史、课程概率和 RNG 保持不变，直到显式提交新概率 |
| 终止 | 边界与跌倒同时发生时失败优先；timeout mask 不包含真实失败；原始标志仍保留 |
| 地形 | 不同副本不改变语义等级与任务数量；本地包装器不按行 difficulty 改难度；原点映射正确；实际障碍与足端/扫描检查通过 |
| 数量与并行度 | N=1、N 大于 env 数均工作；评价试验数不整除槽数不漏计；增加副本不提高 task 概率；600 只是预设的派生结果 |
| 恢复 | 同 N 不同语义拒绝 resume；拆箱/合箱不自动复用历史；运行槽数变化保留试验表；计数无重复评价；半截回合不拼接 |
| HIM 兼容 | 旧 task 配置不变，actor/critic/estimator、terminal target、历史和 clipping 契约保持一致 |
| GPU 短跑 | 8–32 环境运行多次 reset、至少一次 LP 更新及保存恢复；HIM terminal target、数值和 ONNX 检查通过 |

工程验收以正确性和可运行为准，不以“LP 必须胜出”为验收线。实验结论报告多种子差异、全部任务宏平均与分任务结果，不只展示被课程偏好的容易任务。

## 7. 审查发现、修正与实际验证记录

本方案经过论文、架构、地形三路并行分析及交叉审查。重点修正如下：

1. 将隐含的上游 velocity task 依赖纳入复制范围，不能只检查本项目其他 task。
2. 区分静态地形排列和旧学习课程，防止关掉生成器 curriculum 后丢失类型/等级映射。
3. 固定整回合任务和命令，避免换箱之后把累计回报归错任务。
4. 采用 reset 前原始回报，避免使用已经平均、归一化的日志。
5. 边界截断独立分类，不将正常跨边界解释为论文中的摔倒。
6. 初评移到 load 后；独立更新计数避免 off-by-one；评价与 RNG 完整隔离。
7. 修正上一版“manual-reset 环境结束后不 reset 直至批次末”的不可执行流程，按当前 mjlab 的 pending-reset 契约处理 inactive 槽。
8. 明确本地 wrapper 的实际接线路径，以及共享 HIMPPO 的 timeout 近似。
9. 将 root 边界默认值从 0.5 m 修订为 1.1 m，并增加实际扫描点和足部范围验收；0.5 m 不足以保护当前高度扫描范围。
10. 任务数改为有效表长度，支持非等宽箱、可选轴和不同等级数；用语义键查表替代固定四轴 ID 公式。
11. 区分任务、几何副本和并行环境；几何采用固定 family-level 列与副本行，避免改变行数意外改变难度。
12. 新增本地配置编译入口及纯数据预检，防止 registry 早构建与 CLI 晚覆盖产生不一致。
13. 明确 Uniform 的有效任务等概率语义，以及配置改变后的历史作废、warm-start 和评价预算。
14. 补充官方项目页与第三方仓库核查结果，区分论文机制、第三方改编和本项目工程默认。

实际已完成的只读验证与后续测试必须分开看：

- 已检查本地论文、笔记、源代码和安装源码；目标目录 12 个 Python 文件均为空。
- 前版已用内联 Python 验证矩形 600 任务的计数/编码示例，以及 LP 示例 `[0.2, 3.0, -1.0]` 的 softmax 约为 `[0.05635, 0.92668, 0.01697]`；该固定编码不再作为本版通用实现规格。
- 已扫描并确认基线中需要本地化的 `mjlab.tasks.velocity` 导入。
- 前版文档落盘后，7 项临时 CPU 检查全部通过：相对链接存在、12 个空占位文件保持不变、当前源码 reset 顺序、manual-reset 防护、rough API 字段、600 ID 双向映射、文内概率/几何/预算计算。测试文件创建于 `/tmp/ri4438_lpacrl_doc_tests_*`，通过后整个临时测试目录已删除。
- 可配置方案已用内联 Python 验证 42/126 个任务、筛选后 40 个任务、论文预设 600 个任务，以及 14 个几何副本实例与任务数量解耦；预算示例的 38 批、486.4 万槽步计算通过。此检查没有创建测试文件。
- 本版整合后再次通过内联检查：YAML 示例可解析，任务/几何计数一致，95%/19% 槽步预算正确，本地链接存在，Markdown 结构与编号有效，旧的固定数量接线表述已替换。此检查同样未创建测试文件。
- 当前环境未检测到可用 CUDA，未运行 GPU 环境实例、PPO rollout、ONNX 或训练实验。
- 上述是文档与设计检查，不是尚未实现的新 task 的单元测试通过证明。临时检查只使用本次创建的临时路径，完成后删除测试文件。

## 8. 参考文件

- [作者项目主页](https://sites.google.com/view/lp-acrl)
- [arXiv 论文页面](https://arxiv.org/abs/2601.17428)
- [作者主页链接的期刊论文](https://ieeexplore.ieee.org/document/11563605)
- [第三方 Go2 课程实现](https://github.com/FIBO-Physical-AI/curriculum_training)
- [LP-ACRL 阅读笔记](LP-ACRL阅读笔记.md)
- [Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning](<../paper/Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning.pdf>)
- [现有 RI-4438 HIM 环境](../src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py)
- [现有机器人环境配置](../src/tasks/locomotion/ri_4438_him/config/env_cfgs.py)
- [现有 HIM runner](../src/tasks/locomotion/ri_4438_him/rl/runner.py)
- [HIM terminal observations](../src/tasks/locomotion/ri_4438_him/mdp/terminal_observations.py)
- [共享 HIMPPO](../rsl_rl/algorithms/him_ppo.py)
- [共享 HIM 训练循环](../rsl_rl/runners/him_on_policy_runner.py)
- [训练入口](../scripts/train.py)

框架证据来自当前安装位置 `.venv/lib/python3.11/site-packages/mjlab/` 下的 `envs/manager_based_rl_env.py`、`managers/reward_manager.py`、`managers/command_manager.py`、`managers/curriculum_manager.py`、`terrains/terrain_generator.py`、`terrains/heightfield_terrains.py` 和 `rl/vecenv_wrapper.py`。安装包实现可能包含本机改动；仅声明包版本不足以代替重新核对这些生命周期契约。
