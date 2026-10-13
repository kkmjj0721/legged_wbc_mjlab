# legged_wbc_mjlab

基于 [MjLab](https://github.com/mujocolab/mjlab)、MuJoCo 和 RSL-RL 的足式机器人强化学习项目，主要用于 RI-4438 与 Unitree Go2 四足机器人的速度跟踪、复杂地形运动以及策略的 sim-to-sim 验证。

项目目前包含标准 PPO 和 HIM（History Information Model）两类策略，并提供基于 HIM + PPO 的独立 LP-ACRL 任务，用于比较 Uniform、LP 和 ALP 课程采样。支持平地/复杂地形训练、动力学随机化、检查点评价和 ONNX 导出；原 PPO/HIM 任务支持多 GPU 训练，LP-ACRL 当前使用单 GPU。当前较完整的部署链路是 RI-4438 HIM 的 MuJoCo sim2sim。

> 本仓库面向研究与仿真验证。当前部署程序不包含 RI-4438 实机所需的电机、CAN 或 DDS 驱动，请勿直接连接真实机器人。

## 主要功能

- 基于 MuJoCo 的并行强化学习环境
- RI-4438、Unitree Go2 机器人模型与任务配置
- PPO 与带历史编码器/状态估计器的 HIM 策略
- 平地、楼梯、离散障碍等地形及地形课程
- RI-4438 HIM LP-ACRL 的地形与速度联合任务采样、固定试验表评价和课程状态恢复
- 质量、质心、摩擦、执行器增益、控制延迟等随机化
- 单 GPU、多 GPU 和 CPU 训练入口
- TensorBoard 训练记录、模型检查点与自动 ONNX 导出
- RI-4438 HIM 的 MuJoCo sim2sim、起立/趴下状态机和键盘/手柄控制

## 已注册任务

| 任务 ID | 机器人 | 算法 | 地形 |
| --- | --- | --- | --- |
| `Ri-4438-HIM-Rough` | RI-4438 | HIM + PPO | 复杂地形 |
| `Ri-4438-HIM-Flat` | RI-4438 | HIM + PPO | 平地 |
| `Ri-4438-HIM-LPACRL-Rough` | RI-4438 | HIM + PPO + LP-ACRL | 起伏、上下楼梯、上下坡；默认 1560 个任务 |
| `Ri-4438-HIM-LPACRL-Flat` | RI-4438 | HIM + PPO + LP-ACRL | 平地；默认 30 个任务 |
| `Ri-4438-Rough` | RI-4438 | PPO | 复杂地形 |
| `Ri-4438-Flat` | RI-4438 | PPO | 平地 |
| `Unitree-Go2-HIM-Rough` | Unitree Go2 | HIM + PPO | 复杂地形 |
| `Unitree-Go2-HIM-Flat` | Unitree Go2 | HIM + PPO | 平地 |
| `Unitree-Go2-Rough` | Unitree Go2 | PPO | 复杂地形 |
| `Unitree-Go2-Flat` | Unitree Go2 | PPO | 平地 |

LP-ACRL 注册任务分别使用 `ri_joint_smoke` 和 `ri_flat_smoke` 预设；自定义任务空间使用下文的任务内训练入口。

## 环境要求

- Ubuntu 22.04（推荐）
- Python 3.11（`pyproject.toml` 要求 `>=3.11,<3.12`，安装脚本默认使用 `3.11`）
- NVIDIA GPU（推荐用于训练）
- NVIDIA 驱动 550 或更高版本（推荐）

核心 Python 依赖由 `pyproject.toml` 管理，其中包括 `mjlab==1.6.0` 和仓库内置的 `rsl-rl-lib==5.4.2`。安装脚本会检测本机驱动并选择合适的 PyTorch CUDA 后端；没有可用 NVIDIA 驱动时会安装 CPU 版本。

## 快速开始

### 1. 获取代码

```bash
git clone git@github.com:kkmjj0721/legged_wbc_mjlab.git
cd legged_wbc_mjlab
```

### 2. 安装依赖

在仓库根目录执行：

```bash
bash setup.sh
```

脚本会自动安装或复用 `uv`、准备项目要求的 Python、测试可用的软件源、创建 `.venv` 并安装依赖。需要运行 sim2sim 时，安装额外依赖：

```bash
bash setup.sh --extra sim2sim
```

代理、离线安装、自定义镜像和 CUDA 后端等选项见 [安装配置文档](docs/setup.md)。后续命令均建议从仓库根目录通过 `uv run` 执行，无需手动激活虚拟环境。

### 3. 验证安装

```bash
uv run python -c "import mjlab, torch, rsl_rl; print(torch.__version__)"
uv run python scripts/train.py Ri-4438-HIM-Rough --help
```

第二条命令会显示环境、奖励、课程和训练器支持的全部命令行覆盖项。

## 训练

下面以 RI-4438 HIM 复杂地形任务为例：

```bash
uv run python scripts/train.py Ri-4438-HIM-Rough \
  --env.scene.num-envs 4096 \
  --agent.run-name baseline
```

默认使用 GPU 0。显存不足时可减小并行环境数，例如：

```bash
uv run python scripts/train.py Ri-4438-HIM-Rough \
  --env.scene.num-envs 1024
```

使用多张 GPU：

```bash
uv run python scripts/train.py Ri-4438-HIM-Rough \
  --env.scene.num-envs 8192 \
  --gpu-ids 0 1
```

常用训练参数：

```bash
# 修改训练迭代数和保存间隔
uv run python scripts/train.py Ri-4438-HIM-Rough \
  --agent.max-iterations 20000 \
  --agent.save-interval 100

# 从已有运行目录恢复训练
uv run python scripts/train.py Ri-4438-HIM-Rough \
  --agent.resume True \
  --agent.load-run 2026-09-15_15-52-14 \
  --agent.load-checkpoint model_10000.pt
```

命令行参数会覆盖 Python 配置，因此一般不需要为一次实验直接修改源码。复杂地形与平地 HIM 任务保持相同的 critic 观测结构，可以在两者之间加载兼容的检查点。

RI-4438 HIM 训练默认自动复位，并在复位前保存经过运动学刷新后的终止目标，避免重复执行整批碰撞计算和地形射线检测。HIM 的下一步监督只保存 50 维 `estimator` 观测，actor/critic 网络输入保持不变。原训练命令即可使用；用 `--env.auto-reset False` 可切回手动复位路径作对比。

RI-4438 HIM Rough 当前使用平地、正/反金字塔楼梯和随机箱体障碍，基于 mjlab 的 `ROUGH_TERRAINS_CFG` 独立复制后移除坡地、波浪和随机粗糙高度场。楼梯每级高度为 0.05–0.16 m、中央平台约 2.5 m；箱体网格宽 0.8 m、高度为 0.02–0.10 m；平地权重折算为总权重的 15%。地形晋级要求距地形原点超过地形宽度的一半，且目标行进距离至少为 1 m；降级依据实际位移与目标行进距离。默认启用航向命令，纯前进采样关闭航向控制。求解器使用 `cg`，`njmax=300`。具体几何与课程参数以当前任务配置为准。

### 训练输出

通过通用 `scripts/train.py` 训练时，输出默认保存在：

```text
logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/
├── model_<iteration>.pt
├── policy.onnx
├── params/
│   ├── agent.yaml
│   └── env.yaml
└── videos/                 # 启用录制后生成
```

RI-4438 HIM runner 每次保存检查点时都会更新同目录下的 `policy.onnx`。查看 TensorBoard：

```bash
uv run tensorboard --logdir logs/rsl_rl
```

### 选择 best 模型与检查收敛

本节的仿真比较适用于原 RI-4438 HIM/PPO 任务。LP-ACRL 的任务协议和课程评价使用下文的独立评价入口。

完整操作步骤见 [best_model 工具使用](docs/tool使用.md)：包含当前日志的可复制命令、续训用法、并行与种子设置、报告位置和模型回放。

仿真 best 表示本轮第一名，尚不能认定为唯一最优。相同输出目录再次评测时，HTML 会核对上次的模型、测试条件和逐回合成绩，报告排名是否变化；固定种子不能保证 GPU 物理仿真逐位一致。复测方法见上述文档的“常见问题”。

`--run` 支持**实验总目录或具体 run**。传 `logs/rsl_rl/ri_4438_him` 时自动选中最新 run，再追溯原始训练和历次续训，读取各目录的 `params/*.yaml`、TensorBoard
事件和数字 checkpoint。支持当前 HIM/PPO 格式；默认生成完整训练曲线，不启动仿真。
加 `--evaluate` 后统一测试整条训练链的全部有效模型。同名模型按来源区分，页面只显示前 10 名。
历史来源不明确时用 `--runs 原始run 续训run ...` 显式指定；`--single-run` 可仅分析传入的 `--run`。

```bash
# 安装画图与导出验证所需的附加依赖
uv sync --extra analysis

# 生成分析报告，不复制或导出模型
uv run python scripts/best_model.py --run '/path/to/logs/rsl_rl/ri_4438_him/<run>'

# 最新 run 自动追溯历史，统一仿真比较全部模型
uv run --extra analysis python scripts/best_model.py --run '/path/to/<latest_run>' --evaluate --eval-parallel 8

# 直接传实验总目录；以后新增续训 run，无需修改路径
uv run --extra analysis python scripts/best_model.py \
  --run '/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him' --evaluate --eval-parallel 8

# 旧日志仅保存了正则来源时，按原始训练到最新续训的顺序指定
uv run --extra analysis python scripts/best_model.py --runs '/path/to/<original_run>' '/path/to/<resumed_run>' --evaluate

# 复制当前阶段日志评分候选，并从该 checkpoint 重新导出 ONNX（CPU）
uv run python scripts/best_model.py --run '/path/to/<run>' --write-best --export-onnx

# 调整统计窗口，并添加明确的任务目标；不指定目标时最多判为平台期
uv run python scripts/best_model.py --run '/path/to/<run>' \
  --window 100 --min-samples 50 --top-k 3 \
  --goal 'survival_ratio>=0.95' \
  --goal 'track_linear_velocity_index>=0.7' \
  --goal 'track_angular_velocity_index>=0.7'
```

上述目标只是语法示例，需要按任务要求设置。`survival_ratio` 是平均 episode 步数与上限
之比，不是成功率。`track_*_index` 是对应 `Episode_Reward/*` 除以正奖励权重得到的
固定时长积分指标，仍受提前终止影响，不是速度跟踪准确率；奖励未按 dt 缩放或跟踪权重
随课程变化时不生成这些指数。也可通过 `--goal 'TensorBoard标签>=阈值'` 或 `<=` 指定原始指标。

默认评分为窗口内 `Train/mean_reward` 的 **中位数 − MAD**；MAD 是相对中位数的绝对偏差
中位数，惩罚系数由 `--penalty` 控制。这是候选排名启发式，不是统计置信区间。
只比较同一日志分段、同一课程阶段的已保存模型，排除阶段切换后的前 20 轮，并要求至少
50 个有效样本。评分相同优先较新的 checkpoint；未达任务目标的候选仍可成为该阶段 best，
但会单独记录达标结果。NaN/Inf、损坏的 checkpoint、文件编号与内部 iteration 不一致等会被排除。

课程进度优先从 checkpoint 的 `infos.env_state.common_step_counter` 还原；没有该字段时，
只对确认从零开始的训练推断。未知课程不会静默跨阶段排名。续训日志 iteration 回退时分段，
最新 run 的主分析默认选择最后一段，可用 `--segment N` 选择；完整报告另保留各 run 所有分段。
同一分段重复标签/iteration 取时间较晚的值。
多分段的 checkpoint 归属依赖文件 mtime，复制后不能确定归属的模型会被排除。必要时可明确
传入 `--env-step-offset N`，其定义为保存 iteration k 时的环境步数
`N + (k + 1) * num_steps_per_env`；仅在已核对续训计数时使用。

收敛检查独立使用日志末尾的数据，默认要求最终课程适应期之后至少 3 个连续的 100 轮窗口，
每窗覆盖率至少 80%。检查总回报、episode 长度、已有的跟踪奖励和任务目标的原始指标；
地形课程仍变化、关键数据缺失或窗口指标仍变化时不会宣称收敛。

| 状态 | 含义 |
| --- | --- |
| `curriculum_in_progress` | 指令/奖励课程未完成，或地形难度仍在变化 |
| `improving` / `degrading` | 回报持续改善 / 退化 |
| `unstable` | 指标仍在变化、波动偏大或有健康告警 |
| `plateau` | 曲线稳定，但任务目标未指定、未达标或观察不完整 |
| `converged_candidate` | 曲线稳定、跟踪指标可用且显式目标达标，属于疑似收敛 |
| `insufficient_data` / `unknown_curriculum` | 样本不足 / 无法确定课程阶段 |

结果默认写到选中 run 的 `best/`（传总目录也一样），可用 `--output` 指定其他目录。总目录只扫描直接子目录中的 run：按目录名的训练开始时间选择最新一个，自定义名称用 `params/agent.yaml` 修改时间作为备用依据。终端会打印选中的目录，只追溯它的续训链，不自动混合其他独立训练。

```text
best/
├── report.html            # 唯一 HTML 入口：完整训练曲线、来源、各段状态
├── history.png            # 完整曲线；原始 iteration，分别平滑并标记恢复点
├── training_chain.json    # 所有 run 的配置及各段分析
├── chain_candidates.csv   # 所有 run 的日志候选
├── selection.json         # 本轮推荐、日志候选、规则、配置哈希、阶段、收敛依据
├── candidates.csv         # 窗口指标及排除原因
├── report.png             # 曲线、阶段边界、当前阶段 Top K（--no-plot 跳过）
├── model_best.pt           # 日志评分候选；--write-best 或 --export-onnx 时生成
├── model_manifest.json    # 已发布模型来源、SHA256、配套 ONNX 信息
└── policy.onnx            # --export-onnx 时重新导出
```

两种模式都只生成一个 `report.html`，完整训练曲线在页面内；成功生成后清理同目录中工具旧版生成的 `training_history.html`。只分析日志时明确显示未进行仿真评测。

`selection.json` 的 `recommendation` 记录本轮推荐及依据：加 `--evaluate` 时为仿真推荐，与终端、HTML 和 `model_best_eval.pt` 一致；仅分析日志时依据为 `training_log`。
完整评测的 `intended_use` 为 `hardware_trial_priority`，`hardware_recommendation` 记录当前模型间的相对推荐，`hardware_validated=false` 表示尚未完成实机验证。允许楼梯等能力存在短板，不设全部通过门槛。
`top_by_stage` 和 `best_current_stage` 是日志评分候选，作为训练曲线参考，可能与仿真推荐不同。当前阶段样本不足时 `best_current_stage`
为 null，不自动用早期简单任务的模型替代。`published` 为 null 表示这次未发布日志候选，已有日志候选发布
文件以 `model_manifest.json` 为准。重新发布日志候选但不导出 ONNX 时会移除输出目录中旧的
`policy.onnx`，避免配错模型；run 根目录的训练 checkpoint 和 ONNX 不受影响。

终端和 HTML 的“日志候选筛选依据”会说明有效模型数量、当前阶段合格数量、未入选原因、
候选与收敛所需的最早 iteration，并在当前阶段无候选时展示最近有合格候选的历史阶段模型。
这些依据也保存在 `selection.json` 的 `selection_diagnostics` 中。历史阶段模型仅作参考，
不替代 `best_current_stage`。评分窗口止于 checkpoint 的 iteration，后续新增日志不会回填旧模型的评分。
例如最终阶段从 2000 开始，默认适应期后从 2020 计分；连续有效日志下，checkpoint 最早在
2069 满足 50 个样本要求，而收敛观察长度最早在 2319 满足。若每 100 轮保存一次模型，
首个可能合格的模型为 `model_2100.pt`。`--evaluate` 可以直接比较全部有效模型，无需等待日志候选或收敛。

导出复用 runner 使用的 `actor.as_onnx()`，严格加载历史网络配置与参数。HIM 导出包含
观测归一化、估计器及动作裁剪，并要求 checkpoint 有匹配的数值契约。当前只支持现代
`HIMActorModel` / `MLPModel` 和 GaussianDistribution，旧格式或其他网络会明确报错。
导出文件包含模型来源与 HIM 契约；关节、PD、动作缩放等部署配置仍需使用对应任务的配置。

训练窗口包含更新前采样及跨更新的 episode，评分只能近似反映 checkpoint 表现。
日志评分不保证统计收敛或实机表现；跨 run 的训练分数不用于直接排名。
各段收敛使用各自配置，当前不跨续训边界拼接统计窗口。仿真比较使用统一场景和兼容的控制接口。
新训练会保存 `params/lineage.json`，记录实际加载的 checkpoint 及哈希，以便后续自动追溯。

#### 实际仿真对比与中文报告

在项目根目录执行（`--run` 可传实验总目录，或任意一次训练、续训的具体目录）：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run '/path/to/logs/rsl_rl/ri_4438_him/<run>' --evaluate

# 指定要比较的模型编号；降低并行环境数可以节省显存
uv run --extra analysis python scripts/best_model.py \
  --run '/path/to/<run>' --evaluate --eval-checkpoints 500 700 900 \
  --eval-num-envs 12 --eval-seeds 0 1 2 --eval-duration 12

# 默认已评测整条续训链的全部有效 checkpoint；也可显式指定 8 个模型并行
uv run --extra analysis python scripts/best_model.py \
  --run '/path/to/<run>' --evaluate --eval-parallel 8

# 全部模型、3 组种子；上下楼梯均包含固定的 5、10、15 cm 台阶
uv run --extra analysis python scripts/best_model.py \
  --run '/path/to/<run>' --evaluate --eval-seeds 0 1 2 \
  --eval-terrains flat rough stairs_up stairs_down
```

`--evaluate` 默认评测 **选中 run 及其续训祖先中全部有效的数值 checkpoint**，包含较早日志分段的模型。
所有模型完成同一套仿真后再排名，**HTML、Markdown、对比图、场景筛选和单模型诊断只显示前 10 名**。
全部模型排名及逐场景/逐回合结果保留在 CSV、JSON 中；报告中的 best 从全部已测模型中产生。
`--eval-count` 默认 `0`，表示全部；仅在主动进行小规模诊断时，才用正整数限制模型数量，
或用 `--eval-checkpoints` 显式指定编号。HTML 会显示已测/可用数量，数据中记录无效 checkpoint 的原因。
续训 run 的最早 checkpoint 通常就是续训起点，完整数据可用于比较续训前后表现。
即使当前日志窗口不足以选出 best，也能评测有效 checkpoint；完整曲线包含已解析的续训历史，收敛仍按各日志分段独立判断。
每次运行固定本轮 checkpoint 快照，不会持续后台监控；训练新增模型后再次运行即可更新分析。

默认配置为 **每模型 24 个环境、3 个种子、每回合 12 秒，8 个地形场景**：平地、±2 cm 起伏，
以及 **5、10、15 cm 每个高度分别上楼梯、下楼梯**。每模型共 **576 个首次回合**。
种子默认 **0、1、2**；`--eval-seeds` 必须提供恰好 3 个互不相同的非负整数。
平地/起伏各有 6 组指令：站立、前进 0.5/1.0 m/s、后退、侧移、转向；
每个楼梯场景有 3 组向前指令：0.3/0.5/0.8 m/s。每个地形场景样本数相同，默认各占总分的 1/8，
场景内部各指令等权。新增高度改变了评测范围，总分不能直接与旧版单一高度报告比较。
楼梯高度固定在工具中，不提供高度参数；`stairs_up` / `stairs_down` 每个方向均展开全部三个高度。
每段 6 级，累计高差分别为 **30、60、90 cm**，踏面均为 30 cm；出生点距第一台阶 1 m。
楼梯通过要求越过 3.15 m 通过线、到达对应高度（容差 12 cm），且存活至回合结束；
站在原地、未走完、偏出 ±1.2 m 测试通道、越过后跌倒均不算任务通过。

`--eval-parallel 0` 默认根据可用显存估计同时运行 **1～8 个模型**，预留 1.5 GiB；可显式设为 `1`～`16`。
各模型保留各自的权重和归一化参数，通过 `torch.vmap` 并行推理，各自的独立物理环境在同一个 GPU 批次步进。
例如 8 个模型 × 每模型 24 个环境 = **192 个并行环境**，其余模型分批处理；尾批空位不参与计分。
显存估计不是绝对保证，繁忙 GPU 上可用 `--eval-parallel 1 --eval-num-envs 12` 降低占用。
`--eval-num-envs` 必须是 6 的倍数；`--eval-terrains flat` 可仅测平地；
`--eval-device cuda:0` 选择设备；`--eval-task` 可指定 `Ri-4438-HIM-Rough/Flat` 或 `Ri-4438-Rough/Flat`。
任务默认由保存的 experiment_name 推断，目前仿真评测仅支持 RI-4438 HIM/PPO。

评测使用当前工作区的统一机器人资产和终止规则，检查历史观测顺序、缩放、动作接口，
加载保存的 actor、归一化参数及控制周期。课程、推扰、参数随机化、观测噪声、观测与执行器延迟关闭，
同一种子/测试编号的初始姿态与关节状态不依赖模型顺序或并行数量，并须通过哈希一致性检查。
发生跌倒时先记录终止状态，重置后的片段不再计入该次评测。该协议测量名义条件下的运动和楼梯能力，尚未覆盖抗推扰能力。

推荐按基础跟踪、稳定性、动作平滑和楼梯能力的相对分数排序（45/25/10/20），不设全场景通过门槛。
每回合计算指标后平均，不使用训练 reward 混合排名；完整跑完不等于跟踪达标。
相近成绩只提供候选顺序，当前没有显著性检验。样本偏少时报告会提示复核。
稳定性分析包括横滚/俯仰时间标准差、倾角 P95/峰值、机身角速度与垂直速度 RMS、动作一阶/二阶变化、
力矩、机械功率代理、失败原因、最差场景及不同种子的误差范围。`--eval-tilt-limit 20` 设置倾角占比诊断阈值，
不改变仿真终止规则。种子范围不称为置信区间；机械功率代理不等于电池功耗。

报告末尾附上 **排名前 3 名在全部 8 个地形场景中的真实仿真截图**，每个场景一张对比拼图。
每个模型统一记录首个种子、前进 0.5 m/s、首个对应环境，在 2 秒、6 秒和回合结束时的状态；
若提前失败，则保留首次终止状态，之后不再取图。截图使用本次计分回合记录的状态和原始编译场景，
由 MuJoCo 离屏渲染，不重新运行策略，不选择成功回合代替失败回合。图片仅展示单个回合，统计成绩仍以 3 组种子为准。

`best/` 新增以下文件，直接用浏览器打开 `report.html` 即可：

```text
best/
├── report.html               # 中文报告、交互场景筛选、单模型诊断、指标解释
├── report.md                 # 可随图片一起分享的 Markdown 报告
├── comparison.png            # 成功比例、速度误差、姿态、动作变化、存活时间
├── scenarios.png             # 每种地形/指令的对比热图
├── stability.png             # 姿态波动、倾角 P95/峰值、动作二阶变化
├── repeatability.png         # 种子误差范围、跟踪与动作平滑性的取舍
├── stairs.png                # 5/10/15 cm 楼梯剖面与各模型上下楼通过率
├── stairs_progress.png       # 三个高度 × 上下楼梯的真实行进进度
├── tracking.png              # 目标速度与实际速度的时间曲线
├── evaluation.json           # 协议、模型/配置哈希、指标、采样轨迹
├── evaluation.csv            # 每个模型、种子、场景的逐回合结果
├── evaluation_cases.csv      # 各模型/地形/指令的汇总指标
├── evaluation_ranking.csv    # 全部模型的完整排名及汇总指标
├── hardware_readiness.json   # 全部模型的推荐分项、种子分数和权重对照
├── simulation/              # 真实仿真图片、拼图、采样状态、场景 MJB 和图片索引
├── eval_env_flat.json        # 实际采用的统一平地环境描述
├── eval_env_rough.json       # 实际采用的统一起伏环境描述
├── eval_env_stairs_up_5cm.json     # 5 cm 上楼梯；另有 10cm、15cm
├── eval_env_stairs_down_5cm.json   # 5 cm 下楼梯；另有 10cm、15cm
├── model_best_eval.pt        # 本次仿真推荐模型
└── eval_model_manifest.json  # 仿真推荐模型的来源与 SHA256
```

仿真执行期间会写 `evaluation.partial.json`，成功完成后移除；失败时不会发布新的仿真推荐模型。
HTML / Markdown 包含分高度的上下楼梯通过率表，场景筛选、稳定性指标和 CSV 同样区分三个高度。
终端最后的“本轮推荐（基于仿真）”、HTML 页首推荐和 `model_best_eval.pt` 指向同一模型。报告将训练回报选出的模型单独标为“日志评分候选（训练曲线参考）”。
完整评测按相对推荐分选择优先实机验证的模型：基础跟踪 45%、稳定性 25%、动作平滑 10%、楼梯能力 20%，不要求全场景通过。分项先用固定参考尺度归一化，分数与候选数量无关；权重为工程取舍，不是实机成功概率。终端、模型文件、JSON、HTML 排名和图片统一采用该排序。公式、种子表现、权重对照和能力边界见报告及 [实机使用前的判断](docs/tool使用.md#准备上实机时怎么判断)。实机表现未验证单独记录，不再导致推荐恒为空。
`--write-best` / `--export-onnx` 仍针对日志评分候选，分别生成 `model_best.pt` / `policy.onnx`；日常使用仿真推荐请选 `model_best_eval.pt`。
`--no-plot` 跳过统计图，但仍保留真实仿真截图、HTML、Markdown 和数值报告。

若要交互查看推荐模型的动作，可使用现有回放入口（这是 play 环境，条件与上面的固定评测不同）：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --checkpoint-file '/path/to/<run>/best/model_best_eval.pt' --num-envs 1
```

**平台期**指最近几个训练窗口的主要指标变化很小。它可能意味着学得稳定，也可能意味着卡住，
因此报告把训练趋势和实际仿真表现分开展示。课程还在升级、刚续训样本不足时，不会宣布最终收敛。

查看分析工具的完整参数：

```bash
uv run --extra analysis python scripts/best_model.py --help
```

## HIM LP-ACRL

LP-ACRL 在每回合开始时联合采样地形和命令任务。默认 `commands.heading_command = True`：采样相对出生朝向的航向偏差，得到固定世界坐标目标 `heading_target = wrap(spawn_yaw + heading_offset)`；每步生成 `wz = clip(0.5 * wrap(heading_target-current_heading), -1, 1)`，单位 rad/s。任务、vx/vy 和目标航向在回合内保持固定，实际 wz 随反馈更新。`commands.resampling_interval_s = None` 关闭回合内目标重采样，训练拒绝正数间隔；历史变命令 checkpoint 仍可离线评估。`commands.sample_bin_centers = True`：vx / heading 使用分箱中心并采样正负方向，连续 vy 保留范围内采样。评估出界重置时保留同一目标航向。HIM actor 仍使用三维速度命令：47 维单帧观测、6 帧历史；目标角度不增加策略输入，地形高度只进入 critic。

默认参数集中在 [ri_4438_him_lpacrl_config.py](src/config/ri_4438/ri_4438_him_lpacrl_config.py)。Rough 仍为 1560 个任务：5 个纵向速度绝对值箱 × 6 个初始航向误差绝对值箱 ×（8 类楼梯 × 5 档高度 + 3 类其他地形 × 4 档）。航向误差分箱覆盖 ±π，档宽 π/6；反馈系数由 `heading_control_stiffness` 控制，输出由 `ang_vel_z_range` 限幅。任务标签使用 `heading=...`，不把动态 wz 当作固定任务标签。设 `heading_command=False` 可使用原来的六档直接角速度命令。纵向速度在 ±1 m/s，横向速度在 ±0.5 m/s 内连续采样。上下坡仍使用 `gradient = 高差 / 水平距离`：`0、0.4/3、0.8/3、0.4`，对应约 `0°、7.59°、14.93°、21.80°`，高度量化为 1 mm。

楼梯的四种踏步深度为 `0.18/0.21/0.25/0.28 m`，每种深度各有上、下两类，例如 `stairs_018_outward_up` 和 `stairs_018_outward_down`。每类高度为 `0.05/0.08/0.10/0.12/0.16 m` 五档。中心平台实际宽度保持 `2.0 m`：四种深度的外围平地边宽分别约为 `1.20/1.32/1.25/1.32 m`，配合 `8 m` 地块生成 10/8/7/6 层踏步；配置中的 1 nm 余量用于避免浮点取整少生成一层。

课程统计和 teacher 算法对齐 [FIBO teacher_guided.py](https://github.com/FIBO-Physical-AI/curriculum_training/blob/5813fc351d982880e61c5e775be048ee4ee0db33/src/source/curriculum_rl/curriculum_rl/curricula/teacher_guided.py)、[mdp.py](https://github.com/FIBO-Physical-AI/curriculum_training/blob/5813fc351d982880e61c5e775be048ee4ee0db33/src/source/curriculum_rl/curriculum_rl/envs/mdp.py) 和实际生效的课程参数，固定参考提交 `5813fc351d982880e61c5e775be048ee4ee0db33`。多地形任务与 heading 命令是 RI-4438 的扩展；这不是论文总回报 LP 的严格复现。

- 自然结束回合只取 `track_linear_velocity`，计算 `clip(episode_sum / actual_steps / reward_weight, 0, 1)`。按参考代码保留奖励管理器已乘入的 **dt**，不混入 yaw，不再次除 dt。PPO 仍使用原来的全部奖励。
- 每次 PPO 更新统计本轮结束回合的各任务均值，**无样本任务为零**，然后对阶段内的迭代均值等权平均。取消最小回合门槛、按阶段回合总数加权和跨阶段保留稀疏样本。
- `stage_length_updates=50`。保留参考代码首调用启动时钟的语义：第 **51** 次更新建立首次基线，第 **101** 次首次更新概率，以后每 50 次更新一次。基线阶段不改变初始采样。
- 有符号 `LP = 本阶段均值 - 上阶段均值`，`p = 0.85 * softmax(LP / 0.05) + 0.15 / N`。这里 `0.15` 来自实际命令配置的 `eps`，不是 teacher 构造函数单独的默认值 `0.05`。保留 dt 后分数通常远小于 1，因此这个温度下采样可能较平；不擅自调整参考温度来增强选择性。
- 初始 `seed_bin=0`：前两个 |vx| 箱概率质量为 **2/3、1/3**；在每个 vx 箱内对保留的其他任务组合均匀分配。原项目只做 8 个前进速度箱；这里保留 1560 个多地形三轴任务，因此这一先验分配是明确的维度扩展。`uniform` 对照始终均匀，需设置 `uniform_mix=0`。
- reset 前归属任务，手动 reset 不计入统计。HIM、timeout bootstrap、观测历史和数值检查保留。

直接执行上游源码的对齐检查（不需要 Isaac Lab）：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python tools/test_lpacrl_fibo_parity.py \
  --reference-root /path/to/curriculum_training
```

检查要求参考仓库处于上述提交且相关源码未修改；覆盖实际回合统计入口、空箱、稀疏样本、阶段时钟、2:1 初始化和阶段中恢复。仅替换框架对象与 CSV 输出。

### 训练与恢复

```bash
uv run python -m scripts.train Ri-4438-HIM-LPACRL-Rough --env.scene.num-envs 2048

# 旧版评估式课程只加载兼容权重，重新开始在线课程
uv run python -m scripts.train Ri-4438-HIM-LPACRL-Rough \
  --env.scene.num-envs 2048 --warm-start '/path/to/old_run/model_1000.pt'

# 在线课程完整续训；max-iterations 是本次额外更新次数
uv run python -m scripts.train Ri-4438-HIM-LPACRL-Rough \
  --env.scene.num-envs 2048 --agent.resume True \
  --agent.load-run '<run_directory_name>' --agent.load-checkpoint 'model_200.pt' \
  --agent.max-iterations 1000

PYTHONDONTWRITEBYTECODE=1 uv run python -m unittest tools.test_ri4438_lpacrl -v
```

中央默认仍为 1024 个环境，每次 PPO 更新 100 步，每 200 次保存；对照实验明确使用 2048 个环境。目前支持单进程、单 GPU。运行快照记录实际环境数。`--warm-start` 与 `--agent.resume True` 互斥；前者重置优化器、课程统计和迭代数，来源记录在 `params/warm_start.yaml`。

在线 checkpoint 版本 4 保存每迭代统计、阶段均值累计、阶段时钟、初始概率、当前概率与任务采样 RNG。完整续训拒绝旧评估式和版本 3 的 xy/yaw 课程状态；旧权重可播放、离线评估或显式 warm-start。续训重新开始物理回合，未结束的回合不计入统计，因此不承诺仿真轨迹逐位重现。任务、几何、奖励、HIM 和课程配置仍做兼容检查；独立评估参数不再影响训练恢复。

训练不创建评价环境、冻结策略或全量试验表，也不逐步执行几何诊断。`curriculum.jsonl` 和 TensorBoard `LPACRL/*` 记录任务样本数、分数、LP、概率、有效任务数；`Perf/learning_time` 只包含 PPO/HIM 更新，`Perf/curriculum_host_time` 记录统计 CPU 调用与阶段更新耗时（并非独立 GPU kernel 计时），`Perf/iteration_time` 包含采样、优化、课程阶段更新。

### 离线评价与视频

```bash
# 默认恢复 checkpoint 中的任务、地形、命令设置；无需加载课程历史
uv run python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model_1000.pt' --output-dir '/path/to/evaluation'

# 比较不同历史模型时，显式使用当前共同任务空间
# 原地转向可选 turn_left/turn_right，横移可选 left/right，直行可选 forward/backward
MUJOCO_GL=egl uv run python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model_1000.pt' --protocol current \
  --command forward --terrain stairs_025_outward_up --level l0 \
  --num-envs 1 --trials-per-task 1 --video --output-dir '/path/to/stairs_video'
```

默认全任务评价仍为 1560×8 个固定试验、512 个环境、20 秒预算。名义动力学、冻结策略；出界重置同一任务并保留命令时钟，物理失败结束试验。`evaluation.json` 和 `per_task.csv` 保存逐试验、逐任务、分地形与宏平均结果、实际配置和 checkpoint 哈希；视频按任务/试验保存。`--previous-result` 可为相同任务、几何和固定试验添加配对总回报变化的标准误及异常试验敏感性诊断。

heading 模式的试验表记录 `heading_target` / `heading_offset`，`command` 中的 wz 是初始反馈值；评价时每步跟踪实际反馈指令。`--command forward/backward/left/right` 在 heading 模式下保持出生航向，`turn_left/turn_right` 显式测试固定角速度。旧快照缺少 `heading_command` 时按直接 wz 协议恢复。新航向任务与旧角速度任务的 checkpoint 不可直接续训，可用上面的 `--warm-start` 加载旧策略重新建立课程状态。

- EPTE-SP 使用论文稳定性聚合 `(epsilon*k+K-k)/K`；本项目对变指令采用 `abs(error)/max(abs(command),floor)` 的存活段均值，最后截到 `[0,1]`。三个轴的 floor 默认均为 0.1。此近零指令约定属于项目设置。
- `success` / `tracking_success` 保留已有速度跟踪口径：存活比例严格大于 0.9，vx、wz 的 EPTE-SP 严格小于 0.3。vy 指标另列；此成功率不证明地形通过。
- `stair_steps_crossed` 需要身体离开平台并沿楼梯方向改变高度，至少两足在对应或更外侧踏面接触，连续支持至少 0.1 秒；考虑 RI-4438 足球半径，拒绝仅碰竖直立面或平台绕圈。它是可测试的几何近似，仍需结合视频。
- `stair_progress_success` 要求跨过至少两级且试验结束时未跌倒；`terrain_traversal_success` 要求跨过该楼梯全部踏步且未跌倒。后两项只用于楼梯，查看对应 `by_terrain`，不把混合全部地形的宏平均解释为通用通过率。
- 同时记录累计路径、排除重置瞬移后的净位移、偏航误差积分、动作变化 RMS、接触足滑动速度、跌倒、台阶接触和停滞时间。完整奖励分解及逐步几何扫描仅在此路径执行。

### 固定条件对照

`tools/lpacrl_experiment.py` 默认 2048 环境、100 步 rollout、seed 0、1000 次更新。`online` 为 FIBO teacher 课程；`uniform` 仅改变采样方法；两者均为整回合固定命令，已取消 2 秒重采样实验。旧实现须使用修改前源码快照，以免把新课程误当成基线。

```bash
uv run python tools/lpacrl_experiment.py --variant online --output-dir logs/experiments/online_seed0
uv run python tools/lpacrl_experiment.py --variant uniform --output-dir logs/experiments/uniform_seed0
uv run python tools/lpacrl_experiment.py --variant evaluated \
  --source-root logs/experiments/lpacrl_online_20261012/baseline_source \
  --output-dir logs/experiments/evaluated_seed0
```

在同一 GPU 上顺序执行。候选方案用 `--seed 1`、`--seed 2` 补跑；比较时保持评价任务、试验表和命令时序一致。已有第一版共同测试工具 `tools/test_lpacrl_v1_comparison.py` 仍可用于相同 12 个历史任务；它使用旧边界终止口径，不能与上述全时长协议混算。验收须同时检查速度、偏航、动作变化、足滑、跌倒和真实楼梯通过，不能只凭训练提速认定动作改善。

## 策略回放

加载本地检查点：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --checkpoint-file 'logs/rsl_rl/ri_4438_him/<run>/model_<iteration>.pt' \
  --num-envs 1 \
  --viewer native
```

不加载策略，仅检查环境、模型与可视化：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --agent zero \
  --num-envs 1 \
  --viewer native
```

无桌面的服务器可将 `--viewer` 设为 `viser`。录制已训练策略的视频时可添加 `--video True --video-length 500`，视频会写入对应运行目录下的 `videos/play/`。

## URDF 检查工具

`tools/check_urdf.py` 用于检查和生成候选 URDF。工具的固定基准是：STL 内容、每个 link 的几何、质量和装配位置均可信；默认不会修改输入文件，也不会根据外形猜测真实质心或惯量。没有独立 CAD 惯性数据时，质心和惯量问题会进入报告中的待确认项。

一次性运行建议使用 `urdf-tools` 可选依赖：

```bash
uv run --extra urdf-tools python tools/check_urdf.py MODEL.urdf \
  --profile ri4438 \
  --preview \
  --report report.json
```

也可以先把工具依赖同步进当前环境：

```bash
bash setup.sh --extra urdf-tools
```

如果首次运行时当前镜像对某个 wheel 返回 `403`，优先使用上面的 `setup.sh` 让安装助手重新测速选择镜像；临时命令也可以加 `--default-index https://pypi.org/simple` 验证入口，不需要手动改锁文件。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `MODEL.urdf` | 待检查的 URDF。默认只读检查，不写回输入文件。 |
| `--profile generic\|go2\|ri4438` | 选择规则集；默认 `generic`。`go2` 和 `ri4438` 会对四足腿部使用 hip `+X`、thigh/calf `+Y` 的关节轴约定，机械臂不会套用四足规则。 |
| `--mesh-root PATH` | 补充网格搜索根目录，用于解析相对路径、绝对路径和 `package://` 资源。可重复传入。 |
| `--rules FILE.json` | 按关节名覆盖轴、限位和 mimic；`axis` 在工具规范化后的 joint frame 中表达。非共线改轴必须同时给出新的限位；相关 mimic 需要显式覆盖。 |
| `--report report.json` | 写出 JSON 报告，包含 `issues`、`changes`、`joint_coordinate_map` 和 `validation`。问题项记录等级、对象、原值/候选值和待确认状态；变化项记录 `before`、`after`、`kind` 和 `object`。 |
| `--output NEW.urdf` | 导出候选 URDF。禁止覆盖输入文件；导出后仍引用原始 STL 资源。 |
| `--preview` | 启动 Viser 预览，默认监听本机，提供关节滑块、零位复位、正向步进、原模型/候选切换、完整 visual STL、坐标轴、质心和碰撞体显示。 |
| `--host HOST` / `--port PORT` | 调整预览服务监听地址和端口。默认仅本机可访问。 |
| `--validate-mujoco` | 额外执行 MuJoCo 编译检查；惯量错误不会阻断几何预览。报告会区分 strict direct 编译和临时 visual-only auxiliary 编译。 |

工具会聚合报告 URDF 结构、连接关系、数值合法性、网格路径、惯量正定性、主惯量三角关系、惯性坐标变换和质心位置等问题。坐标统一只改变数据表达：在 `q=0` 下使 link 坐标轴对齐机身前 `X`、左 `Y`、上 `Z`，并同步转换 visual、collision、inertial 和子关节局部坐标；零位机器人外形、连杆装配和关节安装位置应保持不变。纯符号翻转会同步转换角度、限位和 mimic；非共线方向修正没有明确规则时只在报告中标为待确认。

MuJoCo 校验的 `direct` 字段表示未拆分 STL 的严格临时编译，工具会关闭惯量自动修复并保留显式惯性参数。若 MuJoCo 因 STL 面数限制等导入器能力问题失败，顶层状态会记录为 `needs_reference`，并把原始错误写入 `validation.mujoco.*.direct.details`。工具可能额外尝试仅拆分 visual STL 的 `auxiliary` 临时副本来定位导入限制；auxiliary 通过不等于原 URDF 直接可编译，也不认证 contact geometry。这个检查不会静默修改原 STL、惯性、碰撞或接触几何。

`--rules` 文件示例：

```json
{
  "joints": {
    "FL_thigh_joint": {
      "axis": [0, 1, 0],
      "limits": {
        "lower": -1.0,
        "upper": 2.7
      }
    },
    "gripper_finger_left_joint": {
      "mimic": {
        "joint": "gripper_finger_right_joint",
        "multiplier": -1.0,
        "offset": 0.0
      }
    }
  }
}
```

退出码约定为：`0` 表示未发现错误，`1` 表示模型中存在错误级问题，`2` 表示命令行参数、资源路径或运行环境错误。碰撞简化会保留已有有效基本体；从 STL 拟合机身和大腿包围盒、髋部和小腿包围圆柱、独立足端球体，未知部件使用包围盒。候选导出不会改写 STL、质量、惯性参数或训练/控制配置。

## RI-4438 HIM sim2sim

先安装可选依赖，并使用训练生成的 ONNX 策略启动部署仿真：

```bash
bash setup.sh --extra sim2sim

uv run python deploy/main.py \
  --task ri_4438_him \
  --policy 'logs/rsl_rl/ri_4438_him/<run>/policy.onnx'
```

不加载 ONNX 的后端与状态机冒烟测试：

```bash
uv run python deploy/main.py \
  --task ri_4438_him \
  --no-policy \
  --headless \
  --auto-stand \
  --steps 800
```

可通过 `--terrain stairs_5cm`、`--terrain heightfield` 等参数切换出生区域。键盘、手柄映射、状态机流程以及观测接口约束见 [RI-4438 HIM sim2sim 文档](deploy/task/ri_4438_him/README.md)。

## 配置说明

以当前主要任务 `Ri-4438-HIM-Rough` 为例：

| 路径 | 作用 |
| --- | --- |
| `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py` | RI-4438 机器人、传感器、平地/复杂地形及 play 模式覆盖 |
| `src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py` | 观测、动作、命令、奖励、终止条件、课程与仿真参数 |
| `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py` | HIM actor、估计器、PPO 与 runner 配置 |
| `src/tasks/locomotion/ri_4438_him/mdp/` | 自定义观测、奖励、课程和速度命令 |
| `src/config/ri_4438/ri_4438_config.py` | RI-4438 控制增益、初始状态与通用 PPO 参数 |
| `src/config/ri_4438/ri_4438_him_config.py` | HIM 历史维度、网络结构、估计器和训练周期 |
| `src/config/ri_4438/ri_4438_him_lpacrl_config.py` | 独立 LP-ACRL 的机器人、HIM、任务空间、课程及训练/评价默认值 |
| `src/tasks/locomotion/ri_4438_him_lpacrl/config/curriculum_cfg.py` | LP-ACRL 配置快照、校验、任务/几何/试验表编译与指纹 |
| `scripts/train.py` | 公共训练、恢复和 LP-ACRL warm-start 入口 |
| `src/tasks/locomotion/ri_4438_him_lpacrl/rl/evaluator.py` | 冻结策略的全任务周期评价及指标导出 |
| `src/assets/robots/ri_4438/` | RI-4438 MuJoCo 模型、网格与资产配置 |
| `rsl_rl/` | 项目内置的 RSL-RL 与 HIM 扩展 |
| `deploy/task/ri_4438_him/` | RI-4438 HIM sim2sim 配置、入口与测试 |

RI-4438 HIM actor 的单帧观测为 47 维，包含机身角速度、投影重力、速度命令、步态相位、关节位置、关节速度和上一时刻动作；策略使用 6 帧历史，ONNX 输入总维度为 282，输出为 12 维关节动作。修改观测项时，需要同步检查训练配置、历史维度和部署端的观测顺序。

## 项目结构

```text
legged_wbc_mjlab/
├── scripts/                 # 训练与策略回放入口
├── src/
│   ├── assets/              # 机器人 XML、网格与资产配置
│   ├── config/              # 机器人和算法公共参数
│   └── tasks/               # locomotion / WBC 任务实现
│       └── locomotion/ri_4438_him_lpacrl/
│           ├── config/      # 预设、配置编译和环境/算法配置
│           ├── mdp/         # 任务表、联合采样、命令、地形与奖励
│           ├── rl/          # 任务内训练入口与 LP-ACRL runner
│           ├── evaluation/  # 固定试验表评价
│           ├── examples/    # 42 任务自定义配置与短跑配置
│           ├── tests/       # 单元测试与可选 GPU 测试
│           └── outputs/     # 任务内训练输出（运行后生成）
├── rsl_rl/                  # 本地 RSL-RL 与 HIM 实现
├── deploy/                  # sim2sim 部署框架与任务配置
├── tools/                   # 安装、模型分析、楼梯诊断和 URDF 工具
├── docs/                    # 安装说明、方案与论文阅读笔记
├── logs/                    # 训练输出（运行后生成）
├── pyproject.toml
└── setup.sh
```

## 常见问题

### 显存不足

通用训练入口优先降低 `--env.scene.num-envs`；LP-ACRL 任务内入口使用 `--num-envs`，并可用 `--set evaluation.num_envs=64` 降低评价批量。视频录制和可视化也会占用额外显存，训练吞吐测试时建议关闭。

### 找不到 `rsl_rl` 或其他依赖

确认已在仓库根目录执行 `bash setup.sh`，并使用 `uv run python ...` 启动脚本。不要混用系统 Python 与项目 `.venv`。

### sim2sim 提示缺少 `onnxruntime`、`pygame` 或 `yaml`

执行 `bash setup.sh --extra sim2sim` 安装部署可选依赖。

### 无图形界面的服务器无法打开 MuJoCo 窗口

训练使用 EGL；回放可选择 `--viewer viser`，部署测试可使用 `--headless --steps <N>`。

## 更多文档

- [安装配置文档](docs/setup.md)
- [best_model 工具使用](docs/tool使用.md)
- [RI-4438 HIM LP-ACRL 使用说明](src/tasks/locomotion/ri_4438_him_lpacrl/USAGE.txt)
- [RI-4438 HIM LP-ACRL 实现与复现方案](docs/RI-4438-HIM-LPACRL实现与复现方案.md)
- [LP-ACRL 阅读笔记](docs/LP-ACRL阅读笔记.md)
- [RI-4438 HIM sim2sim](deploy/task/ri_4438_him/README.md)
- [足臂全身控制项目分析与迁移方案](docs/足臂全身控制项目分析与迁移方案.md)
