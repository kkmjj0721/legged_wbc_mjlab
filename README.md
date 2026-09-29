# legged_wbc_mjlab

基于 [MjLab](https://github.com/mujocolab/mjlab)、MuJoCo 和 RSL-RL 的足式机器人强化学习项目，主要用于 RI-4438 与 Unitree Go2 四足机器人的速度跟踪、复杂地形运动以及策略的 sim-to-sim 验证。

项目目前包含标准 PPO 和 HIM（History Information Model）两类策略，支持平地/复杂地形训练、课程学习、动力学随机化、多 GPU 训练、检查点推理和 ONNX 策略部署。当前较完整的部署链路是 RI-4438 HIM 的 MuJoCo sim2sim。

> 本仓库面向研究与仿真验证。当前部署程序不包含 RI-4438 实机所需的电机、CAN 或 DDS 驱动，请勿直接连接真实机器人。

## 主要功能

- 基于 MuJoCo 的并行强化学习环境
- RI-4438、Unitree Go2 机器人模型与任务配置
- PPO 与带历史编码器/状态估计器的 HIM 策略
- 平地、楼梯、离散障碍等地形及地形课程
- 质量、质心、摩擦、执行器增益、控制延迟等随机化
- 单 GPU、多 GPU 和 CPU 训练入口
- TensorBoard 训练记录、模型检查点与自动 ONNX 导出
- RI-4438 HIM 的 MuJoCo sim2sim、起立/趴下状态机和键盘/手柄控制

## 已注册任务

| 任务 ID | 机器人 | 算法 | 地形 |
| --- | --- | --- | --- |
| `Ri-4438-HIM-Rough` | RI-4438 | HIM + PPO | 复杂地形 |
| `Ri-4438-HIM-Flat` | RI-4438 | HIM + PPO | 平地 |
| `Ri-4438-Rough` | RI-4438 | PPO | 复杂地形 |
| `Ri-4438-Flat` | RI-4438 | PPO | 平地 |
| `Unitree-Go2-HIM-Rough` | Unitree Go2 | HIM + PPO | 复杂地形 |
| `Unitree-Go2-HIM-Flat` | Unitree Go2 | HIM + PPO | 平地 |
| `Unitree-Go2-Rough` | Unitree Go2 | PPO | 复杂地形 |
| `Unitree-Go2-Flat` | Unitree Go2 | PPO | 平地 |

## 环境要求

- Ubuntu 22.04（推荐）
- Python 3.11（仓库当前固定为 `3.11.16`）
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

RI-4438 HIM 训练默认自动复位，并在复位前保存经过运动学刷新后的终止目标，避免重复执行整批碰撞计算和地形射线检测。HIM 的下一步监督只保存 50 维 `estimator` 观测，actor/critic 网络输入保持不变。原训练命令即可使用；用 `--env.auto-reset False` 可切回手动复位路径作对比。性能实测与验证边界见 [性能优化验证](docs/RI4438-HIM性能优化验证.md)。

RI-4438 HIM Rough 使用平地、上楼梯、下楼梯和随机粗糙高度场，基于 mjlab 的 `ROUGH_TERRAINS_CFG` 独立复制后移除坡地和波浪地形。楼梯踏面宽 0.3 m、中央平台宽 3 m，每级台阶高度范围为 0.05–0.15 m；四类地形权重依次为 0.2、0.2、0.2、0.1。求解器使用 `cg`，`njmax=300`。此配置替换了原来的 7 种楼梯和离散障碍，历史性能报告中的耗时不能直接作为当前配置的基准。

### 训练输出

输出默认保存在：

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
uv run python scripts/best_model.py --run /path/to/logs/rsl_rl/ri_4438_him/<run>

# 最新 run 自动追溯历史，统一仿真比较全部模型
uv run --extra analysis python scripts/best_model.py --run /path/to/<latest_run> --evaluate --eval-parallel 8

# 直接传实验总目录；以后新增续训 run，无需修改路径
uv run --extra analysis python scripts/best_model.py \
  --run '/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him' --evaluate --eval-parallel 8

# 旧日志仅保存了正则来源时，按原始训练到最新续训的顺序指定
uv run --extra analysis python scripts/best_model.py --runs /path/to/<original_run> /path/to/<resumed_run> --evaluate

# 复制当前阶段 best，并从选中的 checkpoint 重新导出 ONNX（CPU）
uv run python scripts/best_model.py --run /path/to/<run> --write-best --export-onnx

# 调整统计窗口，并添加明确的任务目标；不指定目标时最多判为平台期
uv run python scripts/best_model.py --run /path/to/<run> \
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
├── training_history.html  # 原始训练和历次续训的完整曲线、来源、各段状态
├── history.png            # 完整曲线；原始 iteration，分别平滑并标记恢复点
├── training_chain.json    # 所有 run 的配置及各段分析
├── chain_candidates.csv   # 所有 run 的日志候选
├── selection.json         # 规则、配置哈希、阶段、所有候选、收敛依据
├── candidates.csv         # 窗口指标及排除原因
├── report.png             # 曲线、阶段边界、当前阶段 Top K（--no-plot 跳过）
├── model_best.pt           # --write-best 或 --export-onnx 时生成
├── model_manifest.json    # 已发布模型来源、SHA256、配套 ONNX 信息
└── policy.onnx            # --export-onnx 时重新导出
```

`selection.json` 的 `top_by_stage` 保留各阶段候选；当前阶段样本不足时 `best_current_stage`
为 null，不自动用早期简单任务的模型替代。`published` 为 null 表示这次只生成报告，已有发布
文件以 `model_manifest.json` 为准。重新发布模型但不导出 ONNX 时会移除输出目录中旧的
`policy.onnx`，避免配错模型；run 根目录的训练 checkpoint 和 ONNX 不受影响。

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
  --run /path/to/logs/rsl_rl/ri_4438_him/<run> --evaluate

# 指定要比较的模型编号；降低并行环境数可以节省显存
uv run --extra analysis python scripts/best_model.py \
  --run /path/to/<run> --evaluate --eval-checkpoints 500 700 900 \
  --eval-num-envs 12 --eval-seeds 0 1 2 --eval-duration 12

# 默认已评测整条续训链的全部有效 checkpoint；也可显式指定 8 个模型并行
uv run --extra analysis python scripts/best_model.py \
  --run /path/to/<run> --evaluate --eval-parallel 8

# 全部模型、3 组种子；上下楼梯均包含固定的 5、10、15 cm 台阶
uv run --extra analysis python scripts/best_model.py \
  --run /path/to/<run> --evaluate --eval-seeds 0 1 2 \
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

推荐顺序依次比较：任务通过率、存活至结束比例、存活时长、平面速度误差、转向速度误差、机身倾斜。
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
`model_best_eval.pt` 和日志选出的 `model_best.pt` 含义不同，报告明确标注两者。
`--write-best` / `--export-onnx` 仍针对日志 best；`policy.onnx` 不代表仿真 best。
`--no-plot` 跳过统计图，但仍保留真实仿真截图、HTML、Markdown 和数值报告。

若要交互查看推荐模型的动作，可使用现有回放入口（这是 play 环境，条件与上面的固定评测不同）：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --checkpoint-file /path/to/<run>/best/model_best_eval.pt --num-envs 1
```

**平台期**指最近几个训练窗口的主要指标变化很小。它可能意味着学得稳定，也可能意味着卡住，
因此报告把训练趋势和实际仿真表现分开展示。课程还在升级、刚续训样本不足时，不会宣布最终收敛。

验证工具：

```bash
uv run --extra analysis python -m unittest discover -s tests -v
```

## 策略回放

加载本地检查点：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --checkpoint-file logs/rsl_rl/ri_4438_him/<run>/model_<iteration>.pt \
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
| `--profile generic|go2|ri4438` | 选择规则集；默认 `generic`。`go2` 和 `ri4438` 会对四足腿部使用 hip `+X`、thigh/calf `+Y` 的关节轴约定，机械臂不会套用四足规则。 |
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
  --policy logs/rsl_rl/ri_4438_him/<run>/policy.onnx
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
├── rsl_rl/                  # 本地 RSL-RL 与 HIM 实现
├── deploy/                  # sim2sim 部署框架与任务配置
├── tools/                   # 安装、模型和机器人描述处理工具
├── docs/                    # 安装说明、方案与论文阅读笔记
├── logs/                    # 训练输出（运行后生成）
├── pyproject.toml
└── setup.sh
```

## 常见问题

### 显存不足

优先降低 `--env.scene.num-envs`。视频录制和可视化也会占用额外显存，训练吞吐测试时建议关闭。

### 找不到 `rsl_rl` 或其他依赖

确认已在仓库根目录执行 `bash setup.sh`，并使用 `uv run python ...` 启动脚本。不要混用系统 Python 与项目 `.venv`。

### sim2sim 提示缺少 `onnxruntime`、`pygame` 或 `yaml`

执行 `bash setup.sh --extra sim2sim` 安装部署可选依赖。

### 无图形界面的服务器无法打开 MuJoCo 窗口

训练使用 EGL；回放可选择 `--viewer viser`，部署测试可使用 `--headless --steps <N>`。

## 更多文档

- [安装配置文档](docs/setup.md)
- [RI-4438 HIM sim2sim](deploy/task/ri_4438_him/README.md)
- [RI-4438 HIM 二阶段蒸馏实施方案](docs/RI4438-HIM二阶段蒸馏实施方案.md)
- [足臂全身控制项目分析与迁移方案](docs/足臂全身控制项目分析与迁移方案.md)
