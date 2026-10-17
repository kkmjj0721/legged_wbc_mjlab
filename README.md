# legged_wbc_mjlab

基于 MjLab、MuJoCo 和 RSL-RL 的足式机器人强化学习项目，支持 RI-4438 与 Unitree Go2 的速度跟踪、复杂地形训练和策略验证。

提供 PPO、HIM + PPO 和独立的 HIM LP-ACRL 课程任务，支持动力学随机化、多 GPU 训练、检查点评估及 ONNX 导出。RI-4438 HIM 已提供 MuJoCo sim2sim 部署流程；当前不包含实机驱动。

## 快速开始

推荐使用 Ubuntu 22.04 与 NVIDIA GPU。项目要求 Python 3.11，核心依赖为 `mjlab==1.6.0` 和仓库内置的 `rsl-rl-lib==5.4.2`。

```bash
git clone git@github.com:kkmjj0721/legged_wbc_mjlab.git
cd legged_wbc_mjlab
bash setup.sh

# 验证安装并查看训练参数
uv run python -c "import mjlab, torch, rsl_rl; print(torch.__version__)"
uv run python -m scripts.train Ri-4438-HIM-Rough --help
```

安装脚本会准备 `uv`、Python 和 `.venv`，并按驱动选择 PyTorch 后端。下文命令均在仓库根目录执行，无需手动激活虚拟环境。镜像、代理和离线安装见 [安装配置](docs/setup.md)。

可选依赖按需安装，例如 `bash setup.sh --extra sim2sim`：

| 扩展 | 用途 |
| --- | --- |
| `analysis` | 训练曲线、模型比较与导出验证 |
| `sim2sim` | ONNX 策略部署仿真 |
| `urdf-tools` | URDF 检查与可视化 |

## 支持的任务

| 机器人 / 算法 | 复杂地形任务 | 平地任务 |
| --- | --- | --- |
| RI-4438 / PPO | `Ri-4438-Rough` | `Ri-4438-Flat` |
| RI-4438 / HIM + PPO | `Ri-4438-HIM-Rough` | `Ri-4438-HIM-Flat` |
| RI-4438 / HIM LP-ACRL | `Ri-4438-HIM-LPACRL-Rough` | `Ri-4438-HIM-LPACRL-Flat` |
| Unitree Go2 / PPO | `Unitree-Go2-Rough` | `Unitree-Go2-Flat` |
| Unitree Go2 / HIM + PPO | `Unitree-Go2-HIM-Rough` | `Unitree-Go2-HIM-Flat` |

## 训练与续训

以 RI-4438 HIM 复杂地形任务为例，替换任务 ID 即可训练其他任务：

```bash
# 单卡训练，默认 GPU 0
uv run python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 4096 --agent.run-name baseline

# 多卡训练；num-envs 为每张卡的环境数
uv run python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 2048 --gpu-ids 0 1

# 从已有运行恢复；将占位符替换为实际目录名和模型编号
uv run python -m scripts.train Ri-4438-HIM-Rough \
  --agent.resume True \
  --agent.load-run '<run>' --agent.load-checkpoint 'model_<iteration>.pt'
```

可用 `--agent.max-iterations` 设置本次训练更新次数，`--agent.save-interval` 设置保存间隔。显存不足时优先降低 `--env.scene.num-envs`；完整参数见对应任务的 `--help`。

输出位于 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`，包括 `model_<iteration>.pt`、`params/` 和 TensorBoard 日志。RI-4438 HIM / LP-ACRL 在保存检查点时更新 `policy.onnx`。

```bash
uv run tensorboard --logdir logs/rsl_rl
```

## 策略回放

```bash
uv run python -m scripts.play Ri-4438-HIM-Rough \
  --checkpoint-file 'logs/rsl_rl/ri_4438_him/<run>/model_<iteration>.pt' \
  --num-envs 1 --viewer native
```

无桌面的服务器使用 `--viewer viser`。仅检查环境时，用 `--agent zero` 替代 `--checkpoint-file`；录制策略视频可添加 `--video True --video-length 500`，输出到运行目录的 `videos/play/`。

LP-ACRL 使用对应任务 ID 回放，推荐 `--viewer viser`。在 `Commands → Twist / Play` 面板选择地形、等级和出生朝向，点击 `Apply & reset`，再调整速度。默认手动站立，也可切换 `Uniform tasks` 或兼容模型的 `Checkpoint teacher` 采样模式；`Replay` 重复当前场景。

## 模型分析与比较

`best_model.py` 可分析训练曲线；其仿真比较支持普通 RI-4438 HIM / PPO 任务。LP-ACRL 使用下一节的独立评估入口。

```bash
# 分析最新 run 及其续训链
uv run --extra analysis python -m scripts.best_model \
  --run logs/rsl_rl/ri_4438_him

# 增加统一场景的仿真比较
uv run --extra analysis python -m scripts.best_model \
  --run logs/rsl_rl/ri_4438_him --evaluate
```

`--run` 也可指定某次训练目录。结果写入选中 run 的 `best/`：打开 `report.html` 查看报告，`model_best_eval.pt` 为本轮仿真推荐模型。`--write-best` / `--export-onnx` 操作的是日志评分候选，分别生成 `model_best.pt` / `policy.onnx`。

日志评分、曲线平台期和仿真推荐有不同含义；仿真排名不代表实机验证结果。评估协议、筛选规则及完整参数见 [best_model 使用说明](docs/tool使用.md)。

## HIM LP-ACRL

LP-ACRL 在回合开始时联合采样地形与命令，按在线学习进展调整任务概率。默认使用航向反馈控制，回合内保持速度目标与目标航向；Rough 包含 1560 个任务，Flat 包含 30 个任务。分组统计、跟踪评分和楼梯覆盖采样属于本项目扩展。

任务空间、地形、HIM 与课程参数集中在 [LP-ACRL 配置](src/config/ri_4438/ri_4438_him_lpacrl_config.py)。

### 训练与恢复

```bash
uv run python -m scripts.train Ri-4438-HIM-LPACRL-Rough \
  --env.scene.num-envs 2048

# 从兼容 HIM 权重开始新课程
uv run python -m scripts.train Ri-4438-HIM-LPACRL-Rough \
  --env.scene.num-envs 2048 --warm-start '/path/to/model.pt'
```

- 多卡训练添加 `--gpu-ids 0 1` 或 `--gpu-ids all`，环境数仍按每卡计算。
- 完整续训使用上文的 `--agent.resume True`，要求兼容的 **v6 课程状态**、配置和相同 GPU 数。
- `--warm-start` 加载兼容策略权重并重置优化器、课程和迭代数，与 resume 互斥。旧版模型或更换 GPU 数时可用此方式启动新训练。

### 离线评估

```bash
uv run python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model.pt' --output-dir '/path/to/new_evaluation'
```

输出目录必须尚不存在。默认恢复检查点中的任务配置，`--suite both` 同时运行任务表评估和平地部署序列；可用 `--suite task` 或 `--suite deployment` 单独执行。用 `--num-envs` 调整评估批量，`--trials-per-task` 调整重复次数；跨历史模型比较可指定 `--protocol current`，统一使用当前任务配置。

任务评估生成 `evaluation.json` 和 `per_task.csv`，部署序列结果位于 `deployment/`。查看指标时区分：

| 指标 | 含义 |
| --- | --- |
| `paper_success` / `success` | 存活比例与 vx、wz 跟踪判据 |
| `tracking_success` | 存活比例与 vx、vy、wz 三轴跟踪判据 |
| `deployment_success` | 三轴误差、物理失败、持续停滞和已观测指令切换的综合判据 |
| `terrain_traversal_success` | 楼梯全部踏步通过且未跌倒 |

速度跟踪达标不等于通过楼梯；没有换向事件时，`reversal_success` 为 null。具体指标定义见 [评估实现](src/tasks/locomotion/ri_4438_him_lpacrl/rl/evaluator.py)。

录制单个楼梯任务：

```bash
uv run python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model.pt' --protocol current --suite task \
  --command forward --terrain stairs_025_outward_up --level l0 \
  --num-envs 1 --trials-per-task 1 --video \
  --output-dir '/path/to/new_stairs_video'
```

视频要求 `--suite task --num-envs 1`。更多选项见该入口的 `--help`。

## RI-4438 HIM sim2sim

```bash
uv run --extra sim2sim python deploy/main.py \
  --task ri_4438_him \
  --policy 'logs/rsl_rl/ri_4438_him/<run>/policy.onnx'
```

不加载策略的后端与状态机检查：

```bash
uv run --extra sim2sim python deploy/main.py \
  --task ri_4438_him --no-policy --headless --auto-stand --steps 800
```

HIM 默认采用 47 维单帧观测、6 帧历史，ONNX 输入为 282 维，输出为 12 维关节动作。修改观测或控制配置后需同步检查部署接口。键盘、手柄、地形选择与状态机说明见 [sim2sim 文档](deploy/task/ri_4438_him/README.md)。

## 项目结构与配置

```text
├── scripts/       # 训练、回放、模型分析入口
├── src/
│   ├── assets/    # 机器人 XML、网格与资产配置
│   ├── config/    # 机器人控制、算法和 LP-ACRL 默认参数
│   └── tasks/     # 任务注册、环境、MDP 与 runner
├── rsl_rl/        # 本地 RSL-RL 与 HIM 扩展
├── deploy/        # sim2sim 部署框架
├── tools/         # 安装、诊断与 URDF 工具
├── docs/          # 使用文档、设计方案和阅读笔记
└── logs/          # 训练输出（运行后生成）
```

任务中的 `config/env_cfgs.py` 配置机器人、地形与回放模式，`config/rl_cfg.py` 配置算法与 runner；观测、动作、奖励和终止条件位于对应的 `*_env_cfg.py`，自定义逻辑位于 `mdp/`。

URDF 检查与预览：

```bash
uv run --extra urdf-tools python tools/check_urdf.py MODEL.urdf \
  --profile ri4438 --preview --report report.json
```

工具默认不修改输入文件；`--output NEW.urdf` 可导出候选模型，完整选项见 `--help`。

## 更多文档

- [安装配置与依赖排查](docs/setup.md)
- [best_model 工具使用](docs/tool使用.md)
- [RI-4438 HIM sim2sim](deploy/task/ri_4438_him/README.md)
- [LP-ACRL 早期设计方案](docs/RI-4438-HIM-LPACRL实现与复现方案.md)（历史设计，当前用法以本 README 和代码为准）
- [LP-ACRL 阅读笔记](docs/LP-ACRL阅读笔记.md)
- [足臂全身控制分析与迁移方案](docs/足臂全身控制项目分析与迁移方案.md)
