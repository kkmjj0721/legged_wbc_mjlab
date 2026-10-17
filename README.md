# legged_wbc_mjlab

基于 MjLab、MuJoCo 和 RSL-RL 的足式机器人强化学习项目，支持 RI-4438 与 Unitree Go2 的速度跟踪、复杂地形训练和策略验证。

提供 PPO、HIM + PPO、HIM LP-ACRL 任务，支持动力学随机化、多 GPU 训练、检查点评估与 ONNX 导出。RI-4438 HIM 已提供 MuJoCo sim2sim 部署流程，当前不包含实机驱动。

## 安装

推荐 Ubuntu 22.04、NVIDIA GPU；Python 要求为 3.11，核心依赖为 `mjlab==1.6.0` 和仓库内置的 `rsl-rl-lib==5.4.2`。

```bash
git clone git@github.com:kkmjj0721/legged_wbc_mjlab.git
cd legged_wbc_mjlab
bash setup.sh

uv run python -c "import mjlab, torch, rsl_rl; print(torch.__version__)"
```

安装脚本自动准备 `uv`、Python、`.venv` 和匹配驱动的 PyTorch 后端。下文命令均在仓库根目录执行，无需手动激活环境。镜像、代理和离线安装见 [安装配置](docs/setup.md)。

可选依赖：`analysis` 用于模型分析，`sim2sim` 用于部署仿真，`urdf-tools` 用于 URDF 检查；通过 `bash setup.sh --extra <扩展名>` 安装，或在命令中添加 `uv run --extra <扩展名>`。

## 任务

| 机器人 / 算法 | 复杂地形 | 平地 |
| --- | --- | --- |
| RI-4438 / PPO | `Ri-4438-Rough` | `Ri-4438-Flat` |
| RI-4438 / HIM + PPO | `Ri-4438-HIM-Rough` | `Ri-4438-HIM-Flat` |
| RI-4438 / HIM LP-ACRL | `Ri-4438-HIM-LPACRL-Rough` | `Ri-4438-HIM-LPACRL-Flat` |
| Unitree Go2 / PPO | `Unitree-Go2-Rough` | `Unitree-Go2-Flat` |
| Unitree Go2 / HIM + PPO | `Unitree-Go2-HIM-Rough` | `Unitree-Go2-HIM-Flat` |

## 训练与续训

替换任务 ID 可训练其他任务，完整参数见 `uv run python -m scripts.train <任务ID> --help`。

```bash
# 单卡训练，默认 GPU 0
uv run python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 4096 --agent.run-name baseline

# 多卡训练，环境数按每张卡计算
uv run python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 2048 --gpu-ids 0 1

# 恢复训练，替换实际运行目录名和模型编号
uv run python -m scripts.train Ri-4438-HIM-Rough \
  --agent.resume True \
  --agent.load-run '<run>' --agent.load-checkpoint 'model_<iteration>.pt'
```

`--agent.max-iterations` 设置本次更新次数，`--agent.save-interval` 设置保存间隔；显存不足时优先降低 `--env.scene.num-envs`。

输出位于 `logs/rsl_rl/<experiment_name>/<timestamp>[_<run_name>]/`，包含检查点、`params/` 和 TensorBoard 日志。RI-4438 HIM / LP-ACRL 保存检查点时同步更新 `policy.onnx`。

```bash
uv run tensorboard --logdir logs/rsl_rl
```

## 策略回放

```bash
uv run python -m scripts.play Ri-4438-HIM-Rough \
  --checkpoint-file 'logs/rsl_rl/ri_4438_him/<run>/model_<iteration>.pt' \
  --num-envs 1 --viewer native
```

服务器使用 `--viewer viser`；仅检查环境时，用 `--agent zero` 替代 `--checkpoint-file <路径>`。添加 `--video True --video-length 500` 可录制策略视频，输出到运行目录的 `videos/play/`。

LP-ACRL 回放使用对应任务 ID 和 `--viewer viser`。在 `Commands → Twist / Play` 面板选择地形、等级和出生朝向，点击 `Apply & reset` 后调整速度；默认手动站立。`Uniform tasks` 均匀采样任务，兼容模型还可使用 `Checkpoint teacher`；`Replay` 重复当前场景。

## 模型评估

### 普通 RI-4438 HIM / PPO

```bash
uv run --extra analysis python -m scripts.best_model \
  --run logs/rsl_rl/ri_4438_him --evaluate
```

工具选择最新 run 并追溯续训链；`--run` 也可指定某次运行目录。打开选中 run 的 `best/report.html` 查看结果，`model_best_eval.pt` 为本轮仿真推荐。去掉 `--evaluate` 仅分析日志。

`--write-best` / `--export-onnx` 分别生成日志评分候选 `model_best.pt` / `policy.onnx`。仿真推荐与日志候选可能不同，实机表现需另行验证。完整说明见 [best_model 工具使用](docs/tool使用.md)。

### HIM LP-ACRL

LP-ACRL 按在线学习进展调整地形与命令任务的采样概率。默认采用第七版混合任务：Rough 保留 1560 个固定任务（520 个 teacher 组），追加 832 个随机过程组，共 1352 个常规 teacher 组；Flat 为 46 个任务、26 个 teacher 组。随机过程的两轴范围独立取 0.2/0.5/0.8/1.0，每 2～10 秒同步重采速度与世界系目标航向。25% 正向直爬覆盖仍单独计数。7×13 连通地图采用 5×11 核心区、外围一圈及 16 m 平地框，允许跨子地块持续训练，只有整图外缘、物理失败或 20 秒时限结束回合。Rough 的两类坡地 l0 实际均为平地。参数见 [LP-ACRL 配置](src/config/ri_4438/ri_4438_him_lpacrl_config.py)。

训练使用公共入口，将任务 ID 替换为 `Ri-4438-HIM-LPACRL-Rough` 或 `Ri-4438-HIM-LPACRL-Flat`。完整续训要求兼容的 **v7 课程状态**、配置和相同 GPU 数；从旧模型或兼容 HIM 权重开始新课程使用：

```bash
uv run python -m scripts.train Ri-4438-HIM-LPACRL-Rough \
  --env.scene.num-envs 2048 --warm-start '/path/to/model.pt'
```

`--warm-start` 重置优化器、课程和迭代数，与 `--agent.resume True` 互斥。分组统计、跟踪评分和楼梯覆盖采样属于本项目扩展。第六版接续必须使用权重 warm-start，全部 teacher 基线重新建立。

对照任务可直接选择 `Ri-4438-HIM-LPACRL-Rough-A0`（旧地图固定任务）、`-A1`（连通地图固定任务）、`-A2`（混合均匀采样）或 `-A3`（混合 signed-LP，默认）。四组均保留覆盖分支。设计见 [第七版混合任务设计](docs/RI-4438_LPACRL第七版混合任务设计.md)。

独立离线评估：

```bash
uv run python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model.pt' --output-dir '/path/to/new_evaluation'
```

输出目录必须尚不存在。默认从检查点恢复任务配置，`--suite both` 运行任务表与平地部署序列；可选 `task` / `deployment` 单独执行。`--protocol current` 统一使用当前任务配置，便于比较历史模型；`--num-envs` 和 `--trials-per-task` 控制批量与重复次数。`--task-mode fixed` 保留固定任务评估，`--task-mode random_process` 使用独立种子的变指令试验；默认 `mixed` 评估两类。

任务结果为 `evaluation.json`、`per_task.csv`，部署序列结果位于 `deployment/`。分别查看论文跟踪判据 `paper_success`、三轴跟踪 `tracking_success`、部署验收 `deployment_success` 、严格变指令通过 `process_success` 和完整出口通过 `terrain_traversal_success`。后者在通过后折返或摔倒仍保留，另报 `terrain_passed_without_failure`；旧跨阶指标另存 `legacy_terrain_traversal_success`。切换失败与窗口未完整观测分开计数，实际支撑地块及每次切换记录保存在试验结果中；定义见 [评估实现](src/tasks/locomotion/ri_4438_him_lpacrl/rl/evaluator.py)。

录制指定楼梯任务：

```bash
uv run python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model.pt' --protocol current --suite task \
  --command forward --terrain stairs_025_outward_up --level l0 \
  --num-envs 1 --trials-per-task 1 --video \
  --output-dir '/path/to/new_stairs_video'
```

视频要求 `--suite task --num-envs 1`，更多参数见该入口的 `--help`。

## RI-4438 HIM sim2sim

```bash
uv run --extra sim2sim python deploy/main.py \
  --task ri_4438_him \
  --policy 'logs/rsl_rl/ri_4438_him/<run>/policy.onnx'
```

后端与状态机检查可将 `--policy <路径>` 替换为 `--no-policy --headless --auto-stand --steps 800`。默认 HIM 接口为 47 维单帧观测 × 6 帧历史，ONNX 输入 282 维、输出 12 维动作。控制方式、地形和接口约束见 [sim2sim 文档](deploy/task/ri_4438_him/README.md)。

## 目录与工具

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 训练、回放与模型分析入口 |
| `src/config/` | 机器人控制、算法与 LP-ACRL 默认参数 |
| `src/tasks/` | 任务注册、环境配置、MDP 与 runner |
| `src/assets/` | 机器人模型和网格 |
| `rsl_rl/` | 本地 RSL-RL 与 HIM 扩展 |
| `deploy/` | sim2sim 部署框架 |
| `tools/` | 安装、诊断与 URDF 工具 |
| `logs/` | 训练输出 |

URDF 检查与预览默认不修改输入文件，完整参数见 `--help`：

```bash
uv run --extra urdf-tools python tools/check_urdf.py MODEL.urdf \
  --profile ri4438 --preview --report report.json
```

更多资料：[安装配置](docs/setup.md) · [模型分析](docs/tool使用.md) · [LP-ACRL 阅读笔记](docs/LP-ACRL阅读笔记.md) · [足臂全身控制方案](docs/足臂全身控制项目分析与迁移方案.md)。
