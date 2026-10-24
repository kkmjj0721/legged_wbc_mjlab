# legged_wbc_mjlab

基于 **MjLab、MuJoCo 和 RSL-RL** 的足式机器人强化学习项目，面向 RI-4438 与 Unitree Go2 的速度跟踪、复杂地形训练和策略验证。

提供从训练、检查点回放、模型评估到 ONNX 导出的仿真工作流。当前可运行任务以四足运动控制为主，足臂全身控制（WBC）仍处于开发骨架与方案阶段。

[快速开始](#快速开始) · [支持的任务](#支持的任务) · [常用操作](#常用操作) · [文档导航](#文档导航)

## 项目能力

- **多种训练方案**：PPO、带历史信息估计器的 HIM + PPO、HIM 对称增强，以及基于学习进展调整任务采样的 HIM LP-ACRL。
- **平地与复杂地形**：提供 Flat / Rough 任务，支持动力学随机化与地形课程。
- **训练管理**：单 GPU / 多 GPU 训练、检查点续训、TensorBoard 日志；Symmetry 与 LP-ACRL 支持 warm-start 权重初始化。
- **策略验证**：本地或浏览器回放、视频录制、RI-4438 模型分析与离线评估、ONNX 策略导出。
- **机器人迁移工作台**：本地 Web 导入 XML 和资源，配置关节、控制与训练参数，交互检查站姿并生成四类独立任务。使用方式见 [迁移工作台](docs/robot_migration_web.md)。

## 快速开始

推荐使用 **Ubuntu 22.04 + NVIDIA GPU**。项目使用 Python **3.11**，固定依赖 `mjlab==1.6.0`，并使用仓库内可编辑的 `rsl-rl-lib==5.4.2`。

### 1. 安装环境

```bash
git clone git@github.com:kkmjj0721/legged_wbc_mjlab.git
cd legged_wbc_mjlab
bash setup.sh

uv run --no-sync python -c "import mjlab, torch, rsl_rl; print(torch.__version__, torch.cuda.is_available())"
```

安装脚本准备 `uv`、Python、虚拟环境，并根据已安装的 CUDA Toolkit 选择 PyTorch 后端。下文命令均在仓库根目录执行，无需手动激活环境；日常运行使用 `uv run --no-sync` 保留安装时选择的后端。镜像、代理、离线安装与可选依赖见 [安装配置](docs/setup.md)。

### 2. 查看机器人与环境

无需训练检查点，可先用零动作策略打开平地环境：

```bash
uv run --no-sync python -m scripts.play Ri-4438-HIM-Flat \
  --agent zero --num-envs 1 --viewer native
```

`native` 打开本地窗口；在服务器上改用 `--viewer viser`，通过终端显示的地址在浏览器中查看。零动作策略仅用于查看环境。

### 3. 启动训练

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 1024 --agent.seed 42 --agent.run-name baseline
```

默认使用 GPU 0；根据显存调整 `--env.scene.num-envs`。使用 `--agent.max-iterations` 设置本次更新次数，使用 `--agent.save-interval` 设置检查点保存间隔。

训练输出位于 `logs/rsl_rl/<experiment_name>/<timestamp>[_<run_name>]/`，包含模型、`params/` 配置和 TensorBoard 日志。上例的实验目录为 `ri_4438_him`；RI-4438 HIM 系列保存检查点时同步更新 `policy.onnx`。

```bash
uv run --no-sync tensorboard --logdir logs/rsl_rl
```

## 支持的任务

将训练或回放命令中的任务 ID 替换为下表中的值。`Flat` 为平地，`Rough` 为复杂地形。

| 机器人 | 训练方案 | Rough 任务 ID | Flat 任务 ID |
| --- | --- | --- | --- |
| RI-4438 | PPO | `Ri-4438-Rough` | `Ri-4438-Flat` |
| RI-4438 | HIM + PPO | `Ri-4438-HIM-Rough` | `Ri-4438-HIM-Flat` |
| RI-4438 | HIM Symmetry | `Ri-4438-HIM-Symmetry-Rough` | `Ri-4438-HIM-Symmetry-Flat` |
| RI-4438 | HIM LP-ACRL | `Ri-4438-HIM-LPACRL-Rough` | `Ri-4438-HIM-LPACRL-Flat` |
| Unitree Go2 | PPO | `Unitree-Go2-Rough` | `Unitree-Go2-Flat` |
| Unitree Go2 | HIM + PPO | `Unitree-Go2-HIM-Rough` | `Unitree-Go2-HIM-Flat` |

LP-ACRL 新增 `Ri-4438-HIM-LPACRL-Rough-Compact`：39 组车体系速度任务，固定技能覆盖，显式变速/启停/换向及停滞惩罚。原 Rough 832 组和 `-A0` 至 `-A3` 历史对照仍可选；新预设训练和消融命令见 [使用指南](docs/usage.md#lp-acrl-任务与课程)。

## 常用操作

### 多 GPU 训练

`--env.scene.num-envs` 按每张 GPU 计算，下面两张卡共运行 4096 个环境：

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 2048 --gpu-ids '[0, 1]'
```

### 从检查点继续训练

将 `<run>` 替换为实验目录下的运行目录名，`<iteration>` 替换为已有模型编号：

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough \
  --agent.resume True \
  --agent.load-run '<run>' --agent.load-checkpoint 'model_<iteration>.pt'
```

完整续训会恢复训练状态。Symmetry 切换实验配置、LP-ACRL 从旧课程开始新训练时，使用 `--warm-start <模型路径>` 初始化权重，并重置优化器、课程和迭代数；它与 `--agent.resume True` 互斥。兼容条件见 [检查点与恢复](docs/usage.md#检查点与恢复)。

### 回放训练好的策略

```bash
uv run --no-sync python -m scripts.play Ri-4438-HIM-Rough \
  --checkpoint-file 'logs/rsl_rl/ri_4438_him/<run>/model_<iteration>.pt' \
  --num-envs 1 --viewer native
```

添加 `--video True --video-length 500` 可录制视频，输出到该运行目录的 `videos/play/`。LP-ACRL 的交互地形与指令控制使用 `--viewer viser`，详见 [策略回放](docs/usage.md#策略回放)。

### 分析与评估模型

普通 RI-4438 HIM / PPO 使用 `best_model`，可追溯续训链并比较检查点。首次使用先安装分析依赖：

```bash
bash setup.sh --extra analysis
uv run --no-sync python -m scripts.best_model \
  --run logs/rsl_rl/ri_4438_him --evaluate
```

结果位于选中 run 的 `best/report.html`，仿真推荐模型为 `best/model_best_eval.pt`。去掉 `--evaluate` 仅分析训练日志；LP-ACRL 使用 [独立离线评估入口](docs/usage.md#lp-acrl-离线评估)。仿真结果与实机表现需分别验证。

完整参数可通过各入口的 `--help` 查看，例如：

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough --help
```

## 项目结构

```text
legged_wbc_mjlab/
├── scripts/       # 训练、回放和模型分析入口
├── src/
│   ├── assets/    # 机器人模型与网格资源
│   ├── config/    # 机器人控制与算法参数
│   ├── envs/      # 项目环境适配
│   └── tasks/     # 任务注册、环境配置、MDP 与 runner
├── rsl_rl/        # 仓库维护的 RSL-RL 与 HIM 扩展
├── tools/         # 安装助手与 URDF 检查工具
├── docs/          # 安装、使用、开发记录与研究资料
└── setup.sh       # 环境安装入口
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [安装配置](docs/setup.md) | 系统环境、镜像、代理、离线安装与 PyTorch 后端 |
| [使用指南](docs/usage.md) | 检查点与恢复、HIM 对称开关、LP-ACRL 课程与评估、URDF 工具 |
| [机器人迁移工作台](docs/robot_migration_web.md) | 本地 Web、模型导入、参数配置、3D 预览与任务生成 |
| [模型分析工具](docs/tool使用.md) | 普通 RI-4438 HIM / PPO 的模型筛选、评估协议与报告 |
| [HIM 对称机制](docs/HIMPPO对称机制分析与接入方案.md) | 对称增强、训练契约与实验方案 |
| [LP-ACRL 阅读笔记](docs/LP-ACRL阅读笔记.md) | 算法背景与方法笔记 |
| [LP-ACRL 设计审查](docs/RI-4438-HIM-LPACRL设计审查与训练退化分析.md) | 设计审查、训练退化分析与历史证据 |
| [足臂全身控制方案](docs/足臂全身控制项目分析与迁移方案.md) | WBC 项目分析与迁移规划 |

开发约定见 [AGENTS.md](AGENTS.md)，任务状态、验证方法和执行记录见 [开发工作流](docs/development.md)。
