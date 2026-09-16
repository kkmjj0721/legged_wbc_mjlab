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
