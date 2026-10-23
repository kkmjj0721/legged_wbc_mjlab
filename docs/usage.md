# 训练、回放与评估指南

本文承接 [README](../README.md) 的详细用法，包含检查点兼容条件、HIM 对称实验、LP-ACRL 课程与评估指标。命令均在仓库根目录执行，默认已按 [安装配置](setup.md) 安装环境；日常使用 `uv run --no-sync` 保留已选 PyTorch 后端。

模型分析需要 `analysis` 可选依赖，URDF 工具需要 `urdf-tools`。先通过 `bash setup.sh --extra <扩展名>` 按需安装，再执行相应命令；同时需要两者时可重复传入 `--extra`：

```bash
bash setup.sh --extra analysis --extra urdf-tools
```

[检查点与恢复](#检查点与恢复) · [策略回放](#策略回放) · [HIM 对称实验](#him-对称实验与训练方案切换) · [LP-ACRL 任务与课程](#lp-acrl-任务与课程) · [普通 HIM / PPO 评估](#普通-ri-4438-him--ppo-评估) · [LP-ACRL 离线评估](#lp-acrl-离线评估)

## 检查点与恢复

基础训练与回放入口见 [README](../README.md)。替换任务 ID 可训练其他任务，完整参数见 `uv run --no-sync python -m scripts.train <任务ID> --help`。

```bash
# 单卡训练，默认 GPU 0
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 4096 --agent.run-name baseline

# 多卡训练，环境数按每张卡计算
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough \
  --env.scene.num-envs 2048 --gpu-ids 0 1

# 恢复训练，替换实际运行目录名和模型编号
uv run --no-sync python -m scripts.train Ri-4438-HIM-Rough \
  --agent.resume True \
  --agent.load-run '<run>' --agent.load-checkpoint 'model_<iteration>.pt'
```

`--agent.max-iterations` 设置本次更新次数，`--agent.save-interval` 设置保存间隔；显存不足时优先降低 `--env.scene.num-envs`。

输出位于 `logs/rsl_rl/<experiment_name>/<timestamp>[_<run_name>]/`，包含检查点、`params/` 和 TensorBoard 日志。RI-4438 HIM / LP-ACRL 保存检查点时同步更新 `policy.onnx`。

LP-ACRL 的训练日志与模型编号从 `0` 开始：第一次 PPO 更新后保存 `model_0.pt`，默认之后按编号 `200、400……` 保存，正常训练结束时另存最后一轮。课程状态中的 `completed_policy_updates` 仍记录实际完成次数，`curriculum.jsonl` 的 `update` 同样使用这个次数；续训依据它定位下一次更新，兼容此前从 `1` 编号的检查点。

```bash
uv run --no-sync tensorboard --logdir logs/rsl_rl
```

完整续训与权重初始化用途不同：

| 方式 | 参数 | 行为与前置条件 |
| --- | --- | --- |
| 完整续训 | `--agent.resume True` 配合 `--agent.load-run`、`--agent.load-checkpoint` | 恢复检查点中的训练状态；任务与训练契约必须兼容。LP-ACRL 还要求兼容的 v7 课程、任务指纹及相同 GPU 数。 |
| 权重初始化 | `--warm-start <模型路径>` | 仅 Symmetry 与 LP-ACRL 支持；加载兼容权重，重新初始化优化器、课程和迭代数，与 resume 互斥。普通 HIM 不支持此参数。 |

Symmetry 的实验切换要求见下方说明，LP-ACRL 的初始化示例见 [任务与课程](#lp-acrl-任务与课程)。

## 策略回放

```bash
uv run --no-sync python -m scripts.play Ri-4438-HIM-Rough \
  --checkpoint-file 'logs/rsl_rl/ri_4438_him/<run>/model_<iteration>.pt' \
  --num-envs 1 --viewer native
```

服务器使用 `--viewer viser`；仅检查环境时，用 `--agent zero` 替代 `--checkpoint-file <路径>`。添加 `--video True --video-length 500` 可录制策略视频，输出到运行目录的 `videos/play/`。

LP-ACRL 回放使用对应任务 ID 和 `--viewer viser`。在 `Commands → Twist / Play` 面板选择地形、等级和出生朝向，点击 `Apply & reset` 后调整速度；默认手动站立。`Uniform tasks` 均匀采样任务，兼容模型还可使用 `Checkpoint teacher`；`Replay` 重复当前场景。

## HIM 对称实验与训练方案切换

`Ri-4438-HIM-Symmetry-Rough` / `-Flat` 默认使用配对镜像增强 PPO，与普通 HIM 一样让 estimator 跟随 PPO 学习率。PPO 镜像增强、actor 镜像约束、critic 镜像约束和 estimator 镜像监督均默认开启；actor 约束系数为 `0.1`（按动作维度求平均），critic 约束系数为 `1.0`。真实 rollout 仍为每个环境 100 步，镜像增强后保存 200 条 transition。设置无对称增强的对照组时，应显式关闭全部对称功能。

实验开关位于 `agent.algorithm`：

| 字段 | 含义 |
| --- | --- |
| `symmetry_cfg.use_data_augmentation` | 默认 `True`。`implementation="ppo"` 时启用配对增强：镜像真实动作及其行为分布，配对 GAE 与终止掩码；调度只读取真实样本 KL。物理对称仍是近似。 |
| `symmetry_cfg.use_mirror_loss` | 默认 `True`，独立开启 actor 镜像约束；默认系数 `0.1`，`actor_mirror_reduction="mean"`，仅更新 actor。 |
| `symmetry_cfg.critic_mirror_loss_coeff` | 默认 `1.0`；大于零时独立开启 critic 约束，不依赖 actor 约束开关。 |
| `estimator_use_mirror` | 默认 `True`，独立开启 estimator 镜像监督；关闭时配对 PPO 仍只使用真实样本训练 estimator。 |
| `estimator_lr_mode` | `follow` 跟随 PPO；`fixed` 使用 `estimator_learning_rate`；`cap` 使用 PPO 学习率与 `estimator_lr_cap` 的较小值。 |
| `reference_update` | 显式选择更新路径，不再由 `homie` 强制开启；正常路径使用 `kl_mode="standard"`。 |

旧 HOMIE 仅用于复现：`HimSymmetryAlgorithmCfg.legacy_homie()` 显式设置独立采样镜像动作、旧求和损失、镜像 estimator 监督、reference 路径和 `kl_mode="legacy_homie"`。这一选项保留原有动作与结果不对应的行为。旧 actor 损失按 12 个动作维度求和，改为求平均时，保持同等强度需将系数乘以 **12**；`0.1` 是新的弱约束候选，并非旧系数的等价迁移。

每轮记录 PPO / estimator 的实际学习率及真实 / 镜像 KL、ratio、clip fraction。检查点保存有效训练契约；symmetry 任务要求相同契约才可续训。Symmetry 切换方案或导入没有新契约的旧检查点时，使用 `--warm-start /path/to/model.pt`，仅加载权重并使用新的优化器、课程和迭代数；旧模型仍可用于推理。这些对称实验配置位于 symmetry 任务，普通 HIM 保持原有配置和旧检查点续训兼容性。Symmetry 日志目录另保存 `effective_him_training.json`。

## LP-ACRL 任务与课程

LP-ACRL 按在线学习进展调整地形与命令任务的采样概率。默认仅使用随机变指令任务：Rough 为 4 个纵向范围 × 4 个横向范围 × 52 种出生地形/等级组合，共 832 个任务及 teacher 组；Flat 为 16 个任务及 teacher 组。全部回合由 teacher 采样，取消默认的 520 个固定指令组和 25% 正向直爬覆盖。两轴范围独立取 0.2/0.5/0.8/1.0，每 2～10 秒同步重采速度与世界系目标航向。7×13 连通地图采用 5×11 核心区、外围一圈及 16 m 平地框，允许跨子地块持续训练，只有整图外缘、物理失败或 20 秒时限结束回合。Rough 的两类坡地 l0 实际均为平地。参数见 [LP-ACRL 配置](../src/config/ri_4438/ri_4438_him_lpacrl_config.py)。

训练使用公共入口，将任务 ID 替换为 `Ri-4438-HIM-LPACRL-Rough` 或 `Ri-4438-HIM-LPACRL-Flat`。完整续训要求兼容的 **v7 课程状态**、配置和相同 GPU 数；从旧模型或兼容 HIM 权重开始新课程使用：

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-LPACRL-Rough \
  --env.scene.num-envs 2048 --warm-start '/path/to/model.pt'
```

`--warm-start` 重置优化器、课程和迭代数，与 `--agent.resume True` 互斥。从原固定/混合任务切换到默认 832 组时，任务指纹和组表已改变，需使用 warm-start，重新建立 teacher 基线。第六版接续同样使用 warm-start。

历史对照任务可选择 `Ri-4438-HIM-LPACRL-Rough-A0`（旧地图固定任务）、`-A1`（连通地图固定任务）、`-A2`（混合均匀采样）或 `-A3`（混合 signed-LP）。四组均保留原有 25% 覆盖分支，A3 不再是默认 Rough 配置。当前默认值与历史对照配置见 [LP-ACRL 配置](../src/config/ri_4438/ri_4438_him_lpacrl_config.py) 和 [课程配置构造](../src/tasks/locomotion/ri_4438_him_lpacrl/config/curriculum_cfg.py)；历史问题与证据见 [设计审查与训练退化分析](RI-4438-HIM-LPACRL设计审查与训练退化分析.md)。

## 普通 RI-4438 HIM / PPO 评估

```bash
uv run --no-sync python -m scripts.best_model \
  --run logs/rsl_rl/ri_4438_him --evaluate
```

工具选择最新 run 并追溯续训链；`--run` 也可指定某次运行目录。打开选中 run 的 `best/report.html` 查看结果，`model_best_eval.pt` 为本轮仿真推荐。去掉 `--evaluate` 仅分析日志。

`--write-best` / `--export-onnx` 分别生成日志评分候选 `model_best.pt` / `policy.onnx`。仿真推荐与日志候选可能不同，实机表现需另行验证。完整说明见 [best_model 工具使用](tool使用.md)。

## LP-ACRL 离线评估

```bash
uv run --no-sync python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model.pt' --output-dir '/path/to/new_evaluation'
```

输出目录必须尚不存在。默认从检查点恢复任务配置，`--suite both` 在 MjLab 环境内运行任务表与平地指令序列；可选 `task` / `deployment` 单独执行。`--protocol current` 统一使用当前任务配置，便于比较历史模型；`--num-envs` 和 `--trials-per-task` 控制批量与重复次数。省略 `--task-mode` 沿用所选协议的任务族；显式选择 `fixed`、`random_process` 或 `mixed` 可分别评估固定指令、独立种子的变指令或两类任务。

任务结果为 `evaluation.json`、`per_task.csv`，部署序列结果位于 `deployment/`。分别查看论文跟踪判据 `paper_success`、三轴跟踪 `tracking_success`、部署验收 `deployment_success` 、严格变指令通过 `process_success` 和完整出口通过 `terrain_traversal_success`。后者在通过后折返或摔倒仍保留，另报 `terrain_passed_without_failure`；旧跨阶指标另存 `legacy_terrain_traversal_success`。切换失败与窗口未完整观测分开计数，实际支撑地块及每次切换记录保存在试验结果中；定义见 [评估实现](../src/tasks/locomotion/ri_4438_him_lpacrl/rl/evaluator.py)。

录制指定楼梯任务：

```bash
uv run --no-sync python -m src.tasks.locomotion.ri_4438_him_lpacrl.rl.evaluate \
  --checkpoint '/path/to/model.pt' --protocol current --suite task \
  --command forward --terrain stairs_025_outward_up --level l0 \
  --num-envs 1 --trials-per-task 1 --video \
  --output-dir '/path/to/new_stairs_video'
```

视频要求 `--suite task --num-envs 1`，更多参数见该入口的 `--help`。

## URDF 检查与预览

URDF 检查与预览默认不修改输入文件，完整参数见 `--help`：

```bash
uv run --no-sync python tools/check_urdf.py MODEL.urdf \
  --profile ri4438 --preview --report report.json
```

## 自定义运行工具

训练、播放和评测入口使用 `src.envs.ManagerBasedRlEnv`：reset 后只刷新被重置环境的执行器控制和初始延迟历史，其他环境的控制、延迟队列和计数保持不变。自行编写运行工具时也应从 `src.envs` 导入环境类；直接使用 `mjlab.envs.ManagerBasedRlEnv` 仍是上游行为。此适配针对当前固定的 `mjlab==1.6.0` 和仓库使用的内置执行器。

新增独立工具时，通过 `import src.tasks` 完成项目任务注册。HIM 观测布局以 [him_layouts.py](../src/config/ri_4438/him_layouts.py) 为依据；调整观测、历史或动作映射时，同时核对 actor、critic、estimator、检查点契约与 ONNX 接口。
