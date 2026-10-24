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
  --env.scene.num-envs 2048 --gpu-ids '[0, 1]'

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
| 完整续训 | `--agent.resume True` 配合 `--agent.load-run`、`--agent.load-checkpoint` | 恢复检查点中的训练状态；任务与训练契约必须兼容。LP-ACRL 还要求兼容的课程版本、任务指纹及相同 GPU 数；Compact 为 v8，并要求每卡环境数相同，旧预设为 v7。 |
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

LP-ACRL 的 `scripts.play` 会在创建环境前自动恢复检查点内的 `task_config`，使用保存的任务、地形、机器人及运动判定协议；旧模型缺少 `motion_tracking` 时使用原相位门限，无需修改中央配置。所选 LP-ACRL 任务 ID 确定 runner 入口，实际播放配置以快照为准，差异会打印提示；`--num-envs`、视频尺寸和 `--no-terminations` 等回放参数随后生效。缺少快照时沿用所选任务配置并执行原兼容性检查；快照与保存指纹不一致时明确拒绝。此恢复仅用于回放，训练切换协议仍须使用 `--warm-start`。

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

同种子 GPU 复现实验可显式开启 `--env.sim.deterministic True`（仅 HIM Symmetry）；默认关闭。该模式固定接触、约束与相关累加顺序，会增加环境步开销，完整续训也需保持相同模式。原因、本地对照结果、回放参数与使用边界见 [GPU 训练复现说明](gpu_reproducibility.md)。

## LP-ACRL 任务与课程

新实验使用 **`Ri-4438-HIM-LPACRL-Rough-Compact`**。它将任务压缩为 39 组：15 组平地/轻粗糙地面稳态、12 组上下楼梯、10 组显式序列及 2 组随机车体系命令。分配固定环境槽位，分别占 35% / 25% / 30% / 10%；按最大余数法取整，1024 环境对应 358 / 256 / 307 / 103 个。LP 只在各族内部调整任务概率，随机族保持均匀，失败 reset 不改变环境所属族。

序列每段 4 秒，包括 `0→±0.1/0.2→0` 的纵横向启停、换向、纯转向，以及 `vx=0.1→0.3→0.5→0.2→0.1 m/s` 同向变速。楼梯为 5 / 8 / 10 cm 级高、30 cm 踏面、四级上下楼梯，命令为 0.3 / 0.5 m/s；使用独立小地块，出口位于行政截断线以内。`vx/vy/wz` 全部为车体系目标，不再将第三轴解释为世界航向。

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-LPACRL-Rough-Compact \
  --env.scene.num-envs 1024 --agent.max-iterations 1000 --agent.seed 42 \
  --agent.run-name compact_v1
```

新预设的停滞惩罚读取仿真真实速度：启动或明显切换后留 0.8 秒宽限，用 0.3 秒滑窗检查沿命令方向的速度，持续不足 0.4 秒后施加有界惩罚。权重为 `-0.5`，在 20 ms 控制步长下每步最多扣 `0.01`；零命令不触发，侧移与纯转向分别判断。持续停滞终止已经接通，**默认关闭**，配置项为 `behavior_monitor.termination_enabled`；启用时在宽限结束后连续停滞 3 秒才作为物理失败终止。阈值是待实验验证的初始参数。

| 任务后缀（前缀均为 `Ri-4438-HIM-LPACRL-Rough-`） | 用途 |
| --- | --- |
| `Compact` | 39 组、固定覆盖、族内 LP、停滞惩罚 |
| `Compact-Rough` | 230 组，保留低速并增加三轴 ±1.0；原 Rough 完整地形与连通地图 |
| `Compact-Coverage` | 与原 Compact 相同任务/覆盖/监测，停滞惩罚权重为 0，用于隔离奖励影响 |
| `Compact-Uniform` | 与原 Compact 相同任务/覆盖/惩罚，族内均匀，用于隔离 LP 影响 |
| `Random-Uniform` | 原 832 组随机任务的均匀对照，原 Rough 为 LP |

Compact 保存 **v8** 课程状态、技能槽位及监测配置指纹。完整续训需要相同预设、有效配置、每卡环境数和 GPU 数；切换上述预设或从 v7 课程进入 Compact，使用 `--warm-start` 初始化兼容权重，或从头训练。HIM 的 47×6 历史、244 维 critic、50 维 estimator 目标及 12 维动作契约保持原样。teacher 仍使用整回合跟踪分数及原有的物理失败时长折扣，不把新增 PPO 惩罚重复计入 teacher。

### Compact-Rough：完整 Rough 地形与 1.0 指令

`Ri-4438-HIM-LPACRL-Rough-Compact-Rough` 是独立新增预设，原 Compact 的 39 组、速度和小地块均保留。新预设保持四类固定环境份额及族内 LP：

| 类别 | 组数 | 环境份额 | 内容 |
| --- | ---: | ---: | --- |
| steady | 41 | 35% | 原平地技能加前后/横移 ±1.0 m/s、转向 ±1.0 rad/s；四档起伏地面练前后 ±0.2/0.5/1.0 m/s |
| stairs | 120 | 25% | 原 Rough 的 8 类楼梯 × 5 档高度 × 0.3/0.5/1.0 m/s 正向固定速度 |
| sequence | 17 | 30% | 原 10 条低速序列，加三轴 1.0 启停/换向及 `vx=0.1→0.5→1.0→0.5→0.1`，每段 4 秒 |
| random | 52 | 10% | 每个 Rough 出生地形/等级各一组；车体系 vx/vy/wz 分别在 [-1,1] 均匀采样，每段保持 2～10 秒 |

三轴上限分别为 1.0，站立及原低速命令继续保留；vx/vy 的单位为 m/s，wz 为 rad/s。该上限按分量计算，不限制平面合速度为 1.0 m/s。新增预设沿用原 Rough 的全部 52 个出生条件、18/21/25/28 cm 踏面、5/8/10/12/16 cm 楼梯及四档起伏/上下坡地；地图为 7×13 连通地块与 16 m 外围平地，跨出生地块不重置，整图外缘才行政截断。

原 Rough 的两类坡地 l0 几何均为平地；steady 中的平地技能和 sequence 从 `slope_outward_up/l0` 出生，保持原地图几何不变。序列运动中可以进入相邻复杂地形，并非整段限制在平地。最大回合仍为 20 秒，摔倒或整图出界仍会提前结束。当前低速跟踪、运动门限与最低进展规则见下节，覆盖所有地形。

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-LPACRL-Rough-Compact-Rough \
  --env.scene.num-envs 1024 --agent.max-iterations 1000 --agent.seed 42 \
  --agent.run-name compact_rough_v1
```

新预设为 230 个任务及 teacher 组，默认每组 8 次离线任务评估，共 1840 次。评估保留默认 `--protocol checkpoint` 以恢复新任务表；`--protocol current` 仍使用原 Rough 的 832 组。独立 deployment v2 序列保持现有定义，1.0 指令由新任务表覆盖。新旧 Compact 都使用 v8 状态，但任务和地图指纹不同，切换时使用 `--warm-start`，不能完整续训旧课程。1024 环境的四类槽位仍为 358/256/307/103；任务增多后每组样本减少，应结合课程日志检查实际样本数与基线年龄。

### 低速跟踪与统一运动门限

当前 LP-ACRL 的 Flat / Rough、Compact 系列及 Random-Uniform 使用同一 `motion_tracking` 配置，规则不按出生族或实际地形筛选：楼梯、粗糙地面、坡地和平地全部生效。A0–A3 保留历史配置。机器人 PD、限矩、随机化和任务表不因本规则改变；这些阈值是训练实验参数，尚未证明提高策略性能。

| 内容 | 当前规则 |
| --- | --- |
| 站立 / 运动 | `norm(vx,vy) > 0.05 m/s` 或 `abs(wz) > 0.05 rad/s` 为运动；等于门限仍在站立死区 |
| 使用相同门限的模块 | 步态、腾空、抬脚（若启用）、滑移、落地、站立惩罚、姿态的站立/行走选择，以及 actor/critic/estimator 相位观测 |
| 平移跟踪误差尺度 σ | 指令合速度 ≤0.1 m/s 时为 0.05 m/s；在 0.1～0.5 m/s 间 smoothstep 插值；≥0.5 m/s 恢复原 0.5 m/s。尺度只取决于指令，零平移与纯转向下也使用低速尺度 |
| 平移最低进展 | 指令合速度 ≤0.2 m/s 时要求沿指令方向达到 70%；0.2～0.5 m/s 平滑回到原 20%；保留 0.02 m/s 下限 |
| 转向最低进展 | 保留原 20% 及 0.02 rad/s 下限，仅激活门限统一为独立的 0.05 rad/s |
| 惩罚时序 | 0.8 s 启动/切换宽限、0.3 s 平滑、持续不足 0.4 s 后处罚；默认不启用停滞终止。由站立进入平移或转向也重新获得宽限，包含小幅及对角启动 |

因此指令 vx=0.1、实际 vx=0.021 m/s 时，线速度核从约 0.9753 降到 0.0824；角速度正确且完整存活时，teacher 的 2:1 跟踪分约从 0.9836 降到 0.3883。过宽限并持续不足后，平移停滞代价约为 0.7；零指令不触发进展惩罚。此处是公式/机制结果，不是性能提升结论。teacher 仍不重复计入停滞扣分，物理失败的完整回合预算折扣保持。

新评分身份为 `speed_adaptive_linear_angular_failure_budget_v4`，课程结构仍按预设使用 v7 / v8。任务、奖励、环境和课程指纹已经改变；旧模型进入新规则须使用 `--warm-start`，不能完整续训旧 teacher 基线。`--protocol checkpoint` 离线评估及 `scripts.play` 回放会按保存快照恢复旧规则，缺少新字段的旧模型不会静默套用新门限；回放仍校验恢复结果与检查点保存的任务/地图指纹。

新参数位于 [中央配置](../src/config/ri_4438/ri_4438_him_lpacrl_config.py) 的 `motion_tracking`。需要原规则对照时，将该节 `enabled=False`；程序化构造也可使用 `make_lpacrl_cfg(..., motion_tracking_enabled=False)`，同时恢复原门限、跟踪核和进展配置。与新规则的对照应固定初始模型、种子、环境数、更新预算及评估协议。

ONNX 接口仍为 282 维历史输入到 12 维动作，但相位由部署端在图外计算。新导出文件写入 `him_motion_tracking` 和 `him_phase_motion_rule` 元数据；使用新模型时，部署侧相位必须采用相同的独立平移/转向门限。本次没有修改实机部署程序。

例如使用已下载的 Compact-Rough 模型初始化新的全地形训练：

```bash
uv run --no-sync python -m scripts.train Ri-4438-HIM-LPACRL-Rough-Compact-Rough \
  --env.scene.num-envs 1024 --agent.max-iterations 1000 --agent.seed 42 \
  --warm-start logs/rsl_rl/ri_4438_him_lpacrl/2026-10-24_10-43-18/model_1200.pt \
  --agent.run-name compact_rough_low_speed_v4
```

### 原任务协议与训练诊断

训练的 `curriculum.jsonl` 每个课程阶段（默认 50 次更新）保存实际技能槽位、按族累计环境步、停滞/惩罚步数及命令切换指标；TensorBoard 每次更新同时记录按族步数占比和平均停滞代价。使用这些数据判断样本覆盖和切换行为。代码与短跑验证记录见[设计审查第十八章](RI-4438-HIM-LPACRL设计审查与训练退化分析.md#十八首批实现wf-022)。

以下是保留的原任务协议：

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

评估 Compact 系列模型时保留默认 `--protocol checkpoint`，即可还原原 Compact 的 39 组或 Compact-Rough 的 230 组及保存的车体系命令；`--protocol current` 仍指原 Rough 832 组。Compact 系列不接受 `--task-mode` 覆盖；单一固定命令使用 `--command`。评估与回放均不启用训练停滞惩罚/终止，因此能完整观察“站着不动”。平地 deployment 序列已升级为 v2，共 10 条，包括新增 0.1 / 0.2 / 0.3 m/s 启停换向；新协议将纵横向显著变化、运动命令门限及绝对跟踪容差设为不超过 0.05 m/s，避免把低速静止判成切换成功。v1 与 v2 结果不可直接混合比较，应让候选模型重跑同一套 v2 协议。

原 Compact 的独立楼梯地块尚未实现连通地图的完整接触出口判据；其楼梯跨阶结果仍是旧式代理指标，不能等同于所有脚完整走出楼梯。训练诊断对不适用的连通地图通过率不输出伪零值。Compact-Rough 使用原 Rough 的完整接触出口判据。

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
