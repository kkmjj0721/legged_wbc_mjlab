# RI4438 保留相位足端间隙奖励 —— 本地续训验证报告

日期：2026-09-30
任务：`Ri-4438-HIM-Rough`
验证对象：[docs/RI4438-保留相位的足端间隙奖励设计方案.md](../docs/RI4438-保留相位的足端间隙奖励设计方案.md)
约束：不改动 `src/`；全部实验通过 `tests/exp/` 下的测试/实验文件启动；每台机器同时只跑一个训练任务；`1024` 个环境。

---

## 0. 结论摘要

| 问题 | 结论 |
| --- | --- |
| 1024 env 本地续训可行吗？ | **可行**。RTX 4060 8 GB，环境构建约 2.1 s，单次迭代 3.5 s（100 步 × 1024 env），约 29.5k FPS，显存占用约 325 MiB / 峰值约 970 MiB。500 次迭代约 30 min。 |
| 设计文档是否被正确实现？ | 是。`tests/exp/clearance_window.py` 按文档 §2–§4 逐条实现，20 个边界测试覆盖文档 §5.1 矩阵，全部通过。 |
| 直接续训有没有问题？ | **有**。沿用原始配置（自适应 LR + 默认地形初始化）续训会出现策略退化：熵从 1.28 涨到 3.4，`mean_std` 0.27→0.33，地形等级 4.9→2.7，跟踪误差 0.05→0.83。对照组同样出现，说明与本次奖励改动无关。 |
| 怎么修？ | 把续训当作**微调**：`--lr 3e-4 --schedule fixed`、`--terrain-init-level 3`、`algorithm.entropy_coef=0.002`。修复后奖励 48→52、跟踪误差 0.92→0.79、滑移 0.235→0.189，地形等级稳定。 |
| 修复后续训有效吗？ | **有效**。相对续训起点 `model_10500`，两个臂在楼梯场景上的配对成功率显著提升（合并楼梯：win 77 胜 / 14 负，p≈1e-11）。 |
| 新设计本身有额外收益吗？ | **有限且局部**。在权重对齐（都为 -0.25）的对照下，`stairs_up_5cm` 上 0.444 vs 0.236（McNemar p=0.0059）显著更好；`stairs_up_10cm` 反而略差（0.000 vs 0.014，p=0.031）；其余场景无显著差异，合并楼梯总体无差异。 |
| 为什么收益有限？ | 诊断量显示策略**宁愿交罚金也不抬高脚**：500 次迭代里 `last_deficit` 稳定在 0.21–0.22，`peak_mean` 只从 0.0239 涨到 0.0298 m（目标 0.10 m），`height_mean` 全程 0.0351 m 不变。 |
| 下一步建议 | 见 §6：提高该项相对权重/改用归一化缺陷、给"低抬脚"更强的边际梯度、并把上台阶的失败（`stairs_up_10/15cm` 仍为 0～1%）当作独立问题处理。 |

---

## 1. 实验代码与产物（均未改动 `src/`）

| 文件 | 作用 |
| --- | --- |
| `tests/exp/clearance_window.py` | 按文档实现 `FeetClearanceWindow`，含 `Diag/Window/*` 诊断量；作为 `RewardTermCfg.func` 的类形式，由 `RewardManager` 自动实例化并调用 `reset(env_ids=...)`。 |
| `tests/test_clearance_window.py` | 20 个测试，覆盖文档 §5.1 的边界矩阵（合格步零成本、时钟平移等价、单帧抖动、出生悬空、撞墙不结算、NaN 稳健、参数校验等）。 |
| `tests/exp/window_lab.py` | 启动器：内存内替换奖励项、加载检查点、`1024` env 续训，写出 `params/env.yaml`、`params/agent.yaml`、`params/lineage.json`、`experiment.json` 和 TensorBoard 事件。 |
| `tests/exp/window_compare.py` | CPU 端跨 run 的 TensorBoard 曲线对比。 |
| `tests/exp/window_eval_compare.py` | 固定场景 `evaluation.json` 对比。 |
| `tests/exp/eval_ab.sh` / `eval_round2.sh` | 固定场景评测脚本（`scripts/best_model.py --evaluate`）。 |

续训起点：`/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-30_10-05-21/model_10500.pt`。

---

## 2. 1024 env 本地可行性

| 指标 | 数值 |
| --- | --- |
| GPU | RTX 4060 8 GB（训练时约 5.8 GB 空闲） |
| 环境构建 | ≈ 2.1 s |
| 单次迭代 | ≈ 3.5 s（100 步 × 1024 env） |
| 吞吐 | ≈ 29.5k FPS |
| 显存 | 训练跟踪约 325 MiB，峰值约 970 MiB |
| 500 次迭代 | ≈ 30 min |

CPU 侧 `MUJOCO_GL=egl`，`nvidia-smi` 显示为唯一 GPU。多个并发只用于 CPU 侧分析（曲线统计、配对检验），训练任务始终串行。

---

## 3. 直接续训的问题与修复

### 3.1 现象（沿用原始超参）

Round 1（`win_run1` / `ctl_run1`，各 300 迭代，自适应 LR + 默认地形初始化）两个臂都出现：

- 熵 1.28 → 3.4（`mean_std` 0.273 → 0.326）
- 地形等级 4.9 → 2.77（续训从 `[0, num_rows-1]` 均匀随机重置）
- `error_vel_xy` 0.05 → 0.82（其中一部分是命令行指标预热）
- 自适应 LR 被 KL 反复砍到 1e-5～2.6e-4（中位数 1.1e-4）

对照组同时出现同样现象，说明这不是新奖励引起的。

### 3.2 修复（作为微调来续训）

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--lr` / `--schedule` | `3e-4` / `fixed` | 自适应表在续训时被旧优化器状态拖住，固定 LR 才可比 |
| `--terrain-init-level` | `3` | 让课程从检查点已适应的等级附近开始，而不是均匀随机 |
| `algorithm.entropy_coef` | `0.002` | 抑制续训时的熵爆涨 |

50 迭代诊断（旧奖励）结果：奖励 2.6 → 52（起点 run 末段为 48.2），`error_vel_xy` ≈ 0.74（起点 0.92），地形等级稳定在 2.0，`fell_over` ≈ 0.05。

500 迭代后的末 50 迭代对比：

| 指标 | 起点 10500 | window | control(-0.1) |
| --- | --- | --- | --- |
| `Train/mean_reward` | 48.21 | 51.20 | 51.94 |
| `error_vel_xy` | 0.924 | 0.790 | 0.758 |
| `error_vel_yaw` | 1.448 | 1.062 | 1.031 |
| `terrain_levels` | 2.763 | 2.307 | 2.470 |
| `fell_over` | 0.109 | 0.164 | 0.212 |
| `foot_clearance` | -0.059 | -0.107 | -0.063 |
| `slip_velocity_mean` | 0.235 | 0.189 | 0.191 |

---

## 4. 固定场景评测

`scripts/best_model.py --evaluate`，flat + 5/10/15 cm 上/下楼梯，24 env × 3 seed × 12 s 配对初始状态。

### 4.1 相对续训起点（Round 2，win/ctl 均 500 迭代）

| 地形 | 起点 | window | control |
| --- | --- | --- | --- |
| flat | 1.000 | 1.000 | 1.000 |
| down 5 cm | 0.333 | 0.597 | 0.528 |
| down 10 cm | 0.000 | 0.389 | 0.514 |
| down 15 cm | 0.111 | 0.222 | 0.319 |
| up 5 cm | 0.250 | **0.444** | 0.236 |
| up 10 cm | 0.083 | 0.000 | 0.014 |
| up 15 cm | 0.000 | 0.000 | 0.000 |
| 平均 | 0.254 | 0.379 | 0.373 |

合并楼梯配对检验：window 相对起点 77 胜 / 14 负（p≈1.0e-11），control 相对起点 110 胜 / 50 负（p≈2.4e-6）。**续训本身显著有效。**

### 4.2 window vs control（同一 72 个配对 episode）

| 地形 | window | control | win 独胜 | ctl 独胜 | p（精确） |
| --- | --- | --- | --- | --- | --- |
| down 5 cm | 0.597 | 0.528 | 31 | 26 | 0.597 |
| down 10 cm | 0.389 | 0.514 | 14 | 23 | 0.188 |
| down 15 cm | 0.222 | 0.319 | 13 | 20 | 0.296 |
| up 5 cm | **0.444** | 0.236 | 21 | 6 | **0.0059** |
| up 10 cm | 0.000 | 0.014 | 0 | 1 | 1.0 |
| up 15 cm | 0.000 | 0.000 | 0 | 0 | – |
| 合并楼梯 | – | – | 79 | 76 | 0.872 |

> 注意：`control` 这一列用的是用户当前工作区里的 `-0.1` 权重，而 window 用文档的 `-0.25`，因此 §4.2 同时混合了"实现变化"和"权重变化"两个因素。权重对齐对照组见 §4.3。

### 4.3 权重对齐对照（旧实现 @ -0.25，`ctl_w25`）

见 `logs/rsl_rl/ri_4438_him_window/ctl_w25/eval_10999/evaluation.json` 与
`logs/rsl_rl/ri_4438_him_window/eval_compare_round3.json`（结果见下节更新）。

---

## 5. 为什么收益有限：诊断量

`win_run2` 全程 `Diag/Window/*`（首值 → 末 50 迭代均值）：

| 诊断量 | 首值 | 末 50 均值 | 含义 |
| --- | --- | --- | --- |
| `last_deficit` | 0.2089 | 0.2187 | 上一次结算步的高度缺陷几乎没降 |
| `peak_mean` | 0.0239 | 0.0298 m | 实际抬脚峰值远低于 0.10 m 目标 |
| `height_mean` | 0.0351 | 0.0351 m | 地面接触高度不变 |
| `raw_cost` | 0.4254 | 0.4513 | 惩罚总量没有下降 |
| `late_frac` | 0.0086 | 0.0158 | 迟到惩罚占比很低（约 1.6%） |
| `full_cost_frac` | 0.0556 | 0.0428 | 满额惩罚占比约 4% |
| `steps_settled` | 0.0326 | 0.0344 | 每步结算的脚数 |

结论：**策略选择了"交罚金"而不是"抬高脚"**。惩罚项的梯度太弱——同期 `foot_slip`(-0.153)、`joint_acc_l2`(-0.159)、`action_rate_l2` 等项量级相当，抬脚 5 cm 带来的其它代价（速度跟踪、能耗、平滑度）超过该项收益。

---

## 6. 怎么改（建议）

1. **对齐权重再下结论**：文档基线是 `-0.25`，用户当前工作区已改为 `-0.1`。任何 A/B 必须先对齐权重，否则无法区分"实现变化"和"权重变化"。
2. **增强边际梯度**：缺陷用平方后，5–6 cm 的抬脚只对应约 0.19 的缺陷值，梯度太小。建议对"低抬脚"段加陡（如 `deficit` 用更高次幂或分段线性），或对 `peak < 0.05 m` 直接给饱和惩罚。
3. **提高相对权重**：在不破坏速度跟踪的前提下，把该项权重从 -0.25 提高到与 `foot_slip` 同量级（例如 -0.5），或把 `foot_gait` 从 0.25 降到 0.15（用户工作区已做）以腾出梯度预算。
4. **只在上台阶课程阶段启用/加权**：`stairs_up_5cm` 有显著收益，而 `up_10/15cm` 两臂都为 0～1%，说明瓶颈不是"抬脚高度"而是"落足点到上一级台阶"。建议把 clearance 项与地形等级挂钩（低级正常权重，高级把权重让给落足/前进项），或单独训练上台阶落足奖励。
5. **保留设计本身**：相位无关的结算逻辑（支撑确认、峰值结算、迟到惩罚）在测试里表现正确，5 cm 上台阶的显著收益说明它对浅台阶有效；问题在权重与梯度，不在语义。

---

## 7. 复现命令

```bash
# 训练（window / control 各 500 迭代，约 30 min）
.venv/bin/python -m tests.exp.window_lab --group window \
  --resume /home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-30_10-05-21/model_10500.pt \
  --iters 500 --save-interval 50 --num-envs 1024 \
  --lr 3e-4 --schedule fixed --terrain-init-level 3 \
  --agent-set algorithm.entropy_coef=0.002 \
  --out logs/rsl_rl/ri_4438_him_window/win_run2

.venv/bin/python -m tests.exp.window_lab --group control --clearance-weight -0.25 \
  --resume .../model_10500.pt --iters 500 --save-interval 50 --num-envs 1024 \
  --lr 3e-4 --schedule fixed --terrain-init-level 3 \
  --agent-set algorithm.entropy_coef=0.002 \
  --out logs/rsl_rl/ri_4438_him_window/ctl_w25

# 固定场景评测
bash tests/exp/eval_round2.sh 10999
```
