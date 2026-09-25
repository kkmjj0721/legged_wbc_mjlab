# RI-4438 HIM 数值保护与续训

本次修改针对极端动作进入 `last_action`、随后长期污染累计归一化统计的问题。参考目录是 `/home/sunteng/projects/HIMLoco`。保留当前 47 维单帧、6 帧历史、actor/critic 网络及 empirical normalization；没有改为原版的 45 维观测或固定缩放训练。

## 动作与观测约定

沿用原版 HIMLoco 的关键顺序：PPO 保存原始高斯采样动作及其 log probability；环境入口对动作做非原地裁剪，`last_action` 和动作历史使用环境收到的裁剪结果。之后才经过关节缩放、默认姿态偏置以及关节目标角限位。

原版动作、观测裁剪阈值均为 100。此工程采用以下初始值，集中在 `src/config/ri_4438/numerics.py`：

| 参数 | 值 | 作用 |
|---|---:|---|
| `ACTION_CLIP` | 10 | 原始策略动作裁剪；不是 10 rad 的关节限位 |
| `OBSERVATION_CLIP` | 100 | 各观测项在噪声之后、scale/历史之前的边界 |
| `RAW_ACTION_ABORT` | 20 | 原始动作 sample 或分布参数绝对值超过此值，停训 |
| `RAW_OBSERVATION_ABORT` | 1000 | 裁剪前原始观测绝对值超过此值，停训 |
| `ACTION_OBSERVATION_SLICE` | 35:47 | 每帧 `last_action` 的位置 |

这些是数值保护的初始阈值，需要结合正常训练分布验证。当前工程的关节速度等观测与原版固定缩放不同，因此观测阈值 100 不代表相同物理边界。用户已有的动作 scale=0.305、hip reduction=0.5、关节目标角 clip，以及奖励/随机化设置均保留。

所有 actor/critic 观测项和独立 estimator 监督组都有边界。观测函数在裁剪之前记录 NaN、Inf 和过大有限值，终止状态也检查；错误标志和原始证据不会被随后的 reset/正常观测覆盖。reset 钩子中不使用分布式 collective，由 runner 在所有 worker 的固定位置统一检查。

## 异常处理与统计保护

- 任一 worker 检出异常，所有 worker 在相同检查点抛出 `FloatingPointError`。不会让一张卡跳过更新、其他卡继续 all-reduce。
- rollout、PPO/HIM loss、梯度范数、更新后的参数和 Adam 状态均检查有限性；loss/ratio/梯度范数还有 `1e8` 的异常门限。
- 故障数据写入训练日志目录下 `numerics/incident_*_rank*.pt`，记录阶段、iteration/step、当前坏值/坐标和最近 8 步所有环境的观测与动作。不会走正常 checkpoint/ONNX 保存流程。
- 这是停训诊断机制，不会回滚已经完成的其他 mini-batch/estimator 更新；恢复需要重新启动并加载正常 checkpoint。对进程被系统杀死、CUDA 驱动崩溃或不属于数值检查的任意 Python 异常，不承诺自动协调恢复。
- Normalizer 用 float64 计算局部和全局候选统计，按样本数加权合并；空局部 batch 仍参与 collective。输入或候选统计非法时，不修改 `count/mean/var/std`。正常统计更新仍在 PPO 更新结束后，避免改变本轮 old/new log probability 的尺度关系。
- 保留累计统计和旧 checkpoint 的 tensor key，不自动清零受到污染的 normalizer。

TensorBoard 增加 `Loss/numerics/raw_action_max`、`Loss/numerics/action_clip_fraction`、`Loss/numerics/last_action_std_max`。保护检查与诊断缓存有额外计算、同步和显存开销；实际多 GPU 吞吐需要在目标机器上测量。

## 加载、导出与续训

加载、开始训练、保存和导出前检查模型、优化器和归一化状态。`last_action` 的 mean 绝对值或 std 超过 10 会拒绝使用。旧 checkpoint 缺少数值约定元数据时给出提示，并进行上述检查；新 checkpoint 的数值约定必须与当前配置相同。

本地实际验证：`2026-09-25_00-24-15/model_11500.pt` 能加载，`model_16300.pt` 因 `last_action` std 异常被拒绝。旧数据已被污染时，单独更新代码无法修复该模型；不要从 16300 继续稀释统计，也不要只清空它的 normalizer。建议从正常的完整 11500 checkpoint 另开日志目录续训，再评估前进、后退、横移和转向。

JIT/ONNX 包含与训练一致的输入裁剪和确定性输出裁剪。导出输入仍是 current-first 的 282 维历史，动作输出仍是 12 维；部署端应把 ONNX 输出的动作写入上一动作历史，再进行关节缩放/偏置。ONNX 元数据记录裁剪阈值和历史约定；使用临时文件完成导出、元数据和模型校验后再替换 `policy.onnx`，导出失败保留原文件。

## 修改文件

| 文件 | 改动 |
|---|---|
| `src/config/ri_4438/numerics.py` | 新增阈值与动作切片约定 |
| `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py` | 启用环境动作裁剪、actor 裁剪和训练保护 |
| `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py` | 接入观测边界，estimator 继承边界 |
| `src/tasks/locomotion/ri_4438_him/mdp/numerics.py` | 新增裁剪前诊断和跨 reset 的故障证据 |
| `src/tasks/locomotion/ri_4438_him/mdp/terminal_observations.py` | 识别包装后的原始观测函数，保留终止状态快速刷新 |
| `rsl_rl/utils/numerics.py` | 新增多 worker 检查、故障现场保存 |
| `rsl_rl/runners/him_on_policy_runner.py` | 接入训练检查和数值指标 |
| `rsl_rl/algorithms/him_ppo.py` | PPO/rollout 检查，接入 estimator 检查 |
| `rsl_rl/modules/him_estimator.py` | estimator loss/梯度/优化器检查，兼容空样本 worker |
| `rsl_rl/modules/normalization.py` | 候选统计校验后提交，支持空 batch，保留 state 格式 |
| `rsl_rl/models/him_actor_model.py` | 统一 Python/JIT/ONNX 输入输出边界 |
| `src/tasks/locomotion/ri_4438_him/rl/runner.py` | checkpoint 校验、约定元数据、原子替换 ONNX |
| `tests/test_him_numerics.py` | 新增异常注入、两 worker、导出和 CUDA 短训练测试 |
| 本文档 | 修改与续训说明 |

HIM 通用组件的新训练保护由 RI-4438 配置启用，其他任务默认不启用；normalizer 的稳定统计实现为共享改动。用户原先修改的 `ri_4438_config.py` 和 `ri_4438_him_env_cfg.py` 未在本次额外修改。

验证命令：

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_him*.py'
```

包含 CPU/Gloo 两进程异常同步、空 batch 的统计/estimator 更新，Python/JIT/ONNX 一致性，PPO/estimator 坏梯度阻断，CUDA 的终止状态与 8 环境两轮短训练、checkpoint 重新加载和导出失败保护。测试产物写入临时目录，不更新部署用模型。当前只有一张可见 GPU，多 GPU/NCCL 尚未实测；短测试不等于已验证长时间训练稳定或全方向运动表现。
