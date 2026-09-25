# RI-4438 HIM 性能优化与验证

日期：2026-09-25。环境：RTX 4060 8 GiB、mjlab 1.6.0、MuJoCo Warp 3.11.0、Warp 1.17.0。

## 本次修改

- RI-4438 rough 当前使用 `nconmax=48`、`ccd_iterations=50`、`contact_sensor_maxmatch=64`、`njmax=600`、`broadphase="nxn"`。首轮曾设成 `64/100/128` 并保留 `njmax=1500`，复核实际训练日志后已修正，见下文。flat 保持 `nconmax=None`、`njmax=300`。
- 首轮曾为 `discrete_obstacles` 开启 0.01 m 高度量化合并，固定地形种子 0 时地形 geom 从 7504 降为 7012。第三轮之前用户另行调整了台阶高度和合并参数，第三轮对比统一使用这份新地形，未再修改地形生成参数。
- 第四轮将 `discrete_obstacles` 改为 mjlab 内置 `HfDiscreteObstaclesTerrainCfg`，保留 7 种楼梯配置；这是地形分布发生变化的优化，详见文末独立对照。
- HIM storage 只保存 estimator 使用的 next-observation groups。Go2 HIM 保存 next critic；RI-4438 HIM 现保存 50 维 next estimator，actor/critic 网络输入及模型接口保持原样。
- 公共手动复位路径取消中间整批 clone 和重复 done 掩码写回；随后只原地更新复位行。RI-4438 HIM 默认改用自动复位及真实终止目标 recorder，省掉复位前的完整碰撞计算和射线检测。
- 自动复位时不再把复位后的整批观测伪装成 terminal observations；保留环境提供的真实 terminal extras，缺失时排除 done 行的 estimator 样本。

本次只调整 RI-4438 的仿真和地形配置；公共 HIM 算法、storage、runner 改动也适用于 Go2 HIM。

## 首轮碰撞筛选与环境短测（64/100/128、njmax=1500）

各方案使用独立进程。固定地形种子 0，256 环境，动作是固定随机种子的 `0.3 * randn`。计时包含 `env.step()` 和必要的手动 reset，不包含策略推理或 PPO 更新。NXN 每组 100 步，丢弃第一组预热，记录后三组平均单步耗时的中位数。

| 方案 | 单步耗时 | 进程 GPU 显存（运行后） | overflow |
|---|---:|---:|---|
| 仓库旧默认容量和地形，NXN | 24.20 ms | 3692 MiB | 0 |
| 首轮容量、1 cm 方块合并，NXN | 21.18 ms | 1644 MiB | 0 |

这里的显存是 `nvidia-smi` 查询到的进程占用，不是峰值，也不是完整训练显存。耗时下降约 12.5%，不能据此推算远程 4090 的训练迭代耗时。**该基准使用的是仓库旧默认值，不是用户已通过启动参数缩小容量后的运行配置，不能用于说明相对用户实际运行的收益。**

- `sap_tile` 无法加载排序 kernel：PTX 报告所需共享内存 `0xe040` 超过允许的 `0xc000`。
- `sap_segmented` 在同一场景的短测中约为 475.40 ms/步（每组 5 步，第一组预热，后三组取中位数），显著慢于 NXN，因此保留 NXN。
- 测量逐步在 GPU 上累计 `Data.overflow`，在复位清除标记之前收集；测试结束统一检查，不仅依赖警告输出。

1024 环境补测：首轮配置运行 400 个 policy steps，累计复位 126 个环境，观测和奖励均有限，累计 overflow 为零；单步中位数 67.72 ms，运行后进程显存 4358 MiB。该测试同样仅包含环境仿真与复位，不包含训练 storage 和模型更新。

## 第二轮复核：实际运行已经缩小过容量

本地 `2026-09-24_11-22-52/params/env.yaml` 已经使用 `nconmax=48`、`ccd_iterations=50`、`contact_sensor_maxmatch=64`、`njmax=600`，1024 环境，nan guard 关闭。首轮修改的容量反而大于这次实际运行。`2026-09-25_01-03-20` 的日志确认首轮参数已经生效，其 3 轮平均 collection 为 9.502 s、learning 为 0.426 s；不能把改善不明显归因于参数未生效。

为隔离容量影响，以下均使用第二轮当时的 HIM runner/storage 和 1 cm 方块合并，RTX 4060、1024 环境、环境/地形种子 42、100 步 rollout、5 epochs × 4 minibatches、20 s episode、初始 episode 长度随机化，关闭 nan guard。各方案独立进程完整训练 4 轮，排除首轮，取余下 3 轮 collection 中位数。不开 viewer、不写训练日志/checkpoint；计时含相同的 CUDA event 分段测量和溢出监控开销。

| 方案 | collection 中位数 | 环境创建后显存 | 训练后进程显存 | overflow | 最大 nefc |
|---|---:|---:|---:|---:|---:|
| 首轮配置：64/100/128，njmax=1500 | 9.780 s | 4322 MiB | 5648 MiB | 0 | 84 |
| 本轮修正：48/50/64，njmax=600 | 9.507 s | 3682 MiB | 5010 MiB | 0 | 76 |
| 修正容量，再增加 AABB 筛选 | 9.951 s | 3682 MiB | 5010 MiB | 0 | 76 |

容量修正降低训练后进程显存 638 MiB（约 11.3%），collection 缩短约 2.8%；**它主要修正了首轮容量相对用户实跑配置变大的问题，不能声称解决了训练速度瓶颈。** AABB 筛选没有采用。

首轮配置的后三轮中，显式 `env.reset()` 合计 8.037 s，占 collection 的约 27.3%；`sim.sense()` 合计 5.647 s，占约 19.2%，包含 step 与 reset 两次路径。这里是包含子调用的时间，reset 和 sense 不可相加。容量修正后物理推进、全环境复位刷新和射线感知仍然存在。

临时 heightfield 实验只替换 `discrete_obstacles`，保留 7 种楼梯、台阶尺寸与 10 级课程。采用 100 个 0.4 m 障碍、0.01–0.12 m 高度范围、0.1 m 水平网格、0.005 m 高度分辨率、1 m 中心平台。此方案会改变障碍密度、正负高度分布和边缘形状，**并不等价于原方块地形，尚未写入默认配置**。

| heightfield 实验 | collection 三轮样本 | 中位数 | 训练后进程显存 | overflow |
|---|---|---:|---:|---:|
| 首测 | 7.454 / 7.264 / 13.193 s | 7.454 s | 3860 MiB | 0 |
| 独立进程复测 | 7.057 / 6.877 / 6.804 s | 6.877 s | 3860 MiB | 0 |

总 geom 从 7002 减为 4307（地形种子 42），训练后进程显存比本轮容量修正再少 1150 MiB。首测出现一次异常高耗时，原因未确定，保留原始样本；复测支持约 6.9–7.5 s 的 collection 中位数范围，不能据此保证长期训练吞吐或收敛质量。

本轮重新运行 10 项 CPU 回归测试，全部通过；容量修正的 GPU 训练为完整 4 轮、400 个 policy steps，各物理子步和 forward 后累计溢出标记，最大 nefc 为 76，损失有限。

测试脚本和原始输出保存在 `/tmp/him-perf-followup-FFvXa9/`；`bench_train.py --mode current/prior/aabb/hf --profile` 对应各组。应顺序运行独立进程，避免相互占用 GPU。显存仍是运行后进程占用，并非峰值；所有结果只代表短程本机测试。

## 首轮正确性审查

CPU 回归测试：

```bash
.venv/bin/python -B -m unittest discover -s tests -p test_him_rollout.py -v
```

10 项测试覆盖：storage 对输入的独立拷贝、真实终止观测、自动复位、完整/仅 done 行 terminal batch、缺失或错误 terminal 数据、非默认 estimator group、部分复位的观测合并，以及 PPO/HIM 更新一致性。固定输入与随机种子时，完整 next-observation storage 和精简 storage 的损失、更新后模型参数完全一致。

另外用真实 GPU 环境各运行 3 轮 HIM 训练，每轮 100 个 policy steps，保持 5 epochs × 4 minibatches。为频繁触发复位，将测试 episode 缩短到 0.2 s，并随机化初始 episode 长度：

| 任务 | 环境数 | 累计复位环境数 | 结果 |
|---|---:|---:|---|
| RI-4438 HIM rough | 64 | 1920 | 通过 |
| RI-4438 HIM flat | 32 | 960 | 通过 |
| Go2 HIM rough | 32 | 960 | 通过 |

每个任务检查了前 10 步 estimator 观测与修改前 resolver 的逐元素一致性、5 次部分复位后未结束环境的观测/历史/延迟状态、storage 中终止观测不被复位覆盖、损失有限、overflow 为零，以及内存中算法 checkpoint 状态导出/载入后的确定性动作一致性。

## 第三轮：真实终止目标与自动复位

用户 `2026-09-25_01-18-22` 日志中容量修正已生效，5 轮 collection 为 9.103 / 9.088 / 9.085 / 9.339 / 9.288 s，均值 9.181 s。耗时改善小来自主要计算路径未改变。

本轮修改 RI-4438 rough/flat 默认复位流程：

- 默认 `auto_reset=True`；`TerminalEstimatorRecorder` 在 reset 前调用 MuJoCo Warp 的 `kinematics → com_pos → com_vel → sensor_vel`，使姿态、速度传感器与最新 qpos/qvel 一致，然后更新终止环境的命令并保存目标。
- 新增无噪声、无延迟、无历史的 `estimator` 观测组，对应 critic 的前 50 维。沿用原有 velocity/target 切片，终止监督不再依赖高度扫描、接触力等未使用的 critic 特征。
- 每步仍执行框架完整的复位后 `forward()`、`sense()` 和 actor/critic 观测更新。对不在精简刷新支持范围内的自定义观测函数或传感器，recorder 回退到完整刷新。
- 模型数组扩展引起模拟 CUDA graph 重建时，终止刷新 graph 也重新捕获，避免继续读取旧 GPU 数组。
- 手动复位仍可通过 `--env.auto-reset False` 使用；Go2 HIM 的复位方式保持原配置。

同一份用户新地形（7 种台阶高度范围 0.05–0.15 m、方块合并采用库默认 0.05 m，种子 42，总 geom=6050）上，独立进程、1024 环境、每轮 100 步、完整训练 4 轮、排除首轮：

| 复位方案 | collection 三轮样本 | 中位数 | 训练后进程显存 | overflow |
|---|---|---:|---:|---:|
| 原手动复位、next critic | 8.497 / 8.515 / 8.429 s | 8.497 s | 4594 MiB | 0 |
| 自动复位、精简终止刷新原型、next estimator | 6.796 / 6.655 / 6.631 s | 6.655 s | 4506 MiB | 0 |

相同地形下 collection 缩短约 21.7%；显存只减少 88 MiB，**本轮主要优化速度**。从 next critic 244 维切换到 next estimator 50 维后，考虑新增的当前 estimator 组，1024 环境 × 100 步的 storage 净减少 56.25 MiB。不能把第三轮与第二轮的全部差值算作复位优化收益。

回归验证：

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_him*.py' -v
```

19 项测试通过（11 项 CPU、8 项 CUDA）。新增检查覆盖 rough/flat 两个任务：精简观测与 critic 前 50 维逐元素一致；施加速度扰动后轻量刷新与完整 forward 的目标一致（atol/rtol=2e-6）；部分自动复位只调用一次完整 forward/sense，未结束环境的历史/延迟只推进一次，命令计时器正确，storage 保留终止值；终止 extras 不泄漏到后续步骤；模拟 graph 重建后终止 graph 更新。固定输入时，完整 critic 与精简 estimator 两种监督产生完全相同的 PPO/HIM 损失和更新后参数，并通过严格模型状态载入。

最终通过正式入口运行，而非只测环境：

```bash
.venv/bin/python -m scripts.train Ri-4438-HIM-Rough --env.scene.num-envs 1024 \
  --agent.max-iterations 5 --agent.run-name reset_perf_review
```

日志目录：`logs/rsl_rl/ri_4438_him/2026-09-25_01-32-18_reset_perf_review`。5 轮全部完成，保存 checkpoint 和 ONNX。排除第一轮后，collection 均值 6.672 s，learning 均值 0.395 s，合计 **7.068 s/轮**，平均 14488 FPS。保存的配置确认 `auto_reset=true`、`him_terminal` recorder 和 `estimator_obs_groups=[estimator]` 已实际生效，原来的训练命令无需添加优化开关。

前两轮保留了手动复位，仅减少拷贝；本轮采用独立终止目标和运动学刷新后才真正省去重复碰撞和感知。自动复位会改变随机采样顺序，不保证与手动复位产生完全相同的训练轨迹。收益也取决于复位频率。实验脚本与原始输出位于 `/tmp/him-reset-perf-fQnrEk/`。

以上是短程正确性和性能检查，尚未验证长程收敛、最终楼梯通过率或远程 2048 环境训练的显存与速度。容量改变仍需在实际训练中监控接触/CCD/传感器匹配溢出。

## 第四轮：随机障碍使用高度场

采用 [mjlab v1.6.0 内置高度场障碍物](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/terrains/heightfield_terrains.py)；只修改 `ri_4438_him_env_cfg.py` 的 `discrete_obstacles` 配置，没有修改依赖库。

- 原配置为 `BoxRandomGridTerrainCfg`，0.4 m 网格，启用库默认 0.05 m 高度量化合并。
- 新配置为 `HfDiscreteObstaclesTerrainCfg`：每块地形一个高度场，0.4 m 方形障碍，100 次随机放置（允许重叠），水平分辨率 0.1 m、垂直量化 0.005 m、中央平台 1 m、平坦边界 0.4 m。
- 保留采样权重 0.2 和高度幅值课程 0.01–0.12 m。`choice` 模式在 `[-h, -h/2, h/2, h]` 中取样，包含凹坑和凸起；中央平台现在位于 z=0，生成器的出生高度也同步为 0，机器人仍按出生点加原来的机身高度偏移复位。
- 保留楼梯、其余地形配置、环境数量、碰撞容量、感知、奖励、课程、复位流程及网络输入。高度场是三角插值表面，边缘有坡面过渡；其稀疏分布与旧的密集网格不同。共享随机数流也会改变后续随机粗糙地形的具体样本，不能视作完全相同场景的等价优化。

对照使用同一台 RTX 4060、同一份当前代码、种子 42、1024 环境、每轮 100 步及完整 PPO/HIM 更新。独立进程各运行 5 轮，排除首轮预热，取其余 4 轮中位数。旧方案通过临时脚本恢复原 `BoxRandomGridTerrainCfg`；两者都已启用第三轮的自动复位优化。

| 随机障碍 | 总 geom 数 | 采样耗时中位数 | 采样 + 更新中位数 | 训练后进程显存 | 累计 overflow |
|---|---:|---:|---:|---:|---|
| 原合并方块网格 | 6050 | 6.293 s | 6.676 s | 4506 MiB | 0 |
| 高度场障碍物 | 4307 | 5.056 s | 5.429 s | 3772 MiB | 0 |

采样耗时下降约 **19.7%**，整轮下降约 **18.7%**，训练后进程显存减少 **734 MiB（16.3%）**。显存来自 `nvidia-smi` 的训练后进程占用，不是峰值；耗时包含相同的有限值和溢出检查开销。两组观测、奖励和损失均为有限值，最大 `nefc` 均为 80。原始采样耗时（不含首轮）为：方块 6.251 / 6.283 / 6.304 / 6.420 s，高度场 5.204 / 5.089 / 4.988 / 5.023 s。

审查与验证：

- 对照修改前后的配置对象，除随机障碍子配置以及每次创建时独立生成的默认工厂函数对象外，其余配置一致。
- 对难度 0 / 0.5 / 1 单独生成并编译地形：每块均为一个 hfield geom，高度有限且分别在 ±0.01 / ±0.065 / ±0.12 m 内；中心和四角射线确认平台、边界及出生高度正确。
- 现有 HIM 回归测试 19 项全部通过，包括 rough/flat 的终止监督、部分自动复位、历史和延迟、checkpoint 兼容性检查。

正式入口另外运行 5 轮：

```bash
.venv/bin/python -B -m scripts.train Ri-4438-HIM-Rough --env.scene.num-envs 1024 \
  --agent.max-iterations 5 --agent.run-name heightfield_perf_review
```

日志目录：`logs/rsl_rl/ri_4438_him/2026-09-25_10-35-21_heightfield_perf_review`。排除首轮后，collection 均值 5.159 s、learning 均值 0.384 s，合计 **5.543 s/轮**。5 轮全部完成，checkpoint 和 ONNX 保存成功；导出的环境配置确认高度场参数已生效，原训练命令无需额外开关。

基准脚本及原始结果位于 `/tmp/him-heightfield-jbwe1rsc/`，`bench.py --box` 测原方块，默认测当前高度场；运行时使用 `PYTHONPATH=. .venv/bin/python -B`，并传入 `--output` 指定 JSON 文件。

这些短测确认运行正确性和性能变化，尚未验证新的障碍分布对长程收敛和通过率的影响。
