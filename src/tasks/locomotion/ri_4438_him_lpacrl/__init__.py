"""RI-4438 HIM + LP 自动课程；训练和播放使用公共 scripts.train / scripts.play。

阅读顺序：
1. src/config/ri_4438/ri_4438_him_lpacrl_config.py：地形等级、速度分箱、LP/HIM 参数。
2. ri_4438_him_lpacrl_env_cfg.py：观测、动作、奖励、事件和终止条件。
3. config/env_cfgs.py：机器人绑定和 Flat/Rough/play 环境工厂。
4. config/rl_cfg.py：HIM/PPO 配置适配；config/__init__.py 注册 task。
5. mdp/curriculums.py：LP 概率更新和 reset 时的任务分配。
6. rl/runner.py + rl/evaluator.py：训练、周期评估、保存和续训。

地形在 mdp/terrains.py 按类型/等级固定，mdp/task_catalog.py 将速度分箱与地形
组合成任务。训练过程中改变的是任务采样概率，不改变每个任务的难度定义。

当前 Rough 为 600 组：|vx| 在 0–1 m/s 内分五档，|wz| 在 0–1 rad/s
内分六档，五类地形各四档。vy 在 ±0.5 m/s 内连续采样，不增加任务数量。
命令每 2 秒在同一分箱内更新。粗糙地形高度上限为 0.02/0.03/0.045/0.06 m，
楼梯高度为 0.05/0.08/0.12/0.16 m；上坡、下坡的梯度为 0/0.10/0.20/0.30，
对应坡角约 0°/5.71°/11.31°/16.70°，坡地高度量化精度为 1 mm。
这是论文的任务组合方式；速度范围、地形高度和 HIM 策略采用本项目设置。
评估 v3 按固定 20 秒累计奖励：出界重置同一任务并保留指令时钟，物理失败
结束试验。增加逐项回报、EPTE-SP、成功率和配对 LP 诊断；LP 仍用总回报差。
EPTE-SP 的稳定性聚合使用论文公式；变指令、近零指令的误差口径见 evaluation
配置。日志还包含行走距离、台阶接触时长和停滞时长；边界率与时限率可重叠。
每次评估覆盖 600 × 8 = 4800 次固定试验，使用 512 个环境分十批执行。
修改速度分箱或地形等级后，旧模型的 LP 历史不能直接 resume，即使任务总数
仍为 600。旧权重可用 scripts.train 的 --warm-start PATH 开始新课程，
不继承优化器和 LP 历史；v3 会从 update 0 重新评估。默认温度 25、均匀混合
0.15 是本项目工程起点，不是论文公开超参数；纯 LP 对照将混合设为 0。
"""
