# RI4438-HIM 真机部署评估与验收清单

评估日期：2026-09-16  
评估范围：RI-4438 HIM 训练产物、ONNX 导出、MuJoCo sim2sim 部署栈、FSM 与真机部署前门禁。  
证据来源：仓库当前代码、`ARCHITECTURE_CONTEXT.md`、`deploy/task/ri_4438_him/README.md`、训练日志快照和本次只读命令检查。本文不把推测写成事实；所有结论按“代码已确认 / 运行验证 / 需要实测 / 当前缺失”区分。

## 1. 总结结论

**当前 RI-4438 HIM 训练产物不能直接部署到真机。**

代码已确认：

- 当前较完整的部署链路是 RI-4438 HIM 的 MuJoCo sim2sim，不是真机 motor/CAN/DDS 部署链路。
- `DeploymentRuntime` 直接实例化 `MujocoBackend`，仓库没有 RI-4438 真机 backend、驱动 SDK 适配、总线收发、硬件 watchdog 或实机日志闭环。
- HIM ONNX 推理契约清楚：输入 `obs_history[batch,282]`，输出 `actions[batch,12]`；ONNX 内部包含观测归一化、HIM estimator encoder 和 deterministic actor。
- sim2sim 默认配置 `deploy/task/ri_4438_him/config/sim2sim.yaml` 指向 `logs/rsl_rl/ri_4438_him/2026-09-15_15-52-14/policy.onnx`，但当前仓库只确认存在 `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx`。默认部署命令会因策略文件缺失失败，除非显式传入现有 ONNX 路径。
- **CRITICAL 真机阻塞**：默认 `sim2sim.yaml` 同时设置 `auto_stand=true` 和 `auto_enable_rl=true`。这适合仿真便捷验证，但真机上会在启动后自动起立并继续进入 RL；上机前必须禁用这两个开关，并禁止任何默认自动站立/自动进入 RL 的真机入口。
- 训练监控报告显示，当前策略在高 command / rough terrain 阶段仍有跌倒、非法接触和性能回撤；没有真机实测、硬件在环、吊挂或低速上机证据。

当前缺失：

- 真机 `RobotBackend` 实现。
- 真机观测链路标定，包括 IMU 坐标、关节方向、单位、延迟、丢包、时间戳、编码器零偏。
- 真机低层控制闭环要求和安全执行器接口，包括电机使能、力矩/速度/位置限位、驱动级急停、失联降级、温度/电流/电压保护。
- 将 sim2sim 默认 ONNX 路径更新到实际存在产物或建立部署产物选择流程。
- 真机入口的 `auto_stand`、`auto_enable_rl` 和任意自动 RL enable 关闭证明。
- ONNX CPU 推理延迟 p50/p99、控制环 jitter 和端到端观测到动作延迟报告。

## 2. 已有训练、ONNX、sim2sim、FSM 边界

### 2.1 训练边界

代码已确认：

- 任务注册和配置路径：
  - `src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py`
  - `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py`
  - `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py`
  - `src/config/ri_4438/ri_4438_him_config.py`
  - `src/config/ri_4438/ri_4438_config.py`
- 训练仿真步长 `0.005 s`，decimation `4`，策略/控制周期 `0.02 s`，即 50 Hz。
- actor 单帧观测 47 维，历史 6 帧，总观测 282 维。
- PPO/HIM runner 保存 checkpoint 时会导出同目录 `policy.onnx`。
- 训练端 actor observation normalization 开启，`clip_actions: null`。

训练快照已确认：

- 当前存在训练产物目录：`logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/`。
- 该目录存在 `model_20000.pt`、`model_20100.pt`、`model_20200.pt`、`params/agent.yaml`、`params/env.yaml` 和 `policy.onnx`。
- `params/agent.yaml` 显示本 run 从 `2026-09-15_09-52-30/model_10500.pt` resume，`num_steps_per_env=100`，`max_iterations=100000`，`save_interval=100`。

需要实测：

- 当前 ONNX 对应的具体 checkpoint iteration 需要从训练保存时的日志或模型导出记录确认。仅从文件名 `policy.onnx` 不能证明它对应 `model_20200.pt`、`model_20100.pt` 或其他保存点。
- 训练监控报告里的性能结论需要与最终 ONNX 对应 checkpoint 对齐复核。

### 2.2 ONNX 边界

运行验证：

- 当前存在的 ONNX：`logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx`
- 文件大小：`991276` bytes
- SHA256：`516dd71d8b004bae90e8342e666a835b78900567aa0095723ab3c64f1b7a9b59`
- ONNX opset：18
- ONNX Runtime 读取到的输入输出：
  - 输入：`obs_history ['batch', 282] tensor(float)`
  - 输出：`actions ['batch', 12] tensor(float)`
- 当前部署证据只覆盖 ONNX Runtime `CPUExecutionProvider`。本地 ONNX Runtime 包可枚举 `AzureExecutionProvider` 和 `CPUExecutionProvider`，但部署代码 `OnnxPolicy` 固定请求 `CPUExecutionProvider`；RKNN、TensorRT、OpenVINO 和 `CUDAExecutionProvider` 均未接入当前部署路径。

代码已确认：

- ONNX wrapper 输入名 `obs_history`，输出名 `actions`。
- ONNX wrapper 接受 frame-major、current-first 的历史观测。
- ONNX 图内包含：
  - actor observation normalizer；
  - HIM estimator encoder；
  - deterministic actor MLP；
  - deterministic distribution output。
- 部署端 `OnnxPolicy` 不再做外部观测归一化。
- `action_clip` 在 sim2sim YAML 中为 `null`，部署端不会裁剪 actor 原始输出。

当前缺失：

- 默认 sim2sim YAML 指向的 `2026-09-15_15-52-14/policy.onnx` 不存在。当前默认 `python deploy/main.py --task ri_4438_him` 不是可用部署证据。
- RKNN/TensorRT/OpenVINO/CUDAExecutionProvider 部署链路未接入；`tools/onnx2rknn.py` 和 `tools/best_policy.py` 当前均为 0 字节占位文件。

### 2.3 sim2sim 边界

代码已确认：

- sim2sim 配置：`deploy/task/ri_4438_him/config/sim2sim.yaml`
- sim2sim 入口：
  - `python deploy/main.py --task ri_4438_him`
  - `python deploy/task/ri_4438_him/main.py`
- MuJoCo XML：`deploy/scene/terrain/scene_ri_4438_terrain.xml`
- 默认生命周期：
  - `PASSIVE -> GETUP -> STAND_SETTLE -> READY -> STAND -> RL -> GETDOWN -> PASSIVE`
- sim2sim README 明确说明该 launcher 当前只使用 MuJoCo simulator backend，不包含 RI-4438 motor/CAN/DDS SDK。
- sim2sim 路径没有复现训练时 observation noise 或 actuator delay；这不是观测契约差异，但是真实动力学差距。
- 默认 `sim2sim.yaml` 的 `input.auto_stand=true`、`input.auto_enable_rl=true` 会让仿真启动后自动起立并自动进入 RL。

真机阻塞：

- **CRITICAL**：`auto_stand=true` 和 `auto_enable_rl=true` 不得带入真机启动路径。上机前必须在真机配置、CLI 覆盖或入口逻辑中明确禁用，并用日志证明启动后停留在人工确认的安全状态。
- **CRITICAL**：任何真机入口都不得复用默认 sim2sim 自动生命周期作为上电流程；至少应从 sensor-only、PASSIVE/DAMPING、人工确认 STAND 逐级推进。

### 2.4 FSM 边界

代码已确认：

- FSM 实现：`deploy/include/fsm.py`
- backend 协议：`deploy/include/types.py`
- 当前 runtime：`deploy/include/runtime.py`
- 当前 backend：`deploy/include/mujoco_bridge.py`
- FSM 有 `FAULT` 状态、operator emergency stop、fall detection、stale/non-finite backend state 检查、transition timeout、退出时 best-effort `emergency_stop`。
- FSM 在 `PASSIVE/DAMPING` 下写 damping command；`GETUP/STAND/RL/GETDOWN` 使用 position command。
- RL 只在 `State.RL` 中调用 ONNX，非 RL 状态 `last_action` 清零。
- runtime 退出时通过 `getattr(self.backend, "emergency_stop", None)` 做 best-effort shutdown；如果 backend 没有实现 `emergency_stop` 或调用失败，代码会降级为 no-op/吞掉异常。这只能算仿真/兼容保护，不能作为真机独立急停。

当前缺失：

- FSM 可复用，但 runtime 目前固定创建 `MujocoBackend`。真机部署前必须改为可注入真机 backend，或提供独立真机入口；不能把当前 sim2sim 进程直接连到电机。

## 3. 精确观测契约

### 3.1 维度、顺序、单位

代码已确认，单帧观测 47 维，按以下顺序拼接：

| 区间 | 项 | 维度 | 单位/定义 | 部署端来源 |
| --- | --- | ---: | --- | --- |
| `0:3` | `base_ang_vel` | 3 | 机身坐标系角速度，rad/s | `RobotState.angular_velocity_body` |
| `3:6` | `projected_gravity` | 3 | 重力向量投影到机身坐标系，无量纲 | `rotate_inverse(quaternion, [0,0,-1])` |
| `6:9` | `command` | 3 | `[vx, vy, wz]`，m/s、m/s、rad/s | `CommandBus.command()` |
| `9:11` | `phase` | 2 | `[sin(phi), cos(phi)]`，无量纲 | period `0.6 s`，命令范数 `<0.1` 时置零 |
| `11:23` | `joint_pos_rel` | 12 | `q - default_joint`，rad | `state.joint_position - cfg.default_joint` |
| `23:35` | `joint_vel` | 12 | 关节速度，rad/s | `state.joint_velocity` |
| `35:47` | `last_action` | 12 | 上一次 actor 原始动作，无量纲 | `fsm.last_action` |

历史契约：

- `history_size = 6`
- `observation_dim = 282 = 47 * 6`
- ONNX 输入历史方向：frame-major、current-first，即 `[t, t-1, t-2, t-3, t-4, t-5]`。
- 训练 observation manager 保存显式时间轴时为 oldest-to-newest；`HIMActorModel` 在导出前反转为 current-first。部署端必须直接提供 current-first。
- 部署首次进入 RL 时清空 history；第一帧观测会回填 6 帧，之后每步 append 当前帧并保留最近 6 帧。

归一化契约：

- 训练 actor `obs_normalization: true`。
- ONNX 导出把 `EmpiricalNormalization` 打进图内。
- 部署端禁止再次做均值方差归一化，也禁止按经验重标定观测输入。
- 训练时 actor observation corruption 覆盖部分观测噪声；部署端输入应是真实测量值，不应人为注入训练噪声。

### 3.2 观测缺口与真机要求

需要实测：

- IMU 角速度必须是机身坐标系 rad/s，轴向和 MuJoCo base frame 一致。
- 四元数必须是 `[w, x, y, z]`，且用于计算 projected gravity 的坐标约定必须与 MuJoCo/Unitree 风格一致。
- 12 个关节位置、速度必须按部署 joint order 排列，单位为 rad 和 rad/s。
- `joint_pos_rel` 使用的 `default_joint` 必须是真机标定后的站立零偏；当前代码值是仿真默认值。
- 观测时间戳需要单调递增，真机 backend 要提供 `monotonic_time` 或递增 `sequence`，否则 FSM stale 检查不能可靠工作。

当前缺失：

- 真机 IMU 到 base_link 的外参和符号验证记录。
- 真机编码器零位、方向和关节顺序的逐关节验证记录。
- 真实观测延迟和 jitter 分布。
- 观测异常值、丢包、重复帧、时钟跳变的硬件日志。

## 4. 精确动作契约

### 4.1 输出、尺度、频率

代码已确认：

- ONNX 输出：`actions[12]`，无量纲 actor 原始动作。
- 部署目标关节位置：
  - `q_des = default_joint + action_scale * action`
- sim2sim 中 `action_clip = null`，不裁剪 ONNX 输出。
- MuJoCo backend 会把最终 `q_des` 裁剪到 joint/actuator position range；真机 backend 必须提供不弱于该逻辑的硬限位和软限位。
- 控制周期：
  - physics timestep `0.005 s`
  - decimation `4`
  - policy/control_dt `0.02 s`
  - policy 频率 `50 Hz`
- RL phase 只在 decimated policy step 后推进 `0.02 s`，不是每个 physics step 都推进。

动作尺度和关节顺序：

| index | joint | default_joint rad | action_scale rad |
| ---: | --- | ---: | ---: |
| 0 | `FL_hip_joint` | 0.0 | 0.25 |
| 1 | `FL_thigh_joint` | 0.9 | 0.5 |
| 2 | `FL_calf_joint` | -1.8 | 0.5 |
| 3 | `FR_hip_joint` | 0.0 | 0.25 |
| 4 | `FR_thigh_joint` | 0.9 | 0.5 |
| 5 | `FR_calf_joint` | -1.8 | 0.5 |
| 6 | `RL_hip_joint` | 0.0 | 0.25 |
| 7 | `RL_thigh_joint` | 0.9 | 0.5 |
| 8 | `RL_calf_joint` | -1.8 | 0.5 |
| 9 | `RR_hip_joint` | 0.0 | 0.25 |
| 10 | `RR_thigh_joint` | 0.9 | 0.5 |
| 11 | `RR_calf_joint` | -1.8 | 0.5 |

低层 PD 契约：

- 位置控制律按代码注释定义为：
  - `tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff`
- 当前部署写入：
  - `dq_des = 0`
  - `tau_ff = 0`
  - `kp = 32.50652905356971`
  - `kd = 2.06942991271352`
  - `effort_limit = 10.0 Nm`
- `PASSIVE/DAMPING` 下写 `kp=0`、`kd=cfg.kd`。

需要实测：

- 真机电机是否接受位置目标、速度目标、kp/kd 和力矩前馈，或是否只能接受厂商模式下的等效参数。
- 真机驱动中的 kp/kd 单位、限幅、饱和行为、命令保持策略和掉线策略。
- `10 Nm` 是否是每个关节、每个电机、驱动内部还是仿真约束；必须与真实额定/峰值力矩、温度限制和减速比对应。
- 当前训练报告指出 `stiffness * action_scale` 对 thigh/calf 的理论比例项约 `16.25 Nm`，超过 `10 Nm` effort limit。这是上机前必须复核的风险。

## 5. HIM estimator、推理时序与热启动

代码已确认：

- HIM actor 是 feed-forward，`is_recurrent = False`。
- estimator 在训练时有独立 optimizer 和 HIM objective；推理时只使用 estimator encoder。
- ONNX 推理流程：
  1. 输入 current-first `obs_history[1,282]`
  2. 图内 observation normalizer 处理历史
  3. estimator encoder 从 282 维历史预测 `base velocity[3]` 和 normalized latent[16]
  4. actor 使用当前帧前 47 维、估计速度 3 维、latent 16 维组成 actor input
  5. deterministic actor 输出 12 维动作
- ONNX `reset()` 是 no-op；没有 RNN hidden state。
- 真正需要热启动的是观测历史和 `last_action`：
  - 进入 RL 前 `last_action=0`
  - 历史为空时用第一帧回填 6 帧
  - reset、passive、fault、stand/getdown 会清空或停止更新策略历史
- RL phase 从进入 RL 时的 `0` 开始，不使用全局仿真时间；只有命令范数达到阈值时 phase 才非零。

运行验证：

- 单元测试覆盖：
  - 首帧 history backfill
  - current-first 历史方向
  - RL phase 从零开始
  - phase 按 `control_dt=0.02 s` 推进
  - reset 清空 history/action/command
  - 非 RL 生命周期不调用策略

需要实测：

- ONNX 在目标真机计算平台上的推理延迟 p50/p95/p99。
- 真机控制线程从传感器采样到电机发包的端到端延迟。
- 进入 RL 前，真实机器人在 `STAND` 状态下的关节速度和机身姿态是否能满足 settle gate。

## 6. 域随机化覆盖、缺口与实测校准

### 6.1 覆盖项

代码已确认：

- 观测噪声：
  - `base_ang_vel`: `[-0.2, 0.2]`
  - `projected_gravity`: `[-0.05, 0.05]`
  - `joint_pos`: `[-0.01, 0.01] rad`
  - `joint_vel`: `[-1.5, 1.5] rad/s`
- reset：
  - base pose offset 当前为 0
  - joint offset 当前为 0
- 外部扰动：
  - 每 `5.0-6.0 s` push
  - linear velocity x/y `[-0.5,0.5] m/s`，z `[-0.4,0.4] m/s`
  - angular roll/pitch `[-0.52,0.52] rad/s`，yaw `[-0.78,0.78] rad/s`
- 地面/接触：
  - foot friction startup randomization `0.3-1.6`，四脚 shared random
- 编码器：
  - encoder bias `[-0.015,0.015] rad`
- 质心/质量/惯量：
  - base COM offset x/y/z `[-0.05,0.05] m`
  - pseudo inertia 对所有 body 做随机化，等效质量范围来自 `link_mass_range=[0.8,1.2]`
- 执行器：
  - kp/kd scale `[0.9,1.1]`
  - effort limit scale `[0.9,1.1]`
  - actuator delay config：`delay_min_lag=0`，`delay_max_lag=4`，`delay_hold_prob=0.3`，`delay_update_period=10`
- 地形：
  - flat、stairs、inverted stairs、random grid obstacles
  - rough terrain curriculum enabled

### 6.2 当前缺口

当前缺失：

- 真机质量、COM、惯量、脚底材料摩擦、地面顺应性、关节摩擦和 backlash 的测量。
- 电机/驱动真实延迟、带宽、饱和、rate limit、温度降额、电压变化和通信 jitter 的测量。
- IMU 噪声谱、bias drift、滤波延迟和 orientation estimator 行为。
- 编码器真实零偏、量化、速度估计滤波延迟。
- 失联、丢包、重复包、乱序包、长尾延迟场景。
- sim2sim 部署端没有复现训练 observation noise 和 actuator delay。
- 当前训练报告指出 command curriculum 扩展后策略性能回撤，terrain level 上升并不等于真机可用能力。

### 6.3 真实测量校准流程

需要实测，建议作为真机门禁前置项：

1. 静态标定：
   - 记录四足悬空、站立、趴下三个姿态的 encoder raw、关节角、IMU quaternion。
   - 确认每个关节正方向、零位、限位和部署 joint order。
   - 将 `default_joint` 与实机自然站立姿态差值写入校准报告。
2. 延迟标定：
   - 给每个关节小幅阶跃或 chirp，记录 command timestamp、driver receive timestamp、encoder response timestamp。
   - 得到观测到动作延迟、动作到关节响应延迟、控制环 jitter 的 p50/p95/p99。
3. 执行器标定：
   - 在吊挂或支撑状态下验证 kp/kd 单位和饱和行为。
   - 建立真实力矩/电流限制、速度限制和温升限制。
4. 接触标定：
   - 用实际脚底材料和目标地面测量静摩擦/动摩擦范围。
   - 用低速脚端拖动或斜坡实验估计摩擦下限。
5. 质量/COM 标定：
   - 称重、支撑点称重或 CAD 修正，得到 body/payload 的质量和质心范围。
6. sim 对齐：
   - 用相同开环关节命令对比真机和 MuJoCo 的 q/dq/IMU 响应。
   - 调整仿真中心值，再把 domain randomization 范围设为覆盖实测 p5-p95 或更保守范围。

## 7. 真机 backend 与安全要求

### 7.1 必须实现的 backend 契约

当前缺失，真机 backend 必须实现 `deploy/include/types.py` 的 `RobotBackend` 协议：

- `read_state() -> RobotState`
- `set_control_mode(mode, damping_kd=None)`
- `write_joint_command(command)`
- `emergency_stop(reason=None)`
- `set_joint_target(target)`，仅兼容旧接口；新真机路径应使用 `write_joint_command`
- `step()`
- `reset()`
- `close()`

`RobotState` 必须提供：

- `time`
- `position[3]`
- `quaternion[4]`，顺序 `[w,x,y,z]`
- `joint_position[12]`
- `joint_velocity[12]`
- `angular_velocity_body[3]`
- `valid`
- `connected`
- `monotonic_time` 或 `sequence`

### 7.2 watchdog、急停、限位、失联

代码已确认：

- FSM stale timeout 默认 `0.25 s`。
- 如果 backend state stale、invalid、disconnected 或 joint 状态非有限，且当前不在 `PASSIVE/DAMPING/FAULT`，FSM 进入 `FAULT`。
- operator emergency stop 会进入 `FAULT` 并调用 backend `emergency_stop`。
- gamepad disconnect 会清零 velocity command，并请求 `estop`。
- `FAULT` 会清空 command/history/action，并阻止继续写普通 command。
- MuJoCo backend 的 emergency stop 是 latched，并禁用 actuator。
- runtime shutdown 的 `emergency_stop` 是 best-effort：backend 未实现时可以退化为 no-op，调用异常也会被忽略。

输入设备风险：

- **HIGH**：raw joystick 的 START/BACK 映射存在设备相关歧义。代码默认/别名中 START 候选包含 `6`，BACK 候选也包含 `6`；按钮分发顺序先检查 reset，再检查 back/estop。因此某些 raw joystick 设备上，物理 BACK/Select 如果上报为 `6`，可能先触发 reset，而不是 back/estop。
- **HIGH**：手柄不得作为唯一急停。每个手柄、每种 SDL Controller/raw Joystick 模式、每个无线接收器固件都必须逐设备实测按钮 index、断连行为、liveness timeout 和 reconnect 行为，并把设备 name/GUID/index 写入验收记录。

当前缺失：

- 真机驱动级 watchdog。Python FSM 的 `0.25 s` stale timeout 不能替代电机侧或通信侧硬 watchdog。
- 真机急停链路，包括硬件按钮、遥控器失联、上位机崩溃、进程退出、电池异常、驱动 fault 的统一 latch 策略。
- 真机软限位和硬限位表，包括位置、速度、力矩、电流、温度、电压。
- 真机控制模式切换约束，例如 DAMPING 到 POSITION 是否需要解锁、预充、零力矩窗口或安全确认。
- 失联后是否制动、阻尼、卸力或趴下的硬件策略。这个必须由实机平台安全策略决定，不能由 sim2sim 推断。

独立急停要求：

- **CRITICAL**：真机必须有独立于 Python、FSM、ONNX 推理和手柄输入的急停链路。
- 独立急停必须直接切断 motor enable 或动力级输出，使驱动进入硬件定义的安全状态，并故障锁存。
- 急停锁存解除必须需要人工复位或硬件安全流程，不能由 Python 进程、策略输出、手柄按钮或自动重连自动恢复。
- runtime/FSM/backend 的 `emergency_stop` 只能作为软件辅助停机路径；它不能替代驱动级和动力级急停。

真机低层控制最低要求：

- backend 必须在驱动层实现命令超时即安全降级，不能等待 Python 下一次循环。
- backend 必须在每次发包前检查 `q_des`、`dq_des`、kp/kd 和 feedforward effort 是否有限且在限幅内。
- backend 必须记录每周期的 state、command、mode、watchdog 状态、通信延迟、driver fault、限幅触发。
- backend 必须保证 emergency stop 是 latched，只有人工 reset 或明确安全流程才能重新使能。
- backend 必须提供 dry-run/sensor-only 模式，允许不开电机只验证观测契约和时序。

## 8. 分阶段部署门禁

任何阶段失败都停止推进，不进入下一阶段。

### Gate 0：离线产物和契约检查（CRITICAL）

目标：确认代码、配置、ONNX 和测试契约一致。

必须通过：

- ONNX 文件存在且输入输出形状为 `[batch,282] -> [batch,12]`。
- sim2sim YAML 的 `policy` 路径存在，或启动命令显式覆盖为存在的 ONNX。
- `action_clip`、`history_order`、`history_size`、`joint_names`、`action_scale` 与本文契约一致。
- `python -m unittest deploy.task.ri_4438_him.test_sim2sim` 通过。

当前状态：

- BLOCKER：默认 YAML 指向的 `2026-09-15_15-52-14/policy.onnx` 不存在。
- `.venv/bin/python3 -m pytest --version` 失败，错误为 `No module named pytest`；本轮验证改用 `unittest` 和 CLI smoke。
- Phase 4 复核记录显示：19 个不依赖真实 ONNX、且不需要写入仓库的 FSM/观测/输入契约 unittest 通过。
- `test_real_onnx_policy_smoke` 因默认配置策略路径缺失被 skipped。
- 默认 CLI `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py --task ri_4438_him --headless --steps 1` 退出码 1，报 `Policy not found`。

### Gate 1：headless sim2sim FSM 冒烟

目标：不加载 ONNX，只验证 MuJoCo backend、FSM 起立/趴下和安全退出。

命令：

```bash
python deploy/main.py --task ri_4438_him --no-policy --headless --auto-stand --steps 800
```

通过条件：

- 进程正常退出。
- 无 `FAULT`。
- `PASSIVE -> GETUP -> STAND_SETTLE -> READY -> STAND` 能完成。
- runtime shutdown 调用安全关闭。

当前状态：

- Phase 4 已通过 `/tmp` 复制配置执行 headless smoke，避免在仓库内生成 runtime XML。
- `no-policy` 10 步退出码 0。
- `auto-stand` 800 步退出码 0。
- 配置和场景复制到 `/tmp/ri4438_sim_validation/sim2sim_tmp.yaml`、`/tmp/ri4438_sim_validation/scene_ri_4438_terrain.xml` 是为了避免生成仓库内 runtime XML；这不能当作默认 YAML 原地运行证据。

### Gate 2：headless ONNX 闭环 sim2sim

目标：加载现有 ONNX，在平坦区域跑固定步数。

命令示例：

```bash
python deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx \
  --headless \
  --auto-stand \
  --terrain checkerboard \
  --steps 2000
```

通过条件：

- ONNX 初始化成功。
- 不出现 policy failure、non-finite observation/action、fall detected during RL。
- 记录每周期 action 范围、q_des 范围和限幅触发次数。

当前状态：

- Phase 4 已通过 `/tmp` 复制配置执行实际 policy 1000 步，退出码 0。
- 最终进入 RL，backend finite，`last_action` finite。
- 该结果证明显式现有 ONNX 和 `/tmp` 验证配置可运行；它不证明默认 `sim2sim.yaml` 的缺失策略路径可运行。

### Gate 3：可视化闭环仿真与地形分级

目标：在可视化 MuJoCo 中确认站立、低速命令、楼梯/heightfield 行为。

命令示例：

```bash
python deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx \
  --terrain stairs_5cm

python deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx \
  --terrain heightfield
```

通过条件：

- 低速 command 下连续运行不跌倒。
- action 不长期贴近极端输出。
- q_des 不频繁触发关节限位。
- 地形能力按 checkerboard、5cm stairs、heightfield 逐级放行。

当前状态：

- 未执行。

### Gate 4：硬件在环，传感器和命令 dry-run（CRITICAL）

目标：不开电机或不使能闭环力矩，只跑真机 backend 观测链路和 ONNX 推理。

必须先实现：

- 真机 `RobotBackend`。
- dry-run/sensor-only 模式。
- 周期日志和 latency 统计。

通过条件：

- 观测维度、单位、方向、历史方向与本文一致。
- `valid/connected/sequence/monotonic_time` 正常。
- ONNX action 有限，且低速命令下分布稳定。
- 失联、传感器重复帧、非有限值会触发 FAULT 或 backend watchdog。

当前状态：

- 当前缺失真机 backend，不能执行。

### Gate 5：单关节低流/低速/低幅阶跃（CRITICAL）

目标：只使能一个关节，在低电流、低速度、低幅度下验证驱动方向、限幅、watchdog 和急停。

必须先具备：

- 真机 backend 或独立低层驱动测试工具。
- 驱动级电流/力矩限制、速度限制、位置窗口限制和温度保护。
- 独立硬件急停已验证，且急停绕过 Python/FSM/ONNX/手柄，能直接切 motor enable 或动力级输出并故障锁存。
- `auto_stand=false`、`auto_enable_rl=false`，且没有任何 RL enable 路径参与该门禁。

通过条件：

- 每个关节逐一验证正负方向、零位、速度反馈、位置反馈和限位。
- 小幅阶跃下无超调失控、无通信超时、无驱动 fault、无异常电流。
- 进程 kill、通信断开、急停按钮、驱动 fault 均进入锁存安全状态。
- 记录每个关节的 command、q、dq、电流/力矩、温度、driver status 和 watchdog status。

当前状态：

- 当前缺少真机 backend、驱动低层测试记录和独立急停验收，不能执行。

### Gate 6：全身非 RL 验收（CRITICAL）

目标：全身上电但禁止策略控制，只验证 PASSIVE/DAMPING/STAND 的低层行为。

强制禁止：

- `auto_stand`
- `auto_enable_rl`
- `enable_rl` / `rl_enable` / `start_rl`
- ONNX policy action 写入电机
- 手柄作为唯一急停

允许状态：

- `PASSIVE`
- `DAMPING`
- 人工确认后的 `STAND`

通过条件：

- 上电默认保持 PASSIVE/DAMPING，不自动站立。
- 人工确认后才允许进入 STAND，且 STAND 只写默认站立姿态或受限站立轨迹。
- 任意 reset、getdown、estop、失联、非有限状态、驱动 fault 都不会进入 RL。
- 关节位置、速度、电流、温度、机身倾角都在保守限幅内。

当前状态：

- 当前缺少真机 backend 和非 RL 硬件验收日志，不能执行。

### Gate 7：吊挂/支撑，限幅闭环（HIGH）

目标：真机离地或部分卸载，验证低层位置控制、急停、限位和小幅策略动作。

必须先具备：

- backend 侧 action scale 限幅或单独受限配置。
- 驱动级位置/速度/力矩/温度/电流保护。
- 硬件急停和软件急停都已验证。
- Gate 5 和 Gate 6 已通过。

建议通过条件：

- 初始只允许很小 command 和很小 action envelope。
- 每个关节 q_des、q、dq、估算 torque/current 都在限幅内。
- 任意通信断开、进程 kill、手柄断开、急停按钮都能进入 latched safe state。
- 至少完成 reset、stand、enable RL、disable RL、getdown、estop 全流程。

当前状态：

- 当前缺少真机 backend 和受限硬件配置。

### Gate 8：低速上机（HIGH）

目标：在平整地面、小速度、小动作范围下短时上机。

前置：

- Gate 0-7 全部通过并留存证据。
- 至少两人现场，一人只负责硬件急停。
- 场地有物理保护，机器人周围清空。

通过条件：

- 首次只运行站立保持，不给行走 command。
- 之后 vx/vy/wz 从极低值开始，逐步增加。
- 任一摔倒趋势、限幅频繁触发、温度/电流异常、通信 jitter 超门限立即停机。
- 每次只改变一个变量：速度、地面、action scale、时长不能同时放大。

当前状态：

- 未达到上机条件。

## 9. 可执行命令与证据路径

### 9.1 运行环境记录

Phase 4 复核环境：

| 项 | 版本/状态 |
| --- | --- |
| Python | 3.11.16 |
| torch | 2.14.0+cu130 |
| MuJoCo | 3.11.0 |
| ONNX | 1.22.0 |
| ONNX Runtime | 1.29.0 |
| pygame | 2.6.1 |
| PyYAML | 6.0.3 |
| numpy | 2.4.6 |
| mjlab | import OK，`__version__` 为 `None` |
| rsl_rl | import OK，`__version__` 为 `None` |
| pytest | `.venv/bin/python3 -m pytest --version` 失败：`No module named pytest` |
| ONNX Runtime deployment provider | 当前部署代码固定 `CPUExecutionProvider` |
| 未接入后端 | RKNN、TensorRT、OpenVINO、`CUDAExecutionProvider` |
| RKNN/策略选择工具 | `tools/onnx2rknn.py` 和 `tools/best_policy.py` 均为 0 字节占位 |

### 9.2 离线检查命令

检查当前 ONNX 文件形状：

```bash
.venv/bin/python3 -B - <<'PY'
from pathlib import Path
import hashlib
import onnxruntime as ort

path = Path("logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx")
print("exists", path.exists())
data = path.read_bytes()
print("size", len(data))
print("sha256", hashlib.sha256(data).hexdigest())
print("available_providers", ort.get_available_providers())
sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
print("session_providers", sess.get_providers())
print("inputs", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
print("outputs", [(o.name, o.shape, o.type) for o in sess.get_outputs()])
PY
```

验证部署契约单元测试：

```bash
.venv/bin/python3 -B -m unittest deploy.task.ri_4438_him.test_sim2sim
```

Phase 4 复核记录：

- 19 个不依赖真实 ONNX、且不需要写入仓库的 FSM/观测/输入契约 unittest 通过。
- `test_real_onnx_policy_smoke` 因默认配置策略路径缺失被 skipped。
- 默认 CLI 失效命令如下，退出码 1，报 `Policy not found`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --headless \
  --steps 1
```

### 9.3 训练和回放命令

训练帮助：

```bash
uv run python scripts/train.py Ri-4438-HIM-Rough --help
```

checkpoint 回放：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --checkpoint-file logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/model_20200.pt \
  --num-envs 1 \
  --viewer native
```

TensorBoard：

```bash
uv run tensorboard --logdir logs/rsl_rl
```

### 9.4 sim2sim 命令

默认启动当前失效，因为 `sim2sim.yaml` 指向不存在的 `2026-09-15_15-52-14/policy.onnx`。需要显式传入当前存在的 ONNX：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
```

后端/FSM 冒烟，不加载 ONNX：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --no-policy \
  --headless \
  --auto-stand \
  --steps 800
```

显式加载当前存在 ONNX：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
```

headless ONNX 闭环：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx \
  --headless \
  --auto-stand \
  --steps 2000
```

地形 spawn：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx \
  --terrain stairs_5cm

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 deploy/main.py \
  --task ri_4438_him \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx \
  --terrain heightfield
```

Phase 4 已执行的 `/tmp` headless smoke 记录：

- `/tmp/ri4438_sim_validation/sim2sim_tmp.yaml`
- `/tmp/ri4438_sim_validation/scene_ri_4438_terrain.xml`
- `no-policy` 10 步退出码 0。
- `auto-stand` 800 步退出码 0。
- 实际 `policy.onnx` 1000 步退出码 0，最终 RL/backend finite，`last_action` finite。
- 这些 `/tmp` 文件是为避免仓库内生成 runtime XML 的复制件，不能当作默认 `deploy/task/ri_4438_him/config/sim2sim.yaml` 原地运行证据。

### 9.5 证据路径

代码路径：

- `README.md`
- `ARCHITECTURE_CONTEXT.md`
- `src/config/ri_4438/ri_4438_config.py`
- `src/config/ri_4438/ri_4438_him_config.py`
- `src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py`
- `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py`
- `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py`
- `src/tasks/locomotion/ri_4438_him/mdp/observations.py`
- `src/tasks/locomotion/ri_4438_him/rl/runner.py`
- `rsl_rl/models/him_actor_model.py`
- `rsl_rl/modules/him_estimator.py`
- `deploy/include/config.py`
- `deploy/include/policy.py`
- `deploy/include/fsm.py`
- `deploy/include/runtime.py`
- `deploy/include/types.py`
- `deploy/include/mujoco_bridge.py`
- `deploy/include/inputs.py`
- `deploy/task/ri_4438_him/config/sim2sim.yaml`
- `deploy/task/ri_4438_him/test_sim2sim.py`
- `deploy/task/ri_4438_him/README.md`
- `/tmp/ri4438_sim_validation/sim2sim_tmp.yaml`，Phase 4 临时验证配置，不是仓库默认配置
- `/tmp/ri4438_sim_validation/scene_ri_4438_terrain.xml`，Phase 4 临时验证场景，不是仓库默认场景

训练与产物路径：

- `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx`
- `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/model_20000.pt`
- `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/model_20100.pt`
- `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/model_20200.pt`
- `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/params/agent.yaml`
- `logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/params/env.yaml`
- `docs/ri_4438_him_training_monitor_report.md`

## 10. 仍不确定项

需要实测或补证：

- 当前 `policy.onnx` 精确对应哪个 checkpoint iteration。
- 真机 IMU/关节观测单位、符号、顺序、外参、延迟、jitter。
- 真实执行器 kp/kd 单位、饱和、力矩限制、温度和电流保护。
- actuator delay 在 mjlab 内部以 physics step 还是 control step 计量；上机前应通过代码追踪或实验确认。
- 当前训练产物在 checkerboard、stairs、heightfield 的 headless 和可视化 sim2sim 通过率。
- 当前策略在低速 command 下 action 分布、限位触发率和跌倒率。
- 真机 backend 的 watchdog、急停和故障 latch 设计。

当前阻塞：

- 默认部署 YAML 的策略路径不存在。
- 默认 `sim2sim.yaml` 的 `auto_stand=true`、`auto_enable_rl=true` 是 CRITICAL 真机阻塞；上机前必须禁用并留存证据。
- 真机 backend 缺失。
- 独立于 Python/FSM/ONNX/手柄的动力级急停和故障锁存缺失。
- 硬件安全和实测校准证据缺失。
