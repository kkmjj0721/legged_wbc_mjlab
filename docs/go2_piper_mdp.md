# Go2 Piper MDP 迁移与实现清单

本文档只解决一件事：把 `/home/kk/LeggedManip_Lab/source/LeggedManip_Lab` 中 Go2 Piper 的 MDP 行为，按 `/home/kk/github/unitree_rl_mjlab` 的 mjlab 组织方式，迁移到当前项目 `/home/kk/legged_wbc_mjlab`。Deep-WBC 只作为 legacy 控制、观测和网络接口的对照来源，不改变当前 mjlab 的迁移边界。

当前项目的 `src/tasks/wbc/go2_piper` 还是 MDP 骨架，本文不是“当前功能说明”，而是实现顺序和核对标准。文档中的来源行为来自已读源码；没有运行 `list_envs.py`、训练或播放，因此不把任何入口写成已验证可用。

## 1. 先区分三个项目

### 1.1 来源：LeggedManip Lab

来源项目提供要保持的任务语义：

```text
/home/kk/LeggedManip_Lab/source/LeggedManip_Lab
└── LeggedManip_Lab/tasks/manager_based/leggedmanip_lab
    ├── leggedmanip_lab_env_cfg.py
    ├── mdp/
    │   ├── rewards.py
    │   ├── observations.py
    │   ├── events.py
    │   ├── curriculums.py
    │   ├── cfg/command_cfg.py
    │   ├── pose_command_b.py
    │   └── pose_command_wbc.py
    └── config/go2_piper/
        ├── flat_env_cfg.py
        ├── wbc_env_cfg.py
        └── __init__.py
```

需要迁移的是 command、observation、reward、termination、event、curriculum 的输入输出和坐标约定，不是文件名本身。

### 1.2 参考组织：unitree_rl_mjlab

`unitree_rl_mjlab` 提供当前 mjlab 风格的目录和入口参考：

| 路径 | 参考作用 |
| --- | --- |
| `src/tasks/velocity/velocity_env_cfg.py` | 共享 EnvCfg factory，集中组装 scene、observation、action、command、event、reward、termination、curriculum、sim 等 |
| `src/tasks/velocity/config/go2/env_cfgs.py` | Go2 的实体、传感器和 train/play 覆盖 |
| `src/tasks/velocity/config/go2/rl_cfg.py` | PPO/RSL-RL runner 配置 |
| `src/tasks/velocity/config/go2/__init__.py` | task 注册，绑定 `env_cfg`、`play_env_cfg`、`rl_cfg`、`runner_cls` |
| `src/tasks/__init__.py` | 自动导入任务包，使 task 可以被枚举 |
| `scripts/list_envs.py` | 列出 task |
| `scripts/train.py` | 加载 task、构造环境、包装向量环境、启动 runner |
| `scripts/play.py` | 加载 play 配置和 checkpoint |

这个仓库使用 `mjlab==1.2.0`，当前项目依赖 `mjlab==1.6.0`。因此只参考分层和调用关系，字段名必须以当前版本 API 为准。

### 1.3 当前目标：legged_wbc_mjlab

当前需要填充的目录是：

```text
src/tasks/wbc/go2_piper/
├── __init__.py
├── wbc_env_cfg.py
└── mdp/
    ├── __init__.py
    ├── observations.py
    ├── rewards.py
    ├── curriculums.py
    ├── events.py
    └── cfg/command_cfg.py
```

已确认的当前状态：

| 文件 | 当前状态 |
| --- | --- |
| `wbc_env_cfg.py` | 已有 mjlab import 骨架，但还没有 EnvCfg 和 manager 组合 |
| `__init__.py` | 空文件 |
| `mdp/__init__.py` | 空文件 |
| `mdp/observations.py` | 空文件 |
| `mdp/rewards.py` | 空文件 |
| `mdp/curriculums.py` | 空文件 |
| `mdp/cfg/command_cfg.py` | 空文件 |
| `mdp/events.py` | 当前只有 3 个空白行，没有 event 配置 |
| `src/tasks/__init__.py` | 空文件 |
| `scripts/train.py` | 空壳 |
| `scripts/play.py` | 空壳 |
| `scripts/list_envs.py` | 空壳 |

所以当前文档的正确表述是“待迁移、待接线、待验证”，不能写成已经有可运行的 Go2 Piper task。

## 2. 建议的文件职责

先按职责分文件，再写 term。不要把所有内容直接塞进 `wbc_env_cfg.py`。

| 文件 | 应该放什么 |
| --- | --- |
| `mdp/__init__.py` | 统一导出当前 Go2 Piper 使用的 MDP term、command cfg 和 generator |
| `mdp/cfg/command_cfg.py` | `base_velocity`、`ee_pose` 的配置类型；绑定 generator 和 ranges |
| `mdp/observations.py` | 自定义 base、joint、feet、末端相对位姿、command passthrough term |
| `mdp/rewards.py` | 末端跟踪、base 跟踪、joint/action/feet penalty |
| `mdp/events.py` | inertia randomization 等自定义 event |
| `mdp/curriculums.py` | base velocity、yaw、ee pose 的 range 更新逻辑 |
| command generator 文件 | Flat/body pose 和 WBC mixed-frame pose 的采样、metric、坐标变换 |
| `wbc_env_cfg.py` | scene、commands、actions、observations、rewards、terminations、events、curriculum 的组合 |
| `go2_piper/__init__.py` | 注册 train/play task，并绑定 env config 和 RL config |
| `config/go2_piper/go2_piper_config.py` | 机器人控制参数、默认姿态、action scale、decimation 等机器人侧配置 |

`unitree_rl_mjlab` 的 `velocity_env_cfg.py` 可以作为 EnvCfg 分层参考，但不要直接复制它的 velocity task 语义。

## 3. 实施顺序

按下面顺序实现。每一步先核对输入和名称，再写 term；否则后面的 observation、reward 会在名称错误上反复返工。

1. asset/entity/name mapping
2. joint position action
3. base velocity 和 ee pose command
4. actor/policy 与 critic observation
5. reward
6. termination、event、curriculum
7. EnvCfg 组合
8. task 注册、RL config、train/play/list 入口
9. 静态核对后再运行入口

## 4. 第一步：资产和命名映射

### 4.1 XML 中已经确认的名称

来源文件：

```text
src/assets/robots/go2_piper/xmls/go2piper.xml
```

已读到的 body/joint 关系：

| 类别 | 名称 |
| --- | --- |
| base body | `base_link` |
| 腿部 body | `FL/FR/RL/RR_hip`、`FL/FR/RL/RR_thigh`、`FL/FR/RL/RR_calf`、`FL/FR/RL/RR_foot` |
| Piper body | `Piper/link1` 到 `Piper/link8`、`Piper/end_effector` 表示嵌套在 `Piper` body 下的 `link1...link8`、`end_effector`；slash 是层级路径表达，不一定是 MJCF body name，entity 查询需确认裸名/路径 |
| 腿部 joint | `FL/FR/RL/RR_{hip,thigh,calf}_joint` |
| Piper joint | `joint1` 到 `joint6` |

来源 MDP 中使用的逻辑名称是：

| 来源名称 | 当前 mjlab 要绑定的对象 |
| --- | --- |
| `robot` | robot entity |
| `base` | base body。当前 XML 实名是 `base_link`，不能不核对就直接写 `base` |
| `link0` | Piper 根参考系。需要确认当前 XML/entity 中对应的 body 或 frame |
| `end_effector` | 当前 XML 中的 `Piper/end_effector`，需确认 entity 查找方式 |
| `.*_foot` | 四个足端 body/site/geom 的实际匹配对象 |
| `FR/FL/RR/RL_*_joint` | 12 个腿关节，顺序必须固定 |
| `joint.*` | 机械臂关节匹配规则；当前 XML 实际为 `joint1...joint6` |

### 4.2 constants 和 actuator 先核对

机器人配置文件：

```text
src/assets/robots/go2_piper/go2_piper_constants.py
src/config/go2_piper/go2_piper_config.py
```

其中已经有 `get_go2_piper_robot_cfg()`、`INIT_STATE`、腿部默认姿态、机械臂默认姿态、action scale、decimation 以及 actuator 参数入口。

实现 MDP 前必须先确认：

- `joint1...joint6` 是否都在 entity 的 joint/actuator 匹配中；
- `GO2_PIPER_ARM_JOINT5` 已定义，但 `GO2PIPER_ARTICULATION.actuators` 当前确实只列到 joint4 和 joint6；joint5 actuator 是当前动作接线的阻塞项，必须先处理；
- 当前 foot regex `^[FR][LR]_foot_collision$` 在原始 XML 命名中没有直接匹配项；必须确认编译/重命名后的 geom 是否使用该名字，或者修正 regex 后再接 feet contact/collision MDP；
- `base_link`、Piper 根 body、末端 body 在当前 mjlab EntityCfg 中分别如何查找；
- action 维度是否确实为 12 个腿关节 + 6 个机械臂关节。

完成标志：名称表、关节顺序、实体查找方式和 action 维度已经写成明确映射，后续 MDP 不再使用未确认的名字。

## 5. 第二步：joint position action

来源项目的 action term：

| 项 | 来源约定 |
| --- | --- |
| 类型 | joint position action |
| 顺序 | 12 个腿关节在前，机械臂 `joint.*` 在后 |
| scale | `0.25` |
| offset | 使用默认 joint position offset |
| preserve order | `True` |
| clip | `(-10.0, 10.0)` |

当前 Go2 Piper 的实际动作顺序应固定为：

```text
FR_hip, FR_thigh, FR_calf,
FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf,
RL_hip, RL_thigh, RL_calf,
joint1, joint2, joint3, joint4, joint5, joint6
```

这份顺序是迁移约束，不代表当前 action 已实现。

参考 `unitree_rl_mjlab` 的写法，mjlab 侧常见配置字段是 `entity_name` 和 `actuator_names`，例如其 velocity task 使用 `entity_name="robot"`、`actuator_names=(".*",)`、`scale=0.25`、`use_default_offset=True`。当前项目使用 `mjlab==1.6.0`，要核对当前版本对应字段，不能只复制旧版本字段名。

完成标志：

- 策略输出维度与实际 18 个关节数一致；
- 输出索引可以逐项映射到上面的关节顺序；
- entity 名称是 `robot` 或当前项目实际注册名，并且所有 actuator 都能解析；
- scale、default offset、clip 的含义已在当前 API 中确认。

## 6. 第三步：commands

来源项目有两个 command：

| key | 输出/用途 |
| --- | --- |
| `base_velocity` | base 的 `vx`、`vy`、yaw rate |
| `ee_pose` | 末端 pose，shape 为 `(num_envs, 7)`，格式为 `(x, y, z, qw, qx, qy, qz)` |

### 6.1 base velocity

来源初始范围：

| 分量 | 范围 |
| --- | --- |
| `lin_vel_x` | `(-0.2, 0.2)` |
| `lin_vel_y` | `(-0.2, 0.2)` |
| `ang_vel_z` | `(-0.2, 0.2)` |

来源极限范围：

| 分量 | 范围 |
| --- | --- |
| `lin_vel_x` | `(-1.0, 1.0)` |
| `lin_vel_y` | `(-0.6, 0.6)` |
| `ang_vel_z` | `(-1.0, 1.0)` |

### 6.2 ee pose

基础 body-frame command 的初始范围：

| 分量 | 范围 |
| --- | --- |
| `pos_x` | `(0.4, 0.45)` |
| `pos_y` | `(-0.05, 0.05)` |
| `pos_z` | `(0.05, 0.05)` |
| `roll/pitch/yaw` | `(0.0, 0.0)` |

基础极限范围：

| 分量 | 范围 |
| --- | --- |
| `pos_x` | `(0.4, 0.7)` |
| `pos_y` | `(-0.35, 0.35)` |
| `pos_z` | `(-0.2, 0.5)` |

WBC command 的初始范围：

| 分量 | 范围 |
| --- | --- |
| `pos_x` | `(0.4, 0.45)` |
| `pos_y` | `(-0.05, 0.05)` |
| `pos_z` | `(0.5, 0.5)` |
| `roll/pitch/yaw` | `(0.0, 0.0)` |

WBC 极限范围：

| 分量 | 范围 |
| --- | --- |
| `pos_x` | `(0.45, 0.7)` |
| `pos_y` | `(-0.35, 0.35)` |
| `pos_z` | `(0.1, 0.8)` |
| `roll` | `(-pi/3, pi/3)` |
| `pitch` | `(-pi/4, pi/4)` |
| `yaw` | `(-pi/6, pi/6)` |

### 6.3 必须保留的坐标约定

这是 Go2 Piper MDP 最容易迁移错的地方。

| 模式 | 位置 command | metric/reward 的比较方式 |
| --- | --- | --- |
| Flat/body | xyz 都在 `link0`/root frame | command 转 world 后与 EE world pose 比较 |
| WBC | x/y 在 `link0` frame，z 在 world frame | xy 误差在 link0 frame，z 误差在 world frame |

WBC 还会根据 `link0` 到目标点方向计算 pitch/yaw，再叠加采样偏移，并对 Euler 角做 `[-pi/4, pi/3]` 限制。来源实现中，位置 xyz 从 `cfg.ranges` 采样，而 roll/pitch/yaw 偏移从 `cfg.limit_ranges` 采样，之后再执行 Euler clamp；迁移时不要将姿态采样逻辑误写成全部使用 `cfg.ranges`。不要把 WBC command 改写成纯 world frame 或纯 body frame。

`unitree_rl_mjlab` 的 velocity command key 是 `twist`，输出是 3 维 body-frame `[vx, vy, wz]`；这是参考项目的 velocity task 约定，不等于当前 Go2 Piper 的 7 维 `ee_pose`，也不要求当前任务把 `base_velocity` 改名成 `twist`。

完成标志：

- command key、输出 shape、单位和 frame 写入 EnvCfg 设计；
- Flat 与 WBC 使用不同 generator/config 或等价的明确分支；
- metric、position reward、observation 中对 frame 的使用完全一致；
- 初始范围和 limit range 分开保存。

## 7. 第四步：observations

来源项目把 observation 分成 `PolicyCfg` 和 `CriticCfg`；`unitree_rl_mjlab` 的 mjlab 参考则使用 `actor` 和 `critic`。当前项目应以 mjlab 1.6.0 的 observation group API 为准，但不能混淆两层语义：

- actor/policy：部署策略真正接收的输入；
- critic：训练时可以包含 privileged/state 信息。

### 7.1 actor/policy term

来源 term：

| term | 来源函数 | noise/scale |
| --- | --- | --- |
| `base_ang_vel` | `base_ang_vel` | noise `[-0.2,0.2]`，scale `0.2` |
| `projected_gravity` | `projected_gravity` | noise `[-0.05,0.05]` |
| `joint_pos` | `joint_pos_rel` | noise `[-0.01,0.01]` |
| `joint_vel` | `joint_vel_rel` | noise `[-1.5,1.5]`，scale `0.05` |
| `actions` | `last_action` | 无额外 noise |
| `velocity_commands` | `generated_commands(base_velocity)` | 直接取 command |
| `pos_commands` | `generated_commands(ee_pose)` | 直接取 command |

来源 policy group：

| 设置 | 值 |
| --- | --- |
| corruption | `True` |
| concatenate | `True` |
| history | `3` |

### 7.2 critic term

critic 至少需要包含 actor/policy 的主要状态，并增加：

| term | 内容 |
| --- | --- |
| `base_lin_vel` | base 线速度 |
| `joint_torques` | 关节力矩 |
| `feet_contact` | 足端接触力范数 |
| `ee_link0_rel_pose` | EE 相对 link0 的 pose，3 维位置 + 4 维四元数 |

critic 的 corruption 为 `False`，并保持 concatenate。

### 7.3 observation 实现顺序

1. 先实现 `robot`、`base_link`、`link0`、`end_effector` 的 entity 查找。
2. 再实现 joint position/velocity，并强制使用 action 的同一关节顺序。
3. 再实现 feet contact；确认传感器返回的维度是四足端还是 body 全集。
4. 再实现 EE 相对 `link0` 的 position/quaternion，固定输出 7 维。
5. 最后组装 actor/critic，配置 noise、scale、history 和 corruption。

不要先写 observation 总维度。先逐项记录每个 term 的 shape，确认拼接顺序后再用运行时打印或检查结果确认最终维度。

完成标志：actor 与 critic 的 term 列表、shape、noise、scale、history 和 corruption 都已经明确，且 critic 的 privileged term 没有进入 actor。

## 8. 第五步：rewards

### 8.1 末端和 base tracking

| 类别 | term | 约定 |
| --- | --- | --- |
| Flat 末端位置 | `position_command_b_error_exp` | link0/body frame |
| WBC 末端位置 | `position_command_error_exp` | xy-link0 + z-world |
| 末端姿态 | `orientation_command_error` | quaternion shortest-path error |
| base 线速度 | `track_lin_vel_xy_exp` | 跟踪 base xy velocity |
| base 角速度 | `track_ang_vel_z_exp` | 跟踪 yaw rate |
| base 高度 | `base_height_tracking` | 指数高度跟踪 |

position reward 必须和 command generator 使用同一个 frame 约定。WBC 环境不能继续使用 `position_command_b_error_exp`。

### 8.2 penalty term

需要从来源项目逐项迁移和核对：

| 类别 | term |
| --- | --- |
| root | `lin_vel_z_l2`、`ang_vel_xy_l2`、`flat_orientation_l2` |
| joint | `joint_torques_l2`、`joint_torques_max`、`joint_acc_l2`、`joint_power`、`joint_deviation_l1`、`joint_mirror`、`joint_pos_limits` |
| action | `action_rate_l2` |
| feet/gait | `feet_slide`、`feet_air_time`、`feet_long_air_penalty`、`air_time_variance_penalty` |

### 8.3 基础权重与 WBC 覆盖

基础配置重要权重：

| term 方向 | weight |
| --- | --- |
| EE position | `3.0` |
| EE orientation | `-1.5` |
| base linear velocity | `3.0` |
| base angular velocity | `1.5` |
| base height | `1.0` |

Go2 Piper WBC 覆盖：

| 项 | WBC 值 |
| --- | --- |
| position function | `position_command_error_exp` |
| position weight | `4.5` |
| orientation weight | `-4.0` |
| linear velocity weight | `3.5` |
| angular velocity weight | `2.5` |
| base height weight | `0.25` |
| flat orientation weight | `-0.5` |
| feet long air weight | `-1.0` |
| air time variance weight | `-1.5` |

来源环境会把 weight 为 0 的 reward term 设为 `None`。迁移时要确认当前 mjlab 的 reward manager 是否也允许后续 curriculum/日志访问被置空的 term。

完成标志：所有 reward 都有函数、输入 entity、frame、单位、weight；Flat/WBC 的 position function 和 WBC 覆盖值可以从配置中直接看出来。

## 9. 第六步：terminations、events、curriculum

### 9.1 Terminations

来源 termination：

| term | 配置 |
| --- | --- |
| `time_out` | episode timeout |
| `base_contact` | `illegal_contact`，sensor 为 `contact_forces`，body 为 `base`，threshold `0.5` |
| `bad_orientation` | `bad_orientation(limit_angle=1.0)` |

Flat Play 会把 `base_contact` 和 `bad_orientation` 设为 `None`。只读分析中没有看到 WBC Play 同样关闭这两项，迁移时不要擅自统一。

### 9.2 Events

基础事件分三类：

| 时机 | event |
| --- | --- |
| startup | `randomize_rigid_body_material`、`randomize_rigid_body_mass`、`randomize_rigid_body_com`、`randomize_rigid_body_inertia` |
| reset | `randomize_actuator_gains`、`apply_external_force_torque`、`reset_root_state_uniform`、`reset_joints_by_scale` |
| interval | `push_by_setting_velocity`，interval `(10.0,15.0)`，xy velocity `(-0.5,0.5)` |

Go2 Piper Flat/WBC 都设置 `push_robot = None`。

自定义 inertia event 只随机 inertia tensor 对角线：

| 分量 | tensor index |
| --- | --- |
| `xx` | `0` |
| `yy` | `4` |
| `zz` | `8` |

支持 `add`、`scale`、`abs`，默认 distribution 为 `uniform`。

### 9.3 Curriculum

来源 curriculum term：

| term | command |
| --- | --- |
| `lin_vel_cmd_levels` | `base_velocity` |
| `ang_vel_cmd_levels` | `base_velocity` |
| `pos_cmd_levels` | `ee_pose` |

默认 `curriculum_enabled=False`。关闭时，来源代码会把当前 `cfg.ranges` 直接设为 `limit_ranges`，返回 `-1.0`；这意味着关闭 curriculum 不等于使用小的初始范围。

开启时，在 episode 边界用 episode reward sum / episode length 与 `reward_term_cfg.weight * 0.8` 比较，达标后扩大 range 并 clamp 到 `limit_ranges`。

扩展步长：

| 项 | 步长 |
| --- | --- |
| linear/angular velocity | `[-0.05, 0.05]` |
| EE position | `[-0.05, 0.05]` |
| EE orientation | `[-pi/18, pi/18]` |

迁移时修正来源代码中的字段拼写兼容问题：它除了 `curriculum_enabled` 还读取了 `currirulum_enabled`。当前实现应只保留一个正确字段，并在配置里明确默认值。

## 10. EnvCfg 和 train/play task 组合

### 10.1 EnvCfg 应该组合什么

参考 `unitree_rl_mjlab/src/tasks/velocity/velocity_env_cfg.py`，EnvCfg 至少要明确这些入口：

```text
scene
observations
actions
commands
events
rewards
terminations
curriculum
metrics
viewer
sim
decimation
episode_length_s
```

来源环境参数：

| 参数 | 值 |
| --- | --- |
| `scene.num_envs` | `4096` |
| `scene.env_spacing` | `2.5` |
| `decimation` | `4` |
| `episode_length_s` | `20.0` |
| `sim.dt` | `0.005` |
| `render_interval` | `decimation` |

contact sensor 来源配置为 `{ENV_REGEX_NS}/Robot/.*`、`history_length=3`、`track_air_time=True`。在 mjlab 中需要按当前 sensor API 找到等价配置，不能直接假定 IsaacLab 的路径格式可用。

### 10.2 Flat/WBC train/play

至少需要四个 task 语义：

| 模式 | 作用 |
| --- | --- |
| Go2 Piper Flat train | body-frame ee pose command |
| Go2 Piper Flat play | Flat 的固定/扩大 command 范围与 play termination |
| Go2 Piper WBC train | mixed-frame WBC ee pose command |
| Go2 Piper WBC play | WBC 的固定/扩大 command 范围与 play 覆盖 |

来源项目的 gym id 是：

```text
GO2-PIPER-Flat
GO2-PIPER-Flat-Play
GO2-PIPER-WBC
GO2-PIPER-WBC-Play
```

当前 mjlab 不一定继续使用 IsaacLab gym id 机制。`unitree_rl_mjlab` 的做法是 task 注册对象绑定 `env_cfg`、`play_env_cfg`、`rl_cfg` 和 `runner_cls`。当前项目应先确认 mjlab 1.6.0 的注册 API，再决定 task id 和注册字段，不能直接复制旧 gym registration。

### 10.3 RL 和脚本入口

这部分不是 MDP term 本身，但不接通就无法验证 MDP。

参考调用关系：

```text
task id
  -> env_cfg / play_env_cfg
  -> ManagerBasedRlEnv
  -> RslRlVecEnvWrapper
  -> runner_cls
  -> train / play
```

当前项目的 `scripts/train.py`、`scripts/play.py`、`scripts/list_envs.py` 仍是空壳，所以只能列为后续接线任务。播放时优先使用显式本地 checkpoint；参考项目的 trained/W&B 自动路径本身存在字段不完整问题，不要把自动下载当成当前功能。

## 11. 不要直接照搬 unitree_rl_mjlab

以下内容必须分开：

| 参考项目内容 | Go2 Piper 不能直接照搬的原因 |
| --- | --- |
| `twist` command | unitree velocity task 的 3 维 body velocity，不是 Go2 Piper 的 7 维 `ee_pose` |
| `actor/critic` 命名 | 只是 mjlab 参考 group 命名；来源项目使用 policy/critic，当前要按当前 API 接口映射 |
| Go2 的 `base_link`/feet sensor 配置 | Go2 Piper XML 同时包含四足和 Piper，entity/body/site 名称不同 |
| `mjlab==1.2.0` 的字段 | 当前项目是 `mjlab==1.6.0`，字段和 manager API 需要重新核对 |
| unitree velocity reward | 任务目标是速度跟踪，不能代替末端 pose tracking reward |
| unitree play 设置 | 只能参考 train/play 覆盖方式，不能替换来源项目 Flat/WBC 的 termination、command 和 reward 语义 |

## 12. 最小实施清单

### A. 资产和动作

- [ ] `base_link`、Piper 根参考系、`Piper/end_effector` 能被当前 entity API 找到。
- [ ] 四个足端的 body/site/geom 实名和 contact sensor 匹配方式已确认。
- [ ] 12 个腿关节 + `joint1...joint6` 的顺序固定。
- [ ] joint5 actuator 是否遗漏已处理。
- [ ] action scale `0.25`、default offset、clip 的当前 API 字段已确认。

### B. Commands

- [ ] `base_velocity` 的 key、shape、frame、range 已定义。
- [ ] `ee_pose` 输出 `(N,7)`，四元数顺序为 `qw,qx,qy,qz`。
- [ ] Flat 为 link0/body frame。
- [ ] WBC 为 xy-link0 + z-world。
- [ ] WBC 的 generator、metric、position reward 使用同一坐标约定。
- [ ] 初始 range、limit range、play range 没有混用。

### C. Observations

- [ ] actor/policy 与 critic group 已分开。
- [ ] actor 的 noise、scale、history=3、corruption 已确认。
- [ ] critic 的 privileged terms 没有进入 actor。
- [ ] joint、feet、EE relative pose 的 shape 已逐项核对。
- [ ] `generated_commands` 取到正确的 velocity 和 ee pose command。

### D. Rewards、termination、events、curriculum

- [ ] Flat/WBC 的 position reward function 没有串用。
- [ ] WBC 的 4.5、-4.0、3.5、2.5、0.25 等覆盖项已单独配置。
- [ ] train/play 的 base contact 和 bad orientation 行为已分别记录。
- [ ] Go2 Piper 的 `push_robot=None` 已确认。
- [ ] inertia 只改 diagonal，operation 为 add/scale/abs 的语义已确认。
- [ ] curriculum 关闭时是否直接使用 limit range 已确认。

### E. 接线和验证

- [ ] `mdp/__init__.py` 能导出所有被 EnvCfg 引用的 term。
- [ ] `wbc_env_cfg.py` 能组合所有 manager。
- [ ] task 注册入口能被枚举。
- [ ] `env_cfg`、`play_env_cfg`、RL config 的关系明确。
- [ ] `list_envs.py`、`train.py`、`play.py` 不再是空壳。
- [ ] 静态检查通过后，再运行 task 枚举、单环境 reset/step、play、短训练。

## 13. 当前未验证项

以下内容本次没有运行，不能写成完成：

- 当前项目的 task registry 是否能枚举 Go2 Piper；
- mjlab 1.6.0 下 command、observation、reward manager 的具体字段是否与参考项目相同；
- action、actor、critic 的最终维度；
- XML 中所有 body/site/geom regex 是否都能被 entity API 解析；
- reset/step 后 command frame 和 reward 数值是否正确；
- train/play 是否能加载环境、checkpoint 和 viewer；
- ONNX 导出、真实部署以及 sim-to-real 行为。

最终顺序只有一句话：先把名称和关节顺序钉死，再接 action；再接 command 的 shape/frame；然后接 actor/critic observation；最后接 reward、termination、event、curriculum，完成 EnvCfg 和 task 注册后才开始运行验证。

## 14. Deep-WBC 参考：只借鉴数据流，不复制结构

Deep-Whole-Body-Control 使用的是 legacy `legged_gym + rsl_rl`，不是当前 mjlab 的 manager-based API。它可以帮助我们核对 action、observation、command、teacher control 和 actor/critic 的数据流，但不能把 `WidowGo1RoughCfg`、legacy env class 或配置继承关系直接搬进 `wbc_env_cfg.py`。当前实现仍以 mjlab 1.6.0 的 manager、term 和 RL config API 为准。

### 14.1 `WidowGo1RoughCfg` 的参考尺寸

下面的数字是 Deep-WBC 的参考值，不是当前 Go2 Piper 必须照搬的 MDP 维度：

| 项 | Deep-WBC 参考值 |
| --- | --- |
| action | 18 维 = 12 个腿部 action + 6 个机械臂 action |
| `num_proprio` | `2+3+20+20+18+4+3+3+3 = 76` |
| `history_len` | `10` |
| `num_priv` | `5+1+18 = 24` |
| `decimation` | `4` |
| `action_delay` | `2` |
| episode | `10s` |
| observation scale | `ang_vel=1`、`dof_pos=1`、`dof_vel=0.05` |
| observation/action clip | observation `100`，action `100` |

`widowGo1.py::compute_observations` 的 proprio 拼接顺序是：

```text
body orientation 2
base angular velocity 3
joint position 20
joint velocity 20
previous action 18
foot contact 4
base command 3
current EE goal 3
EE orientation delta Euler 3
```

合计为 76 维。privileged 部分使用 mass、friction、motor-strength 参数，并加上 history；`num_priv=24` 是这个来源的参考计数。这里的 76/24 不能反推当前 Go2 Piper 的 actor/critic 维度，当前任务仍要按自己的 `observations.py` term、command frame 和 mjlab group API 重新核对。

### 14.2 command、goal 和 curriculum

- base command 每 `3s` 重采样；当前实现只采样 `lin_vel_x` 和 yaw rate，`lin_vel_y` 固定为 `0`，并将过小命令清零。
- EE goal 使用球坐标 `l/p/y`。`traj_time` 在 `1-3s` 采样，`hold_time` 在 `0.5-2s` 采样，并使用 `10` 个 collision-check samples。
- goal 生成时检查 collision bounds 和 underground limit；goal 在 start/target 之间插值，episode/reset 时重采样。
- curriculum 同时扩展底盘速度范围、末端目标范围和 tracking reward scale。

这些是 Deep-WBC 的 command/goal 行为。当前 Go2 Piper 的 `ee_pose` 仍按前文保持 7 维 pose 和 Flat/WBC frame；不能因为 Deep-WBC 使用球坐标，就直接把当前 command 改成 sphere command。

### 14.3 action、delay 和低层控制边界

Deep-WBC 的 `step` 先执行 action reorder、clip 和 delay，再按 `decimation` 调用 `_compute_torques`。PD 目标形式为：

```text
target = action_scale * action + default_joint_pos
```

腿部和机械臂使用不同的 stiffness、damping 和 action scale；配置还支持 adaptive arm gains 和 torque supervision。迁移时先把 action 顺序、18 维 shape、delay 和 decimation 作为接口约束记录下来，不要先假定当前 mjlab 已经有同样的 delay term。

`get_arm_ee_control_torques` 使用 Jacobian、mass matrix 和 gravity compensation 计算 operational-space torque。这属于 teacher/low-level control boundary，不是 MDP 的 joint-position action 输出；当前 action manager 输出什么，仍由当前 mjlab action 配置决定。

### 14.4 reward 如何借鉴

保留“locomotion 与 manipulation 分开统计、分开调权”的思路。Deep-WBC 的 reward 覆盖 EE sphere/cart tracking、EE orientation、linear/yaw tracking、leg/arm energy、leg action、foot contact、survival 等项。

当前 Go2 Piper 仍以来源项目的 Flat/WBC reward 为准。Deep-WBC 的 sphere/cart 目标和权重不能直接替换现有的 `position_command_b_error_exp` / `position_command_error_exp`：先保持 Flat/WBC 各自的 frame 和 reward function，再用单独实验验证新的目标表示。

### 14.5 actor/critic 网络边界

`rsl_rl/modules/actor_critic.py` 的结构包括 priv encoder、history encoder、shared actor backbone、独立的 leg control head 和 arm control head。参考配置为：

| 项 | 参考值 |
| --- | --- |
| actor/critic hidden dims | `128` |
| `priv_encoder_dims` | `[64, 20]` |
| leg/arm head | `[128, 128]` |
| `num_leg_actions` / `num_arm_actions` | `12` / `6` |

actor 输出按 legs + arm 拼接；critic 输入 privileged/state，并输出 leg/arm value。这是 RL/network 层的设计，不是 MDP term。只有当前 mjlab 的 RL config/API 支持相同输入输出契约时，才按这个方向参考；不要为了复刻网络而把 priv encoder 或 control head 写进 `observations.py` 或 `rewards.py`。

### 14.6 Deep-WBC 到当前 mjlab 的映射

| Deep-WBC 来源 | 当前 mjlab 落点 | 迁移要求 |
| --- | --- | --- |
| `compute_observations` | `observations.py` actor/critic | 按当前 group API 重新声明 term、shape、history 和 privileged 输入 |
| `_resample_commands` / `_resample_ee_goal` | `command_cfg.py` + command generator | 保留采样周期、frame、reset/插值边界；不复制 legacy class |
| domain randomization / reset | `events.py` | 用当前 `EventTermCfg` 和 event manager 接口重写 |
| `_reward_*` | `rewards.py` | 维持 Flat/WBC reward 语义，再单独评估 Deep-WBC 目标 |
| `check_termination` | terminations | 映射到当前 termination manager，不把 legacy `done` 逻辑直接搬入 env class |
| policy config | RL config | 仅在 actor/critic API 支持时参考 encoder 和分头 |
| `_compute_torques` | action / low-level control boundary | 明确 PD、delay、decimation 和 teacher torque 的边界 |

## 15. 当前第一阶段怎么做

这是基于上述来源差异做出的工程建议，不是已经运行验证的结果。

第一阶段保持当前 7-D `ee_pose` 和 Flat/WBC frame，不直接引入 Deep-WBC 的 sphere command。先按下面顺序闭环：

1. 固定 actor/critic 的 term shape，确认 action 是 18 维，且 12 个腿部 action 在前、6 个机械臂 action 在后。
2. 固定 command frame，完成 command 生成、observation 输入、reward 比较和 reset 时的 frame 一致性。
3. 让 reset/step 闭环通过，确认 action reorder、clip、低层控制边界和 episode termination 的数据流。
4. 在 shape、frame 和 reset/step 都可核对后，再增加 goal interpolation。
5. 最后再评估 priv encoder、分腿/臂 actor heads 和 torque supervision；它们属于 RL/network 或 low-level control 扩展，不应提前混入 MDP term。

当前 `src/tasks/wbc/go2_piper` 的其余空壳/未实现事实仍然有效：`wbc_env_cfg.py`、`observations.py`、`rewards.py`、`curriculums.py`、`command_cfg.py`、两个 `__init__.py` 仍未完成，task 注册和 train/play/list 入口也未验证。

## 16. 参考来源

- [Deep-Whole-Body-Control 仓库](https://github.com/MarkFzp/Deep-Whole-Body-Control)
- [`widowGo1_config.py`](https://raw.githubusercontent.com/MarkFzp/Deep-Whole-Body-Control/main/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py)
- [`widowGo1.py`](https://raw.githubusercontent.com/MarkFzp/Deep-Whole-Body-Control/main/legged_gym/legged_gym/envs/widowGo1/widowGo1.py)
- [`actor_critic.py`](https://raw.githubusercontent.com/MarkFzp/Deep-Whole-Body-Control/main/rsl_rl/rsl_rl/modules/actor_critic.py)
