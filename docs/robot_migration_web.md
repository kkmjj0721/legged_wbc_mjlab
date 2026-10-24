# 本地四足机器人迁移工作台

工作台将一个标准 12 关节四足机器人接入本项目的训练、回放和导出流程。支持 RI-4438 的 PPO、HIM、HIM Symmetry、HIM LP-ACRL 模板，每种任务生成独立实现。

![配置与 3D 姿态预览](assets/robot_migration_workbench/configure.png)

上图为本地浏览器验收界面，使用 RI-4438 示例模型展示布局与足端标记。

## 安装与启动

首次安装环境时启用 Web 依赖：

```bash
bash setup.sh --extra web
```

已有可用训练环境时，可以只同步可选依赖并保留当前 PyTorch 后端：

```bash
python3 tools/uv_auto_sync.py --extra web --skip-torch -- --no-install-package torch
```

在项目根目录启动：

```bash
uv run --no-sync python -m scripts.robot_migration_web --port 5003
```

打开 <http://127.0.0.1:5003>。服务只绑定本机；3D 预览另使用一个自动分配的本机端口。页面、3D 前端和字体不需要 CDN。停止终端服务会同时释放预览与验证进程。

## 迁移步骤

1. **导入模型**：填写模型根目录和相对入口 XML / URDF，或者上传 ZIP；可直接选择内置 RI-4438 / Go2 的 MJCF 示例。根目录需包含全部 include、网格和纹理资源。URDF 根据文件内容识别，转换后显示报告和可复制的 XML 源码。ZIP 有多个入口时需指定包内入口。
2. **选择模板**：多选 PPO、HIM、HIM Symmetry、HIM LP-ACRL，分别选择 Flat / Rough。LP-ACRL Rough 同时保留默认课程、Compact、Compact-Coverage、Compact-Uniform、A0–A3 和 Random-Uniform；Flat 使用原有课程。
3. **确认映射**：填写模块标识与任务前缀，检查机体、12 个关节的腿与角色、镜像符号、足端 site / body / geom 及机体和腿部碰撞。自动识别的映射是建议，必须人工确认。
4. **配置与姿态**：设置逐关节 PD、力矩和动作限幅、初始姿态、执行器延迟、观测延迟及环境参数。可使用 PD 参数分析，根据模型与电机信息计算候选增益，审阅后填入当前配置。高级区包括指令、重置、随机化、噪声、奖励、地形、课程和训练超参数。公共配置作用于全部选中任务，任务页签可以覆盖；草稿可保存和恢复。
5. **审阅**：检查任务 ID、输出文件树、代码和配置差异。修正阻断问题后勾选人工确认；改变配置后需重新审阅。
6. **生成与验证**：先暂存并检查，再写入新目录并再次验证。已有同名目标会拒绝覆盖；本次失败写入会回滚。完成后提供训练与零动作回放命令，也可单独运行 CPU 检查或固定预算 GPU 短仿真。

无模型 keyframe 时，初始关节角采用模板站姿作为起点，并限制在模型运动范围内；这不是从硬件测得的站姿。PD、延迟和随机化默认值也只是模板参数，应按新机器人重新确认。执行器延迟以物理仿真步计，观测延迟以控制步计。

新模型的接触和约束数量可能不同于 RI-4438。迁移预设将 `environment.nconmax` 和 `environment.njmax` 分别设为至少 256 和 1500，可在环境配置中调整；这些是缓冲区容量，不是已标定的机器人参数。

3D 预览只执行 CPU 前向运动学，关节滑块、根位姿和足端标记用于检查填写结果。它不验证机器人是否能稳定站立；稳定性仍需通过仿真和后续训练判断。

## PD 参数分析

![PD 设计输入与两种候选对比](assets/robot_migration_workbench/pd_analysis.png)

上图使用 RI-4438 演示参数输入和计算结果，示例中的硬件补充值仅用于界面验收，不是实机标定数据。

步骤 04 的 PD 分析卡片复用当前模型、站姿、逐关节动作缩放、髋关节动作系数、目标角范围、力矩上限和执行器惯量。先选择公共配置或某个任务页签，再填写自然频率、阻尼比、姿态采样幅度与用于预算检查的动作幅度。初始的 5 Hz、阻尼比 1 是可修改的设计假设，不是依据实机标定得到的推荐值；自然频率也不等于电机实测带宽。

可补充实机底层 PD 更新频率、反馈到执行的延迟，以及逐关节的电机转子惯量、减速比、连续力矩和速度上限。未知项留空。转子惯量与减速比须成对填写，按 `J_arm = J_rotor × N²` 折算；多级传动应自行计算输出端等效惯量并填入逐关节 `armature`。未填写这对输入时，沿用当前任务配置或 XML 的惯量值，其来源仍可能只是模板假设。

工具提供两组候选：

- **模型负载惯量法**：在当前站姿及各关节类型的正负偏移姿态计算 MuJoCo 质量矩阵，取逐关节 `M[i,i]` 的采样最大值作为设计惯量。它表示基座及其他自由度锁定、无接触约束时的惯量，包含当前配置的 `armature`，不重复相加。结果显示采样最小值、当前姿态值、最大值及候选响应范围；有限姿态采样不能覆盖整个工作空间。
- **电机惯量基线**：仅用电机折算惯量计算增益，供参考项目公式对照；未计入连杆负载。某关节惯量为零时该方法不可用，可继续使用模型法或补齐执行器参数。

两法使用 `Kp = J × (2πf)²`。模型法的 `Kd = max(0, 2ζJ × 2πf − b)` 中，`b` 是配置或 XML 的被动黏性阻尼，发生截零时会提示实际阻尼比与目标不同；电机基线沿用 `Kd = 2ζJ_arm × 2πf`。报告会说明单关节近似、重力/接触、离散控制和饱和的适用边界。

增益分组可选择自动、逐关节或左右成对。自动模式在 HIM Symmetry 页签，或公共配置包含 Symmetry 任务时，对前腿和后腿的左右镜像关节分别共用候选增益。模型法采用配对的最大采样惯量，并取两侧所需阻尼增益的最大值；各关节实际惯量和预计响应仍分别显示。这是满足当前对称训练契约的工程选择，不是论文声称的通用最优整定。原有 armature、被动阻尼、限矩或动作缩放若不满足镜像要求，只提示差异，不替用户修改这些物理输入。

力矩预算按照当前零目标速度、零前馈的位置控制语义计算。在默认关节角、零速度下，检查正负指定动作幅度经实际动作缩放和目标角限幅后的比例项力矩，并与配置限矩及已填连续力矩比较。`Kp × 实际动作缩放` 是单位动作变化对应的比例项力矩增量，不是机器人运行时的力矩上界。动作幅度是本次检查输入，不会更改策略的原始动作裁剪。

比较两法和当前参数后，选择候选并检查应用前后差异，再应用到当前页签。仅填入候选 KP、KD 与计算所用的 armature；动作缩放、力矩上限及其他任务覆盖保持原有值。公共配置仍会被各任务自己的覆盖值覆盖，应在对应页签重新计算和检查。更改配置或分析输入后，旧结果不能继续应用；恢复草稿后也需要重新计算。

分析输入、报告和应用记录随草稿保存，并包含在生成清单 `migration_manifest.json` 的 `spec.pd_analysis` 中；也可下载 JSON 报告单独审阅。这些是设计与估计记录，不是实机标定证明。连续力矩、速度和时序输入用于分析，不自动创建速度相关力矩包络或改写训练时序；当前生成任务仍使用内建位置执行器。支撑接触、反馈滤波、真实延迟、减速器摩擦、柔性和热限制需要后续仿真或实测验证。

方法参考：[BeyondMimic 附录 S1](https://arxiv.org/html/2508.08241v4)、[MjLab 执行器说明](https://mujocolab.github.io/mjlab/main/source/actuators.html)、[Modern Robotics 关节力矩控制](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-4-motion-control-with-torque-or-force-inputs-part-1-of-3/)。仿真与实机响应对齐参考 [Tan 等，2018](https://arxiv.org/html/1804.10332) 和 [Hwangbo 等，2019](https://arxiv.org/html/1901.08652)。

## URDF 转换

例如模型目录如下，根目录填写 `.../my_robot`，入口填写 `urdf/my_robot.urdf`，或上传包含这些文件的 ZIP：

```text
my_robot/
  urdf/my_robot.urdf
  meshes/...
  textures/...
```

转换后的 XML 参考 Go2 与 RI-4438 的组织方式：两空格缩进，依次组织 `compiler`、`default`、`asset`、`worldbody` 与 `sensor`；每个 body 显式列出惯量、关节、视觉和碰撞对象，再嵌套子 body。视觉和碰撞使用独立 class / group，资源使用包内相对路径。生成机器人 XML 与资源可以独立使用。

保留源 URDF 的 link / joint 名称、固定连接层级、坐标变换、关节轴和限位、质量、质心、完整惯量、mesh scale、颜色及碰撞几何。惯量原点的旋转转换到 body 坐标系，不自动修正非法惯量。无质量根节点与固定连接的有质量机身会分别识别，质量随机化绑定有质量机身，腿部惯量随机化排除无质量辅助节点；固定足端子节点也会用于足端绑定。

URDF 的 `limit.effort` 读入逐关节力矩上限，`dynamics` 阻尼和摩擦保留在 XML。PD、执行器延迟、动作缩放等仍需配置，`limit.velocity` 仅记录在转换报告中，不自动变成速度控制器。初始基座位置从原点开始，请设置离地高度并检查站姿。IMU 与足端 site 是依据命名和几何生成的候选，页面会提示人工确认；ROS/Gazebo 控制器和插件不迁移。

资源支持模型根目录内的相对路径、绝对路径和 `package://` 引用；转换输出统一改成相对路径。不读取目录外资源或符号链接。STL / OBJ 可直接处理，ASCII STL 转为二进制；超过 MuJoCo 面数限制的视觉 STL 按原始三角记录拆分，不简化。大型碰撞 STL 明确拒绝拆分，避免改变碰撞凸包。普通碰撞 mesh 仍按 MuJoCo 的凸包语义参与接触，凹形需要人工提供代理。DAE 依赖可用的可选解析器，缺少解析器时提示先导出 STL / OBJ，本功能不新增安装依赖。

导入副本及最终生成的机器人目录中包含 `conversion_report.json`、转换后的 XML / assets，以及 `source/` 下原 URDF 与实际使用的原资源。报告记录源文件摘要、资源、保留项、补充定义和警告；源文件中的 URI 保持原文，运行使用转换后的 XML。恢复草稿或重新导入转换包会保留该来源记录。

## 输出与命名

机器人标识如 `my_bot`，任务前缀如 `MyBot`。输出遵循项目现有结构：

```text
src/assets/robots/my_bot/                  # 模型资源与实体工厂
src/config/my_bot/                         # 参数、绑定、观测契约和生成清单
src/tasks/locomotion/my_bot_ppo/            # PPO
src/tasks/locomotion/my_bot_him/            # HIM
src/tasks/locomotion/my_bot_him_symmetry/   # HIM Symmetry
src/tasks/locomotion/my_bot_him_lpacrl/     # LP-ACRL
```

按所选模板生成对应目录。例如 `MyBot-Rough`、`MyBot-HIM-Flat`、`MyBot-HIM-Symmetry-Rough`、`MyBot-HIM-LPACRL-Rough-Compact`。通过 `import src.tasks` 自动注册，无需手动编辑总注册列表。

新任务拥有模型、完整 MDP、配置及 runner，实现不导入 `web_ui`，关闭或移除 Web 工具后仍可训练。生成清单 `migration_manifest.json` 记录模板和模型摘要、有效参数、输出路径和任务名称。

```bash
uv run --no-sync python -m scripts.train MyBot-HIM-Rough \
  --env.scene.num-envs 1024 --agent.seed 42

uv run --no-sync python -m scripts.play MyBot-HIM-Flat \
  --agent zero --num-envs 1 --viewer viser
```

训练与回放继续使用项目的 `src.envs.ManagerBasedRlEnv`。恢复、warm-start 和 LP-ACRL 指纹要求沿用对应任务契约；新机器人不会自动兼容 RI-4438 的旧检查点。

## 数据和验证结果

- `.artifacts/robot-migration/imports/`：导入资源的本地副本与解析记录，源模型保持原样。
- `.artifacts/robot-migration/drafts/`：版本化 JSON 草稿；保留导入资源才能恢复其预览和生成能力。
- `.artifacts/robot-migration/jobs/`：验证日志、结果、临时目录与生成事务清单。
- 正式训练输出仍由现有入口写入 `logs/rsl_rl/`。

CPU 检查包含注册、原任务配置摘要、模型编译、对象映射、实际执行器参数、初始状态、配置隔离和 HIM 布局。Symmetry 还检查左右配对、镜像符号、初始姿态和几何关系。

GPU 短仿真固定种子 42、4 个环境、200 个控制步，使用零动作检查运行、观测和奖励有限性。任务串行运行，可停止；GPU 不可用时显示“未运行”。页面不启动正式训练。

第一版接受独立浮动机器人 MJCF，或由 fixed / revolute 树组成的 URDF；最终要求 12 个具有有限运动范围的旋转关节，以及可明确绑定的足端和碰撞对象。URDF 的 mimic、continuous、prismatic、闭链和未展开 xacro 会给出错误提示。MJCF 场景入口、外部插件、绝对资源路径和对象名称中的 `/` 不接受。HIM 观测布局固定为 `phase47_v1`；非对称运动学需要选择兼容模板或先修正映射。

短仿真与短训仅验证相关链路，不证明策略已收敛或适合实机。开发验收的具体命令和结果见 [开发记录](development.md)。
