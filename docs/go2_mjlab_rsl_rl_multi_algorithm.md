# Go2 的 mjlab + 外部 rsl_rl 多算法工程方案与审查记录

本文针对仓库
`/home/sunteng/projects/legged_wbc_mjlab`，当前任务为
`src/tasks/locomotion/go2_ppo`，机器人资源为
`src/assets/robots/go2`。目标是让 mjlab 只负责环境和机器人仿真，让外部
`rsl_rl` 负责 PPO 以及后续算法，并且能在同一套 Go2 环境上切换算法。

本次现实审查快照：2026-09-08 18:16（Asia/Shanghai），`HEAD=6b880b0`。
环境版本为 `mjlab==1.6.0`、`rsl-rl-lib==5.4.2`、`torch==2.14.0`、
`mujoco==3.11.0`、`tensordict==0.14.1`。工作树仍有未提交的 `rl_cfg.py`、
`rl/runner.py`、缓存文件和一个未跟踪的 `rsl_rl/__init__.py` shim；因此本文把
`HEAD` 状态、当前工作树状态和建议实现分开描述。后续编辑代码后，应重新运行第 5 节命令，不能沿用旧快照结论。

当前快照的关键状态：

| 检查 | 结果 |
|---|---|
| `src/tasks/locomotion/__init__.py` | 已由 `HEAD` 跟踪，空文件，任务自动发现有效 |
| `rsl_rl/__init__.py` | 当前工作树未跟踪 shim；仓库根解析正常，不能视为已提交修复 |
| `rl_cfg.py` | 当前实时文件可 `py_compile`，actor/critic/algorithm 使用显式 class path |
| `VelocityOnPolicyRunner` | 可构造、可用 `runner.alg.get_policy()` 推理；保存时仍会访问不存在的 `logger.logger_type`（A.6.1 给出修正版） |
| Go2 asset | 未自包含，`spec.assets == 0`，导出验收失败 |

## 1. 当前基线

| 项目 | 当前事实 |
|---|---|
| mjlab | `pyproject.toml` 固定为 `mjlab==1.6.0` |
| rsl_rl | 本地 `rsl_rl/`，以 editable 方式作为 `rsl-rl-lib==5.4.2` 安装；当前工作树另有未跟踪根 shim |
| Go2 环境 | `Unitree-Go2-Flat`、`Unitree-Go2-Rough`，注册在 `src/tasks/locomotion/go2_ppo/config/__init__.py`；`src/tasks/locomotion/__init__.py` 已跟踪为空包标记，当前自动发现有效 |
| 环境配置 | `go_ppo_env_cfg.py` 提供共享配置，`config/env_cfgs.py` 做 Flat/Rough 和 train/play 覆盖 |
| PPO 配置 | `config/rl_cfg.py` 使用 `RslRlOnPolicyRunnerCfg`、显式 `MLPModel`、显式 `PPO` path；当前 dirty 快照的 `experiment_name` 为 `go2_ppo` |
| runner | 注册的是 `VelocityOnPolicyRunner`，继承 `MjlabOnPolicyRunner`；文件中还残留一段已注释的旧 runner 代码 |
| 训练入口 | `scripts/train.py` 已有完整的环境构造、`RslRlVecEnvWrapper`、runner、resume 和日志流程 |
| 播放/列举入口 | `scripts/play.py`、`scripts/list_envs.py` 当前为空文件，不能当作已完成能力 |
| Go2 mesh | 16 个 `.obj` 文件均在 `src/assets/robots/go2/xmls/assets/` 中，MJCF 中有 16 个 mesh |

当前数据流应保持为：

```mermaid
flowchart LR
  T[task id] --> R[mjlab task registry]
  R --> E[ManagerBasedRlEnvCfg]
  E --> M[ManagerBasedRlEnv]
  M --> W[RslRlVecEnvWrapper]
  W --> N[runner]
  N --> A[rsl_rl algorithm]
  A --> P[actor / critic / checkpoint]
```

## 2. 审查结果和阻断项

### 2.1 任务自动发现（历史阻断已修复）

`src/tasks/__init__.py` 调用 `import_packages(__name__, ...)`，mjlab 的扫描器使用
`pkgutil.iter_modules`，因此 `locomotion` 必须是可识别的包。初次审查时该目录没有
`__init__.py`；当前 `HEAD` 已包含一个空的 `src/tasks/locomotion/__init__.py`，它只负责让目录成为可扫描包。

实测结果：

```text
import src.tasks
list_tasks()                         # 当前 HEAD 出现 Unitree-Go2-Flat/Rough
```

在缺少该文件的历史状态下，`scripts/train.py` 启动时只导入 `src.tasks`，任务注册表中没有 Go2；当前
`python -m scripts.train Unitree-Go2-Flat --help` 已能列出 Go2 配置。

后续新任务继续放在该包下即可；不要再依赖手工 `import
src.tasks.locomotion.go2_ppo.config` 才能注册任务。

### 2.2 Python 包路径和外部 rsl_rl 被同名目录遮蔽（历史阻断；当前有未跟踪 shim）

当前仓库有两层目录：

```text
rsl_rl/                 # vendored 项目根，没有顶层 rsl_rl/__init__.py
└── rsl_rl/              # 真正的 Python 包
```

在没有仓库根 shim 的历史状态，从仓库根目录运行时 Python 会把外层目录识别成 namespace package：

```text
import rsl_rl
rsl_rl.__file__ is None
rsl_rl.__path__ == ['.../legged_wbc_mjlab/rsl_rl']
```

`rsl_rl.utils.resolve_callable("PPO")` 会扫描这个路径，误把
`rsl_rl/setup.py` 当成模块导入，随后触发 `setup()` 的 `SystemExit`。这不是算法实现错误，而是包布局错误。

同时，editable 安装当前把 `repo/src` 放进 `sys.path`，而源码和脚本大量使用 `from src...`。于是：

```text
python scripts/train.py       -> ModuleNotFoundError: src
python -m scripts.train       -> 能找到 src，但仍可能被外层 rsl_rl 遮蔽
```

当前工作树新增了未跟踪的 `rsl_rl/__init__.py` shim，将 `__path__` 指向内层
`rsl_rl/rsl_rl`；实测仓库根可以解析 `PPO`。这个 shim 还没有进入 `HEAD`，不应被当作完成的发布方案。稳定方案：

1. 把本地 rsl_rl 项目移到 `_vendor/rsl_rl` 或仓库外，并从该真实包路径 editable 安装；
2. 统一项目导入风格。要么全部使用已安装的 `assets/config/tasks`，要么在打包配置中明确安装 `src` 包并始终使用 `src.*`；
3. 在结构修复完成前，临时运行：

```bash
cd /home/sunteng/projects/legged_wbc_mjlab
export PYTHONPATH="$PWD/rsl_rl:$PWD"
```

这个顺序会让真正的 `rsl_rl/rsl_rl/__init__.py` 优先于外层 namespace。当前 shim 或该环境变量只用于过渡验证；长期应使用独立包布局或正式提交 shim。

### 2.3 Go2 asset 的“本地可编译、导出后丢失”问题（P0）

当前 `src/assets/robots/go2/xmls/go2.xml` 的差异如下：

```xml
<compiler angle="radian" autolimits="true" />
<mesh file="assets/base_0.obj" />
```

参考项目 `/home/sunteng/projects/unitree_rl_mjlab` 的
`src/assets/robots/unitree_go2/go2_constants.py` 和对应 MJCF 使用：

```xml
<compiler angle="radian" meshdir="assets" autolimits="true" />
<mesh file="base_0.obj" />
```

当前 `get_spec()` 只调用 `mujoco.MjSpec.from_file()`，没有把 mesh 字节写入
`spec.assets`。审查得到的状态是：

```text
spec.meshdir == ''
spec.assets == {}
直接从源码目录 MjSpec.compile() 可以成功
```

直接 compile 成功只能说明 MuJoCo 仍能从源 XML 所在目录解析文件，不能说明 mesh 已经进入 `spec.assets`，也不能说明 mjlab attach、场景导出、压缩包或源 XML 被移走后仍可读。对 `Scene.write()` 做导出审查时，当前实现会出现以下风险：

- 导出的 `scene.xml` 没有对应的 mesh 文件；
- 如果 loader key 与 XML 路径混用，导出路径可能出现冗余的 `assets/assets/*.obj`；这属于路径接线风险，是否失败取决于最终 XML 和文件是否仍然对齐；
- 从临时目录重新加载时出现 `Error opening file`。

因此，Go2 资源应按“资源自包含”方式接线：

1. MJCF 使用 `meshdir="assets"`，mesh 的 `file` 使用裸文件名；
2. `get_spec()` 读取 `GO2_XML.parent / "assets"` 下的文件，并填充 `spec.assets`；
3. 在当前 mjlab 1.6.0 中使用 `MjSpec.assets` API。参考项目的
   `mjlab.utils.os.update_assets` 来自不同版本，不能直接复制；当前环境中该函数不存在，应在项目侧写一个等价的小型加载函数，或使用当前 MuJoCo `MjSpec.assets` 接口；
4. 用 `Scene.write(temp_dir)` 后，在没有仓库路径参与的情况下执行
   `mujoco.MjModel.from_xml_path(temp_dir / "scene.xml")`，这才是 asset 验收标准。

推荐的逻辑形态（示意，不是让当前文档修改源码）。下面的 `path.name` 只适用于已经把 MJCF 改成 `meshdir="assets"` + `file="base_0.obj"` 的方案；若保留当前 `file="assets/base_0.obj"`，asset key 必须另外按导出规则匹配，不能混用：

```python
def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(GO2_XML))
    asset_dir = GO2_XML.parent / "assets"
    for path in asset_dir.glob("*.obj"):
        spec.assets[path.name] = path.read_bytes()
    return spec
```

最终 key 的形式必须和 MJCF 的 `file` 属性、`meshdir` 经过导出后的路径一致；不能只凭源码目录下 compile 成功来判断。

### 2.4 Go2 配置和 runner 的工作树漂移风险（P1）

历史审查时 `src/config/go2/go2_config.py` 曾出现未闭合的类定义，导致所有 Go2 任务导入
`SyntaxError`；当前 `HEAD` 已可通过 `py_compile`，该文件当前没有待提交源码 diff。

另外，历史审查时 `control.delay_update_period` 写成了 `10.0`，而
`mjlab.actuator.ActuatorCfg` 的字段类型是 `int`；当前 `HEAD` 已是整数 `10`，CPU
环境 reset/step 已能通过。提交前仍应保留类型检查，避免回归。

`go2_constants.py` 使用 `from config.go2.go2_config import Go2Cfg`，而同一仓库其他位置又使用 `src.config...`。这依赖运行时 `sys.path` 偶然包含仓库根和 `src`，必须统一。

当前工作树的 `src/tasks/locomotion/go2_ppo/rl/runner.py` 还保留一段已注释的旧
`CustomOnPolicyRunner` 示例；它假设 `self.alg.policy`、`act_inference` 和旧 normalizer
字段，与 rsl_rl 5.4.2 不匹配，未被 task registry 使用。应删除这段 dead code，继续使用
注册的 `VelocityOnPolicyRunner(MjlabOnPolicyRunner)`。

`VelocityOnPolicyRunner.save()` 还有一个 mjlab 1.6/rsl_rl 5.4 兼容问题：它访问
`self.logger.logger_type`，但当前 rsl_rl 5.4 的 `Logger` 没有这个属性。保存 checkpoint
时应从 `self.cfg["logger"]` 判断 logger 类型，或暂时只调用基类 `save()`；否则训练到保存
间隔时才会失败。

## 3. 推荐的多算法架构

不要把 `go2_ppo` 做成“环境和 PPO 的单体目录”。Go2 的物理参数、观测、动作、奖励和终止条件应该只实现一次；算法差异放在 runner、模型和 algorithm config 层。

### 3.1 四层职责

| 层 | 负责内容 | 不应负责 |
|---|---|---|
| Robot/Asset | MJCF、mesh、关节名、默认姿态、执行器和碰撞 | PPO/AMP 超参数 |
| Environment | Manager、观测组、动作、command、reward、termination、curriculum | 算法更新循环 |
| Adapter | `RslRlVecEnvWrapper`、TensorDict、timeout、action clipping | 修改物理语义 |
| Algorithm/Runner | PPO、RND、对称性、RNN、蒸馏、AMP 及 checkpoint | 重复创建 Go2 环境 |

建议长期把共享环境根目录命名为 `go2` 或 `go2_velocity`；现有 `go2_ppo` 保留为兼容入口，避免目录名把项目锁死在 PPO。可逐步整理为：

```text
src/tasks/locomotion/go2_ppo/
├── env_cfg.py                 # 共享 Go2 EnvCfg factory
├── config/
│   ├── env_cfgs.py            # flat/rough 与 play 覆盖
│   ├── ppo_cfg.py             # PPO/RND/symmetry 配置
│   ├── distill_cfg.py         # teacher/student 配置
│   ├── amp_cfg.py             # AMP 配置（需要自定义扩展时）
│   └── registry.py            # 根据算法生成 task 注册项
├── algorithms/
│   ├── ppo.py                 # 可选：只放项目自定义算法
│   └── amp.py
├── rl/
│   ├── base_runner.py         # 共用环境、日志、保存接口
│   ├── ppo_runner.py
│   ├── distill_runner.py
│   └── amp_runner.py
└── mdp/
```

现有目录不必一次重命名；可以先保留现有文件，再把 `rl_cfg.py` 拆成多个算法配置工厂。

### 3.2 task ID 不能只表达机器人

mjlab registry 当前把 `task_id`、`env_cfg`、`play_env_cfg`、`rl_cfg` 和
`runner_cls` 绑定在一起。若仍使用单一 ID，切换算法时容易覆盖配置或 checkpoint。建议采用：

```text
Unitree-Go2-Flat-PPO
Unitree-Go2-Rough-PPO
Unitree-Go2-Flat-Distill
Unitree-Go2-Rough-Distill
Unitree-Go2-Flat-AMP       # 规划项，组件完成前不注册
Unitree-Go2-Rough-AMP      # 规划项，组件完成前不注册
```

`Unitree-Go2-Flat` 和 `Unitree-Go2-Rough` 可以作为 PPO 兼容别名保留，但新算法不要复用同一个 `experiment_name`。

每个注册项应只选择一个算法规格。`AlgorithmSpec` 是项目侧组织概念，不是
mjlab registry 的现成类型；落地时每个算法仍要产出能被当前 CLI
`asdict()` 处理的 dataclass（通常继承 `RslRlBaseRunnerCfg`），并提供
`runner_cls`：

```text
AlgorithmSpec
  ├── algorithm_cfg_factory
  ├── model_cfg_factory
  ├── runner_cls
  ├── checkpoint_schema
  └── export_policy_fn
```

### 3.3 算法接入契约

所有算法都必须先通过同一 `VecEnv` 契约，再决定是否复用现有 runner：

| 接口 | 要求 |
|---|---|
| `get_observations()` | 返回带有明确 observation group 的 TensorDict |
| `step(actions)` | 返回 `(obs, rewards, dones, extras)`，并在非有限时域提供 `time_outs` |
| `num_envs/num_actions/device` | 与环境真实维度一致；Go2 action 应为 12 维 |
| observation sets/groups | 环境返回 TensorDict group；`obs_groups` 再把算法所需 set 映射到 group。PPO 可用 `actor:("actor",)`、`critic:("critic",)`；蒸馏可用 `student:("actor",)`、`teacher:("critic",)`；RND 的 `rnd_state` 同理，不要求复制同名环境观测组 |
| action 顺序 | 固定为 actuator/joint 的顺序，所有算法、导出器和部署端共用 |
| checkpoint | 至少保存该算法所需的模型、optimizer、normalizer、iteration 和 schema；不能默认套用 PPO 字段 |
| inference | 每种算法声明自己的 `load_cfg` 和推理模型。PPO 通常加载 `actor`，Distillation 通常加载 `student`；`load_cfg={"actor": True}` 不是通用协议 |

### 3.4 哪些算法可以复用，哪些需要新 runner

当前本地 rsl_rl 已有 `PPO`、`Distillation`、`MLPModel`、`CNNModel`、`RNNModel`，并提供 RND 和 symmetry 扩展。接入方式如下：

| 算法 | 复用策略 |
|---|---|
| PPO | 复用 `MjlabOnPolicyRunner`，只换 algorithm/model cfg |
| PPO + RND/symmetry | 可复用 on-policy runner，但 mjlab 1.6.0 的 `RslRlPpoAlgorithmCfg` 没有 `rnd_cfg`/`symmetry_cfg` 字段；需要项目侧 dataclass 或配置转换，并补齐 `obs_groups` |
| RNN policy | 复用 runner，使用 `RNNModel`，重点验证 hidden state reset |
| Distillation | 需要项目侧 mjlab-aware runner、`student`/`teacher`/`algorithm`/`obs_groups` 配置、teacher checkpoint 参数，并在 `learn()` 前加载 teacher；当前 `scripts/train.py` 没有这条链路 |
| AMP | 当前本地包没有可直接注册的 AMP 算法契约；需要扩展 rsl_rl 的 algorithm、storage、runner 和保存/加载逻辑，不能只改 PPO 配置 |
| 新的 off-policy 算法 | 通常需要自己的 replay buffer、采样循环和 runner，不应强行继承 `MjlabOnPolicyRunner` |

算法类名优先使用显式路径，例如：

```text
rsl_rl.algorithms.ppo:PPO
my_project.algorithms.amp:AMP
```

不要依赖简单名称扫描来解决自定义算法；这既受包遮蔽影响，也会让多版本 rsl_rl 难以复现。

如果自定义算法仍复用 `OnPolicyRunner`，算法类至少要提供当前 runner 调用的生命周期：
`construct_algorithm(obs, env, cfg, device)`、`train_mode`、`eval_mode`、`act`、
`process_env_step`、`compute_returns`、`update`、`get_policy`、`save` 和 `load`。
还要提供 runner 读取的 `learning_rate` 和 `get_policy().output_std`；多 GPU 时要支持
`broadcast_parameters`；启用 RND 时还要提供 `intrinsic_rewards` 和 `rnd.weight`。
缺少这些成员时，应提供算法专用 runner，而不是在 PPO 类中塞入条件分支。

本地 `rsl_rl.runners.DistillationRunner` 直接继承 `OnPolicyRunner`，不会自动获得
`MjlabOnPolicyRunner` 的环境计数保存/恢复、legacy 迁移和项目上传策略。应写一个
项目侧 mjlab-aware distill runner，把这些能力和 teacher 加载流程一起固定下来。

### 3.5 配置和 checkpoint 隔离

机器人配置（关节、mesh、PD、默认姿态）与算法配置必须分开。当前新增的
`GO2CfgPPO` 可以作为迁移兼容层，但不应成为未来 AMP、蒸馏等算法的共同配置源。

日志目录建议至少包含环境和算法：

```text
logs/rsl_rl/
└── go2_ppo/
    └── 2026-09-08_12-00-00/
        ├── params/env.yaml
        ├── params/agent.yaml
        ├── model_*.pt
        └── policy.onnx
```

下面是应新增到 checkpoint 或伴随 manifest 的 metadata。当前 PPO 保存逻辑不会自动写入这些字段，保存后还要在加载前显式比对：

```text
env_id, algorithm_id, observation_groups, action_order,
mjlab_version, rsl_rl_version, git_sha, seed
```

只有实现 schema 校验和拒绝策略，才能阻止“Go2 Flat 的 PPO checkpoint 被误加载到 Rough 的 AMP policy”这类静默错误。

### 3.6 外部 rsl_rl 的版本边界

`mjlab==1.6.0` 与本地 `rsl-rl-lib==5.4.2` 应视为一组兼容矩阵。由于
`pyproject.toml` 使用 `{path="rsl_rl", editable=true}`，实际代码由工作树决定，版本号本身不足。后续替换外部 rsl_rl 时，必须同时记录：

```text
mjlab version, rsl_rl version/commit/dirty state, torch version,
runner API, observation-group API, checkpoint schema
```

不要只在 `pyproject.toml` 中放一个浮动版本。若 AMP 或其他算法必须使用 rsl_rl fork，应让 fork 通过独立包名或明确的 git revision 安装，并在 task 注册中显式指定 runner/algorithm；不能依赖同名目录被 Python 偶然选中。

### 3.7 从 `AMP_mjlab` 提取的通用接入模式

参考仓库：`/home/sunteng/projects/AMP_mjlab`。它的 README 要求先安装主项目，再进入 `rsl_rl/` 单独 editable 安装；这说明它把 rsl_rl 当作独立算法包，而不是把算法代码混在 mjlab 环境目录里。

```bash
cd /home/sunteng/projects/AMP_mjlab
python -m pip install -e .
cd rsl_rl
python -m pip install -e .
```

上面是参考仓库自己的安装记录，不是当前项目应直接执行的命令。
`AMP_mjlab/rsl_rl` 是版本 `2.3.1` 的旧 fork，而且该目录本身就是 Python
包根，直接包含 `rsl_rl/__init__.py`；当前项目是版本 `5.4.2` 的标准嵌套结构
`rsl_rl/rsl_rl/__init__.py`。两者的包布局和 API 都不同。

这套项目可迁移的通用链路是：

```text
ManagerBasedRlEnv
  -> RslRlVecEnvWrapper
  -> mjlab registry 的 runner_cls
  -> 外部 rsl_rl runner
  -> 外部 rsl_rl algorithm/model
```

对应证据在：

| AMP_mjlab 文件 | 可迁移的机制 | AMP 专属内容 |
|---|---|---|
| `scripts/train.py` | 构造 mjlab 环境，包成 `RslRlVecEnvWrapper`，按 task 取 `runner_cls`，调用 `learn()` | motion tracking 判断、AMP 额外状态 |
| `src/tasks/amp_loco/config/g1/__init__.py` | 注册 `env_cfg`、`play_env_cfg`、RL cfg 和 runner | task 名称带 AMP |
| `src/tasks/amp_loco/config/g1/rl_cfg.py` | 用 dataclass 继承 `RslRlOnPolicyRunnerCfg`，向 runner cfg 添加算法字段 | `amp_reward_coef`、动作数据路径、判别器宽度等 |
| `src/tasks/amp_loco/rl/runner.py` | 任务侧 runner 子类负责加载、保存和导出策略 | AMP actor wrapper、AMP normalizer 导出 |
| `rsl_rl/algorithms/amp_ppo.py` | algorithm 类实现 `act`、rollout、`process_env_step`、`compute_returns`、`update` 等契约 | AMP reward/discriminator loss |
| `rsl_rl/runners/amp_on_policy_runner.py` | runner 负责 AMP 状态、训练循环和 checkpoint save/load | AMP loader、判别器、AMP replay buffer |
| `mjlab_patch/` | 对特定 mjlab 版本做局部兼容补丁 | `history_ordering`，不是外部 rsl_rl 的必需步骤 |

`mjlab_patch` 不能直接覆盖当前 mjlab 1.6.0 的 site-packages。它针对旧版
ObservationManager 增加 `history_ordering` 等行为；当前版本没有同名字段。需要该能力时，先按当前 API 重写并单独验证，不能把 patch 当成安装 rsl_rl 的步骤。

因此，当前 Go2 项目如果只使用 PPO，最小链路只需要：

```text
独立可导入的 rsl_rl 包
  + RslRlVecEnvWrapper
  + RslRlOnPolicyRunnerCfg
  + MjlabOnPolicyRunner 或其轻量子类
  + task registry 的 runner_cls
```

只有当新算法改变 rollout、loss、storage、模型输入或 checkpoint 格式时，才需要像
`AMP_mjlab/rsl_rl/algorithms/amp_ppo.py` 和
`AMP_mjlab/rsl_rl/runners/amp_on_policy_runner.py` 那样扩展 rsl_rl。AMP 的 motion loader、判别器、动作数据目录、`amp` observation、延迟恢复 reset 和 `mjlab_patch` 不应加入普通 PPO 或其他不需要它们的算法。

`AMP_mjlab` 的配置继承方式值得保留：

```python
@dataclass
class RslRlMyAlgorithmRunnerCfg(RslRlOnPolicyRunnerCfg):
    my_algorithm_parameter: float = 0.0
```

但新增字段必须同时被项目侧 runner 消费；仅把字段放进 dataclass 不会自动让 rsl_rl 识别。当前 `scripts/train.py` 会对 cfg 执行 `asdict()`，所以 cfg 必须保持为可序列化 dataclass，并且 runner 要明确从字典中取出这些字段。

AMP runner 还展示了一个重要边界：`RslRlVecEnvWrapper` 可以传递 actor、critic 之外的 TensorDict group，算法 runner 再把它们整理成自身需要的 legacy 输入（例如 `amp`、`rnd_state`）。这属于“算法适配层”的职责；Go2 环境只应提供稳定、命名清楚的观测 group，不应为了某一种算法复制整套环境。

这个 legacy bridge 只属于旧 fork。当前 rsl_rl 5.4.2 已原生使用 TensorDict、
`obs_groups` 和 `MLPModel/RNNModel`；普通 Go2 PPO 应直接走
`MjlabOnPolicyRunner`，不需要复制 `_unpack_obs()`、`_unpack_step()` 或旧
`ActorCritic` runner。

版本不能照搬：`AMP_mjlab/setup.py` 使用 `mjlab==1.2.0`，其独立 rsl_rl 的版本和 API 也不同于当前项目的 `mjlab==1.6.0`、`rsl-rl-lib==5.4.2`。应借鉴边界和调用顺序，逐项核对当前版本的 `VecEnv`、runner、model、algorithm 和 export API。

可以用下面的决策表确定工作量：

| 目标 | 需要接入的部分 | 不需要接入的部分 |
|---|---|---|
| 使用现成 PPO | 安装/固定外部 rsl_rl、wrapper、PPO cfg、runner、task 注册 | AMP discriminator、motion loader、AMP patch |
| PPO + RND/symmetry/RNN | 外部 rsl_rl 对应 extension/model、项目侧 cfg 转换和 observation set 映射 | AMP motion 数据和判别器 |
| Distillation | teacher/student model、teacher checkpoint、distillation runner、专用 load schema | AMP replay/discriminator |
| 自定义算法 | rsl_rl algorithm、storage（如需要）、runner、cfg、checkpoint/export | 与算法无关的环境复制 |
| AMP | 上述自定义算法链路，加 AMP observation、motion loader、判别器、replay、恢复 reset | 对其他算法全局修改 |

### 3.8 不做 AMP 时使用外部 rsl_rl 的直接步骤

以当前 Go2 PPO 为例，外部 rsl_rl 的最小接入不是“把 rsl_rl 代码复制进 task”，而是完成下面六个连接点：

| 步骤 | 当前项目对应位置 | 要确认的结果 |
|---|---|---|
| 1. 安装并固定包 | `pyproject.toml`、`rsl_rl/pyproject.toml` | `import rsl_rl` 指向真正包，记录版本、commit 和 dirty 状态 |
| 2. 暴露环境接口 | `scripts/train.py` 中的 `ManagerBasedRlEnv` + `RslRlVecEnvWrapper` | `get_observations()`、`step()`、`num_actions`、timeout 和 device 符合 runner 要求 |
| 3. 写算法 cfg | `src/tasks/locomotion/go2_ppo/config/rl_cfg.py` | cfg 是 dataclass；`asdict(cfg)` 后包含 runner、actor、critic、algorithm 所需字典 |
| 4. 选择 runner | `src/tasks/locomotion/go2_ppo/rl/runner.py`、task registry 的 `runner_cls` | runner 能构造算法、训练、保存、加载和推理 |
| 5. 选择 algorithm/model | cfg 中的 `class_name` 或显式路径 | `PPO` 等现成类可直接解析；自定义类能满足 runner 生命周期 |
| 6. 注册和启动 | `config/__init__.py`、`scripts/train.py`、未来的 `scripts/play.py` | task ID 能同时找到 env cfg、play cfg、RL cfg 和 runner |

PPO 的调用关系可以简化为：

```text
load_env_cfg(task_id)
load_rl_cfg(task_id)
ManagerBasedRlEnv(cfg.env)
RslRlVecEnvWrapper(env)
runner_cls(wrapper, asdict(cfg.agent), log_dir, device)
runner.learn(...)
```

如果只想换算法，通常只需要新增一份 cfg 和 runner 注册：

```python
register_mjlab_task(
    task_id="Unitree-Go2-Flat-MyAlgorithm",
    env_cfg=unitree_go2_flat_env_cfg(),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=my_algorithm_runner_cfg(),
    runner_cls=MyAlgorithmRunner,
)
```

环境的 `ManagerBasedRlEnvCfg`、Go2 asset、reward、termination 和 command 可以保持共享。新增 observation group 首先应修改 `ManagerBasedRlEnvCfg.observations`；`RslRlVecEnvWrapper` 通常会原样封装这些 group，不需要改 wrapper。只有以下情况才需要改 wrapper 或增加专用 adapter：

- 外部算法要求特殊的 timeout、terminal observation 或 recurrent state 语义；
- 外部算法的 action 不是当前 12 维 joint action；
- 外部算法使用的 VecEnv API 与当前 rsl_rl 版本不同。

只有以下情况才需要改 runner：

- rollout 循环不是当前 on-policy 循环；
- 算法需要 teacher、discriminator、replay buffer 或额外状态；
- checkpoint 的保存/加载字段不同；
- policy 导出需要自定义输入或 normalizer。

只有以下情况才需要改外部 rsl_rl 算法包：

- 算法类本身不存在；
- 现有 storage、model 或 update 逻辑不能表达目标算法；
- 需要新增可由 `class_name`/显式路径解析的类。

这三个边界可以避免把 AMP 的专用代码误加到普通 PPO、模仿学习或未来其他算法中。

### 3.9 AMP 参考与当前仓库的差异

| 项 | `/home/sunteng/projects/AMP_mjlab` | 当前 Go2 仓库 |
|---|---|---|
| rsl_rl | 独立的 `2.3.1` fork，旧 `ActorCritic`、plain tensor 和 legacy runner | 本地 `5.4.2`，使用 TensorDict、`obs_groups`、`MLPModel/RNNModel` |
| 算法 | `AMPPPO` + discriminator + AMP `ReplayBuffer` | 当前包没有 `AMPPPO`、`Discriminator`、`AMPLoader` 或 AMP replay 实现 |
| runner | `AmpOnPolicyRunner` 自己做 TensorDict→legacy bridge、AMP reward 和 AMP 状态保存 | `MjlabOnPolicyRunner` 目前按 PPO actor/critic 契约工作 |
| 环境输出 | `actor`、`critic`、`amp`，step 后还要提供 next AMP observation | Go2 当前只有 `actor`、`critic` |
| 数据 | NPZ 中的 joint/body/fps 数据，以及 body/anchor 映射 | 当前 Go2 没有 AMP motion NPZ 数据契约 |
| mjlab | `mjlab==1.2.0`，依赖 `history_ordering` 本地 patch | `mjlab==1.6.0`，`ObservationGroupCfg` 没有该旧字段 |
| checkpoint | legacy `model_state_dict`、discriminator、AMP normalizer、replay 相关状态 | 当前 PPO 使用新的 actor/critic checkpoint schema |

当前仓库不得直接把 `class_name` 改成 `AMPPPO`，也不能只复制
`AMP_mjlab/rsl_rl` 的 import。要做 AMP，必须先决定锁定旧 fork 的独立虚拟环境，或按 5.4.2 的 TensorDict/API 重写 algorithm、storage、runner、observation 和 checkpoint；在此之前不要注册或验收 `Unitree-Go2-Flat-AMP`、`Unitree-Go2-Rough-AMP`。

AMP 的 `amp` observation、motion NPZ、判别器、replay buffer、delayed termination/recovery reset 和旧 mjlab patch 都是算法插件，不能进入 Go2 普通 PPO 的共享环境契约。`AMP_mjlab` 没有 `Scene.write()` 后脱离仓库重新加载 mesh 的测试，不能用它替代本文件的 Go2 asset 验收。

### 3.10 当前项目推荐的外部仓库布局

推荐让算法库成为与本项目并列的独立 Git 仓库：

```text
/home/sunteng/projects/
├── legged_wbc_mjlab/
└── rsl_rl_custom/
    ├── pyproject.toml
    └── rsl_rl/
        └── __init__.py
```

本项目的 `pyproject.toml` 用相对路径指向它：

```toml
[project]
dependencies = [
    "mjlab==1.6.0",
    "rsl-rl-lib==5.4.2",
]

[tool.uv.sources]
rsl-rl-lib = { path = "../rsl_rl_custom", editable = true }
```

也可以用固定 Git revision 替代本地 editable 路径。开发自定义算法时用 editable；复现实验和 CI 时固定 commit。distribution 名称是 `rsl-rl-lib`，Python import 名称是 `rsl_rl`，两者不要混淆。

当前仓库根目录下的 `rsl_rl/` 在没有 shim 时会形成 namespace shadow；当前工作树的未跟踪
`rsl_rl/__init__.py` 已临时解决根目录解析，但它不在 `HEAD`。迁到外部仓库后，项目根下不能继续留一个会被 Python 发现的同名外层目录。否则即使外部包已正确安装，`import rsl_rl` 仍可能先命中本地目录。

安装或同步后必须同时在仓库根和仓库外验证：

```bash
cd /home/sunteng/projects/legged_wbc_mjlab
uv sync

.venv/bin/python - <<'PY'
from importlib.metadata import version
import inspect
import rsl_rl
from rsl_rl.runners import OnPolicyRunner

print("distribution:", version("rsl-rl-lib"))
print("package:", rsl_rl.__file__)
runner_file = inspect.getfile(OnPolicyRunner)
print("runner:", runner_file)
assert rsl_rl.__file__ is not None
assert runner_file.endswith("/rsl_rl/runners/on_policy_runner.py")
PY

cd /tmp
/home/sunteng/projects/legged_wbc_mjlab/.venv/bin/python -c \
  'import rsl_rl; assert rsl_rl.__file__ is not None; print(rsl_rl.__file__)'
```

如果外部仓库仍位于当前项目内部，断言应改为检查预期的绝对路径。核心标准是两个工作目录解析到同一个 `rsl_rl/__init__.py` 和同一个 runner 文件；当前嵌套布局的合法路径包含 `/rsl_rl/rsl_rl/`，不能简单断言“不包含 `/legged_wbc_mjlab/rsl_rl/`”。

### 3.11 当前 Go2 与外部 rsl_rl 的实际边界

当前继承链已经建立：

```text
src.tasks.locomotion.go2_ppo.rl.VelocityOnPolicyRunner
  -> mjlab.rl.runner.MjlabOnPolicyRunner
  -> rsl_rl.runners.OnPolicyRunner
```

其中：

- `ManagerBasedRlEnv` 和 MDP 属于当前项目/mjlab；
- `RslRlVecEnvWrapper` 属于 mjlab，负责满足 rsl_rl 的 VecEnv 契约；
- `MjlabOnPolicyRunner` 属于 mjlab，补充环境状态 checkpoint 和模型导出；
- `OnPolicyRunner`、`PPO`、model、storage、logger 属于外部 rsl_rl；
- `VelocityOnPolicyRunner` 属于当前项目，目前主要补 ONNX 元数据和保存行为。

所以“使用外部 rsl_rl”并不要求重写环境。对现有 PPO，只要外部包来源正确、版本兼容、任务能注册，现有训练链路就应该继续使用。换算法时按变化范围选择：

```text
只改超参数/model       -> 新增 rl cfg
复用 5.4 runner 协议    -> 新增 algorithm 类 + rl cfg
改变 rollout/checkpoint -> 新增 algorithm + runner + cfg
改变环境数据契约        -> 再增加 observation group 或 wrapper 适配
```

不要同时安装当前 5.4.2 和 `AMP_mjlab` 的 2.3.1 fork；它们都提供名为
`rsl_rl` 的顶层包，后安装者会覆盖或混合前者。若以后确实迁移 AMP，有两种明确选择：

1. 为旧 AMP fork 使用独立虚拟环境并锁定整个旧栈；
2. 把 AMP algorithm、storage、runner 和 checkpoint 按 rsl_rl 5.4.2 的 TensorDict/API 重写。

第二种更适合当前项目。旧 fork 使用 `ActorCritic`、plain tensor、`eval()` 和
`model_state_dict`；当前 5.4.2 使用 `MLPModel/RNNModel`、TensorDict、
`resolve_class()` 和分离的 actor/critic checkpoint。不能只改 import 路径或
`class_name="AMPPPO"`。

`AMP_mjlab/scripts/play.py` 可以参考环境、wrapper、runner、checkpoint、viewer 的调用顺序，但不能整文件复制。其配置使用了未声明的 tracking/W&B 字段，且 checkpoint API 属于旧 fork。当前项目应基于 mjlab 1.6.0 和 rsl_rl 5.4.2 的实际签名重新实现 `scripts/play.py`。

## 4. 推荐实施顺序

### 阶段 A：清理包和注册入口

1. 固定一个干净 commit，先处理 `git diff` 中的 Go2 配置修改；
2. 补齐 `src/tasks/locomotion/__init__.py` 或显式注册；
3. 统一 `src.*` 与 `config.*` 的导入策略；
4. 消除顶层 `rsl_rl/` namespace 遮蔽，重新安装 editable 包；
5. 保留 PPO 兼容 task ID，再增加带算法后缀的 ID。

### 阶段 B：修复 Go2 asset

1. 统一 `meshdir` 与 mesh `file` 的相对路径；
2. 在 `get_spec()` 注入 mesh bytes；
3. 验证 Entity attach 后 `scene.spec.assets` 仍包含所有 16 个 mesh；
4. 从临时目录导出、重新加载和 compile；
5. 再进入环境 reset/step 测试。

### 阶段 C：稳定环境契约

1. 断言 action 数为 12，记录 12 个 joint 的顺序；
2. 断言 `actor`、`critic` observation group 的 shape；
3. 验证 Flat/Rough 的 sensor、site、geom、body 名称全部能匹配；
4. 把 actuator delay 字段改成正确的整数类型；
5. CPU、单环境和小批量环境分别完成 reset/step。

### 阶段 D：算法插件化

1. 抽出共享 `AlgorithmSpec`；
2. 将 PPO、Distillation、未来 AMP 分成独立 cfg/runner；
3. 统一 checkpoint metadata 和命名；
4. 每个算法先做 1 iteration smoke test，再做长训练；
5. 最后实现播放脚本、导出脚本和部署验证。

## 5. 验收命令

以下命令在包路径和任务注册修复后执行。它们故意设置小规模，先验证接线，不代表正式训练参数。

### 5.1 静态检查

```bash
cd /home/sunteng/projects/legged_wbc_mjlab
export PYTHONPATH="$PWD/rsl_rl:$PWD"
export PYTHONDONTWRITEBYTECODE=1
export WARP_CACHE_PATH=/tmp/legged_wbc_mjlab-warp-cache

.venv/bin/python -m py_compile \
  src/config/go2/go2_config.py \
  src/assets/robots/go2/go2_constants.py \
  src/tasks/locomotion/go2_ppo/config/env_cfgs.py \
  src/tasks/locomotion/go2_ppo/config/__init__.py \
  scripts/train.py
```

### 5.2 rsl_rl 和任务注册

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
.venv/bin/python - <<'PY'
import inspect
import rsl_rl
from rsl_rl.utils import resolve_callable
import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks
import src.tasks

print(rsl_rl.__file__)
assert rsl_rl.__file__ is not None
assert inspect.getmodule(resolve_callable("rsl_rl.algorithms.ppo:PPO")).__name__.startswith("rsl_rl.")
tasks = list_tasks()
assert "Unitree-Go2-Flat" in tasks
assert "Unitree-Go2-Rough" in tasks
# After adding algorithm-specific registrations, also assert the suffixed IDs.
print(tasks)
PY
```

验收重点是 `rsl_rl.__file__` 指向真正的内层包，且 Go2 task 不需要手工 import 子模块才能出现。

### 5.3 asset 独立导出测试

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
.venv/bin/python - <<'PY'
import tempfile
import mujoco
from src.assets.robots.go2.go2_constants import get_go2_robot_cfg
from mjlab.scene import Scene, SceneCfg

scene = Scene(SceneCfg(entities={"robot": get_go2_robot_cfg()}), device="cpu")
assert len(scene.spec.meshes) == 16
assert len(scene.spec.assets) == 16
with tempfile.TemporaryDirectory() as d:
    from pathlib import Path
    out = Path(d) / "go2"
    scene.write(out)
    assert (out / "scene.xml").exists()
    assert len(list((out / "assets").rglob("*.obj"))) == 16
    model = mujoco.MjModel.from_xml_path(str((out / "scene.xml").resolve()))
    assert model.nmesh == 16
print("go2 asset export/reload: OK")
PY
```

验收条件是临时目录必须包含 XML 和所有 mesh，并能脱离仓库路径重新加载。

### 5.4 环境 reset/step

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
export WARP_CACHE_PATH=/tmp/legged_wbc_mjlab-warp-cache
.venv/bin/python - <<'PY'
import torch
import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

cfg = load_env_cfg("Unitree-Go2-Flat")
cfg.scene.num_envs = 2
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
obs, _ = env.reset()
assert set(("actor", "critic")).issubset(obs)
action = torch.zeros((2, env.action_manager.total_action_dim))
obs, reward, terminated, truncated, _ = env.step(action)
assert action.shape[-1] == 12
assert reward.shape == (2,)
env.close()
print("env reset/step: OK")
PY
```

### 5.5 PPO 最小训练

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
export WARP_CACHE_PATH=/tmp/legged_wbc_mjlab-warp-cache
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python -m scripts.train Unitree-Go2-Flat \
  --gpu-ids 0 \
  --env.scene.num-envs 2 \
  --agent.max-iterations 1 \
  --agent.logger tensorboard
```

该命令依赖包路径和任务注册修复。当前工作树已将 `delay_update_period` 改为整数，CPU
环境 reset/step 和 runner 构造 smoke 已通过；但当前注册的 `VelocityOnPolicyRunner.save()`
仍访问不存在的 `self.logger.logger_type`，因此一轮训练在 checkpoint 保存时会失败，必须先应用 A.6.1。若在旧 commit 上执行，先修复
`10.0` 的类型错误。上面的命令显式关闭 CUDA，适合无 GPU 机器做 CPU smoke；有 GPU
时去掉 `CUDA_VISIBLE_DEVICES=""` 并按实际设备选择。正式训练前再恢复环境数量和
iteration。当前 `scripts/play.py` 为空，因此 checkpoint 播放不能列为已通过；应先实现与训练相同的 task 加载、play env、runner load 和 viewer 流程。

### 5.6 每个新算法的最小验收

每增加一个算法，必须分别完成：

```text
注册 task -> 构造 cfg -> 构造 wrapper -> 构造 runner
-> 1 iteration -> 保存 checkpoint -> 按该算法 schema 重新加载推理模型
-> 用同一 observation 做一次 inference -> 检查 action shape/range
```

AMP、蒸馏等包含额外状态时，还要验证 discriminator、teacher、replay buffer 或 hidden state 的保存和恢复。

## 6. 最终审查清单

### 代码和包

- [ ] `git diff` 中没有未解释的 Go2 配置或生成文件改动。
- [ ] `src.tasks` 一次导入即可发现所有 Go2 task。
- [ ] `rsl_rl.__file__` 不为 `None`，不会扫描外层 `setup.py`。
- [ ] 项目内导入路径不依赖当前工作目录。
- [ ] `mjlab`、`rsl_rl` 版本和 git SHA 写入实验配置。

### 机器人和环境

- [ ] 16 个 Go2 mesh 在 `spec.assets` 中可见。
- [ ] 导出到临时目录后可独立 compile。
- [ ] body、joint、site、geom 的正则表达式都有实际匹配。
- [ ] action 顺序、scale、clip 和部署端一致。
- [ ] actor/critic observation shape 在每个算法中明确记录。

### 算法和实验

- [ ] PPO、Distillation、AMP 使用不同的 algorithm ID 和日志目录。
- [ ] 每个算法的 runner 能保存、加载和推理。
- [ ] normalization、timeout、teacher 或 recurrent state 不会在 resume 时丢失。
- [ ] 新算法没有复制一份 Go2 环境逻辑。
- [ ] 单环境 smoke 通过后才运行大规模训练和多 GPU。

## 7. 已验证和未验证边界

已验证：Go2 MJCF 在源码目录中可被 MuJoCo 解析和直接 compile；16 个 mesh 文件存在；当前 `HEAD` 中的空 `src/tasks/locomotion/__init__.py` 使 `src.tasks` 能自动发现 Flat/Rough task；CPU 单环境 Flat/Rough 的 reset/step 通过；在临时 shim 和当前 runner/config 快照下，`RslRlVecEnvWrapper`、`VelocityOnPolicyRunner` 和外部 `PPO` 可完成构造与 inference（正确推理入口是 `runner.alg.get_policy()`，Flat action 为 `(1,12)`，Rough action 为 `(1,12)`）；`rsl_rl` 在显式修正包路径后可以解析 `PPO`。`WARP_CACHE_PATH` 用于把 mjlab/MJWarp 的编译缓存放到可写目录。

未验证：当前工作树下从零启动的完整 Go2 训练、播放入口、多 GPU 训练，以及未来算法的 checkpoint 兼容性。当前实时 `rl_cfg.py` 已通过 `py_compile`，并使用单一的显式 `PPO` class path；`runner.py` 的旧 `CustomOnPolicyRunner` 已注释但仍应删除，注册的 `VelocityOnPolicyRunner.save()` 仍需应用 A.6.1 才能完成 checkpoint/ONNX 保存。Go2 场景 asset 导出仍失败：`spec.assets` 为 0，`Scene.write()` 只生成 `scene.xml`，从临时目录重载时报 `Error opening file`；必须先应用 A.3/A.4。当前本机没有 CUDA，不能据此判定 GPU 训练状态。只有通过上述分阶段验收后，才能把这些能力标记为完成。

## 附录 A：可直接应用的代码实现（本次只写入文档）

本附录给出按照当前 `mjlab==1.6.0`、`rsl-rl-lib==5.4.2` 接口整理的代码。代码块是建议修改内容，本次请求只修改本文档，没有把任何代码写入项目源码。每个片段都标明了目标文件；应用时按顺序完成，并在每一步运行对应验收命令。

### A.0 先确定外部 rsl_rl 的放置方式

推荐把 rsl_rl 放在并列仓库，并将 `pyproject.toml` 改为真实相对路径：

```toml
[project]
dependencies = [
  "mjlab==1.6.0",
  "rsl-rl-lib==5.4.2",
]

[tool.uv.sources]
rsl-rl-lib = { path = "../rsl_rl_custom", editable = true }
```

如果暂时保留当前仓库内的双层目录 `rsl_rl/rsl_rl`，则需要正式提交下面的
`rsl_rl/__init__.py`，避免仓库根把外层目录当成 namespace package：

```python
"""Checkout shim for the vendored rsl_rl package."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "rsl_rl")]
```

两种方案只选一种。当前工作树已有上面的 shim，但它尚未被 Git 跟踪。无论选择哪种，
下面的检查必须在仓库根和 `/tmp` 得到同一个 runner 文件：

```bash
for workdir in "$PWD" /tmp; do
  (
    cd "$workdir"
    /home/sunteng/projects/legged_wbc_mjlab/.venv/bin/python - <<'PY'
import inspect
import rsl_rl
from rsl_rl.runners import OnPolicyRunner

assert rsl_rl.__file__ is not None
print(rsl_rl.__file__)
print(inspect.getfile(OnPolicyRunner))
PY
  )
done
```

### A.1 让 `locomotion` 被任务扫描器发现

目标文件：`src/tasks/locomotion/__init__.py`。

```python
"""Locomotion task package.

The parent ``src.tasks`` importer recursively discovers the task packages
below this directory.
"""
```

文件可以保持只有这段包说明。不要在这里创建环境实例；注册动作放在各任务的
`config/__init__.py`，否则导入顺序会影响 registry 状态。

当前 `HEAD` 中该文件是空包标记，空文件同样有效；上面的 docstring 只是可读版本。

验收：

```bash
PYTHONPATH="$PWD/rsl_rl:$PWD" .venv/bin/python - <<'PY'
import src.tasks
from mjlab.tasks.registry import list_tasks

tasks = list_tasks()
assert "Unitree-Go2-Flat" in tasks
assert "Unitree-Go2-Rough" in tasks
print("task discovery: OK")
PY
```

### A.2 统一 Go2 配置导入和延迟字段类型

目标文件：`src/assets/robots/go2/go2_constants.py`、
`src/config/go2/go2_config.py`。

当前 `HEAD` 已经把 `delay_update_period` 设为整数 `10`。下面的 diff 是历史修复
记录和回归检查，不要在已经是 `10` 的文件上重复应用：

```diff
--- a/src/assets/robots/go2/go2_constants.py
+++ b/src/assets/robots/go2/go2_constants.py
@@
-from config.go2.go2_config import Go2Cfg
+from src.config.go2.go2_config import Go2Cfg
```

```diff
--- a/src/config/go2/go2_config.py
+++ b/src/config/go2/go2_config.py
@@
-        delay_update_period = 10.0
+        delay_update_period = 10
```

如果最终选择安装后的 `config.*` 导入根，则所有文件都统一改成
`config.*`，不要只修改这一处。`delay_update_period` 是 physics timestep 数，
在当前 mjlab 的 `ActuatorCfg` 中必须是 `int`。

仓库内所有同一配置的 import 也要保持一致。例如选择 `src.*` 时：

```diff
--- a/src/tasks/locomotion/go2_ppo/config/rl_cfg.py
+++ b/src/tasks/locomotion/go2_ppo/config/rl_cfg.py
@@
-from config.go2.go2_config import Go2Cfg
+from src.config.go2.go2_config import Go2Cfg
```

如果该文件没有实际使用 `Go2Cfg`，更好的实现是直接删除这个无用 import，避免
算法配置对机器人 legacy 配置产生隐式依赖。

当前 `rl_cfg.py` 的 `Go2Cfg`/`go2cfg` 没有被使用，建议应用下面的最小清理：

```diff
--- a/src/tasks/locomotion/go2_ppo/config/rl_cfg.py
+++ b/src/tasks/locomotion/go2_ppo/config/rl_cfg.py
@@
-from config.go2.go2_config import Go2Cfg
-
-go2cfg = Go2Cfg()
```

### A.3 修正 Go2 MJCF 的 mesh 路径

目标文件：`src/assets/robots/go2/xmls/go2.xml`。

把 compiler 的 meshdir 设置为 `assets`，并让 mesh 的 `file` 属性只保留文件名：

```diff
--- a/src/assets/robots/go2/xmls/go2.xml
+++ b/src/assets/robots/go2/xmls/go2.xml
@@
-  <compiler angle="radian" autolimits="true" />
+  <compiler angle="radian" meshdir="assets" autolimits="true" />
@@
-    <mesh file="assets/base_0.obj" />
-    <mesh file="assets/base_1.obj" />
-    <mesh file="assets/base_2.obj" />
-    <mesh file="assets/base_3.obj" />
-    <mesh file="assets/base_4.obj" />
-    <mesh file="assets/hip_0.obj" />
-    <mesh file="assets/hip_1.obj" />
-    <mesh file="assets/thigh_0.obj" />
-    <mesh file="assets/thigh_1.obj" />
-    <mesh file="assets/thigh_mirror_0.obj" />
-    <mesh file="assets/thigh_mirror_1.obj" />
-    <mesh file="assets/calf_0.obj" />
-    <mesh file="assets/calf_1.obj" />
-    <mesh file="assets/calf_mirror_0.obj" />
-    <mesh file="assets/calf_mirror_1.obj" />
-    <mesh file="assets/foot.obj" />
+    <mesh file="base_0.obj" />
+    <mesh file="base_1.obj" />
+    <mesh file="base_2.obj" />
+    <mesh file="base_3.obj" />
+    <mesh file="base_4.obj" />
+    <mesh file="hip_0.obj" />
+    <mesh file="hip_1.obj" />
+    <mesh file="thigh_0.obj" />
+    <mesh file="thigh_1.obj" />
+    <mesh file="thigh_mirror_0.obj" />
+    <mesh file="thigh_mirror_1.obj" />
+    <mesh file="calf_0.obj" />
+    <mesh file="calf_1.obj" />
+    <mesh file="calf_mirror_0.obj" />
+    <mesh file="calf_mirror_1.obj" />
+    <mesh file="foot.obj" />
```

这一步必须和下一节的 asset key 形式一起应用。只改 XML 或只改 Python，都会让
`Scene.write()` 的路径匹配失败。

### A.4 在 `MjSpec` 中注入 mesh bytes

目标文件：`src/assets/robots/go2/go2_constants.py`。

把原来的 `get_spec()` 替换为下面实现。它使用当前 MuJoCo 的 `MjSpec.assets`，
没有依赖 `unitree_rl_mjlab` 旧版本中的 `mjlab.utils.os.update_assets`。

该组合方案已在本地以 `mjlab==1.6.0`、MuJoCo 3.11.0 做内存模拟验证：
`spec.assets` 为 16，`Scene.write()` 导出 16 个 OBJ，临时目录重载的
`MjModel.nmesh` 为 16；源码当前仍未应用这两个改动。

```python
GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "go2" / "xmls" / "go2.xml"
)
GO2_ASSET_DIR = GO2_XML.parent / "assets"
assert GO2_XML.exists()
assert GO2_ASSET_DIR.exists()


def get_spec() -> mujoco.MjSpec:
  """Build a self-contained Go2 MuJoCo specification."""
  mesh_files = tuple(sorted(GO2_ASSET_DIR.glob("*.obj")))
  if len(mesh_files) != 16:
    raise FileNotFoundError(
      f"Expected 16 Go2 OBJ meshes in {GO2_ASSET_DIR}, found {len(mesh_files)}"
    )

  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  for mesh_file in mesh_files:
    # The XML now uses meshdir="assets" and file="<name>.obj".
    spec.assets[mesh_file.name] = mesh_file.read_bytes()
  return spec
```

用下面的独立目录测试确认资源已经嵌入 spec：

```python
from pathlib import Path
import tempfile
import mujoco
from mjlab.scene import Scene, SceneCfg
from src.assets.robots.go2.go2_constants import get_go2_robot_cfg

scene = Scene(
  SceneCfg(entities={"robot": get_go2_robot_cfg()}),
  device="cpu",
)
assert len(scene.spec.assets) == 16

with tempfile.TemporaryDirectory() as tmp:
  output = Path(tmp) / "go2"
  scene.write(output)
  assert len(tuple((output / "assets").rglob("*.obj"))) == 16
  model = mujoco.MjModel.from_xml_path(str((output / "scene.xml").resolve()))
  assert model.nmesh == 16
```

### A.5 用显式路径配置 PPO

目标文件：`src/tasks/locomotion/go2_ppo/config/rl_cfg.py`。

下面是一个完整的 PPO 配置工厂。`algorithm.class_name` 使用 qualified path，
这样不会触发外层 `rsl_rl` namespace 的简单名称扫描：

```python
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="rsl_rl.models.mlp_model:MLPModel",
      distribution_cfg={
        "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="rsl_rl.models.mlp_model:MLPModel",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      class_name="rsl_rl.algorithms.ppo:PPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="go2_ppo",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
```

当前 `scripts/train.py` 不使用顶层 `RslRlOnPolicyRunnerCfg.class_name` 来选择
runner；它优先使用 registry 的 `runner_cls`（当前是 `VelocityOnPolicyRunner`）。
`actor.class_name`、`critic.class_name` 和 `algorithm.class_name` 才由 rsl_rl
在构造模型/算法时解析。

### A.6 注册环境、play 配置和 runner

目标文件：`src/tasks/locomotion/go2_ppo/config/__init__.py`。

```python
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.locomotion.go2_ppo.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

新增算法时不要覆盖 PPO 的注册项，使用独立 task ID 和实验目录：

```python
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MyAlgorithm",
  env_cfg=unitree_go2_flat_env_cfg(),
  play_env_cfg=unitree_go2_flat_env_cfg(play=True),
  rl_cfg=my_algorithm_runner_cfg(),
  runner_cls=MyAlgorithmRunner,
)
```

### A.6.1 删除旧 runner，并保留 mjlab 1.6.0 runner

目标文件：`src/tasks/locomotion/go2_ppo/rl/runner.py`。

删除已注释的 `_OnnxPolicyWrapper`、`CustomOnPolicyRunner` 和
`from rsl_rl.rsl_rl...`。rsl_rl 5.4.2 的策略通过
`self.alg.get_policy()` 获取，mjlab 1.6.0 的 `export_policy_to_onnx()` 已调用
`get_policy().as_onnx()`，其中包含 `MLPModel` 自身的 observation normalizer。

下面是按 mjlab 1.6.0/rsl_rl 5.4.2 API 整理、并在 `/tmp` 完成 checkpoint + ONNX
导出验证的实现；应用后应再次跑保存 smoke：

```python
from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils.log_writer import LogWriter


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None) -> None:
    super().save(path, infos)

    export_dir = Path(path).parent
    filename = "policy.onnx"
    onnx_path = export_dir / filename
    self.export_policy_to_onnx(str(export_dir), filename)

    metadata = get_base_metadata(self.env.unwrapped, export_dir.name)
    attach_metadata_to_onnx(str(onnx_path), metadata)

    # TensorBoard writer has no external file upload. A LogWriter backend may.
    if self.cfg.get("upload_model", True) and isinstance(
      self.logger.writer, LogWriter
    ):
      self.logger.writer.save_file(str(onnx_path))
```

不要使用 `self.logger.logger_type`：rsl_rl 5.4.2 的 `Logger` 没有该属性。
不要重新把 normalizer 包一层：当前 `MLPModel.as_onnx()` 已复制
`model.obs_normalizer`。若未来使用完全不同的 model，再为该 model 写专用 exporter。

应用后用这个最小保存测试确认 checkpoint 和 ONNX 都能生成：

```python
from dataclasses import asdict
from pathlib import Path
import tempfile

import src.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

cfg = load_env_cfg("Unitree-Go2-Flat")
cfg.scene.num_envs = 1
agent = load_rl_cfg("Unitree-Go2-Flat")
env = RslRlVecEnvWrapper(
  ManagerBasedRlEnv(cfg=cfg, device="cpu"),
  clip_actions=agent.clip_actions,
)
try:
  runner = VelocityOnPolicyRunner(env, asdict(agent), None, "cpu")
  with tempfile.TemporaryDirectory() as tmp:
    checkpoint = Path(tmp) / "model_0.pt"
    runner.save(str(checkpoint))
    assert checkpoint.exists()
    assert (Path(tmp) / "policy.onnx").exists()
finally:
  env.close()
```

### A.7 训练入口的最小可读实现

下面是 `scripts/train.py` 中与外部 rsl_rl 直接相关的核心实现。现有脚本还包含
Tyro、GPU、视频、resume 和日志处理；新增算法时保持这段调用顺序：

```python
from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)


def train_one_task(task_id: str, device: str, log_dir: str) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)

  env = ManagerBasedRlEnv(
    cfg=env_cfg,
    device=device,
    render_mode=None,
  )
  vec_env = RslRlVecEnvWrapper(
    env,
    clip_actions=agent_cfg.clip_actions,
  )

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(
    vec_env,
    asdict(agent_cfg),
    log_dir,
    device,
  )
  try:
    runner.learn(
      num_learning_iterations=agent_cfg.max_iterations,
      init_at_random_ep_len=True,
    )
  finally:
    vec_env.close()
```

这里 `asdict(agent_cfg)` 是边界：mjlab 侧保存 dataclass，外部 rsl_rl runner
接收普通字典。新增 dataclass 字段不会自动生效，runner 必须显式消费这些字段。

### A.8 空白 `play.py` 的最小 PPO 回放实现

目标文件：`scripts/play.py`。下面是一个不依赖 viewer 的 headless 版本，先用于验证
checkpoint、runner 和 action 输出；viewer 可以在这段逻辑之后接入。

```python
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def play(task_id: str, checkpoint: Path, device: str, steps: int) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  vec_env = RslRlVecEnvWrapper(
    env,
    clip_actions=agent_cfg.clip_actions,
  )

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(
    vec_env,
    asdict(agent_cfg),
    log_dir=None,
    device=device,
  )
  # This load_cfg is for PPO only. Distillation/custom algorithms must declare
  # and implement their own schema; Distillation also needs teacher state.
  runner.load(
    str(checkpoint.resolve()),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  # MjlabOnPolicyRunner.load expects the standard checkpoint ``infos`` key.
  # A bare external rsl_rl checkpoint may need a custom loader.
  policy = runner.get_inference_policy(device=device)

  obs = vec_env.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      actions = policy(obs)
      obs, _, _, _ = vec_env.step(actions)
  vec_env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("task_id")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--steps", type=int, default=1000)
  args = parser.parse_args()
  play(args.task_id, args.checkpoint, args.device, args.steps)
```

运行方式（仓库根目录建议使用模块入口；直接运行脚本时必须补 `PYTHONPATH`）：

```bash
PYTHONPATH="$PWD/rsl_rl:$PWD" \
WARP_CACHE_PATH=/tmp/legged-wbc-warp-cache \
.venv/bin/python scripts/play.py Unitree-Go2-Flat \
  --checkpoint logs/rsl_rl/go2_ppo/<run>/model_100.pt \
  --steps 100
```

### A.9 外部 rsl_rl 自定义算法的最小插件

下面假设外部算法仓库中新增包 `my_rsl_algorithms`。如果算法仍然是 PPO 的同一
rollout/storage 契约，可以先继承 `PPO`，只替换 update；如果改变 storage 或
rollout，则必须实现专用 runner，不能使用这个最小例子。

外部仓库文件：`my_rsl_algorithms/__init__.py`。

```python
from rsl_rl.algorithms.ppo import PPO


class MyPPO(PPO):
  """Example algorithm plugin with the rsl_rl 5.4 PPO contract."""

  def update(self) -> dict:
    loss_dict = super().update()
    # Add custom loss terms here, or replace the update implementation.
    return loss_dict


__all__ = ["MyPPO"]
```

当前项目的 `rl_cfg.py` 中改成：

```python
algorithm=RslRlPpoAlgorithmCfg(
  class_name="my_rsl_algorithms:MyPPO",
  # 其余 PPO 超参数保持不变
)
```

当前 `OnPolicyRunner` 解析 cfg 中的 algorithm class，并调用继承来的静态
`construct_algorithm()`；PPO 的实现再解析 actor/critic class 并实例化算法。
如果新增字段不在 `PPO.__init__` 的参数中，必须同时重写 algorithm 的
`__init__`/`construct_algorithm` 或提供专用 runner。这个例子只替换 `update()`，
所以不需要改环境。若新算法新增模型输入，先在 EnvCfg 增加 observation
group，再在 cfg 的 `obs_groups` 中映射；`RslRlVecEnvWrapper` 通常不需要改。

### A.10 新增 observation group 的实现方式

算法需要额外输入时，先在环境配置中声明 group：

```python
# go_ppo_env_cfg.py（示意）
observations = {
  "actor": ObservationGroupCfg(
    terms=actor_terms,
    concatenate_terms=True,
    history_length=1,
  ),
  "critic": ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    history_length=1,
  ),
  "aux": ObservationGroupCfg(
    terms={
      "aux_state": ObservationTermCfg(
        func=mdp.my_aux_state,
      ),
    },
    concatenate_terms=True,
    history_length=1,
  ),
}
```

再把算法 set 映射到现有 group：

```python
runner_cfg = RslRlOnPolicyRunnerCfg(
  obs_groups={
    "actor": ("actor",),
    "critic": ("critic",),
    "my_aux_set": ("aux",),
  },
  # actor / critic / algorithm ...
)
```

`my_aux_state`、维度、历史顺序和归一化策略必须在算法 runner 中有明确约定。
不要为了满足算法名称，把同一份数据复制成多个环境 group。

### A.11 只写进文档的最终应用顺序

```text
1. 保留已由 `HEAD` 跟踪的 `src/tasks/locomotion/__init__.py`
2. 统一 src.config.* 导入和 delay_update_period=int
3. 修改 go2.xml 的 meshdir/file
4. 修改 get_spec() 注入 16 个 OBJ bytes
5. 运行 Scene.write + 临时目录 MjModel reload
6. 用显式 class_name 配置 PPO
7. 注册 Flat/Rough task 和 runner_cls
8. 运行 CPU reset/step 与 runner/inference smoke
9. 再实现 play.py 并加载 checkpoint
10. 最后按同一 cfg/runner 契约接入新算法
```

本附录中的代码没有在本次请求中写入 `src/`、`scripts/` 或外部 rsl_rl 仓库；源码工作仍需由后续变更单独实施和审查。
