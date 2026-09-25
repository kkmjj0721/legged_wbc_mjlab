# RI-4438-HIM 粗糙地形训练性能与显存分析

> 分析对象：`Ri-4438-HIM-Rough`（2048 环境、单卡训练时显存 16 GB+、每轮 7 s）
> 分析时间：2026-09-24
> 分析方式：**只读**。未修改任何仓库代码、未启动正式训练；全部测量来自临时脚本（已删除）。
> 参考基准：本机安装的 `mjlab 1.6.0` + `mujoco_warp 3.11.0` + 本地 `rsl_rl`（HIM-PPO）。

---

## 0. 结论速览

| 问题 | 结论 | 主要证据 |
|---|---|---|
| 2048 env 占 16 GB+ 显存 | **`ccd_iterations = 500` 引起的 MuJoCo Warp EPA 碰撞缓冲**（瞬时分配、mempool 保留峰值） | 公式计算 2048 env = **16.21 GiB**；本机 1024 env 复现 OOM，失败分配字节数 `3,941,597,184` **逐字节等于**公式算出的 `epa_pr` 数组 |
| 一轮 7 s 慢在哪 | 96% 在采集阶段；其中**地形几何体数量**第一（约 33 ms/步，占 `env.step` 的 ~77%、占 collection 的 ~57%），**HIM runner 的 `auto_reset=False` 复位路径**第二（占 collection 的 20–26%） | 远程 TensorBoard 日志 + 本机真实 runner 分段实测 |
| HIM 算法本身 | **不是瓶颈，也不该改**（learning 只占约 4%） | collection 6.942 s vs learning 0.293 s |
| 与原版 HIMLoco 的差距 | **不是 HIM 移植的问题，是地形几何体表示方式的差异**（legged_gym 用 trimesh 网格 + GPU heightmap 索引；mjlab 这里是 ~7500 个 box geom 参与宽相碰撞与射线 BVH） | 本任务地形 `ngeom=7547`；官方推荐地形 `ngeom=554`；Go2 rough `610` |
| 与本地 `go2_ppo` 对比 | 同一后端、同样开 raycast，Go2 rough `env.step` 17.5 ms vs HIM rough 43 ms，差值几乎全部由地形几何体解释 | 见 §5.3 |
| 最快见效的修法 | **无需改代码**：`--env.sim.mujoco.ccd-iterations 50 --env.sim.nconmax 64 --env.sim.contact-sensor-maxmatch 64`，并且**不要**加 `--enable-nan-guard` | 本机 512 env 实测：iteration 5.9 s → 2.9 s，显存 7.58 GB → 2.97 GB |

一句话：**当前配置把「地形几何体 + 碰撞/CCD 容量」三处参数同时放大，再乘上 2048 个环境，于是显存被 EPA 缓冲吃满、速度被宽相碰撞拖住；而这几处参数恰好是本项目相对 mjlab 官方参考实现偏离最大的地方。**

---

## 1. 测量方法与可复现性

### 1.1 环境

| 项 | 本机测量环境 | 用户远程训练环境 |
|---|---|---|
| GPU | RTX 4060 8 GiB (sm_89) | RTX 4090 24 GiB（8 卡机器，本次为单卡） |
| 软件 | mjlab 1.6.0 / mujoco_warp 3.11.0 / warp 1.17.0 / torch 2.14 / CUDA 13.0 / Python 3.11 | 同 |
| 结论适用性 | **倍数与归因可直接参考；绝对秒数不可搬运**（4090 ≈ 4060 的 2.5–3 倍） | — |

### 1.2 方法

1. **静态读码**：`ri_4438_him` 全部配置 + `.venv` 中 mjlab / mujoco_warp 相关热路径源码 + 本地 `rsl_rl`（HIM runner / HIMPPO / storage / logger）。
2. **真实 runner 分段计时**：用 monkey-patch 包装 `ManagerBasedRlEnv.step`、`HIMOnPolicyRunner._reset_done_envs`、`alg.act`、`alg.process_env_step`、`logger.process_env_step`，跑完整的 `HIMOnPolicyRunner.learn()`，在 256/512/1024/2048 env 下取 1–3 轮迭代。
3. **环境级 A/B**：在不动训练数学的前提下，逐个切换 `nconmax` / `ccd_iterations` / `contact_sensor_maxmatch` / 地形类型，测 `env.step` 单步耗时与显存。
4. **地形几何体归因**：用 `TerrainEntity` 在 CPU 上编译 spec，固定 `seed=0`、`num_rows=10`、`num_cols=20`、`curriculum=True`，逐个替换子地形类型后统计 `ngeom`。
5. **已有远程日志**：解析 `/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/` 下的 TensorBoard 事件文件与 `params/env.yaml`、`params/agent.yaml`。

> 所有临时脚本与输出均位于 `/tmp`，分析结束后**已全部删除**。本文档中的所有数字都可由 §附录B 的步骤重新测量。

---

## 2. 现状配置全景（`Ri-4438-HIM-Rough`）

### 2.1 仿真容量与框架开关

`src/tasks/locomotion/ri_4438_him/config/env_cfgs.py`（rough 覆盖）：

```python
39:  cfg.sim.mujoco.ccd_iterations = 500      # ← 显存主因
40:  cfg.sim.contact_sensor_maxmatch = 500    # ← 显存/速度次因
41:  cfg.sim.nconmax = None                   # ← 走启发式，实际 = 128（≠ base 文件里的 256）
44:  cfg.auto_reset = bool(play)              # 训练时为 False（HIM 需要"真终止观测"）
```

`src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py`（base）：

```python
582:    sim = SimulationCfg(
583:      nconmax = 256,          # 被上面的 rough 覆盖为 None → 128
584:      njmax = 1500,
585:      mujoco = MujocoCfg(timestep = 0.005, iterations = 10, ls_iterations = 20),
589:    ),
591:    decimation = ri_4438_cfg.control.decimation,   # 4
592:    episode_length_s = 20.0,
596:    auto_reset = False,
```

训练时实际生效的仿真参数（本机读取 `env.sim` 验证）：

| 参数 | 取值 | 说明 |
|---|---|---|
| `nconmax` | **None → 启发式 128** | `mujoco_warp/_src/io.py:1282` 的 `_default_nconmax`：`nv*0.35*(nhfield>0)*10+45 = 108` → 向上取到合法档位 **128** |
| `naconmax` / `naccdmax` | `nconmax × nworld` | 2048 env 时 = 262 144 |
| `njmax` / `njmax_pad` | 1500 / 1504 | 与官方一致 |
| `ccd_iterations` | **500** | 官方 flat 用 50 |
| `contact_sensor_maxmatch` | **500** | 官方 flat 用 64 |
| `broadphase` | 未设置 → NXN（默认） | 官方也未设置 |
| `nsensorcontact` | **100** | Go2 rough 为 46 |
| `ngeom` | **7547** | 其中 BOX 7499、HFIELD 10 |

### 2.2 传感器

| 传感器 | 类型 | 关键参数 | 出处于 |
|---|---|---|---|
| `terrain_scan` | `RayCastSensorCfg` | Grid (1.6×1.0)@0.1 → **187 条射线**；`max_distance=5.0`；`exclude_parent_body=True`；`include_geom_groups=(0,)`；`debug_vis=True` + `viz.show_normals=True` | `ri_4438_him_env_cfg.py:50-59`、`config/env_cfgs.py:51` |
| `feet_terrain_height` | `TerrainHeightSensorCfg` | 4 个 foot site × Ring(radius 0.02, 4 samples, include_center) = **5×4 = 20 条射线**；`ray_alignment="world"`；`reduction="min"` | `config/env_cfgs.py:56-81` |
| `feet_ground_contact` | `ContactSensorCfg` | geom→body("terrain")，`track_air_time=True` | `:85-93` |
| `nonfoot_ground_touch` | `ContactSensorCfg` | 仅 `^base_collision$`，`history_length=4` | `:95-111` |
| `leg_ground_contact` | `ContactSensorCfg` | 腿 geoms → terrain，`history_length=1` | `:113-128` |
| `self_collision_contact` | `ContactSensorCfg` | `.*_collision$` → subtree `base_link`，`history_length=4` | `:130-147` |

合计 6 个传感器（官方 Go1 rough 为 7 个），射线总量 207 条/env。

### 2.3 地形

`ri_4438_him_env_cfg.py:516-560`：

```python
curriculum=True, size=(8.0, 8.0), num_rows=10, num_cols=20, border_width=25.0
sub_terrains:
  flat                         0.05   BoxFlatTerrainCfg
  stairs_15/20/25              0.10×3 BoxPyramidStairsTerrainCfg   step_width=0.15/0.20/0.25, h=(0.05,0.20), platform=2.0
  inverted_pyramid_stairs_15/20/25/30  0.10×4 BoxInvertedPyramidStairsTerrainCfg  step_width=0.15/0.20/0.25/0.30, h=(0.05,0.20), platform=2.0
  discrete_obstacles           0.20   BoxRandomGridTerrainCfg   grid_width=0.4, h=(0.01,0.12)
  random_uniform               0.05   HfRandomUniformTerrainCfg
max_init_terrain_level = 9
```

注意第 518 行官方预设是被注释掉的：

```python
518:      #   terrain_generator = replace(ROUGH_TERRAINS_CFG),
```

### 2.4 观测 / 奖励 / 终止 / 课程 / 事件

| 项 | 现状 | 备注 |
|---|---|---|
| actor obs | 7 项 / 47 维，`history_length=6`，`flatten_history_dim=False` | 无 `height_scan`（HIM 用估计器代替） |
| critic obs | 244 维（含 `height_scan` = 187） | 与 `Ri4438HimCfg` 的切片一致 |
| 观测延迟 | `base_ang_vel`、`projected_gravity`、`joint_pos`、`joint_vel` 四项 `delay_max_lag=2, hold_prob=0.3, update_period=10` | `:138-154` |
| rewards | 21 项 | 官方 base 12 项 + Go1 特有 3 项 |
| terminations | `time_out` + `fell_over(80°)` | 官方还有 `out_of_terrain_bounds` |
| curriculum | `terrain_levels` + `command_vel`（7 段，按 `1000*100` 步推进） | `:484-507` |
| events | 9 项（含 `pd_gains` / `effort_limits` / `pseudo_inertia`，均为 `mode="startup"`） | 官方 base 6 项 |
| 执行器 | `BuiltinPositionActuatorCfg`（内置优化路径），带 `delay_*` | 不是 `IdealPdActuatorCfg` |

---

## 3. 与 mjlab 官方参考实现的逐项对比

参考文件：
`.venv/lib/python3.11/site-packages/mjlab/tasks/velocity/velocity_env_cfg.py`（官方 velocity 基类）
`.venv/lib/python3.11/site-packages/mjlab/tasks/velocity/config/go1/env_cfgs.py`（Go1 rough/flat）
`.venv/lib/python3.11/site-packages/mjlab/tasks/velocity/config/g1/env_cfgs.py`（G1 rough/flat）
`.venv/lib/python3.11/site-packages/mjlab/terrains/config.py:282`（`ROUGH_TERRAINS_CFG`）

### 3.1 地形：官方是「hfield 为主 + 少量 box 楼梯」，本项目是「全 box」

官方 `ROUGH_TERRAINS_CFG`（`terrains/config.py:282-301`）：

```python
TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=20.0, num_rows=10, num_cols=20,
  sub_terrains={
    "flat":                 flat(proportion=0.2),
    "pyramid_stairs":       pyramid_stairs(proportion=0.2, step_height_range=(0.0, 0.1)),   # step_width=0.3, platform_width=3.0
    "pyramid_stairs_inv":   pyramid_stairs_inv(proportion=0.2, step_height_range=(0.0, 0.1)),
    "hf_pyramid_slope":     hf_pyramid_slope(proportion=0.1, slope_range=(0.0, 1.0)),        # heightfield
    "hf_pyramid_slope_inv": hf_pyramid_slope_inv(proportion=0.1, slope_range=(0.0, 1.0)),    # heightfield
    "random_rough":         random_rough(proportion=0.1),                                    # heightfield
    "wave_terrain":         wave_terrain(proportion=0.1),                                    # heightfield
  },
  add_lights=True,
)
```

要点：

1. **官方 7 个子地形里有 4 个是 heightfield（`Hf*`）**，1 个 heightfield 子地形整块只占 **1 个 geom**；本项目 9 个里有 8 个是 box。
2. **官方只用 2 个 box 楼梯**（正/反各一），`step_width=0.3`、`platform_width=3.0`、`step_height_range=(0.0, 0.1)`；本项目用了 **6 个 box 楼梯**，`step_width=0.15~0.30`（更密）、`platform_width=2.0`、`step_height_range=(0.05, 0.20)`（更高）。
3. **官方没有 `BoxRandomGridTerrainCfg`**；本项目的 `discrete_obstacles`（0.4 m 网格）是纯 box。
4. 官方 `ROUGH_TERRAINS_CFG` 的 `curriculum` **默认是 False**（`terrain_generator.py:91`），Go1/G1 都显式设成 `True`（`go1/env_cfgs.py:136-137`、`g1/env_cfgs.py:83-84`）；本项目也设为 `True`，这点一致。

**固定 `seed=0 / rows=10 / cols=20 / curriculum=True` 的实测几何体数**（`TerrainEntity` 编译统计，仅地形）：

| 地形方案 | `ngeom` | `nhfield` |
|---|---:|---:|
| **① 现状（全 box）** | **7504** | 10 |
| ② 只把 `discrete_obstacles` 换成 heightfield | 4264 | 20 |
| ③ 只把 6 个 box 楼梯换成官方参数（sw=0.3 / plat=3.0 / h=0~0.1） | 5254 | 10 |
| ④ ②+③ 同时做 | 2014 | 20 |
| ⑤ 楼梯减到 2 个（官方参数）+ 网格换 heightfield | 694 | 20 |
| ⑥ **官方 `ROUGH_TERRAINS_CFG`** | **554** | 40 |

真实构建出的环境（含机器人几何体）实测：

| 任务 | `ngeom` | 构成 |
|---|---:|---|
| `Ri-4438-HIM-Rough` | **7547** | BOX 7499 / HFIELD 10 / MESH 18 / CYL 16 / SPH 4 |
| `Unitree-Go2-Rough`（用官方 `replace(ROUGH_TERRAINS_CFG)`） | **610** | BOX 519 / HFIELD 40 / MESH 33 / CYL 13 / SPH 5 |

> 结论：**同样是 10×20 的课程地形，本项目的 box geom 数量是官方预设的 13 倍以上。** 其中 `discrete_obstacles`（0.4 m box 网格）与 6 个密步 box 楼梯合计贡献了约 73%（见 §6）。

### 3.2 射线传感器：配置基本对齐，只有脚下传感器略有差异

| 项 | 官方 base / Go1 | 本项目 HIM | 评价 |
|---|---|---|---|
| `terrain_scan` 模式 | Grid (1.6,1.0)@0.1 → 187 线 | 同 | 一致（HIMLoco 原版也是 187 点） |
| `terrain_scan.exclude_parent_body` | True | True | 一致 |
| `terrain_scan.include_geom_groups` | `(0,)`（`velocity_env_cfg.py:53`） | `(0,)`（`config/env_cfgs.py:51`） | 一致 |
| `foot_height_scan` 射线数 | `single_ring(radius=0.04, num_samples=4)` = **4/足，16 总计** | `single_ring(radius=0.02, 4, include_center=True)` = **5/足，20 总计** | 差 4 条，可忽略 |
| `foot_height_scan.ray_alignment` | `"yaw"` | `"world"` | 语义差异，非性能项 |
| `debug_vis` / `viz.show_normals` | 官方 base 也设 `debug_vis=True` | 另加 `show_normals=True` | **训练时零成本**（只被 viewer 调用，见 `raycast_sensor.py:607` 的 `debug_vis()`） |

> **澄清**：`debug_vis` / `show_normals` **不是**性能问题。它们的代码路径只在 viewer 拉取调试标记时执行；训练时射线缓冲区的分配与计算与这两个开关无关（`raycast_sensor.py:531-559` 无条件分配，`postprocess_rays` 无条件计算）。

### 3.3 仿真容量：项目是唯一把 rough 的 `nconmax` 设成 `None` 的配置

| 配置 | `nconmax` | `njmax` | `ccd_iterations` | `contact_sensor_maxmatch` |
|---|---:|---:|---:|---:|
| mjlab 框架默认（`mjlab/sim/sim.py:116,157,162,167`） | `None` | `None` | 50 | 64 |
| mjlab 官方 base（`velocity_env_cfg.py:445-453`，只显式给 `nconmax`/`njmax`） | 35 | 1500 | 50（默认） | 64（默认） |
| mjlab 官方 **Go1 rough**（`go1/env_cfgs.py:38-41`） | 35（继承） | 1500 | **500** | **500** |
| mjlab 官方 **G1 rough**（`g1/env_cfgs.py:29-31`） | **70** | 1500 | **500** | **500** |
| mjlab 官方 Go1/G1 **flat** | `None` | 300 | 50 | 64 |
| 宇树官方 fork `unitree_rl_mjlab` G1 rough | **48** | 1500 | 500 | 500 |
| 本地 `go2_ppo` / `ri_4438_ppo` rough | 35（base 继承） | 1500 | 500 | 500 |
| **本项目 `Ri-4438-HIM-Rough`** | **None → 128** | 1500 | **500** | **500** |
| 本项目 `Ri-4438-HIM-Flat` | **256（未改）** | 300 | 50 | 64 |

三条关键观察：

1. **`ccd_iterations=500` 与 `maxmatch=500` 是从官方 rough 抄来的，本身不算错**；错在同时把 `nconmax` 放成 `None`。官方的 rough 都显式给了小值（35 / 70 / 48），只有 flat 才用 `None`——因为 flat 场景没有 hfield，启发式只给 45→48。
2. 本任务 `nhfield = 10 > 0`，`_default_nconmax` 的公式被 ×10 放大：`nv*0.35*10+45 = 108` → **128**，是官方 rough Go1 的 **3.7 倍**、G1 的 1.8 倍。
3. 本项目的 **flat 配置把 `nconmax = None` 这一行注释掉了**（`config/env_cfgs.py:261`），所以 flat 仍然用 base 的 **256**——比官方 flat 的启发式值（48）大 5 倍。flat 本身开销小，问题不突出，但属于同一个疏漏。

### 3.4 RL 超参数：与官方完全一致，不需要动

| 超参 | 本项目 HIM | mjlab 官方 Go1（`go1/rl_cfg.py`） | 判断 |
|---|---|---|---|
| `hidden_dims` | (512, 256, 128) | (512, 256, 128) | 一致 |
| `num_learning_epochs` | 5 | 5 | 一致 |
| `num_mini_batches` | 4 | 4 | 一致 |
| `learning_rate` / `schedule` | 1e-3 / adaptive | 1e-3 / adaptive | 一致 |
| `clip_param` / `entropy_coef` / `desired_kl` | 0.2 / 0.01 / 0.01 | 同 | 一致 |
| `gamma` / `lam` | 0.99 / 0.95 | 同 | 一致 |
| `num_steps_per_env` | **100** | 24 | **刻意不同**：100 是原版 HIMLoco（legged_gym）的设定，保留 |
| `max_iterations` | 见 `ri_4438_him_config.py` | 10 000（Go1）/ 30 000（G1） | — |

> HIM 特有的 estimator（`encoder_hidden_dims=(128,64,16)`、`target_hidden_dims=(128,64)`、`estimator_learning_rate=1e-3`）是 rsl_rl 官方**没有**的扩展（上游 rsl_rl 无 HIM 实现），无法与官方对比，但它每轮只增加约 0.05 s 量级。

### 3.5 本地四个任务的横向对比

| 项 | `ri_4438_him` rough | `ri_4438_ppo` rough | `go2_him` rough | `go2_ppo` rough | `ri_4438_him` flat |
|---|---|---|---|---|---|
| 地形生成器 | **自定义全 box（9 类）** | `replace(ROUGH_TERRAINS_CFG)` | 自定义全 box（8 类） | `replace(ROUGH_TERRAINS_CFG)` | plane |
| `max_init_terrain_level` | **9** | 5 | 5 | 5 | — |
| `ngeom`（实测，num_envs=1） | **7547** | — | ~7550 | **610** | 44 |
| `nconmax` | **None → 128** | 35 | **None → 128** | 35 | **256** |
| `ccd_iterations` / `maxmatch` | 500 / 500 | 500 / 500 | 500 / 500 | 500 / 500 | 50 / 64 |
| `njmax` | 1500 | 1500 | 1500 | 1500 | 300 |
| 传感器数 | 6（含 5 个接触类） | 3 | 4 | 3 | 6 |
| `auto_reset`（训练） | **False** | True（默认） | **False** | True（默认） | **False** |
| 接触传感器 `include_geom_groups` | 已设 `(0,)` | 已设 `(0,)` | 已设 `(0,)` | base 已设 | 已设 |

> `go2_him` 与 `ri_4438_him` 共享同一套「全 box 地形 + `nconmax=None` + `auto_reset=False`」模式，因此**同样会慢、同样吃显存**——本地 `go2_him` 日志（1024 env）collection 5.68 s 印证了这一点。若要修，两套任务应当一起改。

---

## 4. 显存分析：16 GB+ 的精确来源

### 4.1 主因：EPA 碰撞缓冲随 `ccd_iterations` 线性放大

MuJoCo Warp 的凸-凸碰撞在 `mujoco_warp/_src/collision_convex.py`：

```python
1192:  epa_iterations = 16 if nboxbox == ncollision else m.opt.ccd_iterations
1212:  epa_vert       = wp.empty(shape=(d.naccdmax, 10 + 2 * epa_iterations), dtype=wp.vec3)
1214:  epa_vert_index = wp.empty(shape=(d.naccdmax, 10 + 2 * epa_iterations), dtype=int)
1215:  epa_face       = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=int)
1217:  epa_pr         = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=wp.vec3)
1219:  epa_norm2      = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype=float)
1221:  epa_horizon    = wp.empty(shape=(d.naccdmax, MJ_MAX_EPAHORIZON), dtype=int)
```

- `MJ_MAX_EPAFACES = 5`，`MJ_MAX_EPAHORIZON = 24`（本机实测常量）。
- 本任务机器人含 cylinder / sphere / mesh 与地形 box 的接触，**`nboxbox != ncollision`，因此 `epa_iterations = ccd_iterations = 500`**（不是 16）。
- 这 6 个数组按 **`naccdmax = nconmax × nworld`** 分配，**每个 `naccdmax` 单位 66 376 字节**（`I=500`）。

于是：

| `ccd_iterations` | 每单位字节 | 512 env (naccd=65 536) | 1024 env (131 072) | **2048 env (262 144)** |
|---|---:|---:|---:|---:|
| **500（现状）** | 66 376 B | 4.05 GiB | 8.10 GiB | **16.21 GiB** |
| 50（官方 flat 值） | 6 976 B | 0.43 GiB | 0.85 GiB | **1.70 GiB** |

> **2048 env 时这组缓冲合计 16.21 GiB —— 就是用户观察到的「16 G 多」。**

**逐字节交叉验证**：本机以现状配置启动 1024 env 时，构建阶段直接 OOM：

```
RuntimeError: Failed to allocate 3941597184 bytes on device 'cuda:0'
```

而 `naccdmax(131 072) × (6 + 5×500) × 12 B = 131 072 × 2506 × 12 = 3 941 597 184` —— **正好是 `epa_pr` 这一个数组的大小**。根因确认，不是推测。

另外，把 `nconmax` 从 128 降到 64（官方量级）还能再砍一半：

| N=2048 | `nconmax=128` | `nconmax=64` | `nconmax=35`（Go1） | `nconmax=70`（G1） |
|---|---:|---:|---:|---:|
| `ccd=500` | **16.21 GiB** | 8.10 GiB | 4.43 GiB | 8.86 GiB |
| `ccd=50` | 1.70 GiB | 0.85 GiB | 0.47 GiB | 0.93 GiB |

### 4.2 次因：contact sensor 匹配缓冲（本任务 `nsensorcontact=100`）

`mujoco_warp/_src/sensor.py:2600-2604`，每个物理子步都会 `wp.empty` 三个 `[nworld, nsensorcontact, maxmatch]` 数组：

```python
sensor_contact_matchid   = wp.empty((d.nworld, m.nsensorcontact, m.opt.contact_sensor_maxmatch), dtype=int)
sensor_contact_direction = wp.empty((d.nworld, m.nsensorcontact, m.opt.contact_sensor_maxmatch), dtype=float)
sensor_contact_criteria  = wp.empty((d.nworld, m.nsensorcontact, m.opt.contact_sensor_maxmatch), dtype=float)
```

| N=2048 | `maxmatch=500`（现状） | `maxmatch=64`（官方 flat 值） |
|---|---:|---:|
| `Ri-4438-HIM`（`nsensorcontact=100`） | **1 172 MiB** | 150 MiB |
| `Unitree-Go2`（`nsensorcontact=46`） | 539 MiB | 69 MiB |

同时 `_contact_match` 的 launch 维度是 `(nsensorcontact, naconmax)`；N=2048、`nconmax=128` 时是 `(100, 262 144)` ≈ **2600 万线程/子步**，×4 子步。这也是 §5.3 中 `maxmatch` 从 500 降到 64 能省约 2.8 ms/步的原因。

### 4.3 几何体相关显存

| 项目 | 本项目 HIM | Go2 rough | 说明 |
|---|---:|---:|---|
| `ngeom` | 7547 | 610 | |
| 过滤后宽相对数 | **187 848** | 12 742 | `nxn_geom_pair_filtered` |
| 全量 pair 表 | **28 474 831 项（2×217 MiB = 434.5 MiB）** | 185 745 项（2.8 MiB） | `nxn_geom_pair` / `nxn_pairid` 是 `ngeom(ngeom-1)/2`，**与 env 数无关** |
| `geom_xmat` + `geom_xpos`（N=2048） | ~118 MiB | ~10 MiB | 按 `(nworld, ngeom)` 分配 |
| 约束雅可比 `efc.J`（N=2048, njmax_pad=1504, nv_pad=20） | ~235 MiB | ~235 MiB | 与地形无关 |

### 4.4 2048 env 显存逐项汇总（估算）

| 项目 | 现状（`ccd=500, nconmax=128, match=500`） | 仅改 CLI 三项后（`ccd=50, nconmax=64, match=64`） |
|---|---:|---:|
| EPA 碰撞缓冲 | **16.21 GiB** | 0.85 GiB |
| contact 匹配缓冲 | 1.17 GiB | 0.15 GiB |
| 几何体场量 + pair 表 | ~0.6 GiB | ~0.6 GiB |
| 约束/接触/求解器其它 | ~0.6 GiB | ~0.6 GiB |
| rollout storage（100 步 × 2048 env） | ~0.8 GiB | ~0.8 GiB |
| torch/Warp 上下文与其余 | ~0.5 GiB | ~0.5 GiB |
| **量级** | **≈20 GiB**（观测到 16 GB+ 完全合理，部分缓冲被 mempool 复用/回收） | **≈3.5 GiB** |

> 说明：EPA 缓冲是 **mempool 保留的峰值**（`wp.empty` 在每子步申请，peak 被保留），不是常驻数据结构。所以「16 G 多」和「算法本身需要多少显存」不是一回事——但也正因为是峰值，**降低 `ccd_iterations` 会直接、可预测地把占用降下来**。

### 4.5 为什么原版 HIMLoco（Isaac Gym）不吃显存

| 维度 | 原版 HIMLoco（legged_gym + Isaac Gym） | 本项目（mjlab + MuJoCo Warp） |
|---|---|---|
| 地形表示 | **一张 trimesh 网格**（1 个几何体） | **~7500 个 box geom** |
| 地形高度采样 | `self.height_samples` **张量直接索引** | MuJoCo Warp **GPU raycast + BVH refit** |
| 碰撞容量 | PhysX 内部管理，无用户可见的 `nconmax`/`naccdmax` | 所有容量参数 **按 world 批量预分配**，且 `naccdmax = nconmax × nworld` |
| CCD/EPA | PhysX 内部 | 显式 EPA 缓冲，**显存随 `ccd_iterations` 线性增长** |

所以：**在 Isaac Gym 里「换更细的地形」几乎不花显存；在 MuJoCo Warp 里它同时放大几何体数组、宽相对数、pair 表和 EPA 缓冲。** 这是后端结构性差异，不是移植写错了——但当前参数的组合确实把代价放大到了不必要的程度。

---

## 5. 速度分析：一轮 7 s 花在哪

### 5.1 远程日志（2048 env，RTX 4090）

日志：`logs/rsl_rl/ri_4438_him/2026-09-20_10-02-03`（迭代 17 600–17 901，共 302 轮）

| 阶段 | 中位数 | 平均值 |
|---|---:|---:|
| Rollout collection | **6.942 s** | 7.463 s |
| returns + PPO/HIM 更新 | 0.293 s | 0.324 s |

- **约 96% 的时间在采集阶段**，`Perf/total_fps` 中位数 ≈ 28 304 transition/s。
- 尖峰存在（collection 最长 27.7 s），但常规迭代稳定在 ~7 s。
- `params/env.yaml` 确认所有远程 run 都含：`nconmax: null`、`ccd_iterations: 500`、`contact_sensor_maxmatch: 500`、`auto_reset: false`、`nan_guard.enabled: true`、`num_steps_per_env: 100`。

> 注意：2026-09-23/24 的 1024 env 日志实际是 **8 卡**（`multi_gpu.world_size: 8`，aggregate fps ~140k），**不能**与 2048 env 单卡直接比较。

### 5.2 本机实测（RTX 4060，完整 `HIMOnPolicyRunner`，100 步/轮）

| n_envs | 配置 | `env.step`×100 | HIM 复位路径 | collection 合计 | PPO+HIM 更新 | 显存占用 |
|---:|---|---:|---:|---:|---:|---:|
| 256 | **现状**（auto_reset=True 对照） | 2.81 s | 0 | ~2.9 s | 0.17 s | 5 022 MiB |
| 512 | **现状** | 4.28 s | 1.52 s | **~5.9 s** | 0.31 s | **7 582 MiB** |
| 1024 | **现状** | — | — | — | — | **OOM（EPA 分配失败）** |
| 256 | 小容量 + 官方地形 | 1.62 s | 0.48 s | ~2.2 s | 0.17 s | 2 223 MiB |
| 512 | 小容量 + 官方地形 | 2.07 s | 0.72 s | **~2.9 s** | 0.26 s | **2 970 MiB** |
| 1024 | 小容量 + 官方地形 | 3.41 s | 1.49 s | ~5.0 s | 0.52 s | 4 378 MiB |
| 2048 | 小容量 + 官方地形 | 6.18 s | 2.44 s | ~8.7 s | 0.98 s | 7 408 MiB |

（「小容量」= `nconmax=64, ccd_iterations=50, contact_sensor_maxmatch=64`；「官方地形」= `replace(ROUGH_TERRAINS_CFG)`）

**同 GPU、同 env 数（512）对比**：iteration **5.9 s → 2.9 s（2.0×）**，显存 **7.58 GB → 2.97 GB（2.6×）**。
**2048 env 的外推**：现状 OOM 无法在 8 GB 卡上跑；改后 2048 env 本机 8.7 s，按 4090 约为 4060 的 2.5–3 倍折算约 **2.9–3.5 s/轮**，对应现状远程 6.94 s → **约 2–2.4 倍提速**。

### 5.3 `env.step` 内部归因（512 env，单步 ms）

| 变体 | ms/step | 相对增量 |
|---|---:|---|
| HIM **flat**（保留 `terrain_scan`、`height_scan`、全部传感器与奖励） | 9.85 | 公共基线：观测 + raycast + 接触 + 奖励 |
| HIM rough **现状** | 39.2 – 43.0 | **+29~33 ms 全部与地形相关** |
| 只改 `contact_sensor_maxmatch` 500 → 64 | 36.4 | −2.8 ms |
| 只改 `nconmax` 128 → 64 | 39.7 | −0.3 ms |
| 只改 `ccd_iterations` 500 → 50 | 40.7 | **−0.2 ms（噪声内）** |
| 三者都改 | 35.4 | −3.6 ms |
| 再换成官方地形 | **18.7** | **再 −16.7 ms** |
| `Unitree-Go2-Rough`（610 geom） | 17.5 | 参考 |
| `Unitree-Go2-Flat` | 15.2 | 参考 |

结论：

1. **射线传感器不是主因**：HIM flat 同样保留了 187+20 条射线的全部计算，仍然只要 9.85 ms。
2. **地形几何体是第一主因**：rough 与 flat 之间的 ~30 ms 里，绝大部分来自 7500 个 box geom 的宽相碰撞与窄相接触求解；换成官方地形后降到 18.7 ms，与 Go2 rough 的 17.5 ms 基本持平 —— **HIM vs Go2 的差距几乎完全由地形解释**。
3. **`ccd_iterations` 不影响速度**（本机 500 vs 50 在噪声内；子代理在 `go2_him` 上也测得 3.59 ms vs 4.01 ms），它是**纯显存参数**。
4. `contact_sensor_maxmatch` 500 → 64 有 ~7% 的实测收益；`nconmax` 128 → 64 收益较小（~1%）。
5. 官方文档也印证「大部分时间不在物理本身」：mjlab nightly benchmark 的 `overhead_pct`（非物理开销占比）为 52–68%。

### 5.4 HIM runner 的 `auto_reset=False` 复位路径（占 collection 20–26%）

`rsl_rl/runners/him_on_policy_runner.py` 为保留「真终止观测」，训练时 `auto_reset=False`，于是每个 policy step 的执行序列变成：

```
4× 物理子步 sim.step()
  + 1× sim.forward()            ← manager_based_rl_env.py:478（无条件）
  + 1× sim.sense()
  + 1× observation_manager.compute(update_history=True)
  + [有 env 结束] unwrapped.reset(env_ids)
        → 再一次 sim.forward() + sim.sense() + observation_manager.compute()
        → terminal_obs = obs.clone()（整批 TensorDict 克隆）
```

也就是说，**每个 policy step 实际做了 6 次 forward、2 次 sense、2 次 obs compute**（`auto_reset=True` 只需 5/1/1）。实测代价：

| 512 env | 现状 | 小容量 + 官方地形 |
|---|---:|---:|
| `_reset_done_envs` 累计 | 1.52 s/轮 | 0.72 s/轮 |
| 占 collection | **26%** | **25%** |

补充：`manager_based_rl_env.py:478` 的那一次 `sim.forward()` 是**无条件**的（注释说明是为了消除 `mj_step` 留下的一个子步陈旧量），即使没有任何 env 结束也会执行，约占 `env.step` 的 10%。

### 5.5 采样循环里的 GPU→CPU 同步点

每个 policy step（`auto_reset=False` + HIM runner + 默认 `check_for_nan=True`）：

| # | 位置 | 代码 |
|---|---|---|
| 1 | `mjlab/envs/manager_based_rl_env.py:427` | `if not self.cfg.auto_reset and torch.any(self._manual_reset_pending)` |
| 2 | 同文件 `:468` | `reset_env_ids = self.reset_buf.nonzero(...)`（无条件） |
| 3 | `mjlab/managers/command_manager.py:127` | `(self.time_left <= 0.0).nonzero()` |
| 4 | `mjlab/managers/event_manager.py:282` | `push_robot`（interval 事件）的 `nonzero()` |
| 5 | `rsl_rl/runners/him_on_policy_runner.py` + `rsl_rl/utils/utils.py:294` | `check_nan()` 对 actor/critic obs、reward、dones 各一次 `if torch.isnan(...).any()` |
| 6 | `rsl_rl/runners/him_on_policy_runner.py` | `dones.reshape(-1).nonzero()` |
| 7 | `rsl_rl/utils/logger.py:121` | `(dones > 0).nonzero().cpu().numpy().tolist()` |
| 8 | `rsl_rl/algorithms/him_ppo.py` | `_resolve_next_observations()` 的 `done_mask.any()` + 掩码写回 |
| 9 | `mjlab/utils/nan_guard.py:127`（**远程全部开启**） | 每个**物理子步** `if nan_mask.any():` → 每步再 +4 次 |

合计 **约 8–11 次/步**（开 nan_guard 时 12–15 次），100 步就是每轮近千次同步机会。本机实测：

- `check_for_nan` 在 512 env 下约 **1.7%**（不是主因，但可安全关掉）；
- `--enable-nan-guard` 约 **+1 ms/步（~2.4%）**，且 env 越多越贵——**这是纯调试工具，训练时不该开**。

---

## 6. 地形几何体专项分析

### 6.1 逐子地形贡献（seed=0 固定对比）

以现状 7504 个 geom 为基准：

| 改动 | 剩余 geom | 贡献 |
|---|---:|---:|
| 只把 `discrete_obstacles`（0.4 m box 网格）换成 heightfield | 4264 | **−3240（43%）** |
| 只把 6 个 box 楼梯换成官方参数 | 5254 | **−2250（30%）** |
| 两者同时 | 2014 | **−5490（73%）** |
| 楼梯减到 2 个 + 网格换 heightfield | 694 | −6810（91%） |

### 6.2 为什么 box 网格这么贵

`BoxRandomGridTerrainCfg(grid_width=0.4)` 在 8 m × 8 m 的 tile 上划分 0.4 m 网格 → 最多 20×20 个格子，每个格子一个 box geom；而 heightfield 版本（`HfRandomUniformTerrainCfg`）整块 tile **只有 1 个 geom + 1 个 hfield**。实测对比：

| 子地形 | 独占 20 列时 `ngeom` |
|---|---:|
| `BoxFlatTerrainCfg` | 14 |
| `BoxRandomGridTerrainCfg(0.4)` | **3254** |
| `BoxPyramidStairsTerrainCfg(step_width=0.15)` | 814 |
| `HfRandomUniformTerrainCfg` | **14**（+10 hfield） |

同理，楼梯的 geom 数主要由 **step_width**（步宽）与 **platform_width** 决定：`step_width=0.15` 比 `0.30` 多约一倍台阶，`platform_width=2.0` 比 `3.0` 少一圈平台但也意味着更多有效台阶。

### 6.3 建议的地形方案（按改动幅度）

| 方案 | 地形改动 | 预计 `ngeom` | 速度/显存 | 训练难度影响 |
|---|---|---:|---|---|
| B0 | 不改 | 7504 | 基线 | — |
| B1 | `discrete_obstacles` 换成 `HfRandomUniformTerrainCfg` / `HfDiscreteObstaclesTerrainCfg` | 4264 | `env.step` 约 −8 ms | 中：障碍从离散方块变成连续起伏 |
| B2 | 6 个 box 楼梯参数对齐官方（`step_width=0.3`、`platform_width=3.0`、`height=(0,0.1)`） | 5254 | −3 ms | 中：台阶变宽变矮 |
| B3 | B1 + B2 | 2014 | 接近 Go2 rough | 较大 |
| B4 | 只用官方 `replace(ROUGH_TERRAINS_CFG)`（打开第 518 行、去掉自定义块） | 554 | 与 `go2_ppo` 持平 | **最大**：地形种类与难度分布都变 |
| B5 | 保留 2 个楼梯 + heightfield 网格 | 694 | ≈ B4 | 较大 |

> **B 类改动属于「任务重定义」**：会改变接触几何、`critic` 的 `height_scan` 数值分布、以及课程难度曲线，**必须重新训练验证收敛**，不能当作纯性能优化。建议先做 §8 的 A 类（零代码、零语义变化），把性能和显存拉回正常区间，再单独评估 B 类。

---

## 7. 与其它 mjlab 项目的横向对比

| 项目 | 环境数 | 地形 | 射线/height-scan | rough 的 `nconmax` | 公开的速度/显存数据 |
|---|---:|---|---|---:|---|
| mjlab 官方 nightly | 4096 | 仅 flat 任务 | flat 用 raycast | — | Flat-Go1 317k env-SPS，`overhead_pct` 52–68% |
| mjlab 官方 Go1/G1 rough | 4096（示例） | `ROUGH_TERRAINS_CFG` | 有（187 点） | 35 / 70 | 无 |
| 宇树官方 `unitree_rl_mjlab` | 4096 | 官方预设 | 有 | **48** | 无 |
| `asimov-mjlab`（双足） | 4096 | 自定义 + 课程 | **完全没有 raycast** | — | 无 |
| `mjlab-homierl`（HIM-PPO 移植） | 4096 | terrain contact | 接触传感器 | — | 无（且注明 HIM-PPO 只支持单卡） |
| `twist2_mjlab` | 4096 | 动捕跟踪 | — | — | 唯一给出显存估算的项目（4096 env 工作集 ~4 GB） |
| **本项目 `ri_4438_him`** | **2048** | **自定义全 box** | 有（187+20 点） | **None → 128** | 本轮实测 |

可借鉴的两条：

1. **`asimov-mjlab` 证明「rough + 课程」不一定需要 raycast**。本项目 `terrain_scan` 只进 critic 的 `height_scan`（actor 不用），如果策略不需要地形前瞻感知，这是一条可选路径——但会改变 critic 维度与 checkpoint 兼容性，需评估。
2. **宇树官方把 rough 的 `nconmax` 压到 48**，比上游 mjlab 的 70 还低，说明「rough 必须给大 `nconmax`」并不成立；本项目的 128 明显过大。

---

## 8. 修改建议（按风险分级）

### A 类：零代码改动、零训练数学变化 —— 建议立刻验证

`scripts/train.py` 使用 tyro，以下开关均已实测存在（`--help` 可查）：

```bash
python scripts/train.py Ri-4438-HIM-Rough \
  --env.scene.num-envs 2048 \
  --env.sim.mujoco.ccd-iterations 50 \
  --env.sim.nconmax 64 \
  --env.sim.contact-sensor-maxmatch 64
# 关键：不要加 --enable-nan-guard
```

| 改动 | 预期效果 | 风险 |
|---|---|---|
| `ccd_iterations 500 → 50` | **2048 env 省约 14.5 GiB**（16.21 → 1.70 GiB）；实测碰撞耗时无差异 | 极低。只影响凸-凸 EPA 迭代上限；官方 flat 本就用 50 |
| `nconmax None → 64` | 再省一档（1.70 → 0.85 GiB）；对齐官方/宇树量级 | 低。若接触溢出，Warp 默认 `overflow_behavior=error` 会直接报错，易于发现 |
| `contact_sensor_maxmatch 500 → 64` | 省约 1 GiB；`env.step` **−2.8 ms（−7%）** | 低。需监控 `contact match overflow: please increase Option.contact_sensor_maxmatch` 警告 |
| 不加 `--enable-nan-guard` | 每步少 4 次 GPU 同步（约 −2.4% 且随 env 数放大） | 无（纯调试工具） |

**合计**：2048 env 显存从 ~20 GiB 量级降到 ~3.5 GiB 量级，collection 约 −10%。

**验证方法**：
1. 跑 200–300 轮，对比 `Perf/collection_time`、`Perf/learning_time`、`Perf/total_fps` 与训练曲线是否与现状一致（前 100 轮应当几乎重合）。
2. 显存：`nvidia-smi` 峰值 或 在训练脚本里读 `torch.cuda.max_memory_allocated()`。
3. 必须确认日志里**没有** `ncon overflow` 与 `contact match overflow` 警告；如有，就地把 64 提到 96/128 再试。
4. 建议同时记录接触传感器的 `found`/`force` 统计，确认 foot contact 行为没有变化（这是唯一可能受 `maxmatch` 影响的量）。

### B 类：改地形（会改变任务难度，需重新验证）

见 §6.3。建议**一次只改一项**，并保留 A 类改动作为基线。

### C 类：框架/runner 层（需要改代码，语义可论证）

| # | 改动 | 收益 | 注意 |
|---|---|---|---|
| C1 | `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py:93` 打开 `check_for_nan = False`（或降低检查频率） | 每步少 4 次同步（本机 512 env 约 1.7%） | 会推迟 NaN 的发现时机 |
| C2 | `HIMOnPolicyRunner._reset_done_envs`：`auto_reset=False` 引入的额外 `forward()+sense()+obs compute()` | **占 collection 20–26%**，是本轮最大的软件侧可回收项 | 需保留「真终止观测」语义；可考虑让 mjlab 的 `reset()` 只对子集做 forward/sense，或改用别的等价方案 |
| C3 | `HIMRolloutStorage` 不再保存 next **actor** obs（estimator 只用 next critic） | 省约 220 MiB 常驻 + 减少 minibatch 拷贝 | 需要改动 storage 的 TensorDict 结构 |
| C4 | `HIMPPO._resolve_next_observations()` 在 terminal obs 与输入为同一对象时走快路径 | 免去每轮约 411 MiB 的整批克隆写入 | 加明确的等价条件 + 保留通用回退 |
| C5 | `ri_4438_him/config/env_cfgs.py:261` 把 flat 的 `nconmax = None` 恢复（现在是注释状态，flat 仍用 256） | flat 任务省显存 | 与 rough 一致性问题，收益有限 |

### D 类：不要动

`num_steps_per_env=100`、5 epochs × 4 minibatches、`hidden_dims=(512,256,128)`、`decimation=4`、`episode_length_s=20`、网络结构 —— 这些与原版 HIMLoco 及官方 mjlab 一致，且 learning 只占 ~4%，改动只会改变训练动力学而不解决性能问题。

### E 类：需要你确认的配置疑点（非性能）

`ri_4438_him_env_cfg.py:559` 的 `max_init_terrain_level = 9`：官方所有任务都用 **5**（`velocity_env_cfg.py:423`、`go1`/`g1`/`go2_ppo`/`ri_4438_ppo` 均 5）。9 意味着初始就把环境铺到难度第 9 行（最高档），机器人**一开始就站在最难地形上**。这不会显著影响性能，但可能影响早期学习效率与课程曲线，建议核对是否有意为之。

---

## 9. 被排除的假设（澄清，避免误改）

| 假设 | 实测结论 |
|---|---|
| 「rollout storage 太大导致 16 GB」 | **不成立**。2048×100 步的 actor+critic 观测缓冲约 **0.8 GiB**，可精简 ~220 MiB，但远不足以解释 16 GB |
| 「射线传感器是主要瓶颈」 | **不是主因**。HIM flat 保留 187+20 条射线与全部观测/奖励，`env.step` 仍只有 9.85 ms；rough 多出来的 30 ms 来自地形几何体 |
| 「`ccd_iterations=500` 拖慢速度」 | **不成立**。本机 500 vs 50 在噪声内；它是**纯显存参数** |
| 「`debug_vis` / `show_normals` 有开销」 | **无开销**。只在 viewer 的 `debug_vis()` 路径中使用（`raycast_sensor.py:607`），训练时零成本 |
| 「NaN 检查是主要瓶颈」 | **不是主因**。512 env 下 `check_for_nan` 约 1.7%；但 `--enable-nan-guard` 确实每子步强制同步，应关闭 |
| 「`pseudo_inertia` 域随机化拖慢采集（mjlab #757 的 +54%）」 | **本项目不适用**。这里是 `mode="startup"`（`:307-309`），#757 的问题是 `reset` 模式每步触发 `set_const` |
| 「执行器延迟在 Python 里逐步求值导致变慢（mjlab #1035）」 | **不适用**。本项目用 `BuiltinPositionActuatorCfg`（内置优化路径），不是 `IdealPdActuatorCfg` |
| 「观测延迟缓冲引入 CUDA 同步（mjlab PR #1031）」 | **1.6.0 已修复**。`circular_buffer.py` / `delay_buffer.py` 中无 `.item()`/`nonzero()`/`.cpu()`，全部在 GPU 侧完成 |
| 「1080p 观察到的 16 GB 是常驻需求」 | **部分是峰值**。EPA 缓冲是每子步申请、被 mempool 保留的峰值；但降 `ccd_iterations` 仍会直接降低峰值 |

---

## 10. 局限与未验证项

1. **绝对秒数不可搬运**：本文所有分段计时来自 RTX 4060（8 GiB），远程是 RTX 4090（24 GiB）。**倍数与归因可参考**，2048 env 的具体秒数需要你在远程复测。
2. **未在远程 4090 上验证过任何新参数**。A 类改动只在本地 8 GB 卡上验证过「能跑通 + 更快 + 更省」。
3. **地形几何体数依赖难度与随机种子**：§6 的对比固定 `seed=0 / rows=10 / cols=20 / curriculum=True`；默认（无 seed）会有少量浮动（实测官方预设 554–2436 之间浮动，主要受 `curriculum` 与难度采样影响），但**数量级结论不变**。
4. **EPA 字节数是解析计算**（由 `wp.empty` 的 shape × dtype 得出），并与实测 OOM 的精确字节数、以及子代理测得的 Warp pool 增长量交叉验证过；但**未直接读取 Warp 的内存池分配明细**。
5. **`nsensorcontact=100` 的构成未逐项拆解**（4 个接触传感器 × primary/secondary/slot 的组合），只取模型字段总值。
6. **未测量 `njmax=1500` 是否可降**：`efc.J` 在 2048 env 下约 235 MiB，优先级最低；但若要极致压显存，可用 warp 的 `overflow_behavior=error` + `--measure_alloc` 方式逐档下调。
7. **B 类地形改动的训练影响完全未验证**，需要实际训练才能判断收敛与最终性能。

---

## 附录 A：关键文件与行号索引

| 内容 | 位置 |
|---|---|
| rough 容量覆盖（`ccd`/`match`/`nconmax`/`auto_reset`） | `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py:39-44` |
| `terrain_scan` 的 `include_geom_groups` | `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py:51` |
| 脚下地形高度传感器 | `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py:56-81` |
| 4 个接触传感器 | `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py:85-147` |
| flat 容量（`nconmax` 被注释） | `src/tasks/locomotion/ri_4438_him/config/env_cfgs.py:258-261` |
| 传感器与射线定义 | `src/tasks/locomotion/ri_4438_him/ri_4438_him_env_cfg.py:50-59` |
| 观测组与延迟 | `.../ri_4438_him_env_cfg.py:121-154` |
| 21 项奖励 | `.../ri_4438_him_env_cfg.py:330-465` |
| 终止条件 | `.../ri_4438_him_env_cfg.py:470-479` |
| 课程（含 `command_vel` 7 段） | `.../ri_4438_him_env_cfg.py:484-507` |
| 自定义地形 + `max_init_terrain_level=9` | `.../ri_4438_him_env_cfg.py:516-560` |
| 仿真容量 / decimation / episode | `.../ri_4438_him_env_cfg.py:582-596` |
| `check_for_nan` 被注释 | `src/tasks/locomotion/ri_4438_him/config/rl_cfg.py:93` |
| `num_envs` / 观测维度 / `num_steps_per_env=100` | `src/config/ri_4438/ri_4438_him_config.py:15-76` |
| 官方 velocity 基类（传感器/仿真/奖励） | `.venv/.../mjlab/tasks/velocity/velocity_env_cfg.py:46-55, 418-455` |
| 官方 Go1 rough/flat | `.venv/.../mjlab/tasks/velocity/config/go1/env_cfgs.py:38-41, 261-311` |
| 官方 G1 rough/flat | `.venv/.../mjlab/tasks/velocity/config/g1/env_cfgs.py:29-31, 192-195` |
| 官方地形预设 | `.venv/.../mjlab/terrains/config.py:282-301` |
| 地形生成器默认值（`curriculum=False`） | `.venv/.../mjlab/terrains/terrain_generator.py:89-124` |
| EPA / `naccdmax` | `.venv/.../mujoco_warp/_src/collision_convex.py:1192, 1212-1221` |
| `_default_nconmax` 启发式 | `.venv/.../mujoco_warp/_src/io.py:1282-1292` |
| contact sensor 匹配缓冲 | `.venv/.../mujoco_warp/_src/sensor.py:2600-2604` |
| `env.step` 同步点与额外 forward | `.venv/.../mjlab/envs/manager_based_rl_env.py:427, 468, 478, 490-491` |
| nan_guard 每子步同步 | `.venv/.../mjlab/utils/nan_guard.py:127-128` |
| raycast 调试可视化（仅 viewer） | `.venv/.../mjlab/sensor/raycast_sensor.py:531-559, 607-641` |
| HIM runner 复位路径 | `rsl_rl/runners/him_on_policy_runner.py` |
| `check_nan` | `rsl_rl/utils/utils.py:294` |
| logger 的 `.cpu()` | `rsl_rl/utils/logger.py:121` |
| `_resolve_next_observations` | `rsl_rl/algorithms/him_ppo.py` |
| rollout storage | `rsl_rl/storage/him_rollout_storage.py` |

## 附录 B：复现步骤

```bash
cd /home/sunteng/projects/legged_wbc_mjlab
PY=.venv/bin/python

# 1) 读实际生效的仿真容量 / 几何体 / 宽相统计（只读）
$PY - <<'PY'
import mjlab.tasks, src.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
for t in ("Ri-4438-HIM-Rough", "Unitree-Go2-Rough"):
    cfg = load_env_cfg(t); cfg.scene.num_envs = 64
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    d, m, mj = env.sim._wp_data, env.sim._wp_model, env.sim.mj_model
    print(t, "ngeom", mj.ngeom, "nhfield", mj.nhfield,
          "nconmax", d.naconmax // 64, "nsensorcontact", m.nsensorcontact,
          "ccd", m.opt.ccd_iterations, "maxmatch", m.opt.contact_sensor_maxmatch,
          "filtered_pairs", m.nxn_geom_pair_filtered.shape[0],
          "all_pairs", m.nxn_geom_pair.shape[0])
PY

# 2) 地形几何体对比（固定 seed，逐方案统计 ngeom）
#    见本文 §3.1 / §6.1 的表格；用 TerrainEntity(TerrainEntityCfg(...), device="cpu").spec.compile()

# 3) A 类改动的速度 / 显存验证（注意：会启动真实训练）
$PY scripts/train.py Ri-4438-HIM-Rough --env.scene.num-envs 2048 \
    --env.sim.mujoco.ccd-iterations 50 --env.sim.nconmax 64 \
    --env.sim.contact-sensor-maxmatch 64
```

---

## 参考链接（外部资料）

- mjlab nightly benchmark（唯一官方吞吐数据，仅 flat 任务）：https://mujocolab.github.io/mjlab/nightly/
- mjlab issue #719（rough vs flat，8192 env，collection 63.5 s vs 2.19 s）：https://github.com/mujocolab/mjlab/issues/719
- mjlab issue #486（raycast 相对 Isaac 慢 2–3×，BVH 优化）：https://github.com/mujocolab/mjlab/issues/486
- mjlab issue #757（`reset` 模式 DR 触发 `set_const`，collection +54%）：https://github.com/mujocolab/mjlab/issues/757
- mjlab PR #1020 / #1031（消除 Python list 索引与延迟缓冲的隐式 CUDA 同步）：https://github.com/mujocolab/mjlab/pull/1020 · https://github.com/mujocolab/mjlab/pull/1031
- mjlab issue #1056（CUDA graph 被禁用导致 A100 慢 6 倍）：https://github.com/mujocolab/mjlab/issues/1056
- MuJoCo Warp 官方文档（Batch sizes / Contact sensor matching / Memory）：https://mujoco.readthedocs.io/en/latest/mjwarp/index.html
- 宇树官方 fork `unitree_rl_mjlab`（rough `nconmax=48`）：https://github.com/unitreerobotics/unitree_rl_mjlab
- `asimov-mjlab`（rough + 课程、无 raycast）：https://github.com/menloresearch/asimov-mjlab
- `mjlab-homierl`（第三方 HIM-PPO 移植，仅单卡）：https://github.com/Nagi-ovo/mjlab-homierl
