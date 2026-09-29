# best_model 工具使用

入口是 `scripts/best_model.py`。日常使用指定**实验总目录或某次训练的 run 目录**，再加 `--evaluate`：传总目录时自动选择最新 run，再追溯原始训练和历次续训，显示完整训练记录，仿真比较整条训练链的全部有效模型，在选中 run 的 `best/` 中生成中文报告。

## 1. 直接运行：使用当前这份日志

在项目根目录执行。首次使用时安装分析依赖：

```bash
cd /home/sunteng/projects/legged_wbc_mjlab
uv sync --extra analysis
```

可以直接使用实验总目录，以后新增续训目录也不用修改这条命令：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run '/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him' \
  --evaluate --eval-parallel 8
```

终端会先显示“检测到实验目录，选中最新 run”，再列出整条续训链。报告写到**选中 run 的 `best/`**，例如当前会选中 `2026-09-29_21-37-06`，报告是 `ri_4438_him/2026-09-29_21-37-06/best/report.html`。以终端最后打印的路径为准。

如果要固定分析某次训练，可以继续指定具体 run：

```bash
EVAL_RUN_DIR="/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-29_17-21-34"

uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" \
  --evaluate
```

后续续训时，把 `EVAL_RUN_DIR` 改成最新 run 路径，再执行上面的命令。旧 run 应保留，工具会沿保存的续训来源向前查找；无需把 checkpoint 复制到同一目录。仿真使用项目的 MuJoCo/mjlab 环境，默认设备为 `cuda:0`。

终端先列出 `R1、R2…` 对应的 run，再用 `[eval]` 显示当前模型批次、地形和种子；`[images]` 表示正在生成截图。看到 `仿真推荐: R2 / model_... | 报告: .../report.html` 后，本轮评测和报告生成已完成。

完成后打开报告：

```bash
xdg-open "$EVAL_RUN_DIR/best/report.html"
```

也可以在文件管理器中打开 `best/report.html`，用浏览器查看。若页面已经打开，重新评测后刷新页面。

## 2. `--run` 应该填哪个目录

支持两种层级：

- **实验总目录**，例如 `logs/rsl_rl/ri_4438_him/`：只扫描直接子目录中含 `params/agent.yaml` 的 run，按目录名中的 `YYYY-MM-DD_HH-MM-SS` 时间选择最新一个，再追溯其续训来源。自定义目录名没有时间前缀时，用 `agent.yaml` 的修改时间作为备用依据；同一时间按目录名排序。报告目录的修改时间不会影响选择。
- **具体 run 目录**：直接使用该 run，例如下方结构。`--single-run` 用于只分析选中的 run，传总目录时同样先选择最新一个。

```text
2026-09-29_17-21-34/          ← 也可以直接传这一层
├── events.out.tfevents....
├── model_500.pt
├── model_600.pt
├── ...
└── params/
    ├── agent.yaml
    └── env.yaml
```

路径可以在项目外。总目录入口选择的是**最新 run 对应的续训链**，其他独立训练或分支不会自动混入。若最新 run 的来源不明确，会给出原来的来源错误，不会悄悄改选更早的 run。单个 `.pt`、`best/` 或包含多个实验的更上层 `logs/` 不是这个入口需要的路径。完整分析应保留各次训练的 `params/`、事件文件和数字模型文件。需要手动列出多个目录时用第 6 节的 `--runs`。

## 3. 默认会测试哪些内容

| 项目 | 默认行为 |
| --- | --- |
| 模型范围 | 原始训练及历次续训目录中全部有效的 `model_<数字>.pt`，含早期分段及旧分支模型 |
| 同名模型 | 按来源分开，例如 `R1 / model_500` 和 `R2 / model_500`；不会覆盖、合并或只保留其中一个 |
| 页面展示 | 全部评测后排名，HTML、Markdown、对比图和筛选只展示前 10 名 |
| 完整成绩 | CSV、JSON 保留全部已测模型 |
| 并行模型 | 根据可用显存自动选择 1～8 个 |
| 每模型环境数 | 24 个；8 个模型同时运行时共 192 个独立环境 |
| 随机种子 | 3 组：`0、1、2`；所有模型使用相同种子和配对初始状态 |
| 每回合时长 | 最长 12 秒；首次失败后结束计分 |
| 地形 | 平地、±2 cm 起伏，5/10/15 cm 各自的上楼梯和下楼梯，共 8 个场景 |
| 每模型回合数 | `24 × 3 × 8 = 576` 个首次回合 |
| 仿真截图 | 前 3 名在各地形中的真实计分状态，附在报告末尾 |

楼梯高度 **固定为 5、10、15 cm**，无需输入高度参数。每段 6 级，踏面宽 30 cm，总高差分别为 30、60、90 cm。每个高度都分别测上楼、下楼；楼梯指令速度为向前 `0.3、0.5、0.8 m/s`。

楼梯通过要求走过终点、达到对应高度并存活至回合结束。停在台阶前，即使没有跌倒，也不算通过。总排名先比较任务通过率，再依次比较存活比例、存活时间、速度误差、转向误差和机身倾斜。

截图统一取首个种子、前进 `0.5 m/s`、同一个局部环境编号，在 2 秒、6 秒和回合结束采样；提前失败则保存首次终止画面。图片用于直观看姿态与台阶位置，完整通过率仍按 3 组种子的所有回合统计。

## 4. 并行数量、种子和显存设置

以下命令继续使用第 1 节设置的 `EVAL_RUN_DIR`。

显式使用 8 个模型并行、3 组种子：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" --evaluate \
  --eval-parallel 8 \
  --eval-seeds 0 1 2
```

显存紧张或同时训练时，可以降低并行数量和每模型环境数；仍然会评测全部模型：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" --evaluate \
  --eval-parallel 2 \
  --eval-num-envs 12
```

此时每模型有 `12 × 3 × 8 = 288` 个回合。降低样本数会改变评测精度；横向比较报告时应核对测试条件。

| 参数 | 用途与约束 |
| --- | --- |
| `--run /path/to/experiment_or_run` | 总目录自动选择最新 run；具体 run 直接使用，然后追溯续训来源 |
| `--runs /path/to/original /path/to/resumed ...` | 按原始训练到最新续训的顺序显式指定；与 `--run` 二选一 |
| `--single-run` | 与 `--run` 搭配，仅分析传入目录，不追溯历史 |
| `--eval-parallel 0` | 默认值，按显存自动选择；正整数可指定 1～16 个模型并行，实际数量不超过本轮模型总数 |
| `--eval-num-envs 24` | 每个模型的环境数，必须是正的 6 的倍数 |
| `--eval-seeds 0 1 2` | 必须恰好 3 个互不相同的非负整数；可换成其他 3 个值 |
| `--eval-duration 12` | 每回合时长，范围 1～120 秒；时间太短可能来不及完成楼梯 |
| `--eval-device cuda:0` | 选择仿真设备，例如另一张卡 `cuda:1` |
| `--output /path/to/report` | 指定独立输出目录，默认是 `<run>/best/` |
| `--eval-task Ri-4438-HIM-Rough` | 显式指定任务；通常可从保存的配置推断，当前支持 RI-4438 HIM/PPO 的 Rough/Flat 任务 |
| `--no-plot` | 跳过统计图，仍生成真实仿真截图、HTML 和数值结果 |

`--eval-count` 默认是 `0`，即全部模型。`--eval-count 10` 会把实际测试数量限制到 10 个；`--eval-checkpoints 500 700` 只测试这些编号，若多个 run 都有该编号，会全部保留。这两个参数用于主动缩小诊断范围，日常全量比较直接使用默认值。报告的前 10 名展示规则固定，`--top-k` 只控制训练日志候选数量。

## 5. 报告和 best 模型在哪里

默认输出在 `<run>/best/`：

| 文件或目录 | 用途 |
| --- | --- |
| `report.html` | 用浏览器打开：前 10 名排名、地形筛选、稳定性、训练趋势及仿真截图 |
| `report.md` | 可随图片一起分享的 Markdown 报告 |
| `training_history.html` / `history.png` | 原始训练和历次续训的完整曲线；无需 `--evaluate` 也生成 |
| `training_chain.json` | 所有 run 的来源、配置哈希和各日志分段的课程、候选、收敛分析 |
| `chain_candidates.csv` | 所有 run 的日志候选及来源；不同奖励条件的分数不直接混合排名 |
| `model_best_eval.pt` | 本轮实际仿真推荐模型；使用 `--evaluate` 后自动保存 |
| `eval_model_manifest.json` | 仿真推荐模型的原始路径、迭代编号和 SHA256 |
| `evaluation_ranking.csv` | 全部已测模型的排名和汇总指标 |
| `evaluation_cases.csv` | 全部模型按地形、指令划分的成绩 |
| `evaluation.csv` | 全部模型的逐回合结果 |
| `evaluation.json` | 完整协议、模型信息、成绩、轨迹和截图状态 |
| `evaluation.previous.json` | 同一输出目录中上一次完成的评测结果，用于自动复跑核对；只保留上一份 |
| `simulation/` | 仿真原始图片、分地形拼图、记录状态和场景文件 |
| `selection.json` | 训练日志 best、课程阶段、收敛状态及判断依据 |

**日常查看整条训练链的仿真 best，用 `model_best_eval.pt`。** 它可能来自任意一个历史 run，原始来源见 `eval_model_manifest.json`。`model_best.pt` 由最新 run 的当前阶段日志评分选出，需加 `--write-best` 才保存；`--export-onnx` 导出的 `policy.onnx` 也对应这个日志 best。两种模型可能不同。

`model_best_eval.pt` 表示**本轮评测排名第一的候选**，不表示已经证明它是唯一最优。报告的“复跑核对”会说明它领先多少、与上一次比较时条件有没有变化、第一名是否稳定。首次运行没有旧结果时，会明确标为尚未验证重复性。

每次成功评测会更新同一个输出目录中的报告和 `model_best_eval.pt`，原始数字 checkpoint 保持不变。希望保留每次报告时，指定不同的输出目录：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" --evaluate \
  --output "$EVAL_RUN_DIR/best_review_01"
```

上面这次的报告位于 `best_review_01/report.html`。

## 6. 续训之后怎么用

**通常只需传实验总目录或最新 run，不必逐个填写历史目录。** 传总目录时，每次运行会重新选择最新 run。工具优先读取新训练保存的 `params/lineage.json`，其中记录实际加载的来源 run、checkpoint 和 SHA256；旧日志则使用 `agent.yaml` 中明确的 `load_run` 与 `load_checkpoint`，一直向前追到原始训练。

以 `2026-09-29_17-21-34` 这个 run 的此前快照为例：

```text
R1  2026-09-29_16-14-31  （6 个有效模型）
 └─ 从 model_500.pt 续训
R2  2026-09-29_17-21-34  （25 个有效模型）
```

共 31 个模型，两个目录中的 `model_500.pt` 分别参与仿真；显示前 10 名，但 CSV / JSON 保留全部结果。之后再续训一次时，改成最新目录即可：

```bash
EVAL_RUN_DIR="/path/to/logs/rsl_rl/ri_4438_him/新的续训run目录"

uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" --evaluate
```

旧日志如果只保存了 `load_run: .*`、`load_checkpoint: model_.*.pt`，就无法可靠还原当时实际选中的来源。工具会提示显式传入完整目录列表，而不会把同一实验目录里的所有训练都误当成续训：

```bash
uv run --extra analysis python scripts/best_model.py \
  --runs \
  /home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-29_16-14-31 \
  /home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-29_17-21-34 \
  --evaluate --eval-parallel 8
```

目录顺序必须从原始训练到最新续训。来源记录明确时会验证这个顺序；只能依赖手动顺序的连接会在报告中标为“关系未核实”。输出默认放在最后一个目录的 `best/`。如果只想看一个 run：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" --single-run --evaluate
```

完整曲线保留各 run 的原始 iteration，标出续训恢复点，各段分别平滑。比如旧 run 训练到 1000，再从 500 续训：旧的 501～1000 曲线和模型仍保留显示、参与仿真，但不会当作新分支已经经历的训练过程。若从 0 重新计数，两条曲线可能重叠，按颜色和 R 编号区分。

各 run 的各日志分段按自己的奖励、课程配置分析，HTML 列出每段训练状态。页面主收敛结论来自最新 run 的所选分段；即使配置相同，目前也不跨续训边界拼接收敛窗口，因此刚续训时仍可能显示“数据不足”。所有模型的性能比较则使用同一套仿真场景，每个模型按自己的网络配置加载。同名模型的排名、场景成绩、截图和复跑核对均按来源区分。

同一 run 内 iteration 回退也会分段。`--segment N` 选择最新 run 的主分析分段，`--env-step-offset` 只用于最新 run；完整历史仍保留所有分段。旧模型的观测、动作接口、控制步长或动作裁剪与统一测试条件不兼容时会明确报错。原目录缺失或原始 checkpoint 被删，工具无法补回丢失的数据。

同一个 run 继续新增 checkpoint 时，再次执行原命令即可。每轮只处理启动时读取的日志和模型，不会在运行中途加入新保存的文件，也不会持续后台监控。

## 7. 只分析日志，以及怎么看“收敛”

只想看训练趋势时，可以省略 `--evaluate`，另存到 `log_analysis/`：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" \
  --output "$EVAL_RUN_DIR/log_analysis"
```

这会输出完整训练页面 `training_history.html`、完整曲线 `history.png`、`training_chain.json`，以及最新 run 的 `selection.json`、`candidates.csv` 和 `report.png`。打开 `log_analysis/training_history.html` 即可查看；包含模型性能和仿真截图的 `report.html` 需要 `--evaluate`。

| 状态 | 怎么理解 |
| --- | --- |
| 课程仍在推进 | 难度或奖励条件仍在改变，暂不判断最终收敛 |
| 数据不足 | 刚续训或刚进入新阶段，观察窗口还不够 |
| 仍在改善 / 退化 | 近期训练指标还在明显变化 |
| 波动较大 | 近期表现不够稳定 |
| 平台期 | 最近一段时间的训练曲线变化很小，可能学稳了，也可能卡住了 |
| 收敛候选 | 曲线稳定，并且达到显式设置的目标，仍需结合仿真能力判断 |

默认不指定任务目标时，曲线稳定最多判断为平台期；`best` 也只表示参与比较的模型中排名靠前。收敛判断使用训练日志，仿真通过率和稳定性提供独立的性能证据。

日志评分、`--goal` 和 ONNX 导出的详细说明见 [README：选择 best 模型与检查收敛](../README.md#选择-best-模型与检查收敛)。

## 8. 回放仿真推荐模型

以 RI-4438 HIM Rough 为例：

```bash
uv run python scripts/play.py Ri-4438-HIM-Rough \
  --checkpoint-file "$EVAL_RUN_DIR/best/model_best_eval.pt" \
  --num-envs 1
```

使用 `--output` 另存结果时，相应修改这里的模型路径。这个入口使用任务的 play 环境；报告中的固定楼梯、指令和种子由 `best_model.py --evaluate` 的评测协议设置。

## 9. 常见问题

### 同一个 log，为什么两次第一名可能不同？

需要区分三个原因：

1. **参评模型变化**：续训新增了 checkpoint，或覆盖了原来的文件，比较对象就变了。
2. **测试条件变化**：参数、配置、代码、运行库等变化会影响结果；默认并行数量还会随可用显存变化。
3. **GPU 物理仿真数值波动**：即使权重、环境配置、种子和初始状态相同，GPU 接触求解的浮点累加仍可能产生细微差异，随后在动作反馈中放大。固定 3 组种子不等于保证每次轨迹和排名完全一致。

复测时先固定实际并行数量，在模型文件和代码不变的情况下，重复运行相同命令，使用**同一个输出目录**：

```bash
uv run --extra analysis python scripts/best_model.py \
  --run "$EVAL_RUN_DIR" --evaluate \
  --eval-parallel 8 --eval-seeds 0 1 2
```

工具会重新仿真，读取上次的 `evaluation.json`，成功后将它保存为 `evaluation.previous.json`。HTML 的“复跑核对”会检查已记录的模型哈希、配置、编译物理模型、运行库、测试参数和配对初始状态，显示通过判定改变了多少回合、单模型通过率的最大变化以及第一名是否改变。旧版报告未记录的运行库信息会标为未核实。

若续训仍在新增模型，可以暂时用 `--eval-checkpoints 500 600 700` 固定要复测的模型列表，数字替换为自己的列表。使用新的 `--output` 目录时没有旧结果可自动比较；再次使用该目录后才有复跑核对。

如果第一名变化，或领先幅度很小，应把前几名当作待确认候选。当前工具**没有消除 GPU 物理仿真的数值波动**，也不会通过沿用旧成绩来伪装重复性。两次第一名相同，只能说明这两次排名一致，不能证明唯一最优；这与训练日志中的“平台期 / 收敛候选”是不同的判断。

| 现象 | 处理方式 |
| --- | --- |
| `run directory does not exist` | 核对 `--run` 的完整路径，指向具体 run 目录 |
| `no training runs found` | 传具体 run，或直接包含各时间戳 run 的实验目录，例如 `logs/rsl_rl/ri_4438_him`；不递归搜索多个实验 |
| 提示无法还原历史 `latest` / 续训来源是正则 | 使用 `--runs 原始目录 续训目录 ...` 显式列出；后续新训练会保存实际来源到 `params/lineage.json` |
| 来源 run 不存在 | 恢复旧日志目录，或用 `--runs` 指定移动后的路径；仅看当前 run 用 `--single-run` |
| 找不到训练标量或数字 checkpoint | 检查该目录的事件文件与 `model_<数字>.pt`，确认日志复制完整 |
| `saved policy interface differs from current task` | 按错误中列出的差异核对当前任务与日志的观测、动作接口，使用兼容的代码和配套配置 |
| 显存不足 | 降低 `--eval-parallel` 和 `--eval-num-envs`，例如 `2` 和 `12` |
| 没有生成 HTML | 确认使用了 `--evaluate`，并等待终端显示本轮“仿真推荐”和报告路径 |
| 目录里有 `evaluation.partial.json` | 表示评测尚未完成或中途失败；成功后会移除，已有 HTML 可能仍是上次报告 |
| 仿真截图渲染失败 | 查看 `best/simulation/render.log`；指定 `--output` 时查看该输出目录下的同一路径 |
| 想查看第 11 名以后的结果 | 打开 `evaluation_ranking.csv`，完整逐场景成绩在 `evaluation_cases.csv` |

查看脚本全部参数：

```bash
uv run --extra analysis python scripts/best_model.py --help
```
