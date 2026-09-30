"""Chinese benchmark reports and separate publication of the simulation winner."""

from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .training_logs import sha256, model_key, model_label, model_origin
from .training_report import _atomic_text, recommendation, log_candidate_label
from .evaluation_scenarios import REPORT_MODEL_LIMIT, TERRAIN_LABELS, stair_sections
from .evaluation_repeatability import repeatability_notes
from .evaluation_readiness import assess_hardware_readiness, readiness_notes


STATUSES = {
    "unknown_curriculum": "无法确认课程阶段",
    "curriculum_in_progress": "课程仍在推进，暂不判定最终收敛",
    "insufficient_data": "有效数据不足，还需要继续观察",
    "improving": "指标仍在改善",
    "degrading": "近期指标退化",
    "unstable": "近期指标波动较大",
    "plateau": "进入平台期：近期训练指标基本不再改善",
    "converged_candidate": "曲线稳定且已达到所设目标，可作为收敛候选",
}

METRICS = (
    ("task_success", "任务通过率 ↑（楼梯须走完）", "%", 100),
    ("linear_rmse", "平面速度误差 ↓", "m/s", 1),
    ("yaw_rmse", "转向速度误差 ↓", "rad/s", 1),
    ("tilt_rms_deg", "机身倾斜 RMS ↓", "°", 1),
    ("action_rate_rms", "动作变化率 RMS ↓", "1/s", 1),
    ("survival_s", "首次回合存活时间 ↑", "s", 1),
)


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    candidates = ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _plots(data, output):
    plt = _plot_setup()
    from .evaluation_readiness import WEIGHTS
    ranked = data["summary"][:REPORT_MODEL_LIMIT]
    if all("score_components" in r for r in ranked):
        fig, ax = plt.subplots(figsize=(13, 6))
        left = np.zeros(len(ranked))
        for key, label in (("tracking", "基础跟踪 45%"), ("stability", "稳定性 25%"),
                           ("smoothness", "动作平滑 10%"), ("stairs", "楼梯能力 20%")):
            values = np.array([100 * WEIGHTS[key] * r["score_components"][key] for r in ranked])
            ax.barh([model_label(r) for r in ranked], values, left=left, label=label)
            left += values
        for i, value in enumerate(left):
            ax.text(value + .5, i, f"{value:.2f}", va="center")
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), xlabel="相对推荐分（不是实机成功概率）", title="推荐依据 · 固定参考尺度与权重；允许任务存在短板")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output / "recommendation.png", dpi=150)
        plt.close(fig)
    else:
        (output / "recommendation.png").unlink(missing_ok=True)
    summaries = sorted(data["summary"][:REPORT_MODEL_LIMIT], key=lambda r: (r.get("source_index", 0), r["iteration"]))
    colors = {model_key(r): plt.get_cmap("tab10")(i % 10) for i, r in enumerate(summaries)}
    labels = [f"{model_label(r)}" for r in summaries]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, title, unit, scale) in zip(axes.flat, METRICS):
        values = [r[key] * scale for r in summaries]
        bars = ax.bar(labels, values, color=[colors[model_key(r)] for r in summaries], width=0.65)
        digits = 3 if key in ("linear_rmse", "yaw_rmse") else 1 if key == "task_success" else 2
        ax.bar_label(bars, labels=[f"{v:.{digits}f}" for v in values], padding=4, fontsize=8 if len(labels) > 4 else 10)
        ax.set(title=title, ylabel=unit)
        ax.set_ylim(0, max(max(values) * 1.25, 0.01))
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=35 if len(labels) > 4 else 15, labelsize=8 if len(labels) > 4 else 10)
    fig.suptitle(f"统一仿真对比 · 本轮第一 {data.get('best_label', 'model_'+str(data['best_iteration']))}\n"
                 "分项性能与推荐取舍：任务通过率保留地形等权统计，推荐分另计跟踪、稳定性、动作平滑和楼梯", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    cases = [(terrain, c) for terrain, commands in data["scenarios"].items() for c in commands]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(cases) * 0.42)))
    for ax, key, title, scale, cmap in ((axes[0], "task_success", "任务通过率 (%) ↑", 100, "YlGn"),
                                       (axes[1], "linear_rmse", "平面速度误差 (m/s) ↓", 1, "YlOrRd")):
        matrix = np.array([[np.mean([r[key] for r in data["rows"] if model_key(r) == model_key(summary)
                           and r["terrain"] == terrain and r["command"] == c["name"]]) * scale
                           for summary in summaries] for terrain, c in cases])
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=100 if key == "task_success" else None)
        ax.set_xticks(range(len(labels)), labels, rotation=20)
        ax.set_yticks(range(len(cases)), [f"{TERRAIN_LABELS[t]} · {c['label']}" for t, c in cases])
        ax.set_title(title)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.0f}" if key == "task_success" else f"{value:.3f}", ha="center", va="center",
                    fontsize=7 if len(summaries) > 8 else 9,
                    color="white" if value > (100 if key == "task_success" else max(matrix.max(), 0.001)) * 0.65 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("分场景表现：查看模型在哪类动作、哪种地形上退步", fontsize=14)
    fig.tight_layout()
    fig.savefig(output / "scenarios.png", dpi=150)
    plt.close(fig)

    terrain = "flat" if "flat" in data["scenarios"] else next(iter(data["scenarios"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax in axes.flat[len(data['scenarios'][terrain]):]:
        ax.set_visible(False)
    for ax, command in zip(axes.flat, data["scenarios"][terrain]):
        component = "yaw" if command["name"] == "turn" else "vy" if command["name"] == "lateral" else "vx"
        index = {"vx": 0, "vy": 1, "yaw": 2}[component]
        ax.axhline(command["velocity"][index], color="black", linestyle="--", linewidth=1.4, label="目标")
        for summary in summaries:
            samples = [r for r in data["traces"] if model_key(r) == model_key(summary)
                       and r["terrain"] == terrain and r["command"] == command["name"]]
            ax.plot([r["time_s"] for r in samples], [r[component] for r in samples],
                    label=f"{model_label(summary)}", color=colors[model_key(summary)], alpha=0.85)
        ax.set(title=command["label"], xlabel="时间 (s)", ylabel=f"{component} ({'rad/s' if component == 'yaw' else 'm/s'})")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"实际速度跟踪 · {TERRAIN_LABELS[terrain]} · seed={data['options']['seeds'][0]}\n"
                 "每条线为仍在首次回合中的机器人平均值；跌倒后不再计入，不能单独据此判断好坏", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "tracking.png", dpi=150)
    plt.close(fig)
    _stability_plots(data, summaries, colors, output, plt)


def _stability_plots(data, summaries, colors, output, plt):
    labels = [model_label(r) for r in summaries]
    measures = (("roll_std_deg", "横滚波动（时间标准差）", "°"), ("pitch_std_deg", "俯仰波动（时间标准差）", "°"),
                ("tilt_p95_deg", "每回合倾角 P95 的均值", "°"), ("tilt_peak_max_deg", "所有回合的最大倾角", "°"),
                ("body_rate_rms", "机身横滚/俯仰角速度 RMS", "rad/s"), ("action_accel_rms", "动作二阶变化 RMS", "1/s²"))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title, unit) in zip(axes.flat, measures):
        bars = ax.bar(labels, [r[key] for r in summaries], color=[colors[model_key(r)] for r in summaries])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
        ax.set(title=title, ylabel=unit, xlabel="checkpoint iteration")
        ax.margins(y=0.2)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.suptitle("稳定性：姿态摆动、极端倾角与动作抖动（包含启动、台阶和失败前状态）")
    fig.tight_layout()
    fig.savefig(output / "stability.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    means = np.array([r["linear_rmse"] for r in summaries])
    lo, hi = np.array([r["seed_linear_min"] for r in summaries]), np.array([r["seed_linear_max"] for r in summaries])
    axes[0].bar(labels, means, color=[colors[model_key(r)] for r in summaries],
                yerr=np.maximum(0, np.stack([means-lo, hi-means])), capsize=4)
    axes[0].set(title="不同种子的平均速度误差范围（非置信区间）", ylabel="m/s", xlabel="checkpoint iteration")
    axes[0].tick_params(axis="x", rotation=35)
    for r in summaries:
        axes[1].scatter(r["linear_rmse"], r["action_rate_rms"], color=colors[model_key(r)], s=65)
        axes[1].annotate(model_label(r), (r["linear_rmse"], r["action_rate_rms"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set(title="速度跟踪与动作平滑性的取舍", xlabel="速度误差 (m/s) ↓", ylabel="动作变化率 (1/s) ↓")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("综合比较任务能力、误差和平滑性；相近结果不能仅凭一次排名认定显著优劣")
    fig.tight_layout()
    fig.savefig(output / "repeatability.png", dpi=150)
    plt.close(fig)

    stairs = data["stair_presets"]
    if not stairs:
        (output / "stairs.png").unlink(missing_ok=True)
        (output / "stairs_progress.png").unlink(missing_ok=True)
        return
    heights = sorted({p["height_cm"] for p in stairs.values()})
    fig, axes = plt.subplots(len(heights), 2, figsize=(14, 4.2 * len(heights)), squeeze=False)
    for row, cm in enumerate(heights):
        terrains = [t for t, p in stairs.items() if p["height_cm"] == cm]
        profile, rates_ax = axes[row]
        for i, terrain in enumerate(terrains):
            preset = stairs[terrain]
            x, z = [], []
            for left, right, height in stair_sections(9, preset["height_m"], preset["descending"]):
                x += [left, right]; z += [height, height]
            profile.plot(x, z, label=preset["label"])
            rates = [100*np.mean([r["task_success"] for r in data["rows"] if r["terrain"] == terrain and model_key(r) == model_key(s)]) for s in summaries]
            bars = rates_ax.bar(np.arange(len(summaries)) + (i-(len(terrains)-1)/2)*0.36,
                                rates, width=0.36, label=preset["label"])
            rates_ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
        profile.axvline(0, linestyle=":", color="gray", label="出生点")
        profile.axvline(data["stair_goal_m"], linestyle="--", color="black", label="通过线")
        profile.set(xlim=(-0.5, 4), ylim=(-0.04, 1.0), xlabel="前进距离 x (m)", ylabel="地面高度 (m)",
                    title=f"每级 {cm} cm · 总高差 {stairs[terrains[0]]['total_height_m']*100:.0f} cm")
        profile.legend(fontsize=8)
        rates_ax.set_xticks(range(len(labels)), labels, rotation=35, fontsize=8)
        rates_ax.set(title=f"{cm} cm 楼梯任务通过率", ylabel="%", ylim=(0, 115))
        rates_ax.legend(fontsize=8)
        rates_ax.grid(axis="y", alpha=0.2)
    fig.suptitle("固定 5 / 10 / 15 cm 楼梯：6 级、踏面 30 cm；完成后须存活至回合结束")
    fig.tight_layout()
    fig.savefig(output / "stairs.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(heights), 2, figsize=(14, 4 * len(heights)), squeeze=False, sharey=True)
    for row, cm in enumerate(heights):
        for col, down in enumerate((False, True)):
            ax = axes[row, col]
            terrain = next((t for t, p in stairs.items() if p["height_cm"] == cm and p["descending"] == down), None)
            if terrain is None:
                ax.set_visible(False)
                continue
            for r in summaries:
                samples = [t for t in data["traces"] if t["terrain"] == terrain and model_key(t) == model_key(r) and t["command"] == "forward"]
                ax.plot([t["time_s"] for t in samples], [t["progress"] for t in samples], color=colors[model_key(r)], label=model_label(r))
            ax.axhline(data["stair_goal_m"], color="black", linestyle="--", label="通过线")
            ax.set(title=f"{stairs[terrain]['label']} · 0.5 m/s · 首个种子", xlabel="时间 (s)", ylabel="仍存活机器人的平均位置 x (m)")
            ax.legend(fontsize=7, ncol=3)
            ax.grid(alpha=0.2)
    fig.suptitle("各高度行进进度：停在第一阶、途中停住和到达终点分别可见\n曲线只反映当时仍存活的机器人，通过率以上图全部回合为准")
    fig.tight_layout()
    fig.savefig(output / "stairs_progress.png", dpi=150)
    plt.close(fig)


def write_evaluation_report(result, data, output: Path, plot=True):
    selected = recommendation(result, data)
    hardware = assess_hardware_readiness(data)
    if hardware["hardware_recommendation"] and hardware["hardware_recommendation"]["model_id"] != selected["model_id"]:
        raise ValueError("report ranking is stale; apply the recommendation ranking before publication")
    hardware_notes = readiness_notes(hardware)
    _atomic_text(output / "hardware_readiness.json", lambda f: json.dump(hardware, f, indent=2, ensure_ascii=False, allow_nan=False))
    displayed = data["summary"][:REPORT_MODEL_LIMIT]
    displayed_ids = {model_key(r) for r in displayed}
    images = [("recommendation.png", "推荐分项与取舍"), ("comparison.png", "性能总览"), ("scenarios.png", "分场景通过率与误差"),
              ("stability.png", "姿态与动作稳定性"), ("repeatability.png", "种子差异与性能取舍"),
              ("stairs.png", "5 / 10 / 15 cm 楼梯几何与通过率"), ("stairs_progress.png", "各高度上下楼梯行进进度"),
              ("tracking.png", "目标与实际速度"), ("history.png", "完整训练与续训曲线"), ("report.png", "最新 run 训练趋势")]
    if plot:
        _plots(data, output)
    else:
        for name, _ in images:
            if name not in ("report.png", "history.png"):
                (output / name).unlink(missing_ok=True)

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=list(data["rows"][0]))
        writer.writeheader()
        writer.writerows(data["rows"])
    _atomic_text(output / "evaluation.csv", write_csv)
    winner = data["summary"][0]
    timestamp = datetime.now(timezone.utc).isoformat()
    status = STATUSES.get(result["convergence"]["status"], result["convergence"]["status"])
    reliability_notes = repeatability_notes(data)
    command_labels = {(t, c["name"]): c["label"] for t, commands in data["scenarios"].items() for c in commands}
    heading = ["排名 / 模型", "相对推荐分", "任务通过", "存活至结束", "速度误差 m/s", "转向误差 rad/s", "倾角 P95 °", "动作变化率 1/s", "最差场景通过"]
    table = []
    for rank, r in enumerate(displayed, 1):
        table.append([f"{rank} / {model_label(r)}", f"{r['deployment_score']:.2f}" if 'deployment_score' in r else "未计算", f"{r['task_success']:.1%}", f"{r['success']:.1%}",
                      f"{r['linear_rmse']:.3f}", f"{r['yaw_rmse']:.3f}", f"{r['tilt_p95_deg']:.2f}",
                      f"{r['action_rate_rms']:.2f}", f"{r['worst_case_success']:.1%}"])
    baseline = min(displayed, key=lambda r: (r.get("source_index", 0), r["iteration"]))
    worst = min(winner["cases"], key=lambda c: (c["task_success"], -c["linear_rmse"]))
    observations = [
        f"本次评测 {data['selection']['selected']}/{data['selection']['available']} 个有效 checkpoint，{'已覆盖全部' if data['selection']['all_tested'] else '仅对已测候选排名'}；本轮第一名为 {model_label(winner)}，唯一最优尚未确认。",
        f"本页表格、图表、场景筛选和单模型诊断仅展示排名前 {len(displayed)} 个，全部 {len(data['summary'])} 个模型的成绩保存在 CSV / JSON。",
        f"推荐模型最弱场景：{TERRAIN_LABELS[worst['terrain']]} / {command_labels[(worst['terrain'], worst['command'])]}，任务通过 {worst['task_success']:.1%}、速度误差 {worst['linear_rmse']:.3f} m/s。",
        f"推荐模型的各种子平均速度误差为 {winner['seed_linear_min']:.3f}～{winner['seed_linear_max']:.3f} m/s；这是重复试验的范围，不是统计置信区间。",
        f"推荐模型所有回合最大倾角 {winner['tilt_peak_max_deg']:.1f}°；超过 {data['options']['tilt_limit']:.0f}° 的时间占比平均 {winner['tilt_exceed_fraction']:.2%}；该阈值仅用于诊断，不改变跌倒终止条件。",
    ]
    if data.get("ranking_revision"):
        observations.insert(0, "本次复用已有仿真记录，按更新后的相对推荐规则重新排序；没有重新运行物理仿真。旧规则的推荐与分数保存在 evaluation.before_relative_ranking.json。")
    if model_key(baseline) != model_key(winner):
        observations.append(f"相对本页展示模型中最早的 {model_label(baseline)}：任务通过率变化 {(winner['task_success']-baseline['task_success'])*100:+.1f} 个百分点；速度误差变化 {winner['linear_rmse']-baseline['linear_rmse']:+.4f} m/s；动作变化率变化 {winner['action_rate_rms']-baseline['action_rate_rms']:+.2f} 1/s。")
    if winner["task_success"] < 0.95:
        observations.append("当前推荐模型仍有未通过场景，best 只代表本次排序靠前，不能视为能力已经达标。")
    if "deployment_score" in winner:
        near = [r for r in displayed[1:] if winner["deployment_score"] - r["deployment_score"] <= 1.0]
        if near:
            observations.append("接近推荐模型的备选（推荐分相差不超过 1 分）：" + "、".join(model_label(r) for r in near) + "。这是提示复核的启发式，不代表统计等效。")
    if data["options"]["duration"] < 10 or len(data["options"]["seeds"]) < 3:
        observations.insert(0, "本次是短时或少种子评测，仅用于快速筛选；楼梯可能来不及走完。正式比较建议至少 3 个种子、每回合 12 秒。")
    terrain_text = "、".join(TERRAIN_LABELS[t] for t in data["scenarios"])
    notes = [
        f"训练统计截止 iteration={result['as_of_iteration']}，日志收敛分析分段={result['segment']}。仿真默认评测已解析训练链的全部有效数值 checkpoint，含各 run 和早期日志分段；全部采用相同固定条件。新保存的 checkpoint 不会在本轮中途加入，再次运行即可更新。",
        f"每模型 {winner['episodes']} 个首次回合，种子 {data['options']['seeds']}，每回合最长 {data['steps']*data['dt']:.2f} 秒；{terrain_text}。每批 {data['parallel_models']} 个模型、总共 {data['total_parallel_envs']} 个独立物理环境；物理步进批量并行，相同结构与配置的网络使用并行推理，不同网络则分别推理后统一步进。",
        f"平地/起伏测试站立、前进、后退、侧移和转向；上下楼梯仅测试向前 0.3/0.5/0.8 m/s。每个地形场景（楼梯按高度和方向区分）样本数相同、权重相同，本轮各占 1/{len(data['scenarios'])}，场景内各指令等权。新增高度改变了评测范围，总分不能直接与旧版单一高度报告比较。",
        "平地/起伏的任务通过仅要求没有失败并跑满时长，尚未设置速度精度门槛；跟踪误差另列，不能把 100% 存活理解为速度已达标。",
        f"楼梯每级高度固定为 5、10、15 cm，每个方向均测试全部三个高度，不接受高度参数。每段 6 级，总高差分别为 30、60、90 cm，踏面 30 cm；第一台阶在出生点前 1 m。机器人必须越过 x={data['stair_goal_m']:.2f} m 的通过线、达到相应高度（容差 12 cm），且存活到回合结束。绕出 ±1.2 m 通道会终止。单纯站住不算通过。",
        "完整评测按相对推荐分排序：基础跟踪 45%、稳定性 25%、动作平滑 10%、楼梯能力 20%。先用固定参考尺度归一化，再加权；权重是明确的工程取舍，不是实机成功概率或安全门槛。缺少基础行走/楼梯等可比指标时保留原始任务通过率排序并标明未计算推荐分。",
        "各模型使用相同的局部测试编号/种子生成初始姿态和关节状态，并校验哈希。随机初始状态与并行模型数量、模型顺序无关。GPU 接触求解不保证逐位复现，相近成绩需要更多种子复核。",
        "速度误差是 sqrt(mean((vx−目标vx)²+(vy−目标vy)²))；姿态标准差反映摆动，倾角 P95/峰值反映极端状态。所有指标包含启动和失败前状态，按首次回合计算后平均；重置后的片段不参与统计。早期失败的低抖动不能单独解释为稳定。",
        "动作变化率和二阶变化分别是策略输出的一阶、二阶差分 RMS。机械功率代理是 sum(abs(关节执行器力矩×关节速度))，不是电池功耗。各指标和逐回合数据分别展示。",
        "使用当前工作区的统一机器人、终止规则和物理参数；保存的网络、观测/动作接口和控制周期严格匹配。课程、参数随机化、推扰、观测噪声及观测/执行器延迟关闭；尚未测试抗推扰或真实硬件鲁棒性。",
        "训练收敛状态仍根据训练日志判断；这里的多模型性能趋势和稳定性是辅助证据，不会自动把一次仿真通过认定为训练收敛。平台期只表示近期曲线不再明显改善。",
        "model_best_eval.pt 是本轮仿真推荐模型。model_best.pt / policy.onnx（若生成）对应独立的日志评分候选，用于训练曲线分析。HTML、Markdown 和图表只展示前 10 名，evaluation_ranking.csv / evaluation_cases.csv / evaluation.csv / evaluation.json 保留全部模型。",
        "末尾的仿真图片来自排名前 3 名在本次计分回合中的真实状态，并非重新运行或挑选成功回合。统一取首个种子、前进 0.5 m/s、首个对应环境，在 2 s / 6 s / 回合结束采样；提前失败则保留首次终止画面，终止后不再截图。单个回合图片不能代表全部种子的通过率。",
    ]
    failures = []
    reason_labels = {"fell_over": "倾倒", "illegal_contact": "机身触地", "left_stair_lane": "偏离楼梯通道", "stairs_not_completed": "存活但未走完楼梯"}
    for r in displayed:
        counts = Counter(row["failure_reason"].split(";")[0] for row in data["rows"] if model_key(row) == model_key(r) and row["failure_reason"])
        failures.append([f"{model_label(r)}", str(r["failures"]),
                         "；".join(f"{reason_labels.get(k,k)} {v}" for k, v in counts.items()) or "无失败/未完成记录",
                         f"{r['power_abs_w']:.2f}", f"{r['torque_rms']:.2f}", f"{r['action_clip_fraction']:.2%}"])

    def html_table(headers, rows, ident=""):
        return f'<div class="scroll"><table id="{ident}"><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(c))}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'

    hardware_headers = ["地形", "推荐模型任务通过", "提前终止", "存活但未完成", "全部已测模型中的最高通过率"]
    hardware_rows = []
    if hardware["models"]:
        for terrain, limit in zip(hardware["models"][0]["terrains"], hardware["terrain_limits"]):
            total = terrain["episodes"]
            rate = limit["best_observed_pass_rate"]
            hardware_rows.append([terrain["label"], f"{terrain['passed']}/{total}" if total else "未测试",
                                  str(terrain["terminated"]), str(terrain["survived_but_incomplete"]),
                                  f"{rate:.1%}" if rate is not None else "未测试"])
    hardware_html = ('<section id="hardware-readiness"><h2>实机优先模型：相对推荐与能力边界</h2><ul>'
                     + ''.join('<li>' + escape(n) + '</li>' for n in hardware_notes) + '</ul>'
                     + html_table(hardware_headers, hardware_rows, "hardware-terrain-results")
                     + '<details><summary>评分公式、参考尺度与验证范围</summary><p>基础跟踪：平地/起伏各回合的存活标记 × max(0, 1−线速度误差/max(指令平移速度, 0.3 m/s)) × max(0, 1−转向误差/0.5 rad/s)，再取平均。稳定性：全场景存活率与平地/起伏姿态质量各占一半。姿态质量为 1/(1+(倾角 RMS/20°)²)。动作平滑：1/(1+(动作变化率/10)²) 和 1/(1+(动作二阶变化/400)²) 各占一半，取平地/起伏均值。楼梯能力为楼梯回合完成率。四项按 45/25/10/20 加权后乘 100。尺度为比较用参考值，不是硬件限制；分数与候选数量无关。</p><ul>'
                     + ''.join('<li>' + escape(n) + '</li>' for n in hardware["evidence_gaps"])
                     + '</ul></details><p>模型可以带有能力短板而被推荐。推荐表示当前取舍下优先选择，并不表示已经测得实机表现。'
                     '<a href="hardware_readiness.json">全部模型的评分分项、种子分数和权重对照</a></p></section>')

    repeat_headers = ["模型（本轮前 10 名）", "上次排名 → 本轮排名", "上次通过率", "本轮通过率", "变化（百分点）"]
    repeat_rows = [[f"{model_label(r)}", f"{r['previous_rank']} → {r['current_rank']}",
                    f"{r['previous_success']:.2%}", f"{r['current_success']:.2%}", f"{r['delta_percentage_points']:+.2f}"]
                   for r in data.get("repeatability", {}).get("models", [])[:REPORT_MODEL_LIMIT]]

    stairs = sorted(data["stair_presets"], key=lambda t: (data["stair_presets"][t]["height_cm"], data["stair_presets"][t]["descending"]))
    stair_headers = ["模型"] + [data["stair_presets"][t]["label"] for t in stairs]
    stair_rows = []
    for r in displayed:
        cells = [f"{model_label(r)}"]
        for terrain in stairs:
            episodes = [e for e in data["rows"] if model_key(e) == model_key(r) and e["terrain"] == terrain]
            passed = sum(e["task_success"] for e in episodes)
            cells.append(f"{passed}/{len(episodes)} ({passed/len(episodes):.1%})")
        stair_rows.append(cells)
    stair_description = ("每级高度固定为 5 / 10 / 15 cm，各 6 级，总高差分别为 30 / 60 / 90 cm，踏面均为 30 cm。"
                         "表中为通过回合 / 总回合（通过率），合并 0.3 / 0.5 / 0.8 m/s 和全部种子；"
                         "下方场景筛选可进一步比较各速度的跟踪误差、倾角和动作抖动。")
    stair_html = ('<section id="stairs-section"><h2>固定高度楼梯对比</h2><p>' + stair_description + '</p>'
                  + html_table(stair_headers, stair_rows, "stairs-ranking") + '</section>') if stairs else ''

    details = []
    for r in displayed:
        model = next(m for m in data["models"] if model_key(m) == model_key(r))
        worst_case = min(r["cases"], key=lambda c: (c["task_success"], -c["linear_rmse"]))
        details.append(f'<details><summary>{escape(model_label(r))} · {escape(" / ".join(model["reasons"]))} · 任务通过 {r["task_success"]:.1%}</summary>'
                       f'<p>速度误差均值 {r["linear_rmse"]:.3f} m/s，回合 P90 {r["linear_rmse_p90"]:.3f} m/s；不同种子范围 {r["seed_linear_min"]:.3f}～{r["seed_linear_max"]:.3f} m/s。'
                       f'横滚/俯仰时间标准差均值 {r["roll_std_deg"]:.2f}° / {r["pitch_std_deg"]:.2f}°；倾角峰值 {r["tilt_peak_max_deg"]:.1f}°。</p>'
                       f'<p>最弱场景：{TERRAIN_LABELS[worst_case["terrain"]]} / {escape(command_labels[(worst_case["terrain"],worst_case["command"])])}；通过 {worst_case["task_success"]:.1%}。</p>'
                       f'<small>SHA256: {escape(model["sha256"])}</small></details>')
    case_rows = [{"iteration": r["iteration"], **model_origin(r), **c, "terrain_label": TERRAIN_LABELS[c["terrain"]],
                  "command_label": command_labels[(c["terrain"],c["command"])]} for r in data["summary"] for c in r["cases"]]
    def write_cases(stream):
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0])); writer.writeheader(); writer.writerows(case_rows)
    _atomic_text(output / "evaluation_cases.csv", write_cases)
    def write_ranking(stream):
        rows = [{"rank": rank, **{k: v for k, v in r.items() if k not in ("cases", "seeds")}}
                for rank, r in enumerate(data["summary"], 1)]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    _atomic_text(output / "evaluation_ranking.csv", write_ranking)
    image_html = ''.join(f'<section><h2>{escape(label)}</h2><a href="{name}"><img loading="lazy" src="{name}" alt="{escape(label)}"></a></section>' for name, label in images if name not in ("history.png", "report.png") and (output/name).exists())
    model_options = ''.join(f'<option value="{model_key(r)}">{escape(model_label(r))}</option>' for r in displayed)
    terrain_options = ''.join(f'<option value="{t}">{TERRAIN_LABELS[t]}</option>' for t in data["scenarios"])
    case_json = json.dumps([r for r in case_rows if model_key(r) in displayed_ids], ensure_ascii=False, allow_nan=False).replace('<', r'\u003c')
    gallery = data.get("simulation_images", [])
    gallery_html = ''.join(f'<h3>{escape(item["label"])}</h3><a href="{escape(item["image"], quote=True)}">'
                          f'<img loading="lazy" src="{escape(item["image"], quote=True)}" alt="{escape(item["label"])}真实仿真截图"></a>' for item in gallery)
    if gallery:
        gallery_html = ('<section id="simulation-section"><h2>真实仿真截图 · 排名前 3 名</h2>'
                        '<p>来自本轮实际计分状态：同一首个种子、前进 0.5 m/s、同一局部环境编号。'
                        '记录 2 秒、6 秒和回合结束；提前失败则展示首次终止状态。点击图片可放大。</p>'
                        + gallery_html + '</section>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RI-4438 多模型性能与稳定性报告</title><style>
body{font-family:system-ui,"Noto Sans CJK SC",sans-serif;background:#f3f6fa;color:#182d42;max-width:1250px;margin:30px auto;padding:0 20px;line-height:1.7}
section,.card{background:white;border:1px solid #dce5ef;border-radius:12px;padding:22px;margin:20px 0}h1{margin-bottom:0}h2{font-size:21px}
.cards{display:flex;gap:14px;flex-wrap:wrap}.card{flex:1;min-width:160px;margin:12px 0}.card strong{display:block;font-size:25px;color:#006c76}
table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:10px;border-bottom:1px solid #e0e7ef}th{background:#edf4f8}#ranking tbody tr:first-child{background:#e5f5ef}
img{width:100%;height:auto}small{color:#5a6c7e}li{margin:9px 0}a{color:#006c76}.scroll{overflow:auto;max-height:680px}details{padding:14px;border-bottom:1px solid #e0e7ef}summary{cursor:pointer}select{padding:8px;margin:0 12px 10px 4px}
nav{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}.warn{color:#a64713}button{padding:7px;margin:8px;cursor:pointer}
</style><h1>RI-4438 多模型性能与稳定性报告</h1><small>__RUN__ · __TIME__</small>
<nav><a href="#hardware-readiness">推荐依据与能力边界</a><a href="#recommendation">推荐模型</a><a href="#findings">结果解读</a><a href="#repeatability-section">复跑核对</a><a href="#rank-section">前 10 名</a>__STAIRS_NAV__<a href="#case-section">场景筛选</a><a href="#training-history">完整训练曲线</a><a href="#details">单模型诊断</a><a href="#protocol">测试条件</a>__GALLERY_NAV__</nav>
<div class="cards"><div class="card">本轮推荐 · 实机表现未验证<strong>__BEST__</strong></div><div class="card">评测覆盖<strong>__COVERAGE__</strong></div><div class="card">任务通过<strong>__PASS__</strong></div><div class="card">模型并行数<strong>__PARALLEL__</strong></div></div>
__HARDWARE__
<section id="recommendation"><h2>本轮推荐模型</h2><p><b>本轮推荐：__BEST__</b> · <a href="model_best_eval.pt">下载推荐模型 model_best_eval.pt</a> · <a href="eval_model_manifest.json">模型来源与 SHA256</a></p><p>用途：按当前表现优先进行实机验证。完整评测综合基础跟踪、稳定性、动作平滑和楼梯能力；允许能力存在短板，实机表现尚未验证。</p><p><b>日志评分候选（训练曲线参考）：__LOG_CANDIDATE__</b></p><p>日志候选依据最新 run 当前课程阶段的训练回报窗口评分，用于观察训练趋势。它与仿真评测的条件和排序指标不同，因此可能是另一个模型。</p></section>
__CHAIN__
<section id="findings"><h2>本次结果怎么读</h2><ul>__FINDINGS__</ul><p><b>训练状态：</b>__STATUS__</p></section>
<section id="repeatability-section"><h2>重复运行后，第一名可靠吗？</h2><ul>__RELIABILITY__</ul>__REPEAT_TABLE____PREVIOUS_LINK__<p>本工具未启用能保证逐位一致的物理求解后端；仅设置相同随机种子不能消除数值波动。model_best_eval.pt 保存本轮候选，不代表已证明它优于所有其他模型。</p></section>
<section id="rank-section"><h2>模型排名 · 前 10 名</h2><p>全部已测模型参与相对推荐，此处仅显示前 10 名。推荐分兼顾跟踪、稳定性、动作平滑及楼梯能力，不要求全部任务通过；公式和权重见页首。未计算推荐分时仅保留任务通过率排序。<a href="evaluation_ranking.csv">下载全部模型排名</a></p>__RANK__</section>
__STAIRS__
<section id="case-section"><h2>前 10 名：按模型和地形查看表现</h2><label>模型<select id="model-filter"><option value="">全部展示模型</option>__MODELS__</select></label><label>地形<select id="terrain-filter"><option value="">全部地形</option>__TERRAINS__</select></label>
<label>排序<select id="sort-filter"><option value="task_success">任务通过率从低到高（定位弱项）</option><option value="linear_rmse">速度误差从高到低</option><option value="tilt_p95_deg">倾角 P95 从高到低</option><option value="action_rate_rms">动作变化率从高到低</option></select></label>
<p id="case-count"></p><div class="scroll"><table><thead><tr><th>模型</th><th>地形</th><th>指令</th><th>回合数</th><th>任务通过</th><th>存活至结束</th><th>速度误差 m/s</th><th>倾角 P95 °</th><th>动作变化率 1/s</th></tr></thead><tbody id="case-rows"></tbody></table></div><noscript>筛选需要浏览器启用 JavaScript；也可下载下方场景 CSV。</noscript></section>
__IMAGES__
__TRAINING_FIGURES__
<section id="details"><h2>单模型诊断与选入原因</h2>__DETAILS__</section>
<section><h2>失败类型、力矩和机械功率代理</h2><p>同一回合触发多个终止条件时，此表按首个原因计数，完整原因见 CSV。未走完楼梯与跌倒分开记录。</p>__FAILURES__</section>
<section id="protocol"><h2>评测协议与指标含义</h2><ul>__NOTES__</ul><p><a href="evaluation_ranking.csv">全部模型排名 CSV</a> · <a href="evaluation.csv">逐回合 CSV</a> · <a href="evaluation_cases.csv">场景汇总 CSV</a> · <a href="evaluation.json">完整 JSON</a> · <a href="report.md">Markdown</a></p></section>
__GALLERY__
<script id="case-data" type="application/json">__DATA__</script><script>
const cases=JSON.parse(document.getElementById('case-data').textContent);
function renderCases(){const m=document.getElementById('model-filter').value,t=document.getElementById('terrain-filter').value,k=document.getElementById('sort-filter').value;
const selected=cases.filter(r=>(!m||String(r.model_id ?? r.iteration)===m)&&(!t||r.terrain===t)).sort((a,b)=>k==='task_success'?a[k]-b[k]:b[k]-a[k]);
const body=document.getElementById('case-rows');body.replaceChildren();document.getElementById('case-count').textContent=`显示 ${selected.length} 个模型/场景组合`;
for(const r of selected){const tr=document.createElement('tr');for(const v of [r.label || 'model_'+r.iteration,r.terrain_label,r.command_label,r.episodes,(100*r.task_success).toFixed(1)+'%',(100*r.success).toFixed(1)+'%',r.linear_rmse.toFixed(3),r.tilt_p95_deg.toFixed(2),r.action_rate_rms.toFixed(2)]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}body.appendChild(tr);}}
for(const id of ['model-filter','terrain-filter','sort-filter'])document.getElementById(id).addEventListener('change',renderCases);renderCases();
</script></html>'''
    from .training_chain_report import chain_html, training_figures_html, remove_legacy_training_page
    replacements = {"CHAIN": chain_html(result), "HARDWARE": hardware_html, "RUN": escape(data["run"]), "TIME": timestamp, "BEST": escape(selected["label"]),
        "LOG_CANDIDATE": escape(log_candidate_label(result)), "TRAINING_FIGURES": training_figures_html(output),
        "COVERAGE": f"{data['selection']['selected']} / {data['selection']['available']}", "PASS": f"{winner['task_success']:.1%}",
        "PARALLEL": str(data["parallel_models"]), "FINDINGS": ''.join('<li>'+escape(n)+'</li>' for n in observations),
        "RELIABILITY": ''.join('<li>'+escape(n)+'</li>' for n in reliability_notes),
        "REPEAT_TABLE": html_table(repeat_headers, repeat_rows, "repeat-ranking") if repeat_rows else '',
        "PREVIOUS_LINK": '<p><a href="evaluation.previous.json">上次完成的数字结果</a></p>' if (output / "evaluation.previous.json").exists() else '',
        "STATUS": escape(status), "RANK": html_table(heading, table, "ranking"), "MODELS": model_options,
        "TERRAINS": terrain_options, "IMAGES": image_html, "DETAILS": ''.join(details),
        "STAIRS": stair_html, "STAIRS_NAV": '<a href="#stairs-section">5 / 10 / 15 cm 楼梯</a>' if stairs else '',
        "GALLERY": gallery_html, "GALLERY_NAV": '<a href="#simulation-section">仿真截图</a>' if gallery else '',
        "FAILURES": html_table(["模型", "失败回合", "失败/未完成原因", "机械功率代理 W", "力矩 RMS Nm", "动作裁剪比例"], failures),
        "NOTES": ''.join('<li>'+escape(n)+'</li>' for n in notes), "DATA": case_json}
    for key, value in replacements.items():
        html = html.replace('__'+key+'__', value)
    _atomic_text(output / "report.html", lambda f: f.write(html))
    markdown = [f"# RI-4438 多模型评测：本轮推荐 {selected['label']}", "",
                "## 实机优先推荐与能力边界", "", *['- '+n for n in hardware_notes], "",
                '| '+' | '.join(hardware_headers)+' |', '| '+' | '.join(['---']*len(hardware_headers))+' |',
                *['| '+' | '.join(row)+' |' for row in hardware_rows], "",
                *['- '+n for n in hardware['evidence_gaps']], "",
                "全部模型的评分依据见 [hardware_readiness.json](hardware_readiness.json)。", "",
                f"本轮推荐：**{selected['label']}**；模型文件：[model_best_eval.pt](model_best_eval.pt)。",
                f"日志评分候选（训练曲线参考）：**{log_candidate_label(result)}**。日志回报与仿真性能的排序依据不同。",
                "", f"训练状态：{status}", "", *['- '+n for n in observations], "",
                "## 复跑核对与第一名的可靠性", "", *['- '+n for n in reliability_notes]]
    if result.get("chain"):
        markdown += ["", "完整训练来源和各段判断见 [training_chain.json](training_chain.json)。"]
    if repeat_rows:
        markdown += ["", '| '+' | '.join(repeat_headers)+' |', '| '+' | '.join(['---']*len(repeat_headers))+' |']
        markdown += ['| '+' | '.join(row)+' |' for row in repeat_rows]
    markdown += ["", "## 本轮排名", "", '| '+' | '.join(heading)+' |', '| '+' | '.join(['---']*len(heading))+' |']
    markdown += ['| '+' | '.join(row)+' |' for row in table]
    if stairs:
        markdown += ["", "## 固定高度楼梯对比", "", stair_description, "",
                     '| '+' | '.join(stair_headers)+' |', '| '+' | '.join(['---']*len(stair_headers))+' |']
        markdown += ['| '+' | '.join(row)+' |' for row in stair_rows]
    markdown += ["", "## 评测协议", ""] + ['- '+n for n in notes]
    for name, label in images:
        if (output/name).exists():
            markdown += ["", f"## {label}", "", f"![{label}]({name})"]
    if gallery:
        markdown += ["", "## 真实仿真截图 · 排名前 3 名", "", "来自本轮计分回合：首个种子，前进 0.5 m/s，同一局部环境。提前失败则保留终止画面。"]
        for item in gallery:
            markdown += ["", f"### {item['label']}", "", f"![{item['label']}]({item['image']})"]
    _atomic_text(output / "report.md", lambda f: f.write('\n'.join(markdown)+'\n'))
    remove_legacy_training_page(output)
    # These two filenames belonged to the previous single-height protocol.
    # Remove only those obsolete generated descriptions after the report succeeds.
    for terrain in ("stairs_up", "stairs_down"):
        (output / f"eval_env_{terrain}.json").unlink(missing_ok=True)


def publish_evaluation_best(run, data, output: Path):
    model = next(m for m in data["models"] if model_key(m) == data.get("best_model_id", data["best_iteration"]))
    source = Path(model["path"])
    with tempfile.TemporaryDirectory(prefix=".eval-publish-", dir=output) as staging:
        copied = Path(staging) / "model_best_eval.pt"
        shutil.copyfile(source, copied)
        if sha256(copied) != model["sha256"] or sha256(source) != model["sha256"]:
            raise ValueError("evaluated checkpoint changed; simulation winner was not published")
        os.replace(copied, output / "model_best_eval.pt")
    manifest = {**model, "model": str(output / "model_best_eval.pt"), "selection_policy": data["ranking_policy"],
                "unique_best_established": False,
                "intended_use": "hardware_trial_priority" if data.get("hardware_assessment", {}).get("hardware_recommendation") else "simulation_screening",
                "hardware_validated": False, "hardware_recommendation": data.get("hardware_assessment", {}).get("hardware_recommendation"),
                "repeatability_status": data.get("repeatability", {}).get("status", "not_checked"),
                "saved_config_sha256": run.fingerprints, "evaluation": "evaluation.json",
                "generated_at": datetime.now(timezone.utc).isoformat()}
    _atomic_text(output / "eval_model_manifest.json", lambda f: json.dump(manifest, f, indent=2, ensure_ascii=False))
