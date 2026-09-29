"""Complete run history without smoothing across restarts or changed rewards."""

from html import escape
import csv
import json
from pathlib import Path


def chain_html(result):
    from .evaluation_report import STATUSES
    chain = result.get("chain", [])
    if not chain:
        return ""
    rows = []
    sources = {r["path"]: r["label"] for r in chain}
    for run in chain:
        statuses = "；".join(f"段 {s['segment']}: {STATUSES.get(s['convergence']['status'], s['convergence']['status'])}" for s in run["segments"])
        parent = Path(run["parent_checkpoint"]) if run["parent_checkpoint"] else None
        parent_label = f"{sources.get(str(parent.parent), parent.parent.name)} / {parent.name}" if parent else "—"
        cells = [run["label"], Path(run["path"]).name, parent_label,
                 f"{run['first_step']} → {run['last_step']}", str(run["valid_checkpoints"]),
                 "环境配置改变" if run["config_changed"] else "—", statuses]
        rows.append('<tr>' + ''.join('<td title="' + escape(run["parent_checkpoint"] or run["path"], quote=True) + '">' + escape(c) + '</td>' for c in cells) + '</tr>')
    table = '<div class="scroll"><table id="chain-table"><thead><tr>' + ''.join('<th>'+h+'</th>' for h in
            ("来源", "run 目录", "续训起点", "原始迭代范围", "有效模型", "配置变化", "各段训练状态")) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    warnings = ''.join('<li>'+escape(w)+'</li>' for w in result.get("chain_warnings", []))
    evidence = ''.join(f'<li>{r["label"]}: {escape(r["link_evidence"])}</li>' for r in chain)
    return (f'<section id="chain-section"><style>#chain-table{{white-space:normal}}#chain-table td{{min-width:45px}}#chain-table td:last-child{{min-width:190px}}</style><h2>完整训练与续训来源 · {len(chain)} 个 run</h2>'
            '<p>R1、R2…按训练来源顺序标识。同名 model_500.pt 分开评测，模型表中的 R 编号对应下表。所有目录中的有效模型均参与默认仿真，包括恢复点之后仍保存的旧分支模型。</p>'
            + table + '<p>完整曲线保留每个 run 的原始 iteration，恢复、回退与配置变化处分别画线，不跨断点平滑。旧分支的曲线保留显示，不冒充新分支的训练经历。各段收敛按各自配置独立判断；页面主训练状态来自最新 run 的所选分段，不把不同奖励条件的数值直接合并评分。</p>'
            '<p>所有模型在当前代码和最新 run 的控制接口下统一仿真；每个模型用自己的 agent 配置加载。接口或控制步长不兼容时明确报错，不悄悄排除模型。</p>'
            '<details><summary>续训关系依据与日志缺失提示</summary><ul>' + evidence + warnings + '</ul></details>'
            '<p><a href="training_chain.json">全部 run、配置和各段分析</a> · <a href="chain_candidates.csv">所有 run 的日志候选</a> · <a href="training_history.html">单独查看完整训练曲线</a></p></section>')


def write_chain_reports(runs, result, output, plot=True):
    from .training_report import _atomic_text
    from .evaluation_report import _plot_setup
    from .best_model import REWARD, LENGTH, LINEAR, ANGULAR, TERRAIN
    import numpy as np
    output = Path(output)
    _atomic_text(output / "training_chain.json", lambda f: json.dump(result["chain"], f, ensure_ascii=False, indent=2, allow_nan=False))
    def candidates(stream):
        fields = ["run", "source_run", "segment", "iteration", "path", "valid", "eligible", "stage", "score", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for origin in result["chain"]:
            for segment in origin["segments"]:
                for row in segment["candidates"]:
                    if row["segment"] != segment["segment"] and not (row["segment"] is None and segment is origin["segments"][0]):
                        continue
                    writer.writerow({**{key: row.get(key) for key in fields}, "run": origin["label"], "source_run": origin["path"]})
    _atomic_text(output / "chain_candidates.csv", candidates)
    if plot:
        plt = _plot_setup()
        figure, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        tags = (REWARD, LENGTH, LINEAR, ANGULAR, TERRAIN, "Episode_Reward/smoothness")
        labels = ("训练回报（配置改变时不直接比较数值）", "平均回合长度", "线速度跟踪奖励", "转向跟踪奖励", "地形课程等级", "动作平滑奖励")
        for index, (run, origin) in enumerate(zip(runs, result["chain"])):
            color = plt.get_cmap("tab10")(index % 10)
            for segment in run.segments:
                for ax, tag, title in zip(axes.flat, tags, labels):
                    points = sorted(segment.series.get(tag, {}).values(), key=lambda p: p.step)
                    if not points:
                        continue
                    x, y = [p.step for p in points], [p.value for p in points]
                    ax.plot(x, y, alpha=.15, linewidth=.5, color=color)
                    ax.plot(x, [np.median(y[max(0, i-49):i+1]) for i in range(len(y))], color=color,
                            linewidth=1.2, label=f"R{index+1} / 段 {segment.index}")
                    ax.set_title(title, fontsize=10)
            if origin["resume_iteration"] is not None:
                for ax in axes.flat:
                    ax.axvline(origin["resume_iteration"], linestyle="--", color=color, alpha=.55)
                axes.flat[0].annotate(f"R{index+1} 恢复点", (origin["resume_iteration"], .98-index*.07),
                                      xycoords=("data", "axes fraction"), fontsize=8, color=color)
        for ax in axes.flat:
            ax.grid(alpha=.2)
        for ax in axes[-1]:
            ax.set_xlabel("训练 iteration（保留各 run 原始编号，回退时曲线可能重叠）")
        axes.flat[0].legend(fontsize=8)
        figure.suptitle(f"完整训练记录 · {len(runs)} 个 run · 每条曲线单独平滑，虚线标记续训来源 checkpoint")
        figure.tight_layout()
        figure.savefig(output / "history.png", dpi=150)
        plt.close(figure)
    else:
        (output / "history.png").unlink(missing_ok=True)
    picture = '<img src="history.png" alt="完整训练曲线">' if plot else '<p>本次禁用了绘图；逐段分析请下载 JSON。</p>'
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整训练与续训</title><style>'
            'body{font-family:sans-serif;margin:35px;color:#19324b}table{border-collapse:collapse;font-size:13px}td,th{padding:10px;border-bottom:1px solid #ccd}img{width:100%}.scroll{overflow:auto}a{color:#006c76}</style>'
            '<h1>完整训练与续训记录</h1>'+chain_html(result)+picture+'</html>')
    _atomic_text(output / "training_history.html", lambda f: f.write(html))
