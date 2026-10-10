"""First-trial returns and physical errors, without value bootstrapping."""
import numpy as np


def summarize_trials(rows, catalog, trials_per_task):
    if len(rows) != catalog.num_tasks * trials_per_task:
        raise ValueError("Missing evaluation trials")
    grouped = [[] for _ in catalog.records]
    seen = set()
    for row in rows:
        key = (row["task_key"], row["trial_id"])
        if key in seen or catalog.decode(row["task_id"]).key != row["task_key"]:
            raise ValueError("Duplicate/misattributed evaluation trial")
        seen.add(key)
        if row["steps"] <= 0 or not np.isfinite([row["return"], *row["squared_error"]]).all():
            raise ValueError("Nonfinite or empty trial")
        grouped[row["task_id"]].append(row)
    per_task = []
    for tid, group in enumerate(grouped):
        if len(group) != trials_per_task or {r["trial_id"] for r in group} != set(range(trials_per_task)):
            raise ValueError("Incomplete per-task trial coverage")
        returns = [r["return"] for r in group]
        steps = np.array([r["steps"] for r in group])
        rmse = np.sqrt(np.sum([r["squared_error"] for r in group], axis=0) / steps.sum())
        per_task.append(dict(task_id=tid, task_key=catalog.decode(tid).key, count=len(group),
                             mean_return=float(np.mean(returns)), std_return=float(np.std(returns)),
                             mean_length=float(steps.mean()), vx_rmse=float(rmse[0]), vy_rmse=float(rmse[1]), wz_rmse=float(rmse[2]),
                             failure_rate=float(np.mean([r["terminated"] for r in group])),
                             boundary_rate=float(np.mean([r["boundary"] and not r["terminated"] for r in group])),
                             time_limit_rate=float(np.mean([r["time_limit"] and not r["terminated"] and not r["boundary"] for r in group])),
                             raw_boundary_rate=float(np.mean([r["boundary"] for r in group]))))
    numeric = [k for k in per_task[0] if k not in {"task_id", "task_key", "count"}]
    return dict(mean_returns=[r["mean_return"] for r in per_task], counts=[r["count"] for r in per_task],
                per_task=per_task, trials=rows,
                macro={k: float(np.mean([r[k] for r in per_task])) for k in numeric})
