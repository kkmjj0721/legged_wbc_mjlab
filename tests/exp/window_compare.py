"""Compare TensorBoard curves of two or more continued-training runs (CPU only).

Usage::

    python -m tests.exp.window_compare \
        --run win=logs/rsl_rl/ri_4438_him_window/win_run1 \
        --run ctl=logs/rsl_rl/ri_4438_him_window/ctl_run1 \
        --parent /home/sunteng/Downloads/.../2026-09-30_10-05-21 \
        --out logs/rsl_rl/ri_4438_him_window/compare.json

Prints, per metric, the median over the last ``--tail`` logged iterations and
the delta of each run against the parent run's pre-resume level.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

METRICS = [
  "Train/mean_reward",
  "Train/mean_episode_length",
  "Curriculum/terrain_levels",
  "Metrics/twist/error_vel_xy",
  "Metrics/twist/error_vel_yaw",
  "Episode_Reward/track_linear_velocity",
  "Episode_Reward/track_angular_velocity",
  "Episode_Reward/foot_clearance",
  "Episode_Reward/foot_gait",
  "Episode_Reward/stumble",
  "Episode_Reward/action_rate_l2",
  "Episode_Reward/smoothness",
  "Episode_Reward/joint_torques_l2",
  "Episode_Reward/soft_landing",
  "Episode_Reward/foot_slip",
  "Episode_Termination/fell_over",
  "Metrics/slip_velocity_mean",
  "Metrics/mean_action_acc",
  "Perf/total_fps",
  "Diag/Window/raw_cost",
  "Diag/Window/last_deficit",
  "Diag/Window/age",
  "Diag/Window/late_frac",
  "Diag/Window/full_cost_frac",
  "Diag/Window/support_frac",
  "Diag/Window/flight_frac",
  "Diag/Window/air_ok_frac",
  "Diag/Window/peak_mean",
  "Diag/Window/steps_settled",
]


def read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
  from tensorboard.backend.event_processing import event_accumulator as ea

  events = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
  if not events:
    raise FileNotFoundError(f"no tensorboard events in {run_dir}")
  acc = ea.EventAccumulator(events[-1], size_guidance={ea.SCALARS: 0})
  acc.Reload()
  tags = set(acc.Tags()["scalars"])
  return {k: [(s.step, s.value) for s in acc.Scalars(k)] for k in METRICS if k in tags}


def tail_median(series: list[tuple[int, float]], tail: int, after: int | None = None):
  values = [v for step, v in series if after is None or step > after]
  values = values[-tail:]
  if not values:
    return None
  return float(np.median(values))


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--run", action="append", required=True, help="label=path; repeatable")
  p.add_argument("--parent", default=None, help="run dir of the pre-resume parent")
  p.add_argument("--parent-iteration", type=int, default=10500)
  p.add_argument("--tail", type=int, default=25)
  p.add_argument("--out", default=None)
  args = p.parse_args(argv)

  runs = {}
  for spec in args.run:
    label, _, path = spec.partition("=")
    runs[label] = read_scalars(Path(path).expanduser().resolve())

  parent = read_scalars(Path(args.parent).expanduser().resolve()) if args.parent else None

  table: dict[str, dict[str, float | None]] = {}
  for metric in METRICS:
    row: dict[str, float | None] = {}
    if parent is not None and metric in parent:
      row["parent"] = tail_median(parent[metric], args.tail, after=args.parent_iteration)
    for label, data in runs.items():
      row[label] = tail_median(data.get(metric, []), args.tail)
    if any(v is not None for v in row.values()):
      table[metric] = row

  labels = list(runs)
  header = f"{'metric':52s}" + (f"{'parent':>14s}" if parent else "")
  for label in labels:
    header += f"{label:>14s}"
  print(header)
  print("-" * len(header))
  for metric, row in table.items():
    line = f"{metric:52s}"
    if parent:
      line += f"{row.get('parent', float('nan')):14.5f}"
    for label in labels:
      value = row.get(label)
      line += f"{value:14.5f}" if value is not None else f"{'-':>14s}"
    print(line)

  result = {
    "tail": args.tail,
    "metrics": table,
    "input_runs": {spec.split("=")[0]: spec.split("=", 1)[1] for spec in args.run},
  }
  if args.out:
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
