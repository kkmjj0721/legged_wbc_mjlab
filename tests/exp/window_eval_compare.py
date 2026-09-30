"""Compare fixed-scenario evaluation.json files from several checkpoints (CPU only).

Usage::

    python -m tests.exp.window_eval_compare \
        --group parent=logs/.../parent/eval_10500/evaluation.json \
        --group window=logs/.../win_run1/eval_10800/evaluation.json \
        --group control=logs/.../ctl_run1/eval_10800/evaluation.json \
        --out logs/.../eval_compare.json

Reports task success (stairs completed / first episode survived) and the
motion-quality metrics per fixed terrain, averaged over the three seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = (
  "task_success",
  "success",
  "survival_s",
  "linear_rmse",
  "yaw_rmse",
  "tilt_rms_deg",
  "action_rate_rms",
  "vertical_vel_rms",
  "power_abs_w",
)


def load_cases(path: Path) -> dict[str, dict[str, float]]:
  data = json.loads(path.read_text())
  out: dict[str, dict[str, float]] = {}
  for model in data.get("summary", []):
    for case in model.get("cases", []):
      terrain = case["terrain"]
      bucket = out.setdefault(terrain, {k: [] for k in METRICS})
      for key in METRICS:
        if key in case:
          bucket[key].append(case[key])
  return {t: {k: float(np.mean(v)) if v else float("nan") for k, v in m.items()}
          for t, m in out.items()}


def main(argv=None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--group", action="append", required=True, help="label=path/to/evaluation.json")
  p.add_argument("--out", default=None)
  args = p.parse_args(argv)

  groups = {}
  for spec in args.group:
    label, _, raw = spec.partition("=")
    groups[label] = load_cases(Path(raw).expanduser().resolve())

  terrains = sorted({t for g in groups.values() for t in g})
  result: dict[str, object] = {"groups": {}, "terrains": terrains}

  for metric in ("task_success", "linear_rmse"):
    print(f"\n=== {metric} ===")
    header = f"{'terrain':22s}" + "".join(f"{k:>12s}" for k in groups)
    print(header)
    print("-" * len(header))
    for terrain in terrains:
      line = f"{terrain:22s}"
      for label in groups:
        value = groups[label].get(terrain, {}).get(metric, float("nan"))
        line += f"{value:12.4f}"
      print(line)
      result["groups"].setdefault(metric, {})[terrain] = {
        label: groups[label].get(terrain, {}).get(metric) for label in groups
      }

  print("\n=== all metrics (mean over terrains) ===")
  header = f"{'metric':22s}" + "".join(f"{k:>12s}" for k in groups)
  print(header)
  print("-" * len(header))
  for metric in METRICS:
    line = f"{metric:22s}"
    for label in groups:
      values = [groups[label][t][metric] for t in terrains if t in groups[label] and metric in groups[label][t]]
      line += f"{np.mean(values):12.4f}" if values else f"{'-':>12s}"
    print(line)
    result["groups"].setdefault("overall", {})[metric] = {
      label: (float(np.mean([groups[label][t][metric] for t in terrains if t in groups[label]]))
              if any(t in groups[label] for t in terrains) else None)
      for label in groups
    }

  if args.out:
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
