#!/usr/bin/env bash
# Fixed-scenario evaluation of the pre-resume parent and the two continued runs.
#
# Protocol (design doc section 6): same checkpoint start, same budget, fixed
# flat / 5-10-15 cm up / 5-10-15 cm down stair scenarios, three paired seeds,
# paired initial states enforced by src/utils/policy_evaluation.py.
#
# Usage: bash tests/exp/eval_ab.sh [parent_iter] [child_iter]
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PARENT_ITER="${1:-10500}"
CHILD_ITER="${2:-10799}"

PARENT=/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-30_10-05-21
WIN=logs/rsl_rl/ri_4438_him_window/win_run1
CTL=logs/rsl_rl/ri_4438_him_window/ctl_run1

run_eval () {  # label run_dir iteration
  local label="$1" dir="$2" iter="$3"
  local out="$dir/eval_${iter}"
  echo "=== evaluating $label ($dir @ $iter) -> $out"
  $PY scripts/best_model.py --run "$dir" --single-run --evaluate \
    --eval-checkpoints "$iter" \
    --eval-terrains flat stairs_up stairs_down \
    --eval-num-envs 24 --eval-duration 12 --eval-seeds 0 1 2 \
    --output "$out" --no-plot
}

run_eval parent  "$PARENT" "$PARENT_ITER"
run_eval window  "$WIN"    "$CHILD_ITER"
run_eval control "$CTL"    "$CHILD_ITER"

$PY -m tests.exp.window_eval_compare \
  --group "parent=$PARENT/eval_${PARENT_ITER}/evaluation.json" \
  --group "window=$WIN/eval_${CHILD_ITER}/evaluation.json" \
  --group "control=$CTL/eval_${CHILD_ITER}/evaluation.json" \
  --out "logs/rsl_rl/ri_4438_him_window/eval_compare_${CHILD_ITER}.json"
