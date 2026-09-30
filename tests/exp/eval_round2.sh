#!/usr/bin/env bash
# Round 2 fixed-scenario evaluation: pre-resume parent vs the two 500-iteration
# continued runs trained with the stable fine-tuning recipe
# (--lr 3e-4 --schedule fixed --terrain-init-level 3 --entropy_coef 0.002).
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=.venv/bin/python
PARENT=/home/sunteng/Downloads/legged_wbc_mjlab/logs/rsl_rl/ri_4438_him/2026-09-30_10-05-21
WIN=logs/rsl_rl/ri_4438_him_window/win_run2
CTL=logs/rsl_rl/ri_4438_him_window/ctl_run2
ITER="${1:-10999}"

eval_one () {
  local label="$1" dir="$2"
  echo "=== evaluating $label ($dir @ $ITER)"
  $PY scripts/best_model.py --run "$dir" --single-run --evaluate \
    --eval-checkpoints "$ITER" \
    --eval-terrains flat stairs_up stairs_down \
    --eval-num-envs 24 --eval-duration 12 --eval-seeds 0 1 2 \
    --output "$dir/eval_${ITER}" --no-plot
}

eval_one window  "$WIN"
eval_one control "$CTL"

$PY -m tests.exp.window_eval_compare \
  --group "parent=$PARENT/eval_10500/evaluation.json" \
  --group "window=$WIN/eval_${ITER}/evaluation.json" \
  --group "control=$CTL/eval_${ITER}/evaluation.json" \
  --out "$WIN/eval_compare_round2.json"
