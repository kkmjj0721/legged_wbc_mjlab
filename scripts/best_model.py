"""Analyze complete RSL-RL resume histories and compare their saved policies in simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Support both `python scripts/best_model.py` and `python -m scripts.best_model`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.best_model import Goal, Options
from src.utils.training_chain import read_training_chain, analyze_chain, resolve_run_directory
from src.utils.training_chain_report import write_chain_reports, write_training_report
from src.utils.training_report import publish_best, write_reports, write_selection, recommendation, log_candidate_label


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run", type=Path, help="Run or experiment directory; select its newest run and trace saved resume ancestry")
    inputs.add_argument("--runs", type=Path, nargs="+", help="Explicit original/resumed run directories, oldest to newest")
    parser.add_argument("--single-run", action="store_true", help="Analyze only --run, without tracing earlier training")
    parser.add_argument("--output", type=Path, help="Report/artifact directory (default: <run>/best)")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20, help="Iterations excluded after each curriculum transition")
    parser.add_argument("--penalty", type=float, default=1.0, help="MAD penalty multiplier")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--segment", type=int, help="Log segment after a step rewind (default: latest)")
    parser.add_argument("--env-step-offset", type=int, help="Override: env_steps after iteration k = offset + (k+1)*num_steps_per_env")
    parser.add_argument("--convergence-windows", type=int, default=3)
    parser.add_argument("--improvement", type=float, default=0.02, help="Relative trend/span threshold")
    parser.add_argument("--max-relative-mad", type=float, default=0.05)
    parser.add_argument("--terrain-drift", type=float, default=0.1, help="Maximum span in terrain level window medians")
    parser.add_argument("--max-action-clip", type=float, default=0.1)
    parser.add_argument("--goal", action="append", default=[], help="Explicit task target, e.g. 'survival_ratio>=0.95'; repeatable")
    parser.add_argument("--write-best", action="store_true", help="Copy the training-log candidate to model_best.pt (simulation recommendation uses model_best_eval.pt)")
    parser.add_argument("--export-onnx", action="store_true", help="Export the training-log candidate on CPU; implies --write-best")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="Simulate all valid checkpoints, show the top 10 and include real simulation images")
    parser.add_argument("--eval-task", help="Task id; inferred for ri_4438_him / ri_4438_ppo")
    parser.add_argument("--eval-num-envs", type=int, default=24, help="Multiple of 6; command groups run in parallel")
    parser.add_argument("--eval-duration", type=float, default=12.0, help="Seconds per first episode")
    parser.add_argument("--eval-seeds", type=int, nargs=3, default=[0, 1, 2], help="Exactly three distinct seeds (default: 0 1 2)")
    parser.add_argument("--eval-terrains", nargs="+", choices=["flat", "rough", "stairs_up", "stairs_down"], default=["flat", "rough", "stairs_up", "stairs_down"],
                        help="Terrain groups; each stair direction always tests fixed 5, 10 and 15 cm risers")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--eval-checkpoints", type=int, nargs="+", help="Restrict iterations; duplicate iteration numbers in different runs are all retained")
    parser.add_argument("--eval-count", type=int, default=0, help="0 (default) evaluates ALL valid checkpoints; positive values explicitly limit a diagnostic run")
    parser.add_argument("--eval-parallel", type=int, default=0, help="Models simulated concurrently; 0 chooses up to 8 based on free GPU memory")
    parser.add_argument("--eval-tilt-limit", type=float, default=20.0, help="Tilt diagnostic threshold in degrees; does not change termination rules")
    args = parser.parse_args(argv)
    try:
        options = Options(**{key: getattr(args, key) for key in (
            "window", "min_samples", "warmup", "penalty", "top_k", "segment", "env_step_offset",
            "convergence_windows", "improvement", "max_relative_mad", "terrain_drift", "max_action_clip")},
            goals=[Goal.parse(value) for value in args.goal])
        options.validate()
        if args.evaluate:
            from src.utils.policy_evaluation import EvalOptions, evaluate
            eval_options = EvalOptions(task=args.eval_task, num_envs=args.eval_num_envs, duration=args.eval_duration,
                                       seeds=tuple(args.eval_seeds), terrains=tuple(args.eval_terrains),
                                       device=args.eval_device, checkpoints=args.eval_checkpoints,
                                       count=args.eval_count, parallel=args.eval_parallel,
                                       tilt_limit=args.eval_tilt_limit)
            eval_options.validate()
        if args.single_run and args.runs:
            raise ValueError("--single-run only applies to --run")
        print(f"Reading training history: {args.runs or args.run}", flush=True)
        latest = resolve_run_directory(args.run) if args.run is not None else None
        if latest is not None and latest != args.run.expanduser().resolve():
            print(f"检测到实验目录，选中最新 run: {latest}", flush=True)
        runs, links = read_training_chain(latest, args.runs, args.single_run)
        run, result = analyze_chain(runs, links, options)
        if args.run is not None:
            result["input_resolution"] = {"requested": str(args.run.expanduser().resolve()), "selected_run": str(latest),
                                          "kind": "run" if latest == args.run.expanduser().resolve() else "experiment"}
        for index, origin in enumerate(runs, 1):
            print(f"  R{index}: {origin.path} | {sum(c['valid'] for c in origin.checkpoints)} valid models", flush=True)
        output = args.output.expanduser().resolve() if args.output else run.path / "best"
        if output in {origin.path for origin in runs}:
            raise ValueError("output must be a separate directory, not the run root")
        result["published"] = None
        result["recommendation"] = None if args.evaluate else recommendation(result)
        write_reports(runs[-1], result, output, plot=not args.no_plot)
        write_chain_reports(runs, result, output, plot=not args.no_plot)
        if args.evaluate:
            evaluation = evaluate(run, result, output, eval_options)
            result["recommendation"] = recommendation(result, evaluation)
            from src.utils.evaluation_readiness import assess_hardware_readiness
            result["hardware_assessment"] = assess_hardware_readiness(evaluation)
            from src.utils.evaluation_report import write_evaluation_report, publish_evaluation_best
            publish_evaluation_best(run, evaluation, output)
            write_evaluation_report(result, evaluation, output, plot=not args.no_plot)
            result["evaluation"] = {"best_iteration": evaluation["best_iteration"], "best_model_id": evaluation["best_model_id"], "best_label": evaluation["best_label"], "report": str(output / "report.html"),
                                    "data": str(output / "evaluation.json"), "model": str(output / "model_best_eval.pt")}
            result["warnings"] = [w.replace("; no fixed-condition evaluation was run.", "; see evaluation.json for separate fixed-condition simulation results.") for w in result["warnings"]]
            write_selection(result, output)
            from src.utils.evaluation_repeatability import repeatability_notes
            for note in repeatability_notes(evaluation):
                print(f"复跑核对：{note}", flush=True)
        if args.write_best or args.export_onnx:
            result["published"] = publish_best(runs[-1], result, output, export_onnx=args.export_onnx)
            if not args.evaluate:
                result["recommendation"] = recommendation(result)
            write_selection(result, output)
        if not args.evaluate:
            write_training_report(result, output)
        print(f"日志截止 iteration: {result['as_of_iteration']} | segment: {result['segment']}")
        print(f"日志评分候选（训练曲线参考）: {log_candidate_label(result)}")
        if result["current_stage"]:
            print(f"课程阶段: {result['current_stage']['index']} | 最终阶段: {result['current_stage']['final']}")
        print(f"收敛状态: {result['convergence']['status']} — {result['convergence']['reason']}")
        for iteration in result["top_current_stage"]:
            candidate = next(c for c in result["candidates"] if c["iteration"] == iteration)
            print(f"  日志候选 model_{candidate['iteration']}.pt: score={candidate['score']:.4f}, "
                  f"window=[{candidate['window_start']}, {candidate['window_end']}], n={candidate['samples']}")
        if not run.checkpoints:
            print("提示：没有找到数字 checkpoint；已输出曲线和收敛分析。")
        for warning in result["warnings"]:
            print(f"提示：{warning}")
        print(f"日志分析数据: {output / 'selection.json'}")
        if result["published"]:
            print(f"已保存日志候选: {result['published']['model']}")
            if result["published"]["onnx"]:
                print(f"已导出日志候选 ONNX: {result['published']['onnx']}")
        if args.evaluate:
            print(f"本轮推荐（基于仿真）: {result['recommendation']['label']} | 模型: {output / 'model_best_eval.pt'}")
            from src.utils.evaluation_readiness import readiness_notes
            for note in readiness_notes(result["hardware_assessment"]):
                print(note)
        else:
            print("本次未进行仿真评测，暂无仿真推荐。")
        print(f"统一报告（含完整训练曲线）: {output / 'report.html'}")
        return 0
    except (ValueError, OSError, ImportError, KeyError, RuntimeError) as exc:
        print(f"best_model: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
