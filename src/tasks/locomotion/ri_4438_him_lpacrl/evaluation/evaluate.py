"""Evaluate saved HIM weights on their saved or explicitly overridden task space."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models.him_actor_model import HIMActorModel
from ..config.experiment_cfg import compile_experiment, from_dict, resolve_experiment, merge
from ..config.rl_cfg import ri_4438_him_runner_cfg
from .evaluator import TaskEvaluator, export_result, preserve_rng


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Explicit external evaluation config; does not modify saved curriculum")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args(argv)
    # Match rl.train: differing TF32 settings change deterministic actions and
    # can amplify into different contacts/terminations over a long episode.
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state = (checkpoint.get("infos") or {}).get("lpacrl")
    if state is None:
        raise ValueError("Standalone evaluation requires an LPACRL checkpoint with its task/protocol metadata")
    raw = state["resolved_config"]
    if args.config:
        overrides = yaml.safe_load(Path(args.config).read_text())
        raw = merge(raw, overrides)
    cfg = from_dict(raw)
    if args.num_envs is not None:
        cfg.evaluation.num_envs = args.num_envs
    compiled = compile_experiment(cfg)
    import warp as wp
    wp.config.kernel_cache_dir = str((Path(cfg.training.output_dir) / "kernel_cache").resolve())
    if not args.config and compiled.fingerprints != state["fingerprints"]:
        raise ValueError("Saved evaluation protocol no longer matches this implementation")
    evaluator = TaskEvaluator(compiled, args.device)
    try:
        with preserve_rng():
            env = evaluator._get_env()
            obs, _ = env.reset()
            actor_cfg = asdict(ri_4438_him_runner_cfg(cfg).actor)
            actor_cfg.pop("class_name")
            for key in list(actor_cfg):
                if actor_cfg[key] is None:
                    actor_cfg.pop(key)
            actor = HIMActorModel(TensorDict(obs, batch_size=[env.num_envs]), {"actor": ["actor"]}, "actor", env.action_manager.total_action_dim,
                                  **actor_cfg).to(args.device)
            actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        result = evaluator.evaluate(actor)
        result["external_protocol"] = bool(args.config)
        export_result(result, args.output_dir, compiled.catalog, compiled.fingerprints)
        print(result["macro"])
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
