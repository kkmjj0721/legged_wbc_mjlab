"""Seeded, paired locomotion benchmarks, imported only with --evaluate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .training_logs import Run, sha256, model_key, model_label, model_origin, source_run
from .training_report import _atomic_text, load_checkpoint_actor


from .evaluation_metrics import COMMANDS, EpisodeMetrics, summarize
from .evaluation_scenarios import STAIR_COMMANDS, STAIR_GOAL, STAIR_PRESETS, TERRAIN_LABELS, expand_terrains


@dataclass
class EvalOptions:
    task: str | None = None
    num_envs: int = 24
    duration: float = 12.0
    seeds: tuple[int, ...] = (0, 1, 2)
    terrains: tuple[str, ...] = ("flat", "rough", "stairs_up", "stairs_down")
    device: str = "cuda:0"
    checkpoints: tuple[int, ...] | None = None
    count: int = 0
    parallel: int = 0
    tilt_limit: float = 20.0

    def validate(self):
        if self.num_envs < len(COMMANDS) or self.num_envs % len(COMMANDS):
            raise ValueError("--eval-num-envs must be a positive multiple of 6 (one group per command)")
        if not math.isfinite(self.duration) or not 1 <= self.duration <= 120:
            raise ValueError("--eval-duration must be between 1 and 120 seconds")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3 or min(self.seeds) < 0:
            raise ValueError("evaluation requires exactly 3 distinct nonnegative seeds")
        if not self.terrains or len(set(self.terrains)) != len(self.terrains) or set(self.terrains) - {"flat", "rough", "stairs_up", "stairs_down"}:
            raise ValueError("unknown or duplicate evaluation terrain")
        if self.count < 0 or self.parallel < 0 or self.parallel > 16:
            raise ValueError("--eval-count must be >= 0; --eval-parallel must be between 0 and 16")
        if not math.isfinite(self.tilt_limit) or not 1 <= self.tilt_limit <= 90:
            raise ValueError("--eval-tilt-limit must be between 1 and 90 degrees")


def select_candidates(run: Run, result: dict, requested=None, count=0) -> list[dict]:
    """Default to every valid checkpoint in the run, including earlier log segments."""
    available = {model_key(c): c for c in sorted(run.checkpoints, key=lambda c: (c.get("source_index", 0), c["iteration"])) if c["valid"]}
    if not available:
        raise ValueError("no valid checkpoints in the run to simulate")
    if requested is not None:
        requested = list(dict.fromkeys(requested))
        iterations = [key for key, c in available.items() if c["iteration"] in requested]
        if not requested or any(i not in {c["iteration"] for c in available.values()} for i in requested):
            raise ValueError("--eval-checkpoints must name valid numeric checkpoints in the run")
    elif count == 0:
        iterations = list(available)
    else:
        if count < 1:
            raise ValueError("candidate count must be nonnegative")
        # The configured count is a hard cap. Preserve latest, baseline and log
        # shortlist first, then cover the largest gaps in checkpoint positions.
        ordered = list(available)
        top = result.get("top_current_model_ids", result["top_current_stage"])
        priority = top[:1] + [ordered[-1], ordered[0]] + top[1:]
        iterations = [i for i in dict.fromkeys(priority) if i in available][:count]
        while len(iterations) < min(count, len(ordered)):
            selected = [ordered.index(i) for i in iterations]
            index = max((i for i in range(len(ordered)) if ordered[i] not in iterations),
                        key=lambda i: (min(abs(i-j) for j in selected), i))
            iterations.append(ordered[index])
        iterations.sort(key=ordered.index)
    return [available[i] for i in iterations]


def plain(value):
    """Inert configuration description for comparison and provenance."""
    if isinstance(value, Enum):
        # dump_yaml encodes Enum constructors as a one-item argument sequence.
        return [value.value]
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, slice):
        return [value.start, value.stop, value.step]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def validate_interface(cfg, saved: dict):
    """Reject historical observation/action semantics that differ from this checkout."""
    current = plain(cfg.observations["actor"])
    original = saved.get("observations", {}).get("actor", {})
    differences = []
    for key in ("history_length", "flatten_history_dim", "concatenate_terms", "concatenate_dim"):
        if current.get(key) != original.get(key):
            differences.append(f"observations.actor.{key}")
    if list(current["terms"]) != list(original.get("terms", {})):
        differences.append("observations.actor.term_order")
    for name, term in current["terms"].items():
        for key in ("func", "params", "scale", "clip", "history_length", "flatten_history_dim"):
            if term.get(key) != original.get("terms", {}).get(name, {}).get(key):
                differences.append(f"observations.actor.{name}.{key}")
    if plain(cfg.actions) != saved.get("actions"):
        differences.append("actions (joint order, scales, offsets or limits)")
    if differences:
        raise ValueError("saved policy interface differs from current task: " + ", ".join(differences))


def make_env_cfg(run: Run, options: EvalOptions, terrain: str):
    import torch
    import src.tasks  # noqa: F401 -- task registry
    from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.terrains.config import random_rough
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    from mjlab.managers import EventTermCfg, TerminationTermCfg
    from .evaluation_protocol import BenchmarkStairsCfg, left_stair_lane, reset_paired_robot
    if terrain not in TERRAIN_LABELS:
        raise ValueError(f"unknown evaluation scenario: {terrain}")
    stair = STAIR_PRESETS.get(terrain)
    commands = STAIR_COMMANDS if stair else COMMANDS

    @dataclass(kw_only=True)
    class FixedCommandCfg(CommandTermCfg):
        def build(self, env):
            return FixedCommand(self, env)

    class FixedCommand(CommandTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            table = torch.tensor([c[2] for c in commands], device=self.device)
            self._command = table[torch.arange(self.num_envs, device=self.device) % len(commands)]

        @property
        def command(self):
            return self._command

        def _resample_command(self, env_ids):
            pass

        def _update_command(self, env_ids):
            pass

        def _update_metrics(self):
            pass

    task = options.task
    if task is None:
        experiment = run.agent.get("experiment_name", "").lower()
        if experiment not in ("ri_4438_him", "ri_4438_ppo"):
            raise ValueError("cannot infer RI-4438 task; supply --eval-task")
        task = "Ri-4438-HIM-Rough" if experiment.endswith("him") else "Ri-4438-Rough"
    if task not in ("Ri-4438-HIM-Rough", "Ri-4438-HIM-Flat", "Ri-4438-Rough", "Ri-4438-Flat"):
        raise ValueError("simulation benchmark currently supports RI-4438 HIM/PPO tasks")
    cfg = load_env_cfg(task, play=False)
    validate_interface(cfg, run.env)
    cfg.seed = options.seeds[0]
    cfg.scene.num_envs = options.num_envs
    cfg.scene.env_spacing = 0.0
    cfg.auto_reset = False
    cfg.sim.mujoco.timestep = float(run.env["sim"]["mujoco"]["timestep"])
    cfg.decimation = int(run.env["decimation"])
    cfg.episode_length_s = options.duration + 2 * cfg.sim.mujoco.timestep * cfg.decimation
    cfg.curriculum = {}
    cfg.rewards = {}
    cfg.metrics = {}
    cfg.recorders = {}
    cfg.observations = {"actor": cfg.observations["actor"]}
    cfg.observations["actor"].enable_corruption = False
    for term in cfg.observations["actor"].terms.values():
        term.delay_min_lag = term.delay_max_lag = term.delay_update_period = 0
        term.delay_hold_prob = 0.0
    for actuator in cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_min_lag = actuator.delay_max_lag = actuator.delay_update_period = 0
        actuator.delay_hold_prob = 0.0
    cfg.commands = {"twist": FixedCommandCfg(resampling_time_range=(1e9, 1e9))}
    # Nominal physics, paired initial poses; no curriculum, impulse or state-velocity shortcuts.
    cfg.events = {"paired_reset": EventTermCfg(func=reset_paired_robot, mode="reset", params={
        "samples_per_model": options.num_envs, "stairs": terrain.startswith("stairs_")})}
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_type = "plane" if terrain == "flat" else "generator"
    cfg.scene.terrain.terrain_generator = None
    if terrain == "rough":
        # Entire rollout remains inside this patch at commanded speeds.
        side = 2 * options.duration + 8
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20260929, size=(side, side), num_rows=1, num_cols=1,
            curriculum=False, border_width=5, difficulty_range=(1.0, 1.0),
            sub_terrains={"rough": random_rough(noise_range=(-0.02, 0.02), noise_step=0.01)},
        )
    elif terrain.startswith("stairs_"):
        cfg.sim.nconmax = 256
        cfg.sim.njmax = 2048
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=20260929, size=(2 * options.duration + 12, 8), num_rows=1, num_cols=1,
            curriculum=False, border_width=5, difficulty_range=(1.0, 1.0),
            sub_terrains={"stairs": BenchmarkStairsCfg(step_height=stair["height_m"], descending=stair["descending"])},
        )
        cfg.terminations["left_stair_lane"] = TerminationTermCfg(func=left_stair_lane)
    return cfg, task


class ParallelActors:
    """Vectorize independent model parameters, preserving each model's normalizer."""

    def __init__(self, actors, configurations=None):
        import torch
        from tensordict import TensorDict
        signatures = [(type(a), [(k, tuple(v.shape), v.dtype) for k, v in a.state_dict().items()], repr(a)) for a in actors]
        different_configs = configurations is not None and any(c != configurations[0] for c in configurations[1:])
        self.actors = actors if different_configs or any(s != signatures[0] for s in signatures[1:]) else None
        if self.actors is not None:
            return
        self.params, self.buffers = torch.func.stack_module_state(actors)
        template = actors[0]

        def call(params, buffers, observations):
            td = TensorDict({"actor": observations}, batch_size=[observations.shape[0]])
            return torch.func.functional_call(template, (params, buffers), (td,))

        self.forward = torch.vmap(call, in_dims=(0, 0, 0), randomness="error")

    def __call__(self, observations):
        if self.actors is not None:
            import torch
            from tensordict import TensorDict
            return torch.stack([actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))
                                for actor, obs in zip(self.actors, observations)])
        return self.forward(self.params, self.buffers, observations)


def parallel_count(options, model_count):
    import torch
    if options.parallel:
        return min(options.parallel, model_count)
    if options.device.startswith("cuda"):
        free, _ = torch.cuda.mem_get_info(options.device)
        # A conservative scheduling estimate, not an OOM guarantee. Always leave
        # 1.5 GiB free; explicit --eval-parallel 1 is available on busy GPUs.
        budget = max(1, int((free / 2**30 - 1.5) / (0.3 + options.num_envs * 0.01)))
        return min(8, model_count, budget)
    return min(2, model_count)


def evaluate(run: Run, result: dict, output: Path, options: EvalOptions) -> dict:
    import mujoco
    import torch
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from .evaluation_repeatability import compare_evaluations, load_previous, physics_fingerprint, runtime_description

    options.validate()
    previous = load_previous(output)
    candidates = select_candidates(run, result, options.checkpoints, options.count)
    if run.sources:
        # One common benchmark environment, but every actor uses its own saved architecture.
        for origin in run.sources.values():
            cfg, _ = make_env_cfg(origin, options, "flat")
            if origin.agent.get("clip_actions") != run.agent.get("clip_actions"):
                raise ValueError(f"action clipping differs across runs: {origin.path}")
            if (origin.env.get("decimation"), origin.env.get("sim", {}).get("mujoco", {}).get("timestep")) != (run.env.get("decimation"), run.env.get("sim", {}).get("mujoco", {}).get("timestep")):
                raise ValueError(f"control timestep differs across runs: {origin.path}")
    started = time.monotonic()
    parallel = parallel_count(options, len(candidates))
    samples = options.num_envs
    total_envs = parallel * samples
    scenarios = {terrain: STAIR_COMMANDS if terrain in STAIR_PRESETS else COMMANDS for terrain in expand_terrains(options.terrains)}
    available = [c for c in run.checkpoints if c["valid"]]
    data = {"schema_version": 6, "complete": False, "options": asdict(options),
            "runtime": runtime_description(options.device),
            "parallel_models": parallel, "total_parallel_envs": total_envs,
            "selection": {"available": len(available), "selected": len(candidates), "all_tested": len(available) == len(candidates),
                          "scope": "all valid numeric checkpoints across the supplied resume chain and log segments"},
            "chain": result.get("chain", []),
            "excluded_checkpoints": [{"iteration": c["iteration"], **model_origin(c), "error": c["error"]} for c in run.checkpoints if not c["valid"]],
            "commands": [{"name": n, "label": label, "velocity": v} for n, label, v in COMMANDS],
            "scenarios": {t: [{"name": n, "label": label, "velocity": v} for n, label, v in commands] for t, commands in scenarios.items()},
            "terrain_labels": TERRAIN_LABELS, "stair_goal_m": STAIR_GOAL,
            "stair_presets": {t: STAIR_PRESETS[t] for t in scenarios if t in STAIR_PRESETS},
            "run": str(run.path), "log_best": result["best_current_stage"],
            "as_of_iteration": result["as_of_iteration"], "segment": result["segment"],
            "saved_config_sha256": run.fingerprints, "models": [], "rows": [], "traces": [], "environments": {},
            "simulation_frames": [],
            "ranking_policy": "task_success descending, survival success descending, survival_s descending, linear_rmse ascending, yaw_rmse ascending, tilt_rms_deg ascending; exact tie uses source order then lower iteration",
            "scope": "current checkout nominal RI-4438; saved actor/interface/dt; seeded paired resets independent of batch size; no push/domain randomization/noise/delay; first episode only"}
    for candidate in candidates:
        source = Path(candidate["path"])
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (candidate["size"], candidate["mtime_ns"]):
            raise ValueError(f"checkpoint changed since analysis: {source}")
        reasons = []
        iteration = candidate["iteration"]
        if model_key(candidate) in result.get("top_current_model_ids", result["top_current_stage"]):
            reasons.append("日志优选")
        if model_key(candidate) == model_key(available[0]):
            reasons.append("最早/续训起点")
        if model_key(candidate) == model_key(available[-1]):
            reasons.append("最新")
        if options.checkpoints is not None:
            reasons.append("手动指定")
        if not reasons:
            reasons.append("全部评测" if options.count == 0 else "覆盖中间迭代")
        data["models"].append({"iteration": iteration, **model_origin(candidate), "segment": candidate.get("segment"), "path": str(source), "sha256": sha256(source), "reasons": reasons})
    print(f"[eval] {len(candidates)}/{len(available)} models; {parallel} models concurrently, {total_envs} independent worlds", flush=True)
    initial_states = {}
    for terrain, commands in scenarios.items():
        cfg, task = make_env_cfg(run, options, terrain)
        cfg.scene.num_envs = total_envs
        description = plain(cfg)
        fingerprint = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
        _atomic_text(output / f"eval_env_{terrain}.json", lambda f: json.dump(description, f, indent=2, ensure_ascii=False))
        data["environments"][terrain] = {"task": task, "sha256": fingerprint}
        env = ManagerBasedRlEnv(cfg, device=options.device)
        try:
            # Save the exact compiled scene used for scoring. Images are rendered
            # from recorded states afterwards, without rerunning any policy.
            scene_file = output / "simulation" / f"{terrain}.mjb"
            scene_file.parent.mkdir(parents=True, exist_ok=True)
            mujoco.mj_saveModel(env.sim.mj_model, str(scene_file))
            data["environments"][terrain]["render_model"] = str(scene_file.relative_to(output))
            data["environments"][terrain]["render_model_sha256"] = sha256(scene_file)
            data["environments"][terrain]["physics_sha256"] = physics_fingerprint(env.sim.mj_model)
            dt = env.step_dt
            steps = int(math.ceil(options.duration / dt))
            data["dt"], data["steps"] = dt, steps
            robot = env.scene["robot"]
            cpu = lambda tensor: tensor.detach().cpu().numpy().copy()
            for start in range(0, len(data["models"]), parallel):
                batch = data["models"][start:start + parallel]
                observations, _ = env.reset(seed=options.seeds[0])
                actors = []
                for index, model in enumerate(batch):
                    actor, _, action_dim, _ = load_checkpoint_actor(source_run(run, model), Path(model["path"]),
                        TensorDict({"actor": observations["actor"][index*samples:(index+1)*samples]}, batch_size=[samples]), options.device)
                    if action_dim != env.action_manager.total_action_dim:
                        raise ValueError("checkpoint action dimension differs from environment")
                    if sha256(Path(model["path"])) != model["sha256"]:
                        raise ValueError("checkpoint changed while loading simulation actor")
                    actors.append(actor)
                policies = ParallelActors(actors, [source_run(run, m).agent.get("actor", {}) for m in batch])
                valid_envs = len(batch) * samples
                for seed in options.seeds:
                    print(f"[eval] models={[model_label(m) for m in batch]} | {terrain} | seed={seed} | {samples} episodes/model x {steps*dt:.2f}s", flush=True)
                    env.common_step_counter = 0
                    env._evaluation_seed = seed
                    observations, _ = env.reset(seed=seed)
                    positions = cpu(robot.data.root_link_pos_w - env.scene.env_origins)
                    initial = cpu(torch.cat((robot.data.root_link_pos_w - env.scene.env_origins, robot.data.root_link_quat_w,
                                             robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b,
                                             robot.data.joint_pos, robot.data.joint_vel), dim=-1))
                    signatures = []
                    for index in range(len(batch)):
                        signature = hashlib.sha256(initial[index*samples:(index+1)*samples].tobytes()).hexdigest()
                        key = (terrain, seed)
                        if key in initial_states and initial_states[key] != signature:
                            raise ValueError("paired initial state verification failed; refusing unfair comparison")
                        initial_states[key] = signature
                        signatures.append(signature)
                    stair = STAIR_PRESETS.get(terrain)
                    stair_gain = stair["total_height_m"] * (-1 if stair["descending"] else 1) if stair else 0.0
                    metrics = EpisodeMetrics(total_envs, dt, commands, options.tilt_limit,
                                             STAIR_GOAL if terrain.startswith("stairs_") else None, stair_gain)
                    metrics.initial_positions = positions
                    metrics.alive[valid_envs:] = False
                    clip = run.agent.get("clip_actions")
                    failure_causes = [""] * total_envs
                    from .evaluation_images import FrameRecorder
                    recorder = FrameRecorder(batch, samples, commands, terrain, seed, steps, dt) if seed == options.seeds[0] else None
                    # All models infer in one vmap call, and their separate worlds
                    # advance in one GPU simulation step. Partial last batches use idle padding.
                    with torch.no_grad():
                        for step in range(steps):
                            obs = observations["actor"][:valid_envs].reshape(len(batch), samples, *observations["actor"].shape[1:])
                            output_actions = policies(obs).reshape(valid_envs, -1)
                            if not torch.isfinite(output_actions).all():
                                raise ValueError("non-finite policy actions")
                            if clip is not None:
                                output_actions = output_actions.clamp(-clip, clip)
                            actions = torch.zeros(total_envs, action_dim, device=env.device)
                            actions[:valid_envs] = output_actions
                            # State is sampled before manually resetting any failed world.
                            observations, _, terminated, truncated, _ = env.step(actions)
                            term, trunc = cpu(terminated), cpu(truncated)
                            velocity = cpu(robot.data.root_link_lin_vel_b[:, :2])
                            angular = cpu(robot.data.root_link_ang_vel_b)
                            gravity = cpu(robot.data.projected_gravity_b)
                            position = cpu(robot.data.root_link_pos_w - env.scene.env_origins)
                            command = cpu(env.command_manager.get_command("twist"))
                            active = metrics.update(velocity, angular[:, 2], gravity[:, 2], cpu(actions), command, term, trunc, clip,
                                gravity=gravity, angular_velocity=angular, vertical_velocity=cpu(robot.data.root_link_lin_vel_w[:, 2]),
                                torque=cpu(robot.data.qfrc_actuator), joint_velocity=cpu(robot.data.joint_vel), position=position)
                            for name in env.termination_manager.active_terms:
                                if not env.termination_manager.get_term_cfg(name).time_out:
                                    failed = cpu(env.termination_manager.get_term(name)) & active
                                    for index in np.flatnonzero(failed):
                                        failure_causes[index] += name + ";"
                            if recorder is not None:
                                recorder.capture(step, env.sim.data, active, term | trunc, failure_causes,
                                                 robot.data.root_link_pos_w, data["simulation_frames"])
                            if seed == options.seeds[0] and step % max(1, round(0.1 / dt)) == 0:
                                for index, model in enumerate(batch):
                                    for c, (name, _, _) in enumerate(commands):
                                        ids = np.arange(index*samples+c, (index+1)*samples, len(commands))
                                        ids = ids[active[ids]]
                                        if len(ids):
                                            v = velocity[ids].mean(axis=0)
                                            data["traces"].append({"iteration": model["iteration"], **model_origin(model), "terrain": terrain, "seed": seed,
                                                "command": name, "time_s": (step+1)*dt, "vx": float(v[0]), "vy": float(v[1]),
                                                "yaw": float(angular[ids, 2].mean()), "alive": len(ids),
                                                "tilt": float(np.degrees(np.arccos(np.clip(-gravity[ids, 2], -1, 1))).mean()),
                                                "progress": float(position[ids, 0].mean()), "height": float(position[ids, 2].mean())})
                            done = torch.nonzero(terminated | truncated).flatten()
                            if len(done):
                                observations, _ = env.reset(env_ids=done)
                            if not metrics.alive.any():
                                break
                    rows = metrics.rows(steps)[:valid_envs]
                    for row in rows:
                        index, slot = divmod(row["env_id"], samples)
                        reason = failure_causes[row["env_id"]]
                        if row["success"] and not row["task_success"]:
                            reason = "stairs_not_completed"
                        row.update(iteration=batch[index]["iteration"], **model_origin(batch[index]), terrain=terrain, seed=seed, env_id=slot,
                                   initial_state_sha256=signatures[index], failure_reason=reason)
                    data["rows"].extend(rows)
                    _atomic_text(output / "evaluation.partial.json", lambda f: json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False))
                del policies, actors
        finally:
            env.close()
    data["summary"] = summarize(data["rows"])
    data["best_iteration"] = data["summary"][0]["iteration"]
    data["best_model_id"] = model_key(data["summary"][0])
    data["best_label"] = model_label(data["summary"][0])
    data["repeatability"] = compare_evaluations(data, previous)
    _atomic_text(output / "evaluation.partial.json", lambda f: json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False))
    from .evaluation_images import render_simulation_images
    data["simulation_images"] = render_simulation_images(data, output)
    data["elapsed_s"] = time.monotonic() - started
    data["complete"] = True
    if previous is not None:
        _atomic_text(output / "evaluation.previous.json", lambda f: json.dump(previous, f, indent=2, ensure_ascii=False, allow_nan=False))
    _atomic_text(output / "evaluation.json", lambda f: json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False))
    (output / "evaluation.partial.json").unlink(missing_ok=True)
    return data
