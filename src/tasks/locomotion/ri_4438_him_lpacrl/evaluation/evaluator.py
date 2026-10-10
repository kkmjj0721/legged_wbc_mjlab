"""Independent nominal evaluation, paired across curriculum stages and methods."""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from tensordict import TensorDict
from ..config.env_cfgs import build_env_cfg
from ..mdp.curriculums import get_controller
from ..mdp.task_catalog import plain
from .metrics import summarize_trials


def freeze_actor(actor, device):
    """Copy every module, detaching transient distribution/forward caches.

    PPO leaves a Normal distribution with non-leaf loc/scale tensors on the
    actor. PyTorch forbids deepcopy of those tensors. Supplying detached copies
    through deepcopy's memo preserves the complete module without mutating the
    live actor or its optimizer/normalizer state.
    """
    memo, seen = {}, set()
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if not value.is_leaf:
                memo[id(value)] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
        elif not isinstance(value, type) and hasattr(value, "__dict__"):
            visit(vars(value))
    with torch.no_grad():
        visit(actor)
        frozen = deepcopy(actor, memo).to(device).eval()
        frozen.requires_grad_(False)
    return frozen


@contextmanager
def preserve_rng():
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def export_result(result, output_dir, catalog, fingerprints):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {**result, "task_catalog": plain(catalog.records), "fingerprints": fingerprints}
    temporary = path / ".evaluation.json.tmp"
    temporary.write_text(json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path / "evaluation.json")
    with (path / "per_task.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["per_task"][0]))
        writer.writeheader()
        writer.writerows(result["per_task"])


class TaskEvaluator:
    def __init__(self, compiled, device, env_factory=None):
        self.compiled, self.device = compiled, device
        self.env = None
        self.env_factory = env_factory

    def _get_env(self):
        if self.env is None:
            if self.env_factory is None:
                from mjlab.envs import ManagerBasedRlEnv
                self.env_factory = ManagerBasedRlEnv
            self.env = self.env_factory(cfg=build_env_cfg(self.compiled, "evaluation"), device=self.device)
        return self.env

    def evaluate(self, actor):
        start = time.monotonic()
        with preserve_rng():
            frozen = freeze_actor(actor, self.device)
            with torch.inference_mode():
                result = self._evaluate_frozen(frozen)
        result["wall_seconds"] = time.monotonic() - start
        return result

    def _evaluate_frozen(self, actor):
        env = self._get_env()
        controller = get_controller(env)
        from ..mdp.geometry_checks import GeometryMonitor
        monitor = GeometryMonitor(env, controller) if hasattr(controller, "terrain_map") else None
        table = self.compiled.trials
        e = env.num_envs
        device = env.device
        rows, slot_steps = [], 0
        max_steps = math.ceil(self.compiled.cfg.evaluation.horizon_s / env.step_dt) + 1
        for offset in range(0, len(table), e):
            batch = table[offset:offset+e]
            controller.trials_by_slot = batch + [batch[-1]] * (e-len(batch))
            obs, _ = env.reset()
            obs = TensorDict(obs, batch_size=[e], device=device)
            active = torch.arange(e, device=device) < len(batch)
            returns = torch.zeros(e, dtype=torch.float64, device=device)
            lengths = torch.zeros(e, dtype=torch.int64, device=device)
            errors = torch.zeros(e, 3, dtype=torch.float64, device=device)
            for _ in range(max_steps):
                actions = actor(obs)
                next_obs, rewards, terminated, truncated, extras = env.step(actions)
                raw_bad = getattr(env, "_him_numerics_bad", None)
                if raw_bad is not None and raw_bad[active].any():
                    raise ValueError("Raw evaluation observations violated the HIM numerical contract")
                if not torch.isfinite(rewards[active]).all() or not torch.isfinite(actions[active]).all():
                    raise ValueError("Nonfinite evaluation transition")
                actual = torch.cat((env.scene["robot"].data.root_link_lin_vel_b[:, :2],
                                    env.scene["robot"].data.root_link_ang_vel_b[:, 2:3]), dim=1)
                difference = actual - controller.commands
                returns[active] += rewards[active].double()
                lengths[active] += 1
                errors[active] += difference[active].double().square()
                slot_steps += e
                done = terminated | truncated
                if monitor is not None:
                    monitor.observe(done | ~active)
                finished = (done & active).nonzero(as_tuple=False).flatten()
                boundary = env.termination_manager.get_term("boundary_truncation") if "boundary_truncation" in env.termination_manager.active_terms else torch.zeros_like(done)
                time_limit = env.termination_manager.get_term("time_out")
                for slot in finished.tolist():
                    rows.append({**batch[slot], "return": float(returns[slot]), "steps": int(lengths[slot]),
                                 "squared_error": errors[slot].cpu().tolist(),
                                 "terminated": bool(terminated[slot]), "truncated": bool(truncated[slot]),
                                 "boundary": bool(boundary[slot]), "time_limit": bool(time_limit[slot])})
                active[finished] = False
                if not active.any():
                    break
                # All done slots, including inactive/padding slots, must reset.
                obs = TensorDict(next_obs, batch_size=[e], device=device)
                reset_ids = done.nonzero(as_tuple=False).flatten()
                if len(reset_ids):
                    reset_obs, _ = env.reset(env_ids=reset_ids)
                    for key in obs.keys():
                        obs[key][reset_ids] = reset_obs[key][reset_ids]
            else:
                raise RuntimeError("Evaluation horizon exhausted with active trials")
        result = summarize_trials(rows, self.compiled.catalog, self.compiled.cfg.evaluation.trials_per_task)
        result["simulated_slot_steps"] = slot_steps
        result["geometry_checks"] = monitor.summary() if monitor is not None else {}
        result["protocol"] = "nominal, deterministic policy, finite-patch truncation; not paper EPTE-SP"
        return result

    def close(self):
        if self.env is not None:
            try:
                with preserve_rng():
                    self.env.close()
            finally:
                self.env = None
