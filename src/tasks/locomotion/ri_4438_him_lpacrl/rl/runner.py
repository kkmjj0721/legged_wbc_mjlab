# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved. SPDX-License-Identifier: BSD-3-Clause
# Local HIM export adapter and LP-ACRL lifecycle additions.
import inspect
import os
import tempfile
import time
import warnings
from copy import deepcopy
from pathlib import Path

import onnx
import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
    attach_metadata_to_onnx,
    get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

from rsl_rl.runners.him_on_policy_runner import (
    HIMOnPolicyRunner as RslRlHIMOnPolicyRunner,
)
from rsl_rl.utils.numerics import TrainingNumericsGuard, tensor_items
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

from ..mdp.curriculums import get_controller
from ..mdp.task_catalog import fingerprint


def _single_file_onnx_kwargs() -> dict:
    """Request inline ONNX weights when the installed torch supports it."""
    try:
        parameters = inspect.signature(torch.onnx.export).parameters
    except (TypeError, ValueError):
        return {}
    if "external_data" in parameters:
        return {"external_data": False}
    if "use_external_data_format" in parameters:
        return {"use_external_data_format": False}
    return {}


class CurriculumEnvWrapper:
    """记录任务采样，并让失败与超时同时发生时不做价值 bootstrap。"""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def step(self, actions):
        obs, reward, done, extras = self.env.step(actions)
        if "time_outs" in extras:
            extras = dict(extras)
            extras["time_outs"] = (
                self.unwrapped.reset_time_outs & ~self.unwrapped.reset_terminated
            )
        return obs, reward, done, extras

    def reset(self):
        with get_controller(self.unwrapped).discard_episodes():
            return self.env.reset()


class LPACRLRunner(MjlabOnPolicyRunner, RslRlHIMOnPolicyRunner):
    """HIM training with lightweight online tracking-progress curriculum."""

    env: RslRlVecEnvWrapper

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        from ..config.curriculum_cfg import finalize_fingerprints

        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("LPACRL supports single-GPU training only")
        self.controller = get_controller(env.unwrapped)
        self.setup = self.controller.setup
        # Public scripts own CLI options. Record the effective agent directly;
        # do not translate it back into a second learning-config representation.
        training = self.controller.cfg.training
        training.rollout_steps = train_cfg["num_steps_per_env"]
        training.max_iterations = train_cfg["max_iterations"]
        training.save_interval = train_cfg["save_interval"]
        if self.controller.mode == "train" and not env.unwrapped.cfg.auto_reset:
            raise ValueError("LPACRL training requires automatic episode resets")
        if (
            training.rollout_steps * env.num_envs
            < train_cfg["algorithm"]["num_mini_batches"]
        ):
            raise ValueError("PPO batch is smaller than num_mini_batches")
        # HIMPPO owns the estimator optimizer; keep its constructor consistent.
        for key in ("learning_rate", "max_grad_norm"):
            train_cfg["actor"]["estimator_cfg"][key] = train_cfg["algorithm"][
                "estimator_" + key
            ]
        self._validate_agent_layout(train_cfg)
        finalize_fingerprints(self.setup, env.unwrapped.cfg)
        self.setup.fingerprints["agent"] = self._agent_fingerprint(train_cfg)
        self.controller.state.fingerprints = dict(self.setup.fingerprints)
        params = env.unwrapped.cfg.curriculum["lpacrl"].params
        params["task_cfg"] = deepcopy(self.controller.cfg)
        params["fingerprints"] = {k: v for k, v in self.setup.fingerprints.items() if k != "agent"}
        super().__init__(CurriculumEnvWrapper(env), train_cfg, log_dir, device)

    @staticmethod
    def _validate_agent_layout(cfg):
        """CLI 可调整学习参数，但 HIM 的历史排列和监督目标必须匹配观测。"""
        actor, algorithm = cfg["actor"], cfg["algorithm"]
        if (
            actor["num_one_step_obs"],
            actor["history_size"],
            tuple(actor["history_term_dims"]),
            actor["history_order"],
        ) != (47, 6, (3, 3, 3, 2, 12, 12, 12), "frame_major_oldest_first"):
            raise ValueError(
                "Agent history must match the 47 x 6 HIM observation layout"
            )
        if (
            tuple(algorithm["estimator_obs_groups"]) != ("estimator",)
            or tuple(algorithm["estimator_velocity_slice"]) != (47, 50)
            or tuple(map(tuple, algorithm["estimator_target_slices"]))
            != ((0, 6), (9, 47), (47, 50))
        ):
            raise ValueError(
                "Agent estimator targets must match the HIM observation layout"
            )

    @staticmethod
    def _agent_fingerprint(cfg):
        """Only learning semantics matter for resume, not paths or logging options."""
        return fingerprint(
            {
                key: cfg[key]
                for key in (
                    "actor",
                    "critic",
                    "algorithm",
                    "clip_actions",
                    "obs_groups",
                    "numerics",
                )
            }
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.controller.mode != "train":
            raise ValueError("Training requires a train-mode environment")
        try:
            return self._learn(num_learning_iterations, False)
        finally:
            self.close()

    def _numerical_contract(self) -> dict:
        actor = self.alg.get_policy()
        return {
            "version": 1,
            "action_clip": actor.action_clip,
            "observation_clip": actor.observation_clip,
            "action_observation_slice": list(actor.action_observation_slice or ()),
            "history_size": actor.history_size,
            "frame_size": actor.num_one_step_obs,
            "export_history_order": "frame_major_current_first",
            "last_action": "clipped_policy_action_before_joint_scale_and_offset",
        }

    def _validate_numerics(self) -> None:
        """Local check: save/export is called on rank zero only."""
        actor = self.alg.get_policy()
        wrapper_clip = getattr(self.env, "clip_actions", actor.action_clip)
        if wrapper_clip != actor.action_clip:
            raise ValueError(
                f"Environment action clip {wrapper_clip} differs from policy {actor.action_clip}"
            )
        env = getattr(self.env, "unwrapped", None)
        observation_cfg = getattr(getattr(env, "observation_manager", None), "cfg", {})
        for group_name in ("actor", "critic", "estimator"):
            if group_name not in observation_cfg:
                continue
            for name, term in observation_cfg[group_name].terms.items():
                bound = (
                    actor.action_clip if name == "actions" else actor.observation_clip
                )
                expected = (-bound, bound) if bound is not None else None
                if term.clip != expected:
                    raise ValueError(
                        f"{group_name}/{name} observation clip {term.clip} differs from policy {expected}"
                    )
        for label in ("optimizer", "estimator"):
            owner = getattr(self.alg, label, None)
            optimizer = getattr(owner, "optimizer", owner)
            for name, tensor in tensor_items(getattr(optimizer, "state", {})):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid {label} optimizer state: {name}")
        for label, model in (("actor", actor), ("critic", self.alg.critic)):
            for name, tensor in model.state_dict().items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid {label} tensor: {name}")
            normalizer = getattr(model, "obs_normalizer", None)
            if normalizer is None or not hasattr(normalizer, "std"):
                continue
            if (
                (normalizer._var < 0).any()
                or (normalizer._std < 0).any()
                or normalizer.count < 0
            ):
                raise ValueError(f"Invalid {label} normalizer variance/std")
            if actor.action_observation_slice is None:
                continue
            start, stop = actor.action_observation_slice
            mean, std = normalizer.mean, normalizer.std
            if label == "actor":
                mean, std = (
                    mean.reshape(actor.history_size, -1),
                    std.reshape(actor.history_size, -1),
                )
            if actor.action_clip is not None and (
                (std[..., start:stop] > actor.action_clip + 1e-4).any()
                or (mean[..., start:stop].abs() > actor.action_clip + 1e-4).any()
            ):
                raise ValueError(
                    f"{label} last_action normalizer exceeds ±{actor.action_clip}; "
                    "use a checkpoint with valid normalization statistics."
                )

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export normalization, estimator, and deterministic actor as one graph."""
        self._validate_numerics()
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()
        os.makedirs(path, exist_ok=True)
        input_names = list(onnx_model.input_names)
        output_names = list(onnx_model.output_names)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".policy_", suffix=".onnx", dir=path
        )
        os.close(descriptor)
        try:
            self._export_onnx(
                onnx_model, temporary_path, input_names, output_names, verbose
            )
            use_wandb = (
                isinstance(self.logger.writer, WandbLogWriter) and wandb.run is not None
            )
            metadata = get_base_metadata(
                self.env.unwrapped, wandb.run.name if use_wandb else "local"
            )
            metadata.update(
                {
                    f"him_{key}": str(value)
                    for key, value in self._numerical_contract().items()
                }
            )
            attach_metadata_to_onnx(temporary_path, metadata)
            onnx.checker.check_model(temporary_path)
            os.replace(temporary_path, os.path.join(path, filename))
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def _export_onnx(onnx_model, filename, input_names, output_names, verbose):
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),
            filename,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={name: {0: "batch"} for name in (*input_names, *output_names)},
            dynamo=False,
            **_single_file_onnx_kwargs(),
        )

    # Training lifecycle copied from rsl_rl HIM runner (BSD-3-Clause).
    def _learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        """Collect rollouts, reset done environments, and update HIMPPO."""
        self._validate_numerics()
        self._validate_observation_contract()
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        unwrapped = getattr(self.env, "unwrapped", None)
        guard = None
        if self.cfg.get("numerics"):
            guard = TrainingNumericsGuard(
                self.device, self.logger.log_dir, **self.cfg["numerics"]
            )
            self.alg.numerics_guard = guard

        def check_observations(stage, values):
            if guard is None:
                return
            raw_bad = getattr(unwrapped, "_him_numerics_bad", None)
            evidence = getattr(unwrapped, "_him_numerics_evidence", {})
            guard.check(
                stage,
                {"observations": values, "pre_clip": evidence},
                limit=guard.cfg["raw_observation_abort"],
                extra_bad=raw_bad,
            )
            if raw_bad is not None:
                raw_bad.zero_()
                for value in evidence.values():
                    value.zero_()

        start_it = self.controller.state.completed_policy_updates
        total_it = start_it + num_learning_iterations
        for it in range(start_it + 1, total_it + 1):
            start = time.time()
            statistics_before = self.controller.statistics_host_seconds
            action_peak = torch.zeros((), device=self.device)
            action_counts = torch.zeros(3, device=self.device)
            with torch.inference_mode():
                for step in range(self.cfg["num_steps_per_env"]):
                    if guard is not None:
                        guard.iteration, guard.step = it, step
                        guard.remember(observations=obs)
                    check_observations("before_action", obs)
                    actions = self.alg.act(obs)
                    if guard is not None:
                        guard.remember(actions=actions)
                        guard.check(
                            "raw_action",
                            {
                                "actions": actions,
                                "distribution": self.alg.actor.output_distribution_params,
                            },
                        )
                        action_peak = torch.maximum(action_peak, actions.abs().max())
                        action_counts[0] += (
                            actions.abs() > self.alg.actor.action_clip
                        ).sum()
                        action_counts[1] += actions.numel()
                        action_counts[2] += (
                            actions.abs() > guard.cfg["raw_action_warn"]
                        ).sum()
                    next_obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    check_observations(
                        "after_step",
                        {
                            "next": next_obs,
                            "terminal": extras.get("terminal_observations"),
                        },
                    )
                    if guard is not None:
                        guard.check("step_result", {"rewards": rewards, "dones": dones})
                    elif self.cfg.get("check_for_nan", True):
                        from rsl_rl.utils import check_nan

                        check_nan(next_obs, rewards, dones)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    self.alg.process_env_step(next_obs, rewards, dones, extras)
                    self.logger.process_env_step(
                        rewards, dones, extras, self.alg.intrinsic_rewards
                    )
                    obs = next_obs

                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            learn_time = time.time() - start
            curriculum_start = time.perf_counter()
            self._maybe_update_curriculum()
            curriculum_time = time.perf_counter() - curriculum_start
            if guard is not None:
                loss_dict["numerics/raw_action_max"] = action_peak.item()
                loss_dict["numerics/action_clip_fraction"] = (
                    action_counts[0] / action_counts[1].clamp_min(1)
                ).item()
                loss_dict["numerics/raw_action_warn_fraction"] = (
                    action_counts[2] / action_counts[1].clamp_min(1)
                ).item()
                policy = self.alg.get_policy()
                if (
                    policy.obs_normalization
                    and policy.action_observation_slice is not None
                ):
                    slice_start, slice_stop = policy.action_observation_slice
                    action_std = policy.obs_normalizer.std.reshape(
                        policy.history_size, -1
                    )[:, slice_start:slice_stop]
                    loss_dict["numerics/last_action_std_max"] = action_std.max().item()
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None:
                statistics_time = self.controller.statistics_host_seconds - statistics_before
                self.logger.writer.add_scalar("Perf/curriculum_host_time", curriculum_time + statistics_time, it)
                self.logger.writer.add_scalar("Perf/iteration_time", collect_time + learn_time + curriculum_time, it)

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )
            self.logger.stop_logging_writer()

    def _validate_observation_contract(self):
        obs = self.env.get_observations()
        expected = {"actor": 282, "critic": 244, "estimator": 50}
        for name, size in expected.items():
            if name not in obs or obs[name][0].numel() != size:
                raise ValueError(f"HIM {name} observation width must be {size}")
        if self.env.num_actions != 12:
            raise ValueError("HIM requires 12 actions")

    def _maybe_update_curriculum(self):
        import json
        from ..mdp.task_catalog import plain

        state = self.controller.state
        update = state.completed_policy_updates + 1
        stage_closed = self.controller.advance(update)
        if stage_closed and self.logger.log_dir:
            data = {
                "update": update, "score_kind": state.SCORE_KIND,
                "mean_scores": state.mean_scores, "stage_index": state.stage_index,
                "lp": state.lp, "probabilities": state.probabilities,
                "counts": state.counts, "iteration_counts": state.iteration_counts,
                "starts": self.controller.starts, "diagnostics": state.diagnostics(),
            }
            with (Path(self.logger.log_dir) / "curriculum.jsonl").open("a") as handle:
                handle.write(json.dumps(plain(data), allow_nan=False) + "\n")
        if self.logger.writer is not None:
            for key, value in state.diagnostics().items():
                self.logger.writer.add_scalar("LPACRL/" + key, value, update)

    def save(self, path, infos=None):
        self._validate_numerics()
        payload = {
            **(infos or {}),
            "him_numerics": self._numerical_contract(),
            "lpacrl": self.controller.state_dict(),
            "agent_config": deepcopy(self.cfg),
        }
        # Base Mjlab runner persists common_step_counter as well.
        super().save(path, payload)
        self.export_policy_to_onnx(os.path.dirname(path), "policy.onnx")

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        """Support public actor-only playback and strict full training resume."""
        checkpoint = torch.load(
            path, weights_only=False, map_location=map_location or self.device
        )
        infos = checkpoint.get("infos") or {}
        if infos.get("him_numerics") != self._numerical_contract():
            raise ValueError("Checkpoint HIM numerical contract differs")

        if load_cfg is not None:
            if (
                self.controller.mode != "play"
                or not load_cfg.get("actor")
                or any(enabled for name, enabled in load_cfg.items() if name != "actor")
            ):
                raise ValueError(
                    "Actor-only load is for play; use resume or warm_start for training"
                )
            state = infos.get("lpacrl")
            if state is not None:
                for key in ("task", "geometry"):
                    if state["fingerprints"][key] != self.setup.fingerprints[key]:
                        raise ValueError(
                            "Playback task differs; restore its terrain and command settings "
                            "in ri_4438_him_lpacrl_config.py before playing this checkpoint"
                        )
            # HIM estimator and normalizer are already contained in actor_state_dict.
            self.alg.get_policy().load_state_dict(
                checkpoint["actor_state_dict"], strict=strict
            )
            self._validate_numerics()
            return infos

        if self.controller.mode != "train":
            raise ValueError("Full resume requires a train-mode environment")
        if "lpacrl" not in infos:
            raise ValueError(
                "Checkpoint has no LPACRL history; use --warm-start explicitly"
            )
        state = deepcopy(infos["lpacrl"])
        self.controller.load_state_dict(state)
        self.alg.load(checkpoint, None, strict)
        self._validate_numerics()
        self.current_learning_iteration = self.controller.state.completed_policy_updates
        self.env.unwrapped.common_step_counter = infos.get("env_state", {}).get(
            "common_step_counter", 0
        )
        self.env.reset()
        return infos

    def warm_start(self, path):
        """Start a new curriculum from weights, without optimizer or LP history."""
        if (
            self.controller.mode != "train"
            or self.controller.state.completed_policy_updates
            or self.controller.state.counts.any()
        ):
            raise ValueError("Warm-start requires a fresh train-mode runner")
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)
        contract = (checkpoint.get("infos") or {}).get("him_numerics")
        if contract is not None and contract != self._numerical_contract():
            raise ValueError("Warm-start HIM numerical contract differs")
        if contract is None:
            warnings.warn(
                "Legacy HIM weights lack a clipping contract; validating model statistics"
            )
        checkpoint.pop("optimizer_state_dict", None)
        checkpoint.pop("estimator_optimizer_state_dict", None)
        self.alg.load(
            checkpoint,
            {
                "actor": True,
                "critic": True,
                "estimator": True,
                "optimizer": False,
                "iteration": False,
            },
            strict=True,
        )
        self._validate_numerics()
        self.current_learning_iteration = 0
        self.env.unwrapped.common_step_counter = 0
        self.env.reset()

    def close(self):
        # The caller owns the training environment. No second simulation exists.
        pass
