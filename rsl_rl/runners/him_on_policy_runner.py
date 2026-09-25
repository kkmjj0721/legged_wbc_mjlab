# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner for the TensorDict HIM algorithm.

Unlike the legacy HIMLoco runner, this class consumes the current rsl_rl
``VecEnv`` contract. Its special responsibility is handling environments
configured with ``auto_reset=False``: the terminal observation is handed to the
algorithm first, then completed environments are partially reset for the next
rollout action.
"""

from __future__ import annotations

import os
import time

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import HIMPPO
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.numerics import TrainingNumericsGuard


class HIMOnPolicyRunner(OnPolicyRunner):
    """Run HIMPPO while preserving true terminal observations."""

    alg: HIMPPO

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Collect rollouts, reset done environments, and update HIMPPO."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        unwrapped = getattr(self.env, "unwrapped", None)
        manual_reset = not getattr(getattr(unwrapped, "cfg", None), "auto_reset", True)
        guard = None
        if self.cfg.get("numerics"):
            guard = TrainingNumericsGuard(self.device, self.logger.log_dir, **self.cfg["numerics"])
            self.alg.numerics_guard = guard

        def check_observations(stage, values):
            if guard is None:
                return
            raw_bad = getattr(unwrapped, "_him_numerics_bad", None)
            evidence = getattr(unwrapped, "_him_numerics_evidence", {})
            guard.check(stage, {"observations": values, "pre_clip": evidence},
                        limit=guard.cfg["raw_observation_abort"], extra_bad=raw_bad)
            if raw_bad is not None:
                raw_bad.zero_()
                for value in evidence.values():
                    value.zero_()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            action_peak = torch.zeros((), device=self.device)
            action_counts = torch.zeros(2, device=self.device)
            with torch.inference_mode():
                for step in range(self.cfg["num_steps_per_env"]):
                    if guard is not None:
                        guard.iteration, guard.step = it, step
                        guard.remember(observations=obs)
                    check_observations("before_action", obs)
                    actions = self.alg.act(obs)
                    if guard is not None:
                        guard.remember(actions=actions)
                        guard.check("raw_action", {"actions": actions, "distribution": self.alg.actor.output_distribution_params},
                                    limit=guard.cfg["raw_action_abort"])
                        action_peak = torch.maximum(action_peak, actions.abs().max())
                        action_counts[0] += (actions.abs() > self.alg.actor.action_clip).sum()
                        action_counts[1] += actions.numel()
                    next_obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    check_observations("after_step", {
                        "next": next_obs, "terminal": extras.get("terminal_observations"),
                    })
                    if guard is not None:
                        guard.check("step_result", {"rewards": rewards, "dones": dones})
                    elif self.cfg.get("check_for_nan", True):
                        from rsl_rl.utils import check_nan

                        check_nan(next_obs, rewards, dones)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Keep a copy because ``reset()`` replaces the environment's
                    # log dictionary when auto_reset is disabled.
                    step_extras = dict(extras)
                    step_log = dict(step_extras.get("log") or {})

                    # HIMPPO stores next_obs before reset. With auto_reset=False
                    # this is the true terminal physics state for done envs.
                    if manual_reset:
                        step_extras["terminal_observations"] = next_obs
                    self.alg.process_env_step(next_obs, rewards, dones, step_extras)
                    reset_extras: dict = {}
                    if manual_reset:
                        reset_ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
                        if reset_ids.numel() > 0:
                            next_obs, reset_extras = self._reset_done_envs(next_obs, reset_ids)
                    if manual_reset:
                        check_observations("after_reset", next_obs)
                    # With auto_reset=False, MjLab emits Episode_Reward/*,
                    # Episode_Metrics/*, and Episode_Termination/* from the
                    # explicit reset call rather than from env.step(). Merge
                    # those values before handing the step to Logger.
                    log_extras = dict(step_extras)
                    reset_log = reset_extras.get("log")
                    if isinstance(reset_log, dict):
                        step_log.update(reset_log)
                    log_extras["log"] = step_log
                    self.logger.process_env_step(rewards, dones, log_extras, self.alg.intrinsic_rewards)
                    obs = next_obs

                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            if guard is not None:
                # These buffers were created outside inference_mode and can
                # safely be reduced once per rollout, including with NCCL.
                action_peak = action_peak.clone()
                if self.is_distributed:
                    torch.distributed.all_reduce(action_peak, op=torch.distributed.ReduceOp.MAX)
                    torch.distributed.all_reduce(action_counts)
                loss_dict["numerics/raw_action_max"] = action_peak.item()
                loss_dict["numerics/action_clip_fraction"] = (action_counts[0] / action_counts[1].clamp_min(1)).item()
                policy = self.alg.get_policy()
                if policy.obs_normalization and policy.action_observation_slice is not None:
                    slice_start, slice_stop = policy.action_observation_slice
                    action_std = policy.obs_normalizer.std.reshape(policy.history_size, -1)[:, slice_start:slice_stop]
                    loss_dict["numerics/last_action_std_max"] = action_std.max().item()
            learn_time = time.time() - start
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

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def _reset_done_envs(
        self, terminal_obs: TensorDict, reset_ids: torch.Tensor
    ) -> tuple[TensorDict, dict]:
        """Merge reset rows in place after process_env_step has copied the transition.

        Non-reset rows keep their original noisy/delayed observations. Returning
        the complete reset() batch would resample their stateless observations.
        """
        unwrapped = getattr(self.env, "unwrapped", None)
        if unwrapped is None or not hasattr(unwrapped, "reset"):
            raise RuntimeError(
                "HIM training requires an environment exposing reset(env_ids=...) "
                "when auto_reset=False."
            )
        if getattr(getattr(unwrapped, "cfg", None), "auto_reset", False):
            # Auto-reset environments have already replaced terminal entries
            # with the next episode's initial observation.
            return terminal_obs, {}

        reset_result = unwrapped.reset(env_ids=reset_ids.to(unwrapped.device))
        reset_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        reset_extras = reset_result[1] if isinstance(reset_result, tuple) else {}
        reset_obs = TensorDict(reset_obs, batch_size=[self.env.num_envs], device=self.device)
        for key in terminal_obs.keys():
            if key in reset_obs:
                terminal_obs[key][reset_ids] = reset_obs[key].to(self.device)[reset_ids]
        return terminal_obs, reset_extras
