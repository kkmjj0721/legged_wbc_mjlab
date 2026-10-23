# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned HIM PPO, including actor and critic mirror losses.

Copied from the repository's HIMPPO so symmetry experiments can evolve locally.

The default PPO samples come exclusively from real interactions. Optional paired
augmentation pushes forward the real behavior distribution; physical symmetry
remains an approximation. Estimator outputs are detached from the actor loss.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions.symmetry import Symmetry, resolve_symmetry_config
from rsl_rl.models import HIMActorModel, MLPModel
from rsl_rl.storage import HIMRolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets, resolve_class, resolve_obs_groups, resolve_optimizer


class HIMPPO:
    """PPO with the native HIM estimator update lifecycle."""

    def __init__(
        self,
        actor: HIMActorModel,
        critic: MLPModel,
        storage: HIMRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        estimator_learning_rate: float | None = None,
        estimator_lr_mode: str = "follow",
        estimator_lr_cap: float | None = None,
        estimator_use_mirror: bool = False,
        max_grad_norm: float = 1.0,
        estimator_max_grad_norm: float | None = None,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        grad_reduce_bucket_mb: float = 25,
        estimator_obs_groups: list[str] | None = None,
        estimator_velocity_slice: tuple[int, int] | list[int] | None = None,
        estimator_target_slice: tuple[int, int] | list[int] | None = None,
        estimator_target_slices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None,
        symmetry_cfg: dict | None = None,
        fused_optimizer: bool = False,
        reference_update: bool = False,
        kl_mode: str = "standard",
        **kwargs,
    ) -> None:
        """Initialize PPO and the HIM estimator training configuration."""
        self.use_mixed_precision = use_mixed_precision
        del kwargs

        self.device = device
        self.numerics_guard = None
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = int(multi_gpu_cfg.get("global_rank", 0)) if multi_gpu_cfg else 0
        self.gpu_world_size = int(multi_gpu_cfg.get("world_size", 1)) if multi_gpu_cfg else 1
        self.grad_reduce_bucket_mb = grad_reduce_bucket_mb

        self.actor = actor.to(device)
        self.critic = critic.to(device)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("HIMPPO currently supports feed-forward HIM actor and critic models only")
        self._raw_actor = self.actor
        self._raw_critic = self.critic
        self.estimator = self.actor.estimator
        self.fused_optimizer = fused_optimizer
        self._fused_cuda = fused_optimizer and torch.device(device).type == "cuda"
        self.symmetry = None
        self.reference_symmetry = False
        self.paired_augmentation = False
        self.critic_mirror_loss_coeff = 0.0
        self.actor_mirror_reduction = "mean"
        if symmetry_cfg is not None:
            options = dict(symmetry_cfg)
            implementation = options.pop("implementation", "ppo")
            if implementation not in ("ppo", "homie"):
                raise ValueError(f"Unknown HIM symmetry implementation: {implementation!r}")
            self.reference_symmetry = implementation == "homie"
            self.actor_mirror_reduction = options.pop("actor_mirror_reduction", "sum" if self.reference_symmetry else "mean")
            if self.actor_mirror_reduction not in ("mean", "sum"):
                raise ValueError("actor_mirror_reduction must be 'mean' or 'sum'")
            self.critic_mirror_loss_coeff = float(options.pop("critic_mirror_loss_coeff", 0.0))
            coefficient = float(options.get("mirror_loss_coeff", 0.0))
            if any(not math.isfinite(c) or c < 0 for c in (coefficient, self.critic_mirror_loss_coeff)):
                raise ValueError("Symmetry loss coefficients must be finite and nonnegative")
            self.symmetry = Symmetry(**options)
            self.paired_augmentation = not self.reference_symmetry and self.symmetry.use_data_augmentation
        self.reference_update = bool(reference_update)
        if kl_mode not in ("standard", "legacy_homie"):
            raise ValueError("kl_mode must be 'standard' or explicit reproduction mode 'legacy_homie'")
        self.kl_mode = kl_mode
        self.estimator_use_mirror = bool(estimator_use_mirror)
        if self.estimator_use_mirror and self.symmetry is None:
            raise ValueError("estimator_use_mirror requires a symmetry mapping")
        expected_streams = 2 if self.symmetry is not None and self.symmetry.use_data_augmentation else 1
        if storage.transitions_per_step != expected_streams:
            raise ValueError("HIM storage stream count does not match its symmetry configuration")
        storage.paired_mini_batches = self.paired_augmentation
        if self.paired_augmentation:
            # Gaussian std transforms below require a signed permutation, with
            # unit Jacobian determinant and no action offsets/scaling.
            width = storage.actions.shape[-1]
            basis = torch.eye(width, device=device)
            transformed = self._mirror_actions(basis)
            absolute = transformed.abs()
            if (not torch.all((absolute == 0) | (absolute == 1))
                    or not torch.all(absolute.sum(0) == 1) or not torch.all(absolute.sum(1) == 1)
                    or not torch.equal(self._mirror_actions(torch.zeros_like(basis)), torch.zeros_like(basis))
                    or not torch.equal(self._mirror_actions(transformed), basis)):
                raise ValueError("Paired PPO requires an involutive signed-permutation action mapping")
        # Keep the runner/logger contract even when RND is not configured.
        self.rnd = None
        self.intrinsic_rewards = None

        # Native HIMLoco keeps estimator parameters in the PPO optimizer.  The
        # estimator also owns a second optimizer for its HIM objective; since
        # HIMActorModel detaches estimator outputs, PPO contributes no gradient
        # to these parameters, matching the original implementation.
        optimizer_class = resolve_optimizer(optimizer)
        optimizer_options = {"fused": self._fused_cuda} if fused_optimizer else {}
        if fused_optimizer and optimizer_class is not torch.optim.Adam:
            raise ValueError("fused_optimizer requires the Adam optimizer")
        self.optimizer = optimizer_class(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=learning_rate,
            **optimizer_options,
        )
        if fused_optimizer:
            # Construct after moving the estimator parameters onto the target device.
            self.estimator.optimizer = torch.optim.Adam(
                self.estimator.parameters(), lr=self.estimator.learning_rate, fused=self._fused_cuda
            )
        if estimator_lr_mode not in ("follow", "fixed", "cap"):
            raise ValueError("estimator_lr_mode must be 'follow', 'fixed', or 'cap'")
        self.estimator_lr_mode = estimator_lr_mode
        self.estimator_lr_cap = estimator_lr_cap
        if estimator_lr_mode == "cap" and (estimator_lr_cap is None or not math.isfinite(estimator_lr_cap) or estimator_lr_cap <= 0):
            raise ValueError("Capped estimator LR requires a finite positive estimator_lr_cap")
        if estimator_learning_rate is not None:
            self.estimator.learning_rate = float(estimator_learning_rate)
            for param_group in self.estimator.optimizer.param_groups:
                param_group["lr"] = self.estimator.learning_rate
        self.estimator_fixed_learning_rate = self.estimator.learning_rate
        if not math.isfinite(self.estimator_fixed_learning_rate) or self.estimator_fixed_learning_rate <= 0:
            raise ValueError("estimator_learning_rate must be finite and positive")
        if estimator_max_grad_norm is not None:
            self.estimator.max_grad_norm = float(estimator_max_grad_norm)

        self.storage = storage
        self.transition = HIMRolloutStorage.Transition()
        self.transition_sym = HIMRolloutStorage.Transition()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1:
            raise ValueError("num_learning_epochs and num_mini_batches must be positive")
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        if isinstance(estimator_obs_groups, str):
            estimator_obs_groups = [estimator_obs_groups]
        self.estimator_obs_groups = list(estimator_obs_groups or [])
        self.estimator_velocity_slice = self._normalize_slice(estimator_velocity_slice)
        self.estimator_target_slices = self._normalize_target_slices(
            estimator_target_slice,
            estimator_target_slices,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and record the current transition."""
        if self.storage.transitions_per_step == 2:
            return self._act_with_mirror(obs)
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(param.detach() for param in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def _mirror_observations(self, obs: TensorDict) -> TensorDict:
        """Reflect a complete history or the selected next-observation groups."""
        augmented, _ = self.symmetry.data_augmentation_func(env=self.symmetry.env, obs=obs, actions=None)
        count = obs.batch_size[0]
        if augmented.batch_size[0] != 2 * count:
            raise ValueError("HIM symmetry requires exactly one left/right mirror per observation")
        return augmented[count:]

    def _act_with_mirror(self, obs: TensorDict) -> torch.Tensor:
        """Pair the real action/distribution, or explicitly reproduce legacy HOMIE."""
        if self.paired_augmentation:
            # Only this action is sampled and executed. The second stream is its
            # push-forward, never an independently sampled action with copied rewards.
            real, mirror = self.transition, self.transition_sym
            real.observations = obs
            real.actions = self.actor(obs, stochastic_output=True).detach()
            real.values = self.critic(obs).detach()
            real.actions_log_prob = self.actor.get_output_log_prob(real.actions).detach()
            real.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
            if len(real.distribution_params) != 2:
                raise ValueError("Paired HIM PPO requires a diagonal Gaussian policy")
            mean, std = real.distribution_params
            mirror.observations = self._mirror_observations(obs)
            mirror.actions = self._mirror_actions(real.actions)
            mirror.values = real.values.clone()
            mirror.actions_log_prob = real.actions_log_prob.clone()
            mirror.distribution_params = (self._mirror_actions(mean), self._mirror_actions(std).abs())
            return real.actions
        mirrored = self._mirror_observations(obs)
        all_obs = torch.cat((obs, mirrored), dim=0)
        actions = self.actor(all_obs, stochastic_output=True).detach()
        values = self.critic(all_obs).detach()
        log_prob = self.actor.get_output_log_prob(actions).detach()
        params = tuple(p.detach() for p in self.actor.output_distribution_params)
        count = obs.batch_size[0]
        for transition, indices in ((self.transition, slice(0, count)),
                                    (self.transition_sym, slice(count, None))):
            transition.observations = all_obs[indices]
            transition.actions = actions[indices]
            transition.values = values[indices]
            transition.actions_log_prob = log_prob[indices]
            transition.distribution_params = tuple(p[indices] for p in params)
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record rewards, termination flags, and the post-step observation."""
        self.transition.next_observations, self.transition.next_observations_valid = self._resolve_next_observations(
            obs, dones, extras
        )
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones.clone()
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transition(self.transition)
        if self.storage.transitions_per_step == 2:
            self.transition_sym.next_observations = self._mirror_observations(self.transition.next_observations)
            self.transition_sym.next_observations_valid = self.transition.next_observations_valid
            self.transition_sym.rewards = rewards.clone()
            self.transition_sym.dones = dones.clone()
            if "time_outs" in extras:
                self.transition_sym.rewards += self.gamma * torch.squeeze(
                    self.transition_sym.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
                )
            self.storage.add_transition(self.transition_sym)
            self.transition_sym.clear()
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute generalized advantage and return targets."""
        if self.storage.transitions_per_step == 2:
            self._compute_mirrored_returns(obs)
            return
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(self.storage.num_transitions_per_env)):
            next_values = (
                last_values
                if step == self.storage.num_transitions_per_env - 1
                else self.storage.values[step + 1]
            )
            next_is_not_terminal = 1.0 - self.storage.dones[step].float()
            delta = (
                self.storage.rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - self.storage.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            self.storage.returns[step] = advantage + self.storage.values[step]
        self.storage.advantages = self.storage.returns - self.storage.values
        if not self.normalize_advantage_per_mini_batch:
            self.storage.advantages = (self.storage.advantages - self.storage.advantages.mean()) / (
                self.storage.advantages.std() + 1e-8
            )

    def _compute_mirrored_returns(self, obs: TensorDict) -> None:
        """Pair real GAE targets; legacy HOMIE retains its independent mirror GAE."""
        if self.paired_augmentation:
            storage = self.storage
            last_values = self.critic(obs).detach()
            advantage = 0
            for step in reversed(range(storage.num_steps_per_env)):
                index = 2 * step
                next_values = last_values if step == storage.num_steps_per_env - 1 else storage.values[index + 2]
                alive = 1.0 - storage.dones[index].float()
                delta = storage.rewards[index] + alive * self.gamma * next_values - storage.values[index]
                advantage = delta + alive * self.gamma * self.lam * advantage
                storage.returns[index] = advantage + storage.values[index]
                storage.returns[index + 1] = storage.returns[index]
            storage.advantages = storage.returns - storage.values
            if not self.normalize_advantage_per_mini_batch:
                real = storage.advantages[::2]
                storage.advantages = (storage.advantages - real.mean()) / (real.std() + 1e-8)
            return
        count = self.storage.num_steps_per_env
        streams = lambda t: t.reshape(count, 2, self.storage.num_envs, 1)
        values, rewards, dones, returns = map(streams, (
            self.storage.values, self.storage.rewards, self.storage.dones, self.storage.returns,
        ))
        # Bootstrap each stream from its own critic input, including the last step.
        last_obs = torch.cat((obs, self._mirror_observations(obs)), dim=0)
        last_values = self.critic(last_obs).detach().reshape(2, self.storage.num_envs, 1)
        advantage = 0
        for step in reversed(range(count)):
            next_values = last_values if step == count - 1 else values[step + 1]
            alive = 1.0 - dones[step].float()
            delta = rewards[step] + alive * self.gamma * next_values - values[step]
            advantage = delta + alive * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]
        self.storage.advantages = self.storage.returns - self.storage.values
        if not self.normalize_advantage_per_mini_batch:
            self.storage.advantages = (self.storage.advantages - self.storage.advantages.mean()) / (
                self.storage.advantages.std() + 1e-8
            )

    def update(self) -> dict[str, float]:
        """Optimize PPO and HIM losses over the collected rollout."""
        guard = self.numerics_guard
        if guard is not None:
            guard.check("rollout", {
                "obs": self.storage.observations, "next_obs": self.storage.next_observations,
                "actions": self.storage.actions, "rewards": self.storage.rewards,
                "returns": self.storage.returns, "advantages": self.storage.advantages,
                "values": self.storage.values, "log_prob": self.storage.actions_log_prob,
            })
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        mean_symmetry_loss = 0.0
        mean_ppo_samples = 0.0
        mean_estimator_samples = 0.0
        mean_critic_symmetry_loss = 0.0
        diagnostics = torch.zeros(2, 4, device=self.device)
        if self.reference_update:
            mean_value_loss, mean_surrogate_loss, mean_entropy, mean_estimation_loss, mean_swap_loss, \
                mean_symmetry_loss, mean_estimator_samples, mean_critic_symmetry_loss = (
                    torch.zeros((), device=self.device) for _ in range(8)
                )
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    statistics = batch.advantages[~batch.is_mirror] if self.paired_augmentation else batch.advantages
                    batch.advantages = (batch.advantages - statistics.mean()) / (statistics.std() + 1e-8)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy
            if guard is not None:
                guard.check("ppo_forward", {"values": values, "log_prob": actions_log_prob, "entropy": entropy})
                guard.check("ppo_distribution", distribution_params)

            if self.desired_kl is not None and self.schedule == "adaptive":
                if self.paired_augmentation:
                    real = ~batch.is_mirror
                    self._update_learning_rate(tuple(p[real] for p in batch.old_distribution_params),
                                               tuple(p[real] for p in distribution_params))
                else:
                    self._update_learning_rate(batch.old_distribution_params, distribution_params)

            history, velocity_target, target_obs, valid_estimator_samples = self._estimator_batch(batch)
            update_estimator = torch.tensor(
                int(valid_estimator_samples.any()), dtype=torch.int32, device=self.device
            )
            if self.is_multi_gpu:
                torch.distributed.all_reduce(update_estimator, op=torch.distributed.ReduceOp.MAX)
            if bool(update_estimator.item()):
                estimation_loss, swap_loss = self.estimator.update(
                    history[valid_estimator_samples],
                    velocity_target=velocity_target[valid_estimator_samples],
                    target_obs=target_obs[valid_estimator_samples],
                    learning_rate=self._estimator_learning_rate(),
                    gradient_reducer=self._reduce_estimator_gradients if self.is_multi_gpu else None,
                    numerics_guard=guard,
                    **({"return_tensors": True} if self.reference_update else {}),
                )
            else:
                estimation_loss = 0.0
                swap_loss = 0.0

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            with torch.no_grad():
                exact_kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                for index, mask in enumerate((~batch.is_mirror, batch.is_mirror)):
                    diagnostics[index] += torch.stack((
                        exact_kl[mask].sum(), ((ratio[mask]-1).abs() > self.clip_param).float().sum(),
                        ratio[mask].sum(), mask.sum().float(),
                    ))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if self.symmetry is not None:
                # Estimator has stepped. Recompute BOTH actor sides from this
                # version; the pre-step PPO mean would be a stale target.
                symmetry_loss, critic_symmetry_loss = self._reference_symmetry_losses(batch, values)
            if self.symmetry is not None and self.symmetry.use_mirror_loss:
                loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss
            if self.symmetry is not None and self.critic_mirror_loss_coeff:
                loss = loss + self.critic_mirror_loss_coeff * critic_symmetry_loss
            if guard is not None:
                guard.check("ppo_loss", {"loss": loss, "ratio": ratio}, limit=1e8)
            self.optimizer.zero_grad()
            loss.backward()
            if guard is not None:
                guard.gradients("ppo_gradients", self._all_parameters())
            if self.is_multi_gpu:
                self.reduce_parameters()
            grad_norm = nn.utils.clip_grad_norm_(self._all_parameters(), self.max_grad_norm)
            if guard is not None:
                guard.check("ppo_gradient_norm", grad_norm, limit=1e8)
            self.optimizer.step()
            if guard is not None:
                guard.check("ppo_parameters", {
                    "parameters": list(self._all_parameters()), "optimizer": self.optimizer.state,
                })

            mean_value_loss += value_loss.detach() if self.reference_update else value_loss.item()
            mean_surrogate_loss += surrogate_loss.detach() if self.reference_update else surrogate_loss.item()
            mean_entropy += entropy.mean().detach() if self.reference_update else entropy.mean().item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss
            if self.symmetry is not None:
                mean_symmetry_loss += symmetry_loss.detach() if self.reference_update else symmetry_loss.item()
                mean_ppo_samples += batch.observations.batch_size[0]
                mean_estimator_samples += (valid_estimator_samples.sum() if self.reference_update
                                           else valid_estimator_samples.sum().item())
                mean_critic_symmetry_loss += (critic_symmetry_loss.detach() if self.reference_update
                                              else critic_symmetry_loss.item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if num_updates < 1:
            raise ValueError("num_learning_epochs and num_mini_batches must be positive")
        # Preserve this project's normalization contract: only real observations
        # update running statistics, even when the buffer includes a mirror stream.
        observations = self.storage.observations[::self.storage.transitions_per_step].flatten(0, 1)
        try:
            self.actor.update_normalization(observations)
            self.critic.update_normalization(observations)
        except FloatingPointError:
            if guard is not None:
                guard.fail("normalization", observations)
            raise
        self.storage.clear()
        if self.reference_update:
            (mean_value_loss, mean_surrogate_loss, mean_entropy, mean_estimation_loss, mean_swap_loss,
             mean_symmetry_loss, mean_estimator_samples, mean_critic_symmetry_loss) = torch.stack((
                 mean_value_loss, mean_surrogate_loss, mean_entropy, mean_estimation_loss, mean_swap_loss,
                 mean_symmetry_loss, mean_estimator_samples, mean_critic_symmetry_loss,
             )).detach().cpu().tolist()
        losses = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "estimation": mean_estimation_loss / num_updates,
            "swap": mean_swap_loss / num_updates,
            "learning_rate/ppo": self.optimizer.param_groups[0]["lr"],
            "learning_rate/estimator": self.estimator.optimizer.param_groups[0]["lr"],
        }
        if self.is_multi_gpu:
            torch.distributed.all_reduce(diagnostics)
        for name, row in zip(("real", "mirror"), diagnostics.cpu().tolist()):
            if row[3]:
                for key, value in zip(("kl", "clip_fraction", "ratio"), row[:3]):
                    losses[f"{key}/{name}"] = value / row[3]
        if self.symmetry is not None:
            losses.update({
                "symmetry": mean_symmetry_loss / num_updates,
                "symmetry_weighted": (mean_symmetry_loss / num_updates * self.symmetry.mirror_loss_coeff
                                      if self.symmetry.use_mirror_loss else 0.0),
                "ppo_samples": mean_ppo_samples / num_updates,
                "estimator_samples": mean_estimator_samples / num_updates,
            })
            losses.update({
                "critic_symmetry": mean_critic_symmetry_loss / num_updates,
                "critic_symmetry_weighted": mean_critic_symmetry_loss / num_updates * self.critic_mirror_loss_coeff,
            })
        return losses

    def _reference_symmetry_losses(self, batch, values):
        """Detached actor targets; critic and actor switches are independent."""
        obs = batch.observations if self.reference_symmetry else batch.observations[~batch.is_mirror]
        mirrored = self._mirror_observations(obs)
        with torch.no_grad():
            target = self._mirror_actions(self.actor.get_action_mean(obs))
        with torch.set_grad_enabled(self.symmetry.use_mirror_loss):
            prediction = self.actor.get_action_mean(mirrored)
            error = (prediction - target).square()
            actor_loss = error.sum(dim=-1).mean() if self.actor_mirror_reduction == "sum" else error.mean()
        critic_loss = values.new_zeros(())
        if self.critic_mirror_loss_coeff:
            with torch.no_grad():
                target_value = values.detach() if self.reference_symmetry else self.critic(obs)
            critic_loss = (self.critic(mirrored) - target_value).square().mean()
        if self.numerics_guard is not None:
            self.numerics_guard.check("symmetry_loss", {
                "actor": actor_loss, "critic": critic_loss,
            }, limit=1e8)
        return actor_loss, critic_loss

    def _mirror_actions(self, actions):
        _, augmented = self.symmetry.data_augmentation_func(env=self.symmetry.env, obs=None, actions=actions)
        if augmented.shape != (2 * actions.shape[0], actions.shape[1]):
            raise ValueError("HIM action augmentation requires exactly one original and one mirror")
        return augmented[actions.shape[0]:]

    def _estimator_learning_rate(self):
        if self.estimator_lr_mode == "fixed":
            return self.estimator_fixed_learning_rate
        if self.estimator_lr_mode == "cap":
            return min(self.learning_rate, self.estimator_lr_cap)
        return self.learning_rate

    def _estimator_batch(self, batch):
        observations, targets = batch.observations, batch.next_observations
        valid = batch.next_observations_valid.reshape(-1).bool()
        if self.storage.transitions_per_step == 2 and not self.estimator_use_mirror:
            real = ~batch.is_mirror
            observations, targets, valid = observations[real], targets[real], valid[real]
        elif self.storage.transitions_per_step == 1 and self.estimator_use_mirror:
            observations = torch.cat((observations, self._mirror_observations(observations)), dim=0)
            targets = torch.cat((targets, self._mirror_observations(targets)), dim=0)
            valid = valid.repeat(2)
        history = self.actor.obs_normalizer(self.actor.get_observation_tensor(observations))
        velocity, target = self._get_estimator_targets(targets)
        return history, velocity, target, valid

    def train_mode(self) -> None:
        """Set all HIM and PPO modules to training mode."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Set all HIM and PPO modules to evaluation mode."""
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """Return actor, critic, estimator, and optimizer states."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "estimator_state_dict": self.estimator.state_dict(),
            "estimator_optimizer_state_dict": self.estimator.optimizer.state_dict(),
            "him_training_contract": self.training_contract(),
            "him_learning_rates": {
                "ppo": self.optimizer.param_groups[0]["lr"],
                "estimator": self.estimator.optimizer.param_groups[0]["lr"],
            },
        }

    def training_contract(self) -> dict:
        """Effective training choices, independent of mutable optimizer rates."""
        symmetry = self.symmetry
        mapping = None if symmetry is None else symmetry.data_augmentation_func
        return {
            "version": 1,
            "ppo_data": ("paired" if self.paired_augmentation else "legacy_independent"
                         if self.storage.transitions_per_step == 2 else "real"),
            "symmetry_implementation": "homie" if self.reference_symmetry else "ppo",
            "mapping": None if mapping is None else f"{mapping.__module__}:{mapping.__qualname__}",
            "actor_mirror_enabled": bool(symmetry and symmetry.use_mirror_loss),
            "actor_mirror_coefficient": symmetry.mirror_loss_coeff if symmetry and symmetry.use_mirror_loss else 0.0,
            "actor_mirror_reduction": self.actor_mirror_reduction,
            "critic_mirror_coefficient": self.critic_mirror_loss_coeff,
            "estimator_use_mirror": self.estimator_use_mirror,
            "estimator_lr_mode": self.estimator_lr_mode,
            "estimator_learning_rate": self.estimator_fixed_learning_rate,
            "estimator_lr_cap": self.estimator_lr_cap,
            "reference_update": self.reference_update,
            "kl_mode": self.kl_mode,
            "optimizer": type(self.optimizer).__name__,
            "fused_optimizer": self.fused_optimizer,
            "real_steps_per_rollout": self.storage.num_steps_per_env,
            "num_learning_epochs": self.num_learning_epochs,
            "num_mini_batches": self.num_mini_batches,
            "normalize_advantage_per_mini_batch": self.normalize_advantage_per_mini_batch,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load selected model and optimizer states from a checkpoint."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "estimator": True, "iteration": True}
        # Symmetry modes need a strict sample-source/training contract. Ordinary
        # HIM keeps support for historical checkpoints without this metadata.
        check_training_contract = self.symmetry is not None
        if check_training_contract and (load_cfg.get("optimizer") or load_cfg.get("iteration")):
            if loaded_dict.get("him_training_contract") != self.training_contract():
                raise ValueError(
                    "HIM training contract missing or different; use weights-only warm-start "
                    "with fresh optimizers when changing training modes or importing legacy checkpoints"
                )
        if check_training_contract and load_cfg.get("optimizer"):
            owners = [("optimizer_state_dict", self.optimizer)]
            if load_cfg.get("estimator"):
                owners.append(("estimator_optimizer_state_dict", self.estimator.optimizer))
            for key, optimizer in owners:
                saved_groups = loaded_dict.get(key, {}).get("param_groups", [])
                if ([len(g.get("params", [])) for g in saved_groups]
                        != [len(g["params"]) for g in optimizer.param_groups]):
                    raise ValueError(f"Missing or incompatible {key}; use weights-only warm-start")
        if load_cfg.get("actor"):
            self._load_actor_state(loaded_dict, strict, load_cfg.get("estimator", True))
        if load_cfg.get("critic"):
            self._load_critic_state(loaded_dict, strict)
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            optimizer_state = loaded_dict["optimizer_state_dict"]
            # Older ordinary HIM optimizers excluded estimator parameters.
            # Preserve their existing fallback to a fresh PPO optimizer; the
            # symmetry path has already rejected incompatible state above.
            saved_groups = optimizer_state.get("param_groups", [])
            saved_count = len(saved_groups[0].get("params", [])) if saved_groups else 0
            if len(self.optimizer.param_groups[0]["params"]) == saved_count:
                self.optimizer.load_state_dict(optimizer_state)
                if self.fused_optimizer:
                    for group in self.optimizer.param_groups:
                        group["fused"] = self._fused_cuda
                self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("estimator"):
            self._load_estimator_state(loaded_dict, strict)
            if load_cfg.get("optimizer") and "estimator_optimizer_state_dict" in loaded_dict:
                self.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
                self.estimator.learning_rate = self.estimator.optimizer.param_groups[0]["lr"]
                if self.fused_optimizer:
                    for group in self.estimator.optimizer.param_groups:
                        group["fused"] = self._fused_cuda
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> HIMActorModel:
        """Return the HIM actor used for inference."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Leave HIM uncompiled because its TensorDict estimator is stateful."""
        del mode

    def broadcast_parameters(self) -> None:
        """Broadcast actor, critic, and estimator parameters in distributed training."""
        if not self.is_multi_gpu:
            return
        for model in (self._raw_actor, self._raw_critic):
            for tensor in model.state_dict().values():
                torch.distributed.broadcast(tensor, src=0)

    def reduce_parameters(self) -> None:
        """Average PPO and estimator gradients across distributed workers."""
        if self.is_multi_gpu:
            reduce_gradients_in_buckets(
                self._all_parameters(),
                self.gpu_world_size,
                self.grad_reduce_bucket_mb,
            )

    def _reduce_estimator_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
        """Average estimator gradients across distributed workers."""
        reduce_gradients_in_buckets(parameters, self.gpu_world_size, self.grad_reduce_bucket_mb)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "HIMPPO":
        """Construct HIMPPO from the same configuration contract as modern PPO."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        alg_cfg = resolve_symmetry_config(alg_cfg, env)
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        actor: HIMActorModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        if not alg_cfg.get("estimator_obs_groups"):
            alg_cfg["estimator_obs_groups"] = list(cfg["obs_groups"]["critic"])
        elif isinstance(alg_cfg["estimator_obs_groups"], str):
            alg_cfg["estimator_obs_groups"] = [alg_cfg["estimator_obs_groups"]]
        storage = HIMRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device,
            next_observation_groups=alg_cfg["estimator_obs_groups"],
            transitions_per_step=(2 if alg_cfg.get("symmetry_cfg") is not None
                                  and alg_cfg["symmetry_cfg"].get("use_data_augmentation") else 1),
        )
        algorithm = alg_class(
            actor,
            critic,
            storage,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **alg_cfg,
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def _all_parameters(self) -> Iterable[nn.Parameter]:
        """Yield all actor/critic parameters, including the HIM estimator."""
        return chain(self.actor.parameters(), self.critic.parameters())

    def _get_estimator_targets(self, next_observations: TensorDict | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Build velocity and one-step targets from post-step observation groups."""
        if next_observations is None:
            raise ValueError("HIM batches must contain next_observations")
        groups = self.estimator_obs_groups
        if not groups:
            groups = self.actor.obs_groups
        next_critic_obs = torch.cat([next_observations[group] for group in groups], dim=-1)
        velocity_slice = self.estimator_velocity_slice or (
            self.actor.num_one_step_obs,
            self.actor.num_one_step_obs + 3,
        )
        target_slices = self.estimator_target_slices or ((3, self.actor.num_one_step_obs + 3),)
        velocity_target = self._slice_features(next_critic_obs, velocity_slice)
        target_obs = torch.cat([self._slice_features(next_critic_obs, slc) for slc in target_slices], dim=-1)
        if velocity_target.shape[-1] != 3 or target_obs.shape[-1] != self.actor.num_one_step_obs:
            raise ValueError(
                "Invalid HIM estimator target configuration: expected velocity width 3 and target width "
                f"{self.actor.num_one_step_obs}, got {velocity_target.shape[-1]} and {target_obs.shape[-1]}"
            )
        return velocity_target, target_obs

    def _update_learning_rate(self, old_params, new_params) -> None:
        """Adapt PPO learning rate from the Gaussian KL divergence."""
        with torch.inference_mode():
            if self.kl_mode == "legacy_homie":
                old_mean, old_std = old_params
                mean, std = new_params
                kl = torch.sum(
                    torch.log(std / old_std + 1e-5)
                    + (old_std.square() + (old_mean - mean).square()) / (2.0 * std.square()) - 0.5,
                    dim=-1,
                )
            else:
                kl = self.actor.get_kl_divergence(old_params, new_params)
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                learning_rate = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(learning_rate, src=0)
                self.learning_rate = learning_rate.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    def _resolve_next_observations(
        self,
        obs: TensorDict,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> tuple[TensorDict, torch.Tensor]:
        """Resolve true post-step observations and the samples valid for estimator training."""
        # Storage copies these tensors before the runner resets any environments.
        # Keep only the estimator inputs and avoid cloning the actor's history.
        next_observations = obs.select(*self.storage.next_observation_groups)
        terminal_observations = self._get_terminal_observations(extras)
        if terminal_observations is obs:
            # The manual-reset runner supplies the unmodified post-step batch.
            # Every row is valid, including true terminal states; no mask/sync
            # or intermediate copy is needed before storage.add_transition().
            valid = torch.ones((dones.numel(), 1), dtype=torch.bool, device=self.device)
            return next_observations, valid

        done_mask = dones.reshape(-1).bool().to(self.device)
        valid = (~done_mask).view(-1, 1)
        if terminal_observations is None or not done_mask.any():
            return next_observations, valid

        # Auto-reset adapters may provide separate full-batch or done-only
        # terminal observations. Do not overwrite their post-reset input batch.
        next_observations = next_observations.clone()
        terminal_observations = terminal_observations.to(self.device)
        copied = False
        required_groups = self.estimator_obs_groups or self.actor.obs_groups
        terminal_batch = terminal_observations.batch_size[0] if terminal_observations.batch_size else 0
        for group, value in next_observations.items():
            if group not in terminal_observations:
                continue
            terminal_value = terminal_observations[group]
            if terminal_batch == next_observations.batch_size[0]:
                value[done_mask] = terminal_value[done_mask]
            elif terminal_batch == int(done_mask.sum().item()):
                value[done_mask] = terminal_value
            else:
                raise ValueError(
                    "terminal observations must contain either all environments or only done environments, got "
                    f"batch {terminal_batch} for {next_observations.batch_size[0]} envs"
                )
            copied = True
        if copied and all(group in terminal_observations for group in required_groups):
            valid[done_mask] = True
        return next_observations, valid

    @staticmethod
    def _get_terminal_observations(extras: dict) -> TensorDict | None:
        """Return terminal observations from common VecEnv extras keys, if present."""
        for key in ("terminal_observations", "terminal_observation", "final_observations", "final_observation"):
            value = extras.get(key)
            if isinstance(value, TensorDict):
                return value
            if isinstance(value, dict):
                tensors = list(value.values())
                if tensors and all(isinstance(item, torch.Tensor) for item in tensors):
                    return TensorDict(value, batch_size=[tensors[0].shape[0]])
        return None

    @staticmethod
    def _slice_features(tensor: torch.Tensor, slc: tuple[int, int]) -> torch.Tensor:
        start, stop = slc
        if start < 0 or stop <= start or stop > tensor.shape[-1]:
            raise ValueError(f"Invalid slice {(start, stop)} for tensor width {tensor.shape[-1]}")
        return tensor[:, start:stop]

    @staticmethod
    def _normalize_slice(value: tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"Expected a two-element slice, got {value}")
        return int(value[0]), int(value[1])

    @classmethod
    def _normalize_target_slices(
        cls,
        target_slice: tuple[int, int] | list[int] | None,
        target_slices: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None,
    ) -> tuple[tuple[int, int], ...] | None:
        if target_slices is not None:
            return tuple(cls._normalize_slice(slc) for slc in target_slices)  # type: ignore[arg-type]
        if target_slice is not None:
            if len(target_slice) == 2 and all(isinstance(item, int) for item in target_slice):
                return (cls._normalize_slice(target_slice),)  # type: ignore[return-value]
            return tuple(cls._normalize_slice(slc) for slc in target_slice)  # type: ignore[arg-type]
        return None

    def _load_actor_state(self, loaded_dict: dict, strict: bool, load_estimator: bool = True) -> None:
        if "actor_state_dict" in loaded_dict:
            state_dict = loaded_dict["actor_state_dict"]
            if not load_estimator:
                state_dict = {
                    key: value for key, value in state_dict.items() if not key.startswith("estimator.")
                }
            self._raw_actor.load_state_dict(state_dict, strict=strict if load_estimator else False)
            return
        if "model_state_dict" not in loaded_dict:
            if strict:
                raise KeyError("actor_state_dict")
            return
        state_dict = self._legacy_actor_state_dict(loaded_dict["model_state_dict"], load_estimator)
        self._raw_actor.load_state_dict(state_dict, strict=False if state_dict else strict)

    def _load_critic_state(self, loaded_dict: dict, strict: bool) -> None:
        if "critic_state_dict" in loaded_dict:
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
            return
        if "model_state_dict" not in loaded_dict:
            if strict:
                raise KeyError("critic_state_dict")
            return
        state_dict = self._legacy_critic_state_dict(loaded_dict["model_state_dict"])
        self._raw_critic.load_state_dict(state_dict, strict=False if state_dict else strict)

    def _load_estimator_state(self, loaded_dict: dict, strict: bool) -> None:
        if "estimator_state_dict" in loaded_dict:
            self.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=strict)
            return
        if "model_state_dict" not in loaded_dict:
            return
        state_dict = self._legacy_prefixed_state_dict(loaded_dict["model_state_dict"], "estimator.", "")
        if state_dict:
            self.estimator.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _legacy_prefixed_state_dict(state_dict: dict, old_prefix: str, new_prefix: str) -> dict:
        return {
            f"{new_prefix}{key[len(old_prefix):]}": value
            for key, value in state_dict.items()
            if key.startswith(old_prefix)
        }

    def _legacy_actor_state_dict(self, state_dict: dict, load_estimator: bool = True) -> dict:
        actor_state = self._legacy_prefixed_state_dict(state_dict, "actor.", "mlp.")
        if load_estimator:
            actor_state.update(self._legacy_prefixed_state_dict(state_dict, "estimator.", "estimator."))
        if "std" in state_dict and hasattr(self._raw_actor.distribution, "std_param"):
            actor_state["distribution.std_param"] = state_dict["std"]
        return actor_state

    def _legacy_critic_state_dict(self, state_dict: dict) -> dict:
        return self._legacy_prefixed_state_dict(state_dict, "critic.", "mlp.")
