# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM-enhanced PPO implementation for the modern rsl_rl API."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import HIMActorModel, MLPModel
from rsl_rl.storage import HIMRolloutStorage
from rsl_rl.utils import reduce_gradients_in_buckets, resolve_class, resolve_obs_groups, resolve_optimizer


class HIMPPO:
    """Proximal policy optimization with a separately trained HIM estimator."""

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
        **kwargs,
    ) -> None:
        """Initialize PPO and the HIM estimator training configuration."""
        self.use_mixed_precision = use_mixed_precision
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            print(f"HIMPPO.__init__ got unexpected arguments, which will be ignored: {unexpected}")

        self.device = device
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
        # Keep the runner/logger contract even when RND is not configured.
        self.rnd = None
        self.intrinsic_rewards = None

        estimator_params = list(self.estimator.parameters())
        estimator_param_ids = {id(param) for param in estimator_params}
        ppo_params = [
            param
            for param in chain(self.actor.parameters(), self.critic.parameters())
            if id(param) not in estimator_param_ids
        ]
        self.optimizer = resolve_optimizer(optimizer)(ppo_params, lr=learning_rate)
        if estimator_learning_rate is not None:
            self.estimator.learning_rate = float(estimator_learning_rate)
            for param_group in self.estimator.optimizer.param_groups:
                param_group["lr"] = self.estimator.learning_rate
        if estimator_max_grad_norm is not None:
            self.estimator.max_grad_norm = float(estimator_max_grad_norm)

        self.storage = storage
        self.transition = HIMRolloutStorage.Transition()
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
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
        self.estimator_velocity_slice = tuple(estimator_velocity_slice) if estimator_velocity_slice else None
        self.estimator_target_slice = tuple(estimator_target_slice) if estimator_target_slice else None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and record the current transition."""
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(param.detach() for param in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record rewards, termination flags, and the post-step observation."""
        self.transition.next_observations = obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute generalized advantage and return targets."""
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

    def update(self) -> dict[str, float]:
        """Optimize PPO and HIM losses over the collected rollout."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimation_loss = 0.0
        mean_swap_loss = 0.0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                self._update_learning_rate(batch.old_distribution_params, distribution_params)

            history = self.actor.get_observation_tensor(batch.observations)
            history = self.actor.obs_normalizer(history)
            next_critic_obs = self._get_estimator_target(batch.next_observations)
            estimation_loss, swap_loss = self.estimator.update(
                history,
                next_critic_obs,
                velocity_slice=self.estimator_velocity_slice,
                target_slice=self.estimator_target_slice,
                gradient_reducer=self._reduce_estimator_gradients if self.is_multi_gpu else None,
            )

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
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
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self._ppo_parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_estimation_loss += estimation_loss
            mean_swap_loss += swap_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        if num_updates < 1:
            raise ValueError("num_learning_epochs and num_mini_batches must be positive")
        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "estimation": mean_estimation_loss / num_updates,
            "swap": mean_swap_loss / num_updates,
        }

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
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load selected model and optimizer states from a checkpoint."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "estimator": True, "iteration": True}
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("estimator") and "estimator_state_dict" in loaded_dict:
            self.estimator.load_state_dict(loaded_dict["estimator_state_dict"], strict=strict)
            if "estimator_optimizer_state_dict" in loaded_dict:
                self.estimator.optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
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
                self._ppo_parameters(),
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
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])
        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        actor: HIMActorModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        storage = HIMRolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        if not alg_cfg.get("estimator_obs_groups"):
            alg_cfg["estimator_obs_groups"] = list(cfg["obs_groups"]["critic"])
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

    def _ppo_parameters(self) -> Iterable[nn.Parameter]:
        """Yield parameters owned by the PPO optimizer, excluding estimator weights."""
        estimator_param_ids = {id(param) for param in self.estimator.parameters()}
        return (
            param
            for param in chain(self.actor.parameters(), self.critic.parameters())
            if id(param) not in estimator_param_ids
        )

    def _get_estimator_target(self, next_observations: TensorDict | None) -> torch.Tensor:
        """Concatenate configured next-observation groups for estimator supervision."""
        if next_observations is None:
            raise ValueError("HIM batches must contain next_observations")
        groups = self.estimator_obs_groups
        if not groups:
            groups = self.actor.obs_groups
        return torch.cat([next_observations[group] for group in groups], dim=-1)

    def _update_learning_rate(self, old_params, new_params) -> None:
        """Adapt PPO learning rate from the Gaussian KL divergence."""
        with torch.inference_mode():
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