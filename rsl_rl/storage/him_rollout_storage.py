# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage with next observations required by the HIM estimator."""

from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
    """Feed-forward RL storage that keeps the observation after every transition."""

    class Transition(RolloutStorage.Transition):
        """Transition with the post-step TensorDict required by the estimator."""

        def __init__(self) -> None:
            super().__init__()
            self.next_observations: TensorDict | None = None

        def clear(self) -> None:
            """Reset all transition fields, including the next observation."""
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        """Allocate the standard rollout fields and a next-observation buffer."""
        if training_type != "rl":
            raise ValueError("HIMRolloutStorage only supports reinforcement-learning rollouts")
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        self.next_observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )

    def add_transition(self, transition: Transition) -> None:
        """Store a transition and its post-step observation."""
        next_observations = getattr(transition, "next_observations", None)
        if next_observations is None:
            raise ValueError("HIM transitions must provide next_observations")
        super().add_transition(transition)
        self.next_observations[self.step - 1].copy_(next_observations)

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Yield shuffled batches containing both current and next observations."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size < 1:
            raise ValueError("num_mini_batches cannot exceed the rollout batch size")
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]
                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    next_observations=next_observations[batch_idx],
                )
                yield batch
