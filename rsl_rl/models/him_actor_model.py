# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HIM policy model compatible with the modern rsl_rl model API."""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HIMEstimator, MLP
from rsl_rl.modules.distribution import Distribution, GaussianDistribution
from rsl_rl.utils import resolve_class


class HIMActorModel(nn.Module):
    """Actor that conditions a one-step policy on HIM velocity and latent estimates."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        init_noise_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
        num_one_step_obs: int | None = None,
        num_one_step_observations: int | None = None,
        history_size: int | None = None,
        estimator_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the HIM actor from TensorDict observation groups."""
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            print(f"HIMActorModel.__init__ got unexpected arguments, which will be ignored: {unexpected}")
        if obs_set not in obs_groups:
            raise KeyError(f"Observation set '{obs_set}' is not present in obs_groups")

        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = self._get_obs_dim(obs)
        if num_one_step_obs is None:
            num_one_step_obs = num_one_step_observations
        if num_one_step_obs is None:
            raise ValueError("HIMActorModel requires num_one_step_obs")
        self.num_one_step_obs = int(num_one_step_obs)
        if self.obs_dim % self.num_one_step_obs != 0:
            raise ValueError(
                f"HIM history dimension {self.obs_dim} is not divisible by num_one_step_obs "
                f"{self.num_one_step_obs}"
            )
        inferred_history_size = self.obs_dim // self.num_one_step_obs
        self.history_size = int(history_size or inferred_history_size)
        if self.history_size != inferred_history_size:
            raise ValueError("history_size does not match the configured actor history dimension")

        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()

        estimator_cfg = copy.deepcopy(estimator_cfg or {})
        estimator_cfg.pop("class_name", None)
        estimator_cfg.setdefault("temporal_steps", self.history_size)
        estimator_cfg.setdefault("num_one_step_obs", self.num_one_step_obs)
        if estimator_cfg["temporal_steps"] != self.history_size:
            raise ValueError("estimator temporal_steps must match the actor history size")
        if estimator_cfg["num_one_step_obs"] != self.num_one_step_obs:
            raise ValueError("estimator num_one_step_obs must match the actor one-step observation dimension")
        self.estimator = HIMEstimator(**estimator_cfg)

        actor_input_dim = self.num_one_step_obs + 3 + self.estimator.num_latent
        if distribution_cfg is None:
            self.distribution: Distribution = GaussianDistribution(
                output_dim,
                init_std=init_noise_std,
                std_range=std_range,
                std_type=std_type,
                learn_std=learn_std,
            )
        else:
            distribution_class, distribution_args = resolve_class(distribution_cfg)
            self.distribution = distribution_class(output_dim, **distribution_args)
        self.mlp = MLP(self._actor_input_dim(actor_input_dim), self.distribution.input_dim, hidden_dims, activation)
        self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Compute deterministic or sampled actions from a TensorDict observation."""
        del masks, hidden_state
        latent = self.get_latent(obs)
        mlp_output = self.mlp(latent)
        if stochastic_output:
            self.distribution.update(mlp_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_output)

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        """Build the policy input from current history and HIM estimates."""
        del masks, hidden_state
        obs_history = self.get_observation_tensor(obs)
        obs_history = self.obs_normalizer(obs_history)
        with torch.no_grad():
            velocity, latent = self.estimator(obs_history)
        one_step_obs = obs_history[..., : self.num_one_step_obs]
        return torch.cat((one_step_obs, velocity, latent), dim=-1)

    def get_observation_tensor(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate the configured actor observation groups."""
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
        """Reset policy state; HIM is feed-forward and has no recurrent state."""
        del dones, hidden_state

    def get_hidden_state(self):
        """Return the recurrent state (always ``None`` for HIM)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent state; this is a no-op for HIM."""
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the current action mean."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the current action standard deviation."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return entropy summed over action dimensions."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return parameters needed to reconstruct the current distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute action log probability under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        """Compute KL divergence between old and new action distributions."""
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update actor observation statistics when normalization is enabled."""
        if self.obs_normalization:
            self.obs_normalizer.update(self.get_observation_tensor(obs))

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Return deterministic actions for deployment."""
        return self(obs, stochastic_output=False)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-friendly deterministic policy wrapper."""
        return _TorchHIMActorModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX wrapper with a pre-concatenated history tensor input."""
        del verbose
        return _OnnxHIMActorModel(self)

    def _get_obs_dim(self, obs: TensorDict) -> int:
        """Validate configured actor groups and compute their flattened dimension."""
        obs_dim = 0
        for group in self.obs_groups:
            if group not in obs:
                raise KeyError(f"Observation group '{group}' is not present")
            if obs[group].ndim != 2:
                raise ValueError(f"HIMActorModel only supports 1D observations, got {obs[group].shape}")
            obs_dim += obs[group].shape[-1]
        return obs_dim

    @staticmethod
    def _actor_input_dim(input_dim: int) -> int:
        """Keep actor input dimension explicit for readable model construction."""
        return input_dim


class HIMActorCritic(HIMActorModel):
    """Backward-compatible name for the HIM actor model."""


class _TorchHIMActorModel(nn.Module):
    """Deterministic TorchScript export wrapper for HIM."""

    def __init__(self, model: HIMActorModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.estimator_encoder = copy.deepcopy(model.estimator.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = copy.deepcopy(model.distribution.as_deterministic_output_module())
        self.num_one_step_obs = model.num_one_step_obs
        self.num_latent = model.estimator.num_latent

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Compute deterministic actions from a concatenated history tensor."""
        obs_history = self.obs_normalizer(obs_history)
        parts = self.estimator_encoder(obs_history)
        velocity = parts[..., :3]
        latent = torch.nn.functional.normalize(parts[..., 3:], dim=-1, p=2.0)
        actor_input = torch.cat((obs_history[..., : self.num_one_step_obs], velocity, latent), dim=-1)
        return self.deterministic_output(self.mlp(actor_input))

    @torch.jit.export
    def reset(self) -> None:
        """Reset export state (no-op)."""
        pass


class _OnnxHIMActorModel(_TorchHIMActorModel):
    """ONNX export wrapper with standard metadata used by the runner."""

    input_names = ["obs_history"]
    output_names = ["actions"]

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return a dummy history input for ONNX tracing."""
        return (torch.zeros(1, self.estimator_encoder[0].in_features),)
