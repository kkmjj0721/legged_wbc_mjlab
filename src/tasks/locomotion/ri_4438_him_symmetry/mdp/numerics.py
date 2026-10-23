"""Check observations before clipping, including pre-reset estimator targets."""

from copy import deepcopy

import torch

from src.config.ri_4438.numerics import ACTION_CLIP, OBSERVATION_CLIP, RAW_OBSERVATION_ABORT


def checked_observation(env, *, source_func, numerics_tag, raw_limit, **params):
  value = source_func(env, **params)
  bad = (~torch.isfinite(value) | (value.abs() > raw_limit)).flatten(1).any(1)
  if not hasattr(env, "_him_numerics_bad"):
    env._him_numerics_bad = torch.zeros(env.num_envs, dtype=torch.bool, device=value.device)
    env._him_numerics_evidence = {}
  env._him_numerics_bad.logical_or_(bad)
  # No host synchronization and no collectives in reset hooks: ranks can reset
  # different numbers of episodes. Keep evidence until the runner consumes it.
  previous = env._him_numerics_evidence.get(numerics_tag)
  if previous is None:
    previous = torch.zeros_like(value)
  mask = bad.reshape((-1,) + (1,) * (value.ndim - 1))
  env._him_numerics_evidence[numerics_tag] = torch.where(mask, value, previous)
  return value


def configure_numerical_observations(cfg):
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    # Base actor/critic dictionaries share term objects. Own each copy before
    # adding the wrapper, otherwise a shared term would wrap itself twice.
    group.terms = {name: deepcopy(term) for name, term in group.terms.items()}
    for name, term in group.terms.items():
      bound = ACTION_CLIP if name == "actions" else OBSERVATION_CLIP
      term.clip = (-bound, bound)
      term.params = {
        **term.params, "source_func": term.func,
        "numerics_tag": f"{group_name}/{name}", "raw_limit": RAW_OBSERVATION_ABORT,
      }
      term.func = checked_observation
