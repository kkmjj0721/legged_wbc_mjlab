"""Evaluation diagnostics; these scores do not replace episodic-return LP."""

from __future__ import annotations

import numpy as np
import torch


def reward_terms_for_step(env):
    """Read MjLab 1.6's weighted reward rates and apply dt exactly once.

    Read immediately after step(), before any explicit reset. Using per-step
    contributions also preserves totals across boundary-reset segments.
    """
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    return env.reward_manager._step_reward.double() * scale


def relative_tracking_error(actual, commands, floors):
    """Project convention for changing commands, including near-zero commands.

    Clamp the episode's mean later, not each sample: isolated overshoots still
    count. Floors have velocity units (m/s, m/s, rad/s).
    """
    return (actual - commands).abs() / torch.maximum(commands.abs(), floors)


def epte_sp(relative_error_sum, alive_steps, horizon_steps):
    """Paper Eq. (8), with failure-step samples excluded from the alive portion.

    A physical failure on the first transition has zero alive steps and score
    one. A full time-limit episode has K alive steps and no stability penalty.
    Boundary resets consume no extra steps and do not end the trial.
    """
    error = np.asarray(relative_error_sum, dtype=np.float64)
    if (
        horizon_steps <= 0
        or not 0 <= alive_steps <= horizon_steps
        or error.shape != (3,)
        or not np.isfinite(error).all()
        or (error < 0).any()
    ):
        raise ValueError("Invalid EPTE-SP trial")
    # min(sum, k) == clip(mean, 0, 1) * k, also well-defined when k == 0.
    return (np.minimum(error, alive_steps) + horizon_steps - alive_steps) / horizon_steps


def annotate_progress(result, previous_trial_returns):
    """Add paired-trial uncertainty without changing LP or sampling scores.

    Pair by canonical (task_id, trial_id), not completion order. Standard error
    describes these fixed trials only; it is not a generalization guarantee.
    """
    current = np.asarray(result["trial_returns"], dtype=np.float64)
    if previous_trial_returns is None:
        result["paired_progress_available"] = False
        return
    previous = np.asarray(previous_trial_returns, dtype=np.float64)
    if previous.shape != current.shape:
        raise ValueError("Paired evaluation trial shape changed")
    delta = current - previous
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired progress")
    n = current.shape[1]
    lp = delta.mean(axis=1)
    largest = np.abs(delta).argmax(axis=1)
    largest_delta = delta[np.arange(len(delta)), largest]
    total_abs = np.abs(delta).sum(axis=1)
    share = np.divide(
        np.abs(largest_delta), total_abs,
        out=np.zeros_like(total_abs), where=total_abs > 0,
    )
    se = delta.std(axis=1, ddof=1) / np.sqrt(n) if n > 1 else None
    without = (delta.sum(axis=1) - largest_delta) / (n - 1) if n > 1 else None
    for i, row in enumerate(result["per_task"]):
        row.update(
            lp=float(lp[i]),
            lp_standard_error=float(se[i]) if se is not None else None,
            lp_largest_trial_abs_share=float(share[i]),
            lp_without_largest_trial=float(without[i]) if without is not None else None,
        )
    result["paired_progress_available"] = True
    result["macro"]["lp_largest_trial_abs_share"] = float(share.mean())
    if se is not None:
        result["macro"]["lp_standard_error"] = float(se.mean())
        # Positive raw LP accompanied by a nonpositive leave-one-out LP is an
        # outlier-sensitivity diagnostic, not a hypothesis test or rejection rule.
        result["macro"]["lp_positive_outlier_sensitive_tasks"] = int(
            ((lp > 0) & (without <= 0)).sum()
        )
