"""Project-owned options for the pinned MjLab simulation backend."""

from dataclasses import dataclass

from mjlab.sim import SimulationCfg as MjlabSimulationCfg


@dataclass(kw_only=True)
class SimulationCfg(MjlabSimulationCfg):
    """Optional reproducibility controls for supported project tasks.

    The initial experimental mode targets the RI-4438 HIM Symmetry rigid,
    dense-CG, pyramidal configuration with the pinned backend versions. Its
    runtime validates supported model features; enabling it is not a guarantee
    for other tasks, GPU architectures, or complete checkpoint continuation.
    """

    deterministic: bool = False
    """Use canonical contact/constraint ordering and supported serial reductions.

    The selected mode is fixed for the lifetime of the Python process: default
    and deterministic environments cannot be mixed, including after close().
    Full training resume requires the checkpoint's execution mode; warm-start
    and actor-only playback may explicitly use a different mode.
    """
