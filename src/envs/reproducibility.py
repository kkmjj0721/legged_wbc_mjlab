"""Opt-in, version-scoped canonical GPU execution for RI-4438 HIM Symmetry.

GPU contact/constraint slot allocation and floating-point tree reductions have
no stable order in the pinned backend. This adapter canonicalizes completed
outputs and orders the affected reductions without changing collision formulas.
It is process-wide: choose one mode before constructing the first environment.
Default mode installs no hooks. Separate processes are required for comparison.

This is a bounded reproducibility aid, not cross-device/version determinism or
checkpoint-exact simulation continuation. See docs/gpu_reproducibility.md.
"""
from __future__ import annotations

from functools import wraps
from importlib.metadata import version
import os

_BACKEND = {"mjlab": "1.6.0", "mujoco-warp": "3.11.0", "warp-lang": "1.17.0"}
_PROCESS_MODE: bool | None = None


def execution_contract(cfg) -> dict:
    if not getattr(cfg, "deterministic", False):
        return {"mode": "default"}
    return {"mode": "canonical_v1", "backend": dict(_BACKEND)}


def configure_simulation(cfg, device: str) -> None:
    """Pin process mode before any model creation, DR, or graph capture."""
    global _PROCESS_MODE
    enabled = bool(getattr(cfg, "deterministic", False))
    if _PROCESS_MODE is not None and _PROCESS_MODE != enabled:
        raise RuntimeError("Simulation execution mode is process-wide; use a separate process to change env.sim.deterministic")
    if enabled:
        if not str(device).startswith("cuda"):
            raise ValueError("env.sim.deterministic currently supports CUDA only")
        actual = {name: version(name) for name in _BACKEND}
        if actual != _BACKEND:
            raise RuntimeError(f"Canonical GPU adapter requires {_BACKEND}, got {actual}")
        import torch
        import warp as wp
        from mjlab.utils.torch import configure_torch_backends
        if wp.config.deterministic != wp.DeterministicMode.NOT_GUARANTEED:
            raise RuntimeError("Canonical mode requires the ordinary Warp backend; do not enable global Warp deterministic atomics")
        bad_modules = _nonordinary_mjwarp_modules()
        if bad_modules:
            raise RuntimeError(f"Canonical GPU mode cannot mix native deterministic Warp modules: {bad_modules}")
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in (":4096:8", ":16:8"):
            raise ValueError("Canonical GPU mode needs CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
        configure_torch_backends(allow_tf32=False, deterministic=True)
        torch.use_deterministic_algorithms(True)
        if _PROCESS_MODE is None:
            _install()
    _PROCESS_MODE = enabled


def validate_simulation(sim) -> None:
    """Read a small sticky status vector once per environment step/reset."""
    enabled = bool(getattr(sim.cfg, "deterministic", False))
    if enabled != _PROCESS_MODE:
        raise RuntimeError("Do not change sim.cfg.deterministic after environment construction")
    if enabled:
        sim.wp_data._project_canonical.validate()


def _install() -> None:
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp._src import collision_driver, constraint
    from mjlab.sim.sim import Simulation
    from ._deterministic import tree, tree_extra, sensor
    from ._deterministic.state import CanonicalState

    tree.install()
    tree_extra.install()
    sensor.install(world_sorted=True)

    def state(m, d):
        result = getattr(d, "_project_canonical", None)
        if result is None:
            # Simulation.create_graph prepares this before CUDA capture.
            if d.nacon.device.is_capturing:
                raise RuntimeError("Prepare canonical state before capturing a custom MuJoCo Warp graph")
            result = CanonicalState(m, d)
            d._project_canonical = result
        return result

    original_collision = collision_driver.collision
    @wraps(original_collision)
    def collision(m, d, *args, **kwargs):
        result = original_collision(m, d, *args, **kwargs)
        state(m, d).contacts.sort()
        return result

    original_constraint = constraint.make_constraint
    @wraps(original_constraint)
    def make_constraint(m, d, *args, **kwargs):
        result = original_constraint(m, d, *args, **kwargs)
        state(m, d).constraints.sort()
        return result

    original_graph = Simulation.create_graph
    @wraps(original_graph)
    def create_graph(sim):
        state(sim.wp_model, sim.wp_data)
        return original_graph(sim)

    collision_driver.collision = mjw.collision = collision
    constraint.make_constraint = mjw.make_constraint = make_constraint
    Simulation.create_graph = create_graph

    # Capture the same kernels as upstream; collect sticky status immediately
    # after each call, before a later reset can clear backend overflow bits.
    for name in ("step", "forward"):
        original = getattr(Simulation, name)
        def collected(sim, _original=original):
            result = _original(sim)
            state(sim.wp_model, sim.wp_data).collect()
            return result
        setattr(Simulation, name, wraps(original)(collected))


def _nonordinary_mjwarp_modules() -> list[str]:
    """Cover existing unique factory kernels as well as shared modules."""
    import warp as wp
    from warp._src import context
    bad = []
    for module in tuple(context.user_modules.values()):
        items = (*tuple(module.kernels.values()), *tuple(module.functions.values()))
        owners = {getattr(getattr(item, "func", None), "__module__", "") for item in items}
        owned = module.name.startswith("mujoco_warp.") or any(
            owner.startswith("mujoco_warp.") for owner in owners)
        if owned and wp.get_module_options(module)["deterministic"] != wp.DeterministicMode.NOT_GUARANTEED:
            bad.append(module.name)
    return sorted(bad)
