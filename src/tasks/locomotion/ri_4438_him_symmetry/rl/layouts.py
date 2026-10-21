"""Validate environment factories against the versioned HIM contracts."""

from src.config.ri_4438.him_layouts import HimLayout, PHASE47, get_layout


def validate_env_cfg(cfg, layout: HimLayout) -> None:
    if layout != PHASE47:
        raise ValueError("RI-4438 HIM symmetry requires phase47_v1")
    for group_name in ("actor", "critic", "estimator"):
        group = cfg.observations[group_name]
        if tuple(group.terms) != layout.terms(group_name):
            raise ValueError(f"{layout.name}: unexpected {group_name} term order: {tuple(group.terms)}")
        if not group.concatenate_terms:
            raise ValueError(f"{group_name} must concatenate observation terms")
    actor = cfg.observations["actor"]
    if actor.history_length != layout.history_size or actor.flatten_history_dim:
        raise ValueError("HIM symmetry requires explicit oldest-first [B,6,F] history")
    estimator = cfg.observations["estimator"]
    if estimator.history_length or estimator.enable_corruption:
        raise ValueError("Estimator supervision must have no history or noise")
    for term in estimator.terms.values():
        if term.history_length or term.delay_min_lag or term.delay_max_lag or term.noise is not None:
            raise ValueError("Estimator supervision must have no term history, delay or noise")
    gait = cfg.rewards["foot_gait"].params
    for group_name in ("actor", "critic", "estimator"):
        phase = cfg.observations[group_name].terms["phase"].params
        for name in ("period", "command_name", "command_threshold"):
            if phase[name] != gait[name]:
                raise ValueError(f"{group_name} phase and foot_gait disagree on {name}")
    if "feet_air_time" not in cfg.rewards:
        raise ValueError("phase47_v1 requires the original gait and air-time rewards")
