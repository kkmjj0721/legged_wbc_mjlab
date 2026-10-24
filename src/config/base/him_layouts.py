"""Versioned HIM input contracts shared by every robot and deployment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HimLayout:
    name: str
    with_phase: bool
    history_size: int = 6

    @property
    def actor_terms(self) -> tuple[str, ...]:
        return ("base_ang_vel", "projected_gravity", "command") + (
            ("phase",) if self.with_phase else ()
        ) + ("joint_pos", "joint_vel", "actions")

    @property
    def term_dims(self) -> tuple[int, ...]:
        return (3, 3, 3) + ((2,) if self.with_phase else ()) + (12, 12, 12)

    @property
    def frame_size(self) -> int:
        return sum(self.term_dims)

    @property
    def history_width(self) -> int:
        return self.frame_size * self.history_size

    @property
    def critic_size(self) -> int:
        return self.frame_size + 3 + 3 + 4 + 187

    @property
    def estimator_size(self) -> int:
        return self.frame_size + 3

    @property
    def action_slice(self) -> tuple[int, int]:
        return self.frame_size - 12, self.frame_size

    @property
    def velocity_slice(self) -> tuple[int, int]:
        return self.frame_size, self.estimator_size

    @property
    def target_slices(self) -> tuple[tuple[int, int], ...]:
        return (0, 6), (9, self.frame_size), self.velocity_slice

    @property
    def deployment_terms(self) -> tuple[str, ...]:
        aliases = {"joint_pos": "joint_pos_rel", "actions": "last_action"}
        return tuple(aliases.get(name, name) for name in self.actor_terms)

    def terms(self, group: str) -> tuple[str, ...]:
        extra = {"actor": (), "estimator": ("base_lin_vel",),
                 "critic": ("base_lin_vel", "base_com", "foot_contact", "height_scan")}
        return self.actor_terms + extra[group]

    def widths(self, group: str) -> tuple[int, ...]:
        return self.term_dims + {"actor": (), "estimator": (3,), "critic": (3, 3, 4, 187)}[group]

    def descriptor(self) -> dict:
        return {"name": self.name, "history_size": self.history_size,
                "groups": {g: list(zip(self.terms(g), self.widths(g)))
                           for g in ("actor", "critic", "estimator")},
                "action_slice": list(self.action_slice), "velocity_slice": list(self.velocity_slice),
                "target_slices": [list(s) for s in self.target_slices],
                "phase_half_cycle": self.with_phase}


PHASE47 = HimLayout("phase47_v1", True)
LAYOUTS = {PHASE47.name: PHASE47}


def get_layout(name: str) -> HimLayout:
    try:
        return LAYOUTS[name]
    except KeyError:
        raise ValueError(f"Unknown HIM observation layout {name!r}; expected {tuple(LAYOUTS)}") from None

# Keep the historical pickle path while both import locations share one class.
HimLayout.__module__ = "src.config.ri_4438.him_layouts"
