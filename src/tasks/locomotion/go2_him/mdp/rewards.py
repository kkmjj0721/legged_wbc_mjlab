"""Reward terms used by Go2 HIM."""

from src.tasks.locomotion.ri_4438_him.mdp.rewards import *  # noqa: F401, F403

# Go2 keeps its terrain-height sensor based clearance reward.
from mjlab.tasks.velocity.mdp.rewards import feet_clearance as feet_clearance
