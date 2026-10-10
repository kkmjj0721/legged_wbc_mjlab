"""RI-4438 HIM numerical contract (raw policy units, before joint scaling)."""

ACTION_CLIP = 10.0
OBSERVATION_CLIP = 100.0
# The environment clips executed actions to ACTION_CLIP. Larger finite samples
# are diagnostic signals; only non-finite policy values should abort training.
RAW_ACTION_WARN = 20.0
RAW_OBSERVATION_ABORT = 1000.0
ACTION_OBSERVATION_SLICE = (35, 47)
