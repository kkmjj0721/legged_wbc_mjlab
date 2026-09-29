"""Fixed benchmark scenarios, shared by simulation and offline reports."""

STAIR_HEIGHTS_CM = (5, 10, 15)
REPORT_MODEL_LIMIT = 10
IMAGE_MODEL_LIMIT = 3
STAIR_START = 1.0
STAIR_WIDTH = 0.30
STAIR_COUNT = 6
STAIR_GOAL = STAIR_START + STAIR_WIDTH * STAIR_COUNT + 0.35
LANE_HALF_WIDTH = 1.2
STAIR_COMMANDS = (
    ("forward_slow", "前进 0.3 m/s", (0.3, 0.0, 0.0)),
    ("forward", "前进 0.5 m/s", (0.5, 0.0, 0.0)),
    ("forward_fast", "前进 0.8 m/s", (0.8, 0.0, 0.0)),
)
STAIR_PRESETS = {
    f"stairs_{direction}_{cm}cm": {
        "height_cm": cm, "height_m": cm / 100, "descending": direction == "down",
        "steps": STAIR_COUNT, "tread_m": STAIR_WIDTH, "total_height_m": STAIR_COUNT * cm / 100,
        "label": f"{cm} cm {'下' if direction == 'down' else '上'}楼梯",
    }
    for cm in STAIR_HEIGHTS_CM for direction in ("up", "down")
}
TERRAIN_LABELS = {"flat": "平地", "rough": "起伏地形", **{t: p["label"] for t, p in STAIR_PRESETS.items()}}


def expand_terrains(groups):
    """Each stair direction always includes all three fixed riser heights."""
    scenarios = []
    for group in groups:
        if group in ("flat", "rough"):
            scenarios.append(group)
        elif group in ("stairs_up", "stairs_down"):
            scenarios.extend(f"{group}_{cm}cm" for cm in STAIR_HEIGHTS_CM)
        else:
            raise ValueError(f"unknown evaluation terrain group: {group}")
    return scenarios


def stair_sections(length: float, height: float, descending: bool):
    """Non-overlapping x intervals and top heights, measured from spawn x=0."""
    top = STAIR_COUNT * height
    sections = [(-length / 2, STAIR_START, top if descending else 0.0)]
    for i in range(STAIR_COUNT):
        left = STAIR_START + i * STAIR_WIDTH
        z = top - (i + 1) * height if descending else (i + 1) * height
        sections.append((left, left + STAIR_WIDTH, max(0.0, z)))
    sections.append((STAIR_START + STAIR_COUNT * STAIR_WIDTH, length / 2, 0.0 if descending else top))
    return sections
