from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .math_utils import normalize_quaternion


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "logs/rsl_rl/ri_4438_ppo/2026-09-09_14-05-41/policy.onnx"
DEFAULT_JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
DEFAULT_JOINT = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
DEFAULT_OBSERVATION_TERMS = (
    "base_ang_vel", "projected_gravity", "command", "phase",
    "joint_pos_rel", "joint_vel", "last_action",
)
SUPPORTED_OBSERVATION_TERMS = frozenset(DEFAULT_OBSERVATION_TERMS)
# Width of each observation term emitted by :class:`LocomotionFSM`.  Keeping
# this table next to the config parser catches an accidental ONNX contract
# drift before a simulation is started.
OBSERVATION_TERM_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "command": 3,
    "phase": 2,
    "joint_pos_rel": 12,
    "joint_vel": 12,
    "last_action": 12,
}


_DEFAULT_GAMEPAD_ENTER_DEADZONE = 0.18
_DEFAULT_GAMEPAD_EXIT_DEADZONE = 0.15
_DEFAULT_CALIBRATION_CENTER_LIMIT = 0.15
_DEFAULT_CALIBRATION_RELEASE_MARGIN = 0.005


def _validate_gamepad_center_safety(options: dict[str, Any]) -> None:
    """Reject calibration settings that can turn stick release into motion."""

    try:
        enter = float(
            options.get(
                "deadzone_enter",
                options.get("deadzone", _DEFAULT_GAMEPAD_ENTER_DEADZONE),
            )
        )
        exit_deadzone = float(
            options.get("deadzone_exit", _DEFAULT_GAMEPAD_EXIT_DEADZONE)
        )
        center_limit = float(options["calibration_center_limit"])
        margin = float(options["calibration_release_margin"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "gamepad deadzone and calibration center safety values must be numeric"
        ) from exc
    if (
        not np.isfinite(enter)
        or not np.isfinite(exit_deadzone)
        or not 0.0 < exit_deadzone <= enter < 1.0
    ):
        raise ValueError(
            "gamepad deadzones must satisfy 0 < deadzone_exit <= deadzone_enter < 1"
        )
    if not np.isfinite(center_limit) or not 0.0 < center_limit < 1.0:
        raise ValueError("gamepad calibration center_limit must be in (0, 1)")
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("gamepad calibration release_margin must be finite and positive")
    # This is the per-axis/yaw configuration bound.  AxisCalibrator applies
    # the stronger radial translation check to the measured center vector.
    release_at_limit = center_limit / (1.0 + center_limit)
    if not release_at_limit + margin < exit_deadzone:
        raise ValueError(
            "unsafe gamepad calibration: center_limit/(1+center_limit) + "
            "release_margin must be strictly below deadzone_exit "
            f"({release_at_limit:.6f} + {margin:.6f} !< {exit_deadzone:.6f}); "
            "diagnose the physical center, then lower center_limit or "
            "explicitly increase both deadzones"
        )


def _array(value: Iterable[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(list(value), dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {array.shape}")
    return array


def _yaw_rotate(quaternion: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a MuJoCo [w, x, y, z] quaternion about the world z axis."""
    half_yaw = 0.5 * yaw
    yaw_quaternion = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    w1, x1, y1, z1 = yaw_quaternion
    w2, x2, y2, z2 = quaternion
    return normalize_quaternion(np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]))


@dataclass
class SimConfig:
    task_name: str
    xml: Path
    policy: Path | None
    joint_names: tuple[str, ...]
    observation_dim: int
    action_dim: int
    command_dim: int
    observation_terms: tuple[str, ...]
    timestep: float = 0.005
    decimation: int = 4
    kp: float = 30.0
    kd: float = 0.6
    effort_limit: float = 10.0
    action_scale: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.5, 0.5] * 4))
    default_joint: np.ndarray = field(default_factory=lambda: DEFAULT_JOINT.copy())
    lying_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.085]))
    lying_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    lying_joint: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.25, -2.25] * 4))
    standing_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.25]))
    standing_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    stand_duration: float = 3.0
    phase_period: float = 0.6
    command_limit: np.ndarray = field(default_factory=lambda: np.array([2.0, 1.0, 1.0]))
    fall_height: float = 0.055
    fall_tilt: float = 1.20
    command_deadzone: float = 0.08
    joystick_axes: tuple[int, int, int] = (0, 1, 3)
    input_backend: str = "both"
    auto_stand: bool = False
    render: bool = True
    noise: bool = False
    camera_follow: bool = True
    camera_track_body: str = "base_link"
    camera_distance: float = 1.5
    camera_azimuth: float = 90.0
    camera_elevation: float = -20.0

    # Fields below were added after the original SimConfig API.  Keep all of
    # them after the historical fields above so positional construction by
    # older callers retains its exact argument order.
    history_size: int = 1
    # Options forwarded verbatim (after validation/filtering) to the SDL
    # joystick/controller adapter.  Keeping this as a mapping preserves
    # compatibility with older callers that only supplied ``joystick_axes``.
    gamepad_options: dict[str, Any] = field(default_factory=dict)
    # When enabled, the startup stand sequence proceeds automatically into
    # policy control after the measured settle gate. Manual ``stand`` requests
    # still stop in STAND until an explicit enable_rl event.
    auto_enable_rl: bool = False
    terrain_name: str = "combined"
    available_terrains: tuple[str, ...] = ("combined",)
    # HIM export metadata.  Kept at the end to preserve positional
    # construction compatibility with the original SimConfig dataclass.
    history_order: str = "oldest-first"
    getdown_duration: float = 3.0
    # ``None`` means preserve the policy output as-is.  A finite positive
    # value clips each action component symmetrically before scaling.
    action_clip: float | None = 1.0
    # Real-robot style transition/control parameters.  ``control_dt`` is the
    # policy/update period (normally timestep * decimation), while the
    # trajectory speeds are joint-space radians per second.
    control_dt: float = 0.02
    getup_joint_speed: float = 1.5
    getdown_joint_speed: float = 1.5
    settle_duration: float = 0.5
    settle_height_tolerance: float = 0.04
    settle_tilt_tolerance: float = 0.20
    settle_joint_tolerance: float = 0.12
    settle_velocity_tolerance: float = 0.35
    stale_timeout: float = 0.25
    # Hard lifecycle deadlines prevent a real backend from remaining in a
    # powered transition indefinitely.  ``load_config`` derives these from
    # the associated nominal durations when they are not specified.
    settle_timeout: float = 2.0
    getup_timeout: float = 5.0
    getdown_timeout: float = 5.0


def _yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Install with `uv sync --extra sim2sim`.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(
    path: Path,
    task_name: str,
    policy_override: Path | None = None,
    terrain_override: str | None = None,
) -> SimConfig:
    data = _yaml(path)
    model = data.get("model", {})
    control = data.get("control", {})
    pose = data.get("pose", {})
    safety = data.get("safety", {})
    camera = data.get("camera", {})
    input_cfg = data.get("input", {})
    policy_cfg = data.get("policy", {})
    terrain_cfg = data.get("terrain", {})
    xml = Path(model.get("xml", REPO_ROOT / "src/assets/robots/ri_4438/xmls/ri_4438.xml"))
    if not xml.is_absolute():
        xml = REPO_ROOT / xml
    policy_value = policy_override if policy_override is not None else model.get("policy", DEFAULT_POLICY if DEFAULT_POLICY.exists() else None)
    policy = None if policy_value is None else Path(policy_value)
    if policy is not None and not policy.is_absolute():
        policy = REPO_ROOT / policy
    joystick_axes = tuple(int(axis) for axis in input_cfg.get("joystick_axes", [0, 1, 3]))
    if len(joystick_axes) != 3:
        raise ValueError("joystick_axes must contain [left_x, left_y, right_x]")
    if any(axis < 0 for axis in joystick_axes) or len(set(joystick_axes)) != 3:
        raise ValueError("joystick_axes must contain three distinct non-negative indices")
    # ``input.gamepad`` is the preferred explicit section.  For compatibility
    # with the original YAML, values may also be provided directly under
    # ``input``.  Filter the mapping so unrelated input keys never reach
    # JoystickInput's constructor.
    gamepad_cfg = input_cfg.get("gamepad", {})
    if gamepad_cfg is None:
        gamepad_cfg = {}
    if not isinstance(gamepad_cfg, dict):
        raise ValueError("input.gamepad must be a mapping")
    calibration_cfg = gamepad_cfg.get("calibration", {})
    if calibration_cfg is None:
        calibration_cfg = {}
    if not isinstance(calibration_cfg, dict):
        raise ValueError("input.gamepad.calibration must be a mapping")
    neutral_cfg = gamepad_cfg.get("neutral_rearm", {})
    if neutral_cfg is None:
        neutral_cfg = {}
    if not isinstance(neutral_cfg, dict):
        raise ValueError("input.gamepad.neutral_rearm must be a mapping")
    gamepad_options: dict[str, Any] = {
        "device_name": gamepad_cfg.get("device_name", input_cfg.get("device_name")),
        "device_guid": gamepad_cfg.get("device_guid", input_cfg.get("device_guid")),
        "device_index": gamepad_cfg.get("device_index", input_cfg.get("device_index")),
        "axis_config": gamepad_cfg.get("axis_config", input_cfg.get("axis_config")),
        "button_config": gamepad_cfg.get("button_config", input_cfg.get("button_config")),
        "deadzone": gamepad_cfg.get(
            "deadzone",
            input_cfg.get("gamepad_deadzone", _DEFAULT_GAMEPAD_ENTER_DEADZONE),
        ),
        "deadzone_enter": gamepad_cfg.get("deadzone_enter"),
        "deadzone_exit": gamepad_cfg.get(
            "deadzone_exit",
            input_cfg.get("gamepad_deadzone_exit", _DEFAULT_GAMEPAD_EXIT_DEADZONE),
        ),
        "yaw_rescale": gamepad_cfg.get("yaw_rescale", input_cfg.get("yaw_rescale", 1.0)),
        "reconnect_interval": gamepad_cfg.get("reconnect_interval", 1.0),
        "liveness_timeout": gamepad_cfg.get("liveness_timeout", 2.0),
        "prefer_controller": gamepad_cfg.get("prefer_controller", True),
        "calibration_samples": calibration_cfg.get(
            "samples", gamepad_cfg.get("calibration_samples", 25)
        ),
        "calibration_duration": calibration_cfg.get(
            "duration", gamepad_cfg.get("calibration_duration", 0.25)
        ),
        "calibration_max_mad": calibration_cfg.get(
            "max_mad", gamepad_cfg.get("calibration_max_mad", 0.01)
        ),
        "calibration_max_peak_to_peak": calibration_cfg.get(
            "max_peak_to_peak",
            gamepad_cfg.get("calibration_max_peak_to_peak", 0.04),
        ),
        "calibration_center_limit": calibration_cfg.get(
            "center_limit",
            gamepad_cfg.get(
                "calibration_center_limit", _DEFAULT_CALIBRATION_CENTER_LIMIT
            ),
        ),
        "calibration_extreme_limit": calibration_cfg.get(
            "extreme_limit", gamepad_cfg.get("calibration_extreme_limit", 0.90)
        ),
        "calibration_release_margin": calibration_cfg.get(
            "release_margin",
            gamepad_cfg.get(
                "calibration_release_margin", _DEFAULT_CALIBRATION_RELEASE_MARGIN
            ),
        ),
        "neutral_rearm_samples": neutral_cfg.get(
            "samples", gamepad_cfg.get("neutral_rearm_samples", 5)
        ),
        "neutral_rearm_duration": neutral_cfg.get(
            "duration", gamepad_cfg.get("neutral_rearm_duration", 0.10)
        ),
        "input_debug": gamepad_cfg.get(
            "input_debug", input_cfg.get("input_debug", False)
        ),
        "debug_interval": gamepad_cfg.get("debug_interval", 0.25),
    }
    # Drop unset optional selectors; JoystickInput treats None as no filter.
    gamepad_options = {key: value for key, value in gamepad_options.items() if value is not None}
    _validate_gamepad_center_safety(gamepad_options)
    required_policy_keys = ("joint_names", "observation_dim", "action_dim", "command_dim", "observation_terms")
    missing_policy_keys = [key for key in required_policy_keys if key not in policy_cfg]
    if missing_policy_keys:
        raise ValueError(f"Missing required policy config fields: {', '.join(missing_policy_keys)}")
    joint_names = tuple(str(name) for name in policy_cfg["joint_names"])
    action_dim = int(policy_cfg["action_dim"])
    observation_dim = int(policy_cfg["observation_dim"])
    command_dim = int(policy_cfg["command_dim"])
    observation_terms = tuple(str(term) for term in policy_cfg["observation_terms"])
    history_size = int(policy_cfg.get("history_size", 1))
    is_him_task = "him" in task_name.lower()
    history_order = str(policy_cfg.get("history_order", "current-first" if is_him_task else "oldest-first"))
    if not joint_names:
        raise ValueError("policy.joint_names must not be empty")
    if action_dim != len(joint_names):
        raise ValueError(f"policy.action_dim={action_dim} must equal joint_names length={len(joint_names)}")
    if is_him_task and action_dim != 12:
        raise ValueError(f"HIM policy.action_dim must be 12, got {action_dim}")
    if observation_dim <= 0:
        raise ValueError("policy.observation_dim must be a positive integer")
    if history_size <= 0:
        raise ValueError("policy.history_size must be a positive integer")
    if is_him_task and history_order != "current-first":
        raise ValueError("HIM policy.history_order must be 'current-first'")
    unsupported_terms = set(observation_terms) - SUPPORTED_OBSERVATION_TERMS
    if unsupported_terms:
        raise ValueError(f"Unsupported policy.observation_terms: {sorted(unsupported_terms)}")
    if is_him_task and observation_terms != DEFAULT_OBSERVATION_TERMS:
        raise ValueError(
            "HIM policy.observation_terms must exactly match the exported ONNX order: "
            + ", ".join(DEFAULT_OBSERVATION_TERMS)
        )
    if command_dim != 3:
        raise ValueError("The current keyboard/gamepad velocity interface requires policy.command_dim=3")
    frame_dim = sum(OBSERVATION_TERM_DIMS.get(term, 0) for term in observation_terms)
    if is_him_task and frame_dim != 47:
        raise ValueError(
            "HIM observation terms must produce one 47-value frame; "
            f"configured terms produce {frame_dim} values"
        )
    expected_observation_dim = frame_dim * history_size
    if is_him_task and observation_dim != expected_observation_dim:
        raise ValueError(
            f"policy.observation_dim={observation_dim} must equal "
            f"history_size({history_size}) * frame_dim({frame_dim}) = {expected_observation_dim}"
        )
    terrain_spawns = terrain_cfg.get("spawns", {"combined": {}})
    if not isinstance(terrain_spawns, dict) or not terrain_spawns:
        raise ValueError("terrain.spawns must be a non-empty mapping")
    terrain_name = str(terrain_override or terrain_cfg.get("default", "combined"))
    if terrain_name not in terrain_spawns:
        choices = ", ".join(str(name) for name in terrain_spawns)
        raise ValueError(f"Unknown terrain '{terrain_name}'. Available terrains: {choices}")
    spawn = terrain_spawns[terrain_name] or {}
    if not isinstance(spawn, dict):
        raise ValueError(f"terrain.spawns.{terrain_name} must be a mapping")
    spawn_offset = _array(spawn.get("offset", [0.0, 0.0, 0.0]), 3, f"terrain.spawns.{terrain_name}.offset")
    spawn_yaw = float(spawn.get("yaw", 0.0))
    lying_pos = _array(pose.get("lying_pos", [0.0, 0.0, 0.085]), 3, "lying_pos") + spawn_offset
    standing_pos = _array(pose.get("standing_pos", [0.0, 0.0, 0.25]), 3, "standing_pos") + spawn_offset
    lying_quat = _yaw_rotate(
        normalize_quaternion(_array(pose.get("lying_quat", [1, 0, 0, 0]), 4, "lying_quat")),
        spawn_yaw,
    )
    standing_quat = _yaw_rotate(
        normalize_quaternion(_array(pose.get("standing_quat", [1, 0, 0, 0]), 4, "standing_quat")),
        spawn_yaw,
    )
    action_clip_raw = control.get("action_clip", policy_cfg.get("action_clip", 1.0))
    action_clip = None if action_clip_raw is None else float(action_clip_raw)
    if action_clip is not None and (not np.isfinite(action_clip) or action_clip <= 0.0):
        raise ValueError("control.action_clip must be null or a finite positive number")
    # Transition settings may be grouped under ``trajectory``/``settle`` in
    # newer task files, or supplied directly in ``control``/``safety`` for
    # backwards compatibility.  Resolve all forms here so the FSM has one
    # stable interface.
    trajectory_cfg = data.get("trajectory", {})
    if not isinstance(trajectory_cfg, dict):
        raise ValueError("trajectory must be a mapping")
    settle_cfg = data.get("settle", {})
    if not isinstance(settle_cfg, dict):
        raise ValueError("settle must be a mapping")
    timestep = float(model.get("timestep", 0.005))
    decimation = int(model.get("decimation", 4))
    control_dt = float(
        control.get("control_dt", model.get("control_dt", timestep * decimation))
    )
    if not np.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control.control_dt must be a finite positive number")

    def _positive(mapping: dict[str, Any], keys: tuple[str, ...], default: float, name: str) -> float:
        value: Any = default
        for key in keys:
            if key in mapping:
                value = mapping[key]
                break
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
        return value

    getup_joint_speed = _positive(
        {**control, **trajectory_cfg},
        ("getup_joint_speed", "stand_joint_speed", "joint_speed", "getup_speed"),
        1.5,
        "trajectory.getup_joint_speed",
    )
    getdown_joint_speed = _positive(
        {**control, **trajectory_cfg},
        ("getdown_joint_speed", "getdown_speed", "joint_speed"),
        getup_joint_speed,
        "trajectory.getdown_joint_speed",
    )
    settle_duration = _positive(
        {**pose, **settle_cfg}, ("duration", "settle_duration"), 0.5, "settle.duration"
    )
    stand_duration = _positive(
        pose, ("stand_duration",), 3.0, "pose.stand_duration"
    )
    getdown_duration = _positive(
        pose, ("getdown_duration",), stand_duration, "pose.getdown_duration"
    )
    settle_timeout = _positive(
        {**control, **safety, **settle_cfg},
        ("timeout", "settle_timeout"),
        max(3.0 * settle_duration, 2.0),
        "settle.timeout",
    )
    getup_timeout = _positive(
        {**control, **safety, **trajectory_cfg},
        ("getup_timeout", "stand_timeout"),
        stand_duration + 2.0,
        "trajectory.getup_timeout",
    )
    getdown_timeout = _positive(
        {**control, **safety, **trajectory_cfg},
        ("getdown_timeout",),
        getdown_duration + 2.0,
        "trajectory.getdown_timeout",
    )

    def _nonnegative(mapping: dict[str, Any], keys: tuple[str, ...], default: float, name: str) -> float:
        value: Any = default
        for key in keys:
            if key in mapping:
                value = mapping[key]
                break
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        return value

    settle_height_tolerance = _nonnegative(
        {**safety, **settle_cfg}, ("height_tolerance", "settle_height_tolerance"), 0.04,
        "settle.height_tolerance",
    )
    settle_tilt_tolerance = _nonnegative(
        {**safety, **settle_cfg}, ("tilt_tolerance", "settle_tilt_tolerance"), 0.20,
        "settle.tilt_tolerance",
    )
    settle_joint_tolerance = _nonnegative(
        {**safety, **settle_cfg}, ("joint_tolerance", "settle_joint_tolerance"), 0.12,
        "settle.joint_tolerance",
    )
    settle_velocity_tolerance = _nonnegative(
        {**safety, **settle_cfg}, ("velocity_tolerance", "settle_velocity_tolerance"), 0.35,
        "settle.velocity_tolerance",
    )
    stale_timeout = _positive(
        {**safety, **settle_cfg}, ("stale_timeout", "state_stale_timeout"), 0.25,
        "safety.stale_timeout",
    )
    return SimConfig(
        task_name=task_name,
        xml=xml,
        policy=policy,
        joint_names=joint_names,
        observation_dim=observation_dim,
        action_dim=action_dim,
        command_dim=command_dim,
        observation_terms=observation_terms,
        history_size=history_size,
        history_order=history_order,
        timestep=timestep,
        decimation=decimation,
        kp=float(control.get("kp", 30.0)),
        kd=float(control.get("kd", 0.6)),
        effort_limit=float(control.get("effort_limit", 10.0)),
        action_scale=_array(control["action_scale"], action_dim, "action_scale"),
        default_joint=_array(pose["default_joint"], action_dim, "default_joint"),
        lying_pos=lying_pos,
        lying_quat=lying_quat,
        lying_joint=_array(pose["lying_joint"], action_dim, "lying_joint"),
        standing_pos=standing_pos,
        standing_quat=standing_quat,
        stand_duration=stand_duration,
        getdown_duration=getdown_duration,
        phase_period=float(control.get("phase_period", 0.6)),
        action_clip=action_clip,
        command_limit=_array(input_cfg.get("command_limit", [2.0, 1.0, 1.0]), 3, "command_limit"),
        fall_height=float(safety.get("fall_height", 0.055)),
        fall_tilt=float(safety.get("fall_tilt", 1.20)),
        command_deadzone=float(input_cfg.get("deadzone", 0.08)),
        joystick_axes=joystick_axes,
        gamepad_options=gamepad_options,
        input_backend=str(input_cfg.get("backend", "both")),
        auto_stand=bool(input_cfg.get("auto_stand", False)),
        auto_enable_rl=bool(input_cfg.get("auto_enable_rl", input_cfg.get("auto_stand", False))),
        render=bool(model.get("render", True)),
        noise=bool(control.get("noise", False)),
        camera_follow=bool(camera.get("follow", True)),
        camera_track_body=str(camera.get("track_body", "base_link")),
        camera_distance=float(camera.get("distance", 1.5)),
        camera_azimuth=float(camera.get("azimuth", 90.0)),
        camera_elevation=float(camera.get("elevation", -20.0)),
        terrain_name=terrain_name,
        available_terrains=tuple(str(name) for name in terrain_spawns),
        control_dt=control_dt,
        getup_joint_speed=getup_joint_speed,
        getdown_joint_speed=getdown_joint_speed,
        settle_duration=settle_duration,
        settle_height_tolerance=settle_height_tolerance,
        settle_tilt_tolerance=settle_tilt_tolerance,
        settle_joint_tolerance=settle_joint_tolerance,
        settle_velocity_tolerance=settle_velocity_tolerance,
        stale_timeout=stale_timeout,
        settle_timeout=settle_timeout,
        getup_timeout=getup_timeout,
        getdown_timeout=getdown_timeout,
    )
