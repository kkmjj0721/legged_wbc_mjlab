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


def _array(value: Iterable[float], size: int, name: str) -> np.ndarray:
    array = np.asarray(list(value), dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {array.shape}")
    return array


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
    joystick_axes: tuple[int, int, int] = (0, 1, 2)
    input_backend: str = "both"
    auto_stand: bool = False
    render: bool = True
    noise: bool = False
    camera_follow: bool = True
    camera_track_body: str = "base_link"
    camera_distance: float = 1.5
    camera_azimuth: float = 90.0
    camera_elevation: float = -20.0


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


def load_config(path: Path, task_name: str, policy_override: Path | None = None) -> SimConfig:
    data = _yaml(path)
    model = data.get("model", {})
    control = data.get("control", {})
    pose = data.get("pose", {})
    safety = data.get("safety", {})
    camera = data.get("camera", {})
    input_cfg = data.get("input", {})
    policy_cfg = data.get("policy", {})
    xml = Path(model.get("xml", REPO_ROOT / "src/assets/robots/ri_4438/xmls/ri_4438.xml"))
    if not xml.is_absolute():
        xml = REPO_ROOT / xml
    policy_value = policy_override if policy_override is not None else model.get("policy", DEFAULT_POLICY if DEFAULT_POLICY.exists() else None)
    policy = None if policy_value is None else Path(policy_value)
    if policy is not None and not policy.is_absolute():
        policy = REPO_ROOT / policy
    joystick_axes = tuple(int(axis) for axis in input_cfg.get("joystick_axes", [0, 1, 2]))
    if len(joystick_axes) != 3:
        raise ValueError("joystick_axes must contain [left_x, left_y, right_x]")
    required_policy_keys = ("joint_names", "observation_dim", "action_dim", "command_dim", "observation_terms")
    missing_policy_keys = [key for key in required_policy_keys if key not in policy_cfg]
    if missing_policy_keys:
        raise ValueError(f"Missing required policy config fields: {', '.join(missing_policy_keys)}")
    joint_names = tuple(str(name) for name in policy_cfg["joint_names"])
    action_dim = int(policy_cfg["action_dim"])
    observation_dim = int(policy_cfg["observation_dim"])
    command_dim = int(policy_cfg["command_dim"])
    observation_terms = tuple(str(term) for term in policy_cfg["observation_terms"])
    if not joint_names:
        raise ValueError("policy.joint_names must not be empty")
    if action_dim != len(joint_names):
        raise ValueError(f"policy.action_dim={action_dim} must equal joint_names length={len(joint_names)}")
    if observation_dim <= 0:
        raise ValueError("policy.observation_dim must be a positive integer")
    unsupported_terms = set(observation_terms) - SUPPORTED_OBSERVATION_TERMS
    if unsupported_terms:
        raise ValueError(f"Unsupported policy.observation_terms: {sorted(unsupported_terms)}")
    if command_dim != 3:
        raise ValueError("The current keyboard/gamepad velocity interface requires policy.command_dim=3")
    return SimConfig(
        task_name=task_name,
        xml=xml,
        policy=policy,
        joint_names=joint_names,
        observation_dim=observation_dim,
        action_dim=action_dim,
        command_dim=command_dim,
        observation_terms=observation_terms,
        timestep=float(model.get("timestep", 0.005)),
        decimation=int(model.get("decimation", 4)),
        kp=float(control.get("kp", 30.0)),
        kd=float(control.get("kd", 0.6)),
        effort_limit=float(control.get("effort_limit", 10.0)),
        action_scale=_array(control["action_scale"], action_dim, "action_scale"),
        default_joint=_array(pose["default_joint"], action_dim, "default_joint"),
        lying_pos=_array(pose.get("lying_pos", [0.0, 0.0, 0.085]), 3, "lying_pos"),
        lying_quat=normalize_quaternion(_array(pose.get("lying_quat", [1, 0, 0, 0]), 4, "lying_quat")),
        lying_joint=_array(pose["lying_joint"], action_dim, "lying_joint"),
        standing_pos=_array(pose.get("standing_pos", [0.0, 0.0, 0.25]), 3, "standing_pos"),
        standing_quat=normalize_quaternion(_array(pose.get("standing_quat", [1, 0, 0, 0]), 4, "standing_quat")),
        stand_duration=float(pose.get("stand_duration", 3.0)),
        phase_period=float(control.get("phase_period", 0.6)),
        command_limit=_array(input_cfg.get("command_limit", [2.0, 1.0, 1.0]), 3, "command_limit"),
        fall_height=float(safety.get("fall_height", 0.055)),
        fall_tilt=float(safety.get("fall_tilt", 1.20)),
        command_deadzone=float(input_cfg.get("deadzone", 0.08)),
        joystick_axes=joystick_axes,
        input_backend=str(input_cfg.get("backend", "both")),
        auto_stand=bool(input_cfg.get("auto_stand", False)),
        render=bool(model.get("render", True)),
        noise=bool(control.get("noise", False)),
        camera_follow=bool(camera.get("follow", True)),
        camera_track_body=str(camera.get("track_body", "base_link")),
        camera_distance=float(camera.get("distance", 1.5)),
        camera_azimuth=float(camera.get("azimuth", 90.0)),
        camera_elevation=float(camera.get("elevation", -20.0)),
    )
