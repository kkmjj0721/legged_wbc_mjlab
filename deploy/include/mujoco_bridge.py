from __future__ import annotations

import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, SimConfig
from .types import ControlMode, Gain, JointCommand, RobotState

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MuJoCo is required. Install with `uv sync --extra sim2sim`.") from exc


def _runtime_xml(source: Path, cfg: SimConfig) -> Path:
    """Build a temporary simulation scene without modifying the task asset."""
    tree = ET.parse(source)
    root = tree.getroot()
    # MuJoCo resolves mesh paths relative to the included file.  Resolve the
    # scene include before moving the generated XML to a temporary filename so
    # nested RI-4438 mesh assets remain valid for every working directory.
    for include in root.findall("include"):
        include_path = include.get("file")
        if include_path and not Path(include_path).is_absolute():
            include.set("file", str((source.parent / include_path).resolve()))
    for hfield in root.findall("asset/hfield"):
        hfield_path = hfield.get("file")
        if not hfield_path:
            continue
        asset_path = Path(hfield_path)
        if asset_path.is_absolute():
            resolved = asset_path.resolve()
        else:
            candidates = (
                (source.parent / asset_path).resolve(),
                (REPO_ROOT / asset_path).resolve(),
                (REPO_ROOT / "deploy/scene/terrain/assets" / asset_path.name).resolve(),
            )
            resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
        if resolved is None or not resolved.is_file():
            raise FileNotFoundError(f"Heightfield asset not found: {hfield_path}")
        hfield.set("file", str(resolved))
    option = root.find("option")
    if option is None:
        option = ET.Element("option", {"timestep": str(cfg.timestep), "integrator": "implicitfast"})
        root.insert(0, option)
    else:
        option.set("timestep", str(cfg.timestep))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    has_plane = any(geom.get("type") == "plane" for geom in worldbody.findall(".//geom"))
    if worldbody.find("geom[@name='sim2sim_ground']") is None and not has_plane:
        worldbody.insert(0, ET.Element("geom", {
            "name": "sim2sim_ground", "type": "plane", "pos": "0 0 0",
            "size": "20 20 0.1", "friction": "0.8 0.01 0.001",
            "contype": "1", "conaffinity": "1",
        }))
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.Element("actuator")
        root.append(actuators)
    existing: set[str | None] = set()
    for node in actuators.findall("position"):
        joint_name = node.get("joint")
        existing.add(joint_name)
        if joint_name in cfg.joint_names:
            node.set("kp", str(cfg.kp))
            node.set("kv", str(cfg.kd))
            node.set("forcerange", f"{-cfg.effort_limit} {cfg.effort_limit}")
            node.set("forcelimited", "true")
    for name in cfg.joint_names:
        if name not in existing:
            ET.SubElement(actuators, "position", {
                "name": f"{name}_position", "joint": name,
                "kp": str(cfg.kp), "kv": str(cfg.kd),
                "ctrlrange": "-3.5 3.5", "ctrllimited": "true",
                "forcerange": f"{-cfg.effort_limit} {cfg.effort_limit}",
                "forcelimited": "true",
            })
    handle = tempfile.NamedTemporaryFile(prefix="ri4438_sim2sim_", suffix=".xml", dir=source.parent, delete=False)
    path = Path(handle.name)
    handle.close()
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)
    return path

class MujocoBackend:
    """Simulation implementation of the deployment ``RobotBackend`` API."""

    def __init__(self, cfg: SimConfig, damping_kd: Gain | None = None):
        if not cfg.xml.exists():
            raise FileNotFoundError(f"MJCF not found: {cfg.xml}")
        self.cfg = cfg
        self.timestep = cfg.timestep
        self.runtime_xml = _runtime_xml(cfg.xml, cfg)
        try:
            self.model = mujoco.MjModel.from_xml_path(str(self.runtime_xml))
        except Exception:
            self.runtime_xml.unlink(missing_ok=True)
            raise
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.timestep
        self.joint_ids = np.array([self.model.joint(name).id for name in cfg.joint_names], dtype=int)
        self.joint_qpos = self.model.jnt_qposadr[self.joint_ids].astype(int)
        self.joint_qvel = self.model.jnt_dofadr[self.joint_ids].astype(int)
        actuator_ids = []
        for name in cfg.joint_names:
            joint_id = self.model.joint(name).id
            matches = np.flatnonzero(self.model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                raise ValueError(f"Expected one actuator for joint '{name}', got {matches.size}")
            actuator_ids.append(int(matches[0]))
        self.actuator_ids = np.asarray(actuator_ids, dtype=int)
        joint_limited = self.model.jnt_limited[self.joint_ids].astype(bool)
        actuator_limited = self.model.actuator_ctrllimited[self.actuator_ids].astype(bool)
        joint_range = self.model.jnt_range[self.joint_ids]
        actuator_range = self.model.actuator_ctrlrange[self.actuator_ids]
        lower = np.where(joint_limited, joint_range[:, 0], -np.inf)
        upper = np.where(joint_limited, joint_range[:, 1], np.inf)
        lower = np.where(actuator_limited, np.maximum(lower, actuator_range[:, 0]), lower)
        upper = np.where(actuator_limited, np.minimum(upper, actuator_range[:, 1]), upper)
        if np.any(lower > upper):
            invalid = [cfg.joint_names[index] for index in np.flatnonzero(lower > upper)]
            raise ValueError(f"Joint and actuator position ranges do not overlap: {invalid}")
        self.joint_target_range = np.column_stack((lower, upper))
        self.effort_range = self.model.actuator_forcerange[self.actuator_ids].copy()
        self.imu_adr = int(self.model.sensor_adr[self.model.sensor("imu_ang_vel").id])
        if self.model.nu != cfg.action_dim:
            raise ValueError(f"Expected {cfg.action_dim} actuators, got {self.model.nu}")
        self._target = cfg.lying_joint.copy()
        self._velocity_target = np.zeros(cfg.action_dim, dtype=np.float64)
        self._position_kp = np.full(cfg.action_dim, cfg.kp, dtype=np.float64)
        self._position_kd = np.full(cfg.action_dim, cfg.kd, dtype=np.float64)
        damping = cfg.kd if damping_kd is None else damping_kd
        self._damping_kd = self._gain_vector(damping, "damping_kd")
        self._control_mode = ControlMode.DISABLED
        self._emergency_latched = False
        self._emergency_reason: str | None = None
        self._sequence = 0
        self._reset_pose()

    def _gain_vector(self, value: Gain, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self.cfg.action_dim, float(array), dtype=np.float64)
        if array.shape != (self.cfg.action_dim,):
            raise ValueError(f"{name} must be scalar or have shape ({self.cfg.action_dim},), got {array.shape}")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must contain finite, non-negative values")
        return array.copy()

    def _apply_servo(
        self,
        kp: np.ndarray,
        kd: np.ndarray,
        velocity: np.ndarray | None = None,
        effort: np.ndarray | None = None,
    ) -> None:
        """Configure MuJoCo's affine position actuators as a joint PD law."""

        velocity = np.zeros(self.cfg.action_dim) if velocity is None else velocity
        effort = np.zeros(self.cfg.action_dim) if effort is None else effort
        ids = self.actuator_ids
        self.model.actuator_gainprm[ids, 0] = kp
        self.model.actuator_biasprm[ids, 0] = kd * velocity + effort
        self.model.actuator_biasprm[ids, 1] = -kp
        self.model.actuator_biasprm[ids, 2] = -kd

    def _disable_actuators(self) -> None:
        zeros = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self._apply_servo(zeros, zeros)
        self.data.ctrl[self.actuator_ids] = 0.0

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    def _reset_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._set_base_pose_for_reset(self.cfg.lying_pos, self.cfg.lying_quat)
        self.data.qpos[self.joint_qpos] = self.cfg.lying_joint
        self.data.qvel[:] = 0.0
        self._target = self.cfg.lying_joint.copy()
        self._velocity_target.fill(0.0)
        self._emergency_latched = False
        self._emergency_reason = None
        self._control_mode = ControlMode.DAMPING
        self._apply_servo(np.zeros(self.cfg.action_dim), self._damping_kd)
        self.data.ctrl[self.actuator_ids] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_base_pose_for_reset(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("reset base pose must contain position[3] and quaternion[4]")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ValueError("reset base pose must be finite")
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("reset base quaternion must be non-zero")
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = quaternion / norm

    def read_state(self) -> RobotState:
        valid = self.finite()
        sequence = self._sequence
        self._sequence += 1
        return RobotState(
            time=float(self.data.time),
            position=self.data.qpos[:3].copy(),
            quaternion=self.data.qpos[3:7].copy(),
            joint_position=self.data.qpos[self.joint_qpos].copy(),
            joint_velocity=self.data.qvel[self.joint_qvel].copy(),
            angular_velocity_body=self.data.sensordata[self.imu_adr:self.imu_adr + 3].copy(),
            valid=valid,
            connected=True,
            monotonic_time=time.monotonic(),
            sequence=sequence,
        )

    def set_control_mode(self, mode: ControlMode, damping_kd: Gain | None = None) -> None:
        try:
            requested = mode if isinstance(mode, ControlMode) else ControlMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported control mode: {mode!r}") from exc
        if self._emergency_latched and requested is not ControlMode.EMERGENCY_STOP:
            raise RuntimeError("Emergency stop is latched; call reset() before enabling actuators")
        if requested is ControlMode.EMERGENCY_STOP:
            self.emergency_stop()
            return
        if requested is ControlMode.DAMPING:
            if damping_kd is not None:
                self._damping_kd = self._gain_vector(damping_kd, "damping_kd")
            self._apply_servo(np.zeros(self.cfg.action_dim), self._damping_kd)
            self.data.ctrl[self.actuator_ids] = 0.0
        elif requested is ControlMode.POSITION:
            self._apply_servo(self._position_kp, self._position_kd, self._velocity_target)
            self.data.ctrl[self.actuator_ids] = self._target
        else:  # DISABLED
            self._disable_actuators()
        self._control_mode = requested

    def write_joint_command(self, command: JointCommand) -> None:
        if not isinstance(command, JointCommand):
            raise TypeError(f"command must be JointCommand, got {type(command).__name__}")
        if self._control_mode is ControlMode.EMERGENCY_STOP or self._emergency_latched:
            raise RuntimeError("Cannot write a command while emergency stop is latched")
        if self._control_mode is ControlMode.DISABLED:
            raise RuntimeError("Cannot write a command while control is disabled")
        if self._control_mode is ControlMode.DAMPING:
            validated = command.validate(self.cfg.action_dim)
            if validated.kp is not None and np.any(validated.kp != 0.0):
                raise ValueError("DAMPING mode requires kp=0")
            if validated.velocity is not None and np.any(validated.velocity != 0.0):
                raise ValueError("DAMPING mode does not accept a non-zero velocity target")
            if validated.feedforward_effort is not None and np.any(validated.feedforward_effort != 0.0):
                raise ValueError("DAMPING mode does not accept feedforward effort")
            if validated.kd is not None:
                self._damping_kd = validated.kd
            self._apply_servo(np.zeros(self.cfg.action_dim), self._damping_kd)
            self.data.ctrl[self.actuator_ids] = 0.0
            return

        validated = command.validate(self.cfg.action_dim, require_position=True)
        assert validated.position is not None
        self._target = np.clip(
            validated.position,
            self.joint_target_range[:, 0],
            self.joint_target_range[:, 1],
        )
        self._velocity_target = (
            np.zeros(self.cfg.action_dim, dtype=np.float64)
            if validated.velocity is None
            else validated.velocity
        )
        self._position_kp = (
            np.full(self.cfg.action_dim, self.cfg.kp, dtype=np.float64)
            if validated.kp is None
            else validated.kp
        )
        self._position_kd = (
            np.full(self.cfg.action_dim, self.cfg.kd, dtype=np.float64)
            if validated.kd is None
            else validated.kd
        )
        effort = (
            np.zeros(self.cfg.action_dim, dtype=np.float64)
            if validated.feedforward_effort is None
            else np.clip(validated.feedforward_effort, self.effort_range[:, 0], self.effort_range[:, 1])
        )
        self._apply_servo(self._position_kp, self._position_kd, self._velocity_target, effort)
        self.data.ctrl[self.actuator_ids] = self._target

    def set_joint_target(self, target: np.ndarray) -> None:
        """Compatibility adapter for the original position-only backend API."""

        if self._control_mode is not ControlMode.POSITION:
            self.set_control_mode(ControlMode.POSITION)
        self.write_joint_command(JointCommand(position=target))

    def set_base_pose(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        del position, quaternion
        raise RuntimeError("set_base_pose is reset-only; runtime FSM transitions must use joint commands")

    def emergency_stop(self, reason: str | None = None) -> None:
        self._emergency_latched = True
        self._emergency_reason = reason
        self._control_mode = ControlMode.EMERGENCY_STOP
        self._disable_actuators()

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        self._reset_pose()

    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

    def close(self) -> None:
        self.runtime_xml.unlink(missing_ok=True)
