from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .config import SimConfig
from .types import RobotState

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("MuJoCo is required. Install with `uv sync --extra sim2sim`.") from exc


def _runtime_xml(source: Path, cfg: SimConfig) -> Path:
    """Build a temporary simulation scene without modifying the task asset."""
    tree = ET.parse(source)
    root = tree.getroot()
    option = root.find("option")
    if option is None:
        option = ET.Element("option", {"timestep": str(cfg.timestep), "integrator": "implicitfast"})
        root.insert(0, option)
    else:
        option.set("timestep", str(cfg.timestep))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    if worldbody.find("geom[@name='sim2sim_ground']") is None:
        worldbody.insert(0, ET.Element("geom", {
            "name": "sim2sim_ground", "type": "plane", "pos": "0 0 0",
            "size": "20 20 0.1", "friction": "0.8 0.01 0.001",
            "contype": "1", "conaffinity": "1",
        }))
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.Element("actuator")
        root.append(actuators)
    existing = {node.get("joint") for node in actuators.findall("position")}
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

    def __init__(self, cfg: SimConfig):
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
        self.joint_qpos = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in cfg.joint_names], dtype=int)
        self.joint_qvel = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in cfg.joint_names], dtype=int)
        self.actuator_ids = np.array([self.model.actuator(f"{name}_position").id for name in cfg.joint_names], dtype=int)
        self.imu_adr = int(self.model.sensor_adr[self.model.sensor("imu_ang_vel").id])
        if self.model.nu != cfg.action_dim:
            raise ValueError(f"Expected {cfg.action_dim} actuators, got {self.model.nu}")
        self._target = cfg.lying_joint.copy()
        self._reset_pose()

    def _reset_pose(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.cfg.lying_pos
        self.data.qpos[3:7] = self.cfg.lying_quat
        self.data.qpos[self.joint_qpos] = self.cfg.lying_joint
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def read_state(self) -> RobotState:
        return RobotState(
            time=float(self.data.time),
            position=self.data.qpos[:3].copy(),
            quaternion=self.data.qpos[3:7].copy(),
            joint_position=self.data.qpos[self.joint_qpos].copy(),
            joint_velocity=self.data.qvel[self.joint_qvel].copy(),
            angular_velocity_body=self.data.sensordata[self.imu_adr:self.imu_adr + 3].copy(),
        )

    def set_joint_target(self, target: np.ndarray) -> None:
        ctrlrange = self.model.actuator_ctrlrange[self.actuator_ids]
        self._target = np.clip(np.asarray(target, dtype=np.float64), ctrlrange[:, 0], ctrlrange[:, 1])
        self.data.ctrl[self.actuator_ids] = self._target

    def set_base_pose(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = quaternion
        self.data.qvel[:6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        self._reset_pose()

    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel)))

    def close(self) -> None:
        self.runtime_xml.unlink(missing_ok=True)
