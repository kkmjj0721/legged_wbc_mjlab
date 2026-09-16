from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECK_URDF = ROOT / "tools" / "check_urdf.py"
REAL_RI_PLUS_URDF = ROOT / "src/assets/robots/ri_plus_4438/xmls/urdf/ri_plus_4438.urdf"


def _run_check(args: list[object], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-B", str(CHECK_URDF), *[str(arg) for arg in args]]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _free_tcp_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError:
        return 18080 + (os.getpid() % 1000)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        return
    process.terminate()
    try:
        process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5.0)


def _wait_for_http(port: int, process: subprocess.Popen[str], timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1.0)
            raise AssertionError(f"preview process exited early with {process.returncode}\n{stdout}\n{stderr}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=1.0) as response:
                if response.status < 500:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(f"preview HTTP server did not start: {last_error}")


def _start_preview_process(urdf: Path, report: Path, port: int, *extra_args: object) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-B",
            str(CHECK_URDF),
            str(urdf),
            "--preview",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--report",
            str(report),
            *[str(arg) for arg in extra_args],
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_box_stl(path: Path, half_extents: tuple[float, float, float]) -> None:
    hx, hy, hz = half_extents
    v = {
        "nbl": (-hx, -hy, -hz),
        "nbr": (hx, -hy, -hz),
        "ntl": (-hx, hy, -hz),
        "ntr": (hx, hy, -hz),
        "fbl": (-hx, -hy, hz),
        "fbr": (hx, -hy, hz),
        "ftl": (-hx, hy, hz),
        "ftr": (hx, hy, hz),
    }
    faces = [
        ("nbl", "nbr", "ntr"),
        ("nbl", "ntr", "ntl"),
        ("fbl", "ftr", "fbr"),
        ("fbl", "ftl", "ftr"),
        ("nbl", "fbl", "fbr"),
        ("nbl", "fbr", "nbr"),
        ("ntl", "ntr", "ftr"),
        ("ntl", "ftr", "ftl"),
        ("nbl", "ntl", "ftl"),
        ("nbl", "ftl", "fbl"),
        ("nbr", "fbr", "ftr"),
        ("nbr", "ftr", "ntr"),
    ]
    lines = ["solid box"]
    for face in faces:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for key in face:
            lines.append("      vertex %.9f %.9f %.9f" % v[key])
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid box")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture_robot(tmp: Path) -> Path:
    meshes = tmp / "meshes"
    meshes.mkdir()
    mesh_extents = {
        "base_link.STL": (0.18, 0.09, 0.045),
        "FL_hip.STL": (0.035, 0.035, 0.045),
        "FL_thigh.STL": (0.040, 0.025, 0.120),
        "FL_calf.STL": (0.018, 0.018, 0.130),
        "FL_foot.STL": (0.022, 0.018, 0.015),
        "arm_link.STL": (0.025, 0.030, 0.080),
        "mystery_link.STL": (0.050, 0.020, 0.030),
    }
    for name, half_extents in mesh_extents.items():
        _write_box_stl(meshes / name, half_extents)

    urdf = """<?xml version="1.0"?>
<robot name="check_urdf_fixture">
  <checker_extension keep="yes"><child value="42" /></checker_extension>
  <link name="base_link">
    <inertial>
      <origin xyz="0.001 0.002 0.003" rpy="0.1 -0.2 0.3" />
      <mass value="4.0" />
      <inertia ixx="0.20" ixy="0.01" ixz="0.02" iyy="0.30" iyz="-0.015" izz="0.40" />
    </inertial>
    <visual name="base_visual">
      <origin xyz="0.01 -0.02 0.03" rpy="0.05 0.02 -0.03" />
      <geometry><mesh filename="meshes/base_link.STL" /></geometry>
    </visual>
    <collision name="base_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/base_link.STL" /></geometry>
    </collision>
  </link>
  <link name="FL_hip">
    <inertial>
      <origin xyz="0.005 0.001 -0.002" rpy="0.2 0.1 -0.1" />
      <mass value="0.8" />
      <inertia ixx="0.030" ixy="0.001" ixz="0.0" iyy="0.032" iyz="0.0" izz="0.034" />
    </inertial>
    <visual name="hip_visual">
      <origin xyz="0.01 0.00 0.00" rpy="0.01 0.02 0.03" />
      <geometry><mesh filename="meshes/FL_hip.STL" /></geometry>
    </visual>
    <collision name="FL_hip_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/FL_hip.STL" /></geometry>
    </collision>
  </link>
  <joint name="FL_hip_joint" type="revolute">
    <origin xyz="0.17 0.08 0.02" rpy="0.15 0.10 -0.20" />
    <parent link="base_link" />
    <child link="FL_hip" />
    <axis xyz="1 0 0" />
    <limit lower="-0.7" upper="0.9" effort="12" velocity="4" />
  </joint>
  <link name="FL_thigh">
    <inertial>
      <origin xyz="-0.003 0.008 -0.050" rpy="0.4 -0.3 0.2" />
      <mass value="1.2" />
      <inertia ixx="0.080" ixy="-0.003" ixz="0.004" iyy="0.090" iyz="0.002" izz="0.060" />
    </inertial>
    <visual name="thigh_visual">
      <origin xyz="-0.01 0.02 -0.01" rpy="-0.04 0.03 0.02" />
      <geometry><mesh filename="meshes/FL_thigh.STL" /></geometry>
    </visual>
    <collision name="FL_thigh_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/FL_thigh.STL" /></geometry>
    </collision>
  </link>
  <joint name="FL_thigh_joint" type="revolute">
    <origin xyz="0.02 0.07 -0.03" rpy="-0.10 0.05 0.18" />
    <parent link="FL_hip" />
    <child link="FL_thigh" />
    <axis xyz="0 -1 0" />
    <limit lower="-0.25" upper="1.10" effort="14" velocity="5" />
  </joint>
  <link name="FL_calf">
    <inertial>
      <origin xyz="0.006 -0.002 -0.070" rpy="-0.2 0.25 0.1" />
      <mass value="0.4" />
      <inertia ixx="0.050" ixy="0.0" ixz="0.003" iyy="0.052" iyz="-0.001" izz="0.012" />
    </inertial>
    <visual name="calf_visual">
      <origin xyz="0.005 -0.006 -0.015" rpy="0.03 -0.02 0.04" />
      <geometry><mesh filename="meshes/FL_calf.STL" /></geometry>
    </visual>
    <collision name="FL_calf_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/FL_calf.STL" /></geometry>
    </collision>
  </link>
  <joint name="FL_calf_joint" type="revolute">
    <origin xyz="-0.01 0.00 -0.16" rpy="0.07 -0.03 0.02" />
    <parent link="FL_thigh" />
    <child link="FL_calf" />
    <axis xyz="0 -1 0" />
    <limit lower="-0.70" upper="0.90" effort="10" velocity="6" />
    <mimic joint="FL_thigh_joint" multiplier="2.0" offset="0.1" />
  </joint>
  <link name="FL_foot">
    <inertial>
      <origin xyz="0.0 0.0 -0.010" rpy="0 0 0" />
      <mass value="0.1" />
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002" />
    </inertial>
    <visual name="foot_visual">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/FL_foot.STL" /></geometry>
    </visual>
    <collision name="FL_foot_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/FL_foot.STL" /></geometry>
    </collision>
  </link>
  <joint name="FL_foot_joint" type="fixed">
    <origin xyz="0 0 -0.15" rpy="0 0 0" />
    <parent link="FL_calf" />
    <child link="FL_foot" />
  </joint>
  <link name="arm_link">
    <inertial>
      <origin xyz="0.0 0.0 0.04" rpy="0.1 0.0 0.2" />
      <mass value="0.5" />
      <inertia ixx="0.020" ixy="0" ixz="0" iyy="0.021" iyz="0" izz="0.022" />
    </inertial>
    <visual name="arm_visual">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/arm_link.STL" /></geometry>
    </visual>
    <collision name="arm_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/arm_link.STL" /></geometry>
    </collision>
  </link>
  <joint name="shoulder_joint" type="revolute">
    <origin xyz="-0.10 0.0 0.08" rpy="0.2 0.1 0.0" />
    <parent link="base_link" />
    <child link="arm_link" />
    <axis xyz="0 -1 0" />
    <limit lower="-0.30" upper="0.80" effort="7" velocity="3" />
  </joint>
  <link name="sensor_link">
    <visual name="sensor_visual">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><box size="0.02 0.03 0.04" /></geometry>
    </visual>
    <collision name="sensor_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><box size="0.02 0.03 0.04" /></geometry>
    </collision>
  </link>
  <joint name="sensor_joint" type="fixed">
    <origin xyz="0.0 0.0 0.12" rpy="0 0 0" />
    <parent link="base_link" />
    <child link="sensor_link" />
  </joint>
  <link name="mystery_link">
    <inertial>
      <origin xyz="0.001 0.002 0.003" rpy="0 0 0" />
      <mass value="0.25" />
      <inertia ixx="0.010" ixy="0" ixz="0" iyy="0.011" iyz="0" izz="0.012" />
    </inertial>
    <visual name="mystery_visual">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/mystery_link.STL" /></geometry>
    </visual>
    <collision name="mystery_collision">
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry><mesh filename="meshes/mystery_link.STL" /></geometry>
    </collision>
  </link>
  <joint name="mystery_joint" type="fixed">
    <origin xyz="0.0 -0.13 0.02" rpy="0.1 0 0" />
    <parent link="base_link" />
    <child link="mystery_link" />
  </joint>
</robot>
"""
    path = tmp / "fixture.urdf"
    path.write_text(urdf, encoding="utf-8")
    _set_fixture_root_axes(
        path,
        {
            "FL_hip_joint": (1.0, 0.0, 0.0),
            "FL_thigh_joint": (0.0, -1.0, 0.0),
            "FL_calf_joint": (0.0, -1.0, 0.0),
            "shoulder_joint": (0.0, -1.0, 0.0),
        },
    )
    return path


def _bad_robot(tmp: Path) -> Path:
    meshes = tmp / "meshes"
    meshes.mkdir()
    _write_box_stl(meshes / "valid.STL", (0.05, 0.04, 0.03))
    urdf = """<?xml version="1.0"?>
<robot name="bad_fixture">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="1.0" />
      <inertia ixx="-0.1" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual><geometry><mesh filename="meshes/missing.STL" /></geometry></visual>
    <collision><geometry><mesh filename="meshes/valid.STL" /></geometry></collision>
  </link>
  <link name="sensor_link">
    <visual><geometry><box size="0.01 0.01 0.01" /></geometry></visual>
  </link>
  <joint name="sensor_joint" type="fixed">
    <parent link="base_link" />
    <child link="sensor_link" />
    <origin xyz="0 0 0.1" rpy="0 0 0" />
  </joint>
</robot>
"""
    path = tmp / "bad.urdf"
    path.write_text(urdf, encoding="utf-8")
    return path


def _invalid_inertia_preview_robot(tmp: Path) -> Path:
    urdf = _fixture_robot(tmp)
    tree, root, _, _ = _parse_model(urdf)
    inertia = root.find("link/inertial/inertia")
    if inertia is None:
        raise AssertionError("fixture lost base inertia")
    inertia.set("ixx", "-0.20")
    tree.write(urdf, encoding="utf-8", xml_declaration=True)
    return urdf


def _mimic_chain_robot(tmp: Path, mode: str) -> Path:
    mimic = {"j1": "", "j2": "", "j3": ""}
    if mode == "self_loop":
        mimic["j1"] = '<mimic joint="j1" multiplier="1.0" offset="0.1" />'
    elif mode == "indirect_loop":
        mimic["j1"] = '<mimic joint="j2" multiplier="1.0" offset="0.1" />'
        mimic["j2"] = '<mimic joint="j1" multiplier="1.0" offset="0.1" />'
    elif mode == "long_chain":
        mimic["j2"] = '<mimic joint="j1" multiplier="2.0" offset="0.1" />'
        mimic["j3"] = '<mimic joint="j2" multiplier="-0.5" offset="0.2" />'
    elif mode == "rules_loop":
        pass
    else:
        raise AssertionError(f"unknown mimic mode {mode}")
    links = "\n".join(
        f"""
  <link name="{name}">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <mass value="1.0" />
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01" />
    </inertial>
    <visual><geometry><box size="0.02 0.02 0.02" /></geometry></visual>
    <collision><geometry><box size="0.02 0.02 0.02" /></geometry></collision>
  </link>"""
        for name in ("base", "l1", "l2", "l3")
    )
    joints = f"""
  <joint name="j1" type="revolute">
    <origin xyz="0.1 0 0" rpy="0.1 0.0 0.0" />
    <parent link="base" /><child link="l1" />
    <axis xyz="0 0 1" />
    <limit lower="-1" upper="1" effort="1" velocity="1" />
    {mimic["j1"]}
  </joint>
  <joint name="j2" type="revolute">
    <origin xyz="0.1 0 0" rpy="0.0 0.2 0.0" />
    <parent link="l1" /><child link="l2" />
    <axis xyz="0 0 1" />
    <limit lower="-1" upper="1" effort="1" velocity="1" />
    {mimic["j2"]}
  </joint>
  <joint name="j3" type="revolute">
    <origin xyz="0.1 0 0" rpy="0.0 0.0 -0.3" />
    <parent link="l2" /><child link="l3" />
    <axis xyz="0 0 1" />
    <limit lower="-1" upper="1" effort="1" velocity="1" />
    {mimic["j3"]}
  </joint>
"""
    path = tmp / f"mimic_{mode}.urdf"
    path.write_text(f'<?xml version="1.0"?><robot name="mimic_{mode}">{links}{joints}</robot>\n', encoding="utf-8")
    return path


def _vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(x) for x in text.split()], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray | None = None, rpy: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if rpy is not None:
        out[:3, :3] = _rpy_matrix(rpy)
    if xyz is not None:
        out[:3, 3] = xyz
    return out


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    origin = element.find("origin")
    if origin is None:
        return np.eye(4, dtype=np.float64)
    return _transform(_vec(origin.get("xyz"), (0.0, 0.0, 0.0)), _vec(origin.get("rpy"), (0.0, 0.0, 0.0)))


def _format_vec(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _set_fixture_root_axes(path: Path, root_axes: dict[str, tuple[float, float, float]]) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    children = {joint.find("child").get("link") for joint in joints.values()}
    roots = [name for name in links if name not in children]
    if not roots:
        raise AssertionError("fixture has no root link")
    world_rot = {roots[0]: np.eye(3, dtype=np.float64)}
    pending = set(joints)
    while pending:
        progressed = False
        for name in list(pending):
            joint = joints[name]
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in world_rot:
                continue
            origin = joint.find("origin")
            rpy = _vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            joint_rot = world_rot[parent] @ _rpy_matrix(rpy)
            if name in root_axes:
                desired = np.array(root_axes[name], dtype=np.float64)
                desired = desired / np.linalg.norm(desired)
                axis_local = joint_rot.T @ desired
                axis_local = axis_local / np.linalg.norm(axis_local)
                axis = joint.find("axis")
                if axis is None:
                    axis = ET.SubElement(joint, "axis")
                axis.set("xyz", _format_vec(axis_local))
            world_rot[child] = joint_rot
            pending.remove(name)
            progressed = True
        if not progressed:
            raise AssertionError(f"unconnected joints remain while setting axes: {sorted(pending)}")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _motion_transform(joint: ET.Element, q: float) -> np.ndarray:
    jtype = joint.get("type", "fixed")
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis_element = joint.find("axis")
    if axis_element is not None:
        axis = _vec(axis_element.get("xyz"), (1.0, 0.0, 0.0))
    if jtype in ("revolute", "continuous"):
        return _transform(rpy=None) @ np.block(
            [
                [_axis_angle(axis, q), np.zeros((3, 1))],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )
    if jtype == "prismatic":
        return _transform(axis * q, None)
    return np.eye(4, dtype=np.float64)


def _parse_model(path: Path) -> tuple[ET.ElementTree, ET.Element, dict[str, ET.Element], dict[str, ET.Element]]:
    tree = ET.parse(path)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    return tree, root, links, joints


def _joint_value(joints: dict[str, ET.Element], name: str, q_values: dict[str, float], seen: set[str] | None = None) -> float:
    if seen is None:
        seen = set()
    if name in seen:
        raise AssertionError(f"mimic cycle involving {name}")
    seen.add(name)
    joint = joints[name]
    mimic = joint.find("mimic")
    if mimic is None:
        return float(q_values.get(name, 0.0))
    source = mimic.get("joint")
    if source not in joints:
        raise AssertionError(f"mimic source {source!r} missing")
    multiplier = float(mimic.get("multiplier", "1"))
    offset = float(mimic.get("offset", "0"))
    return multiplier * _joint_value(joints, source, q_values, seen) + offset


def _link_world_transforms(path: Path, q_values: dict[str, float]) -> dict[str, np.ndarray]:
    _, _, links, joints = _parse_model(path)
    children = {joint.find("child").get("link") for joint in joints.values()}
    roots = [name for name in links if name not in children]
    if not roots:
        raise AssertionError("URDF has no root link")
    world = {roots[0]: np.eye(4, dtype=np.float64)}
    pending = set(joints)
    while pending:
        progressed = False
        for name in list(pending):
            joint = joints[name]
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in world:
                continue
            q = _joint_value(joints, name, q_values)
            world[child] = world[parent] @ _origin_transform(joint) @ _motion_transform(joint, q)
            pending.remove(name)
            progressed = True
        if not progressed:
            raise AssertionError(f"unconnected joints remain: {sorted(pending)}")
    return world


def _structural_zero_link_world_transforms(path: Path) -> dict[str, np.ndarray]:
    _, _, links, joints = _parse_model(path)
    children = {joint.find("child").get("link") for joint in joints.values()}
    roots = [name for name in links if name not in children]
    if not roots:
        raise AssertionError("URDF has no root link")
    world = {roots[0]: np.eye(4, dtype=np.float64)}
    pending = set(joints)
    while pending:
        progressed = False
        for name in list(pending):
            joint = joints[name]
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in world:
                continue
            world[child] = world[parent] @ _origin_transform(joint)
            pending.remove(name)
            progressed = True
        if not progressed:
            raise AssertionError(f"unconnected joints remain: {sorted(pending)}")
    return world


def _assert_structural_zero_rotations_identity(test: unittest.TestCase, path: Path) -> None:
    for link_name, transform in _structural_zero_link_world_transforms(path).items():
        np.testing.assert_allclose(
            transform[:3, :3],
            np.eye(3),
            atol=1.0e-6,
            rtol=0.0,
            err_msg=f"{link_name} structural q=0 rotation should be identity",
        )


def _visual_world_transforms(path: Path, q_values: dict[str, float]) -> dict[str, np.ndarray]:
    _, _, links, _ = _parse_model(path)
    link_world = _link_world_transforms(path, q_values)
    visual_world: dict[str, np.ndarray] = {}
    for link_name, link in links.items():
        visuals = link.findall("visual")
        if not visuals:
            visual_world[f"{link_name}:link"] = link_world[link_name]
            continue
        for index, visual in enumerate(visuals):
            visual_world[f"{link_name}:visual:{index}"] = link_world[link_name] @ _origin_transform(visual)
    return visual_world


def _collision_world_transforms(path: Path, q_values: dict[str, float], *, primitive_only: bool = False) -> dict[str, np.ndarray]:
    _, _, links, _ = _parse_model(path)
    link_world = _link_world_transforms(path, q_values)
    collision_world: dict[str, np.ndarray] = {}
    for link_name, link in links.items():
        for index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            if primitive_only and (geometry is None or geometry.find("mesh") is not None):
                continue
            name = collision.get("name") or str(index)
            collision_world[f"{link_name}:collision:{name}"] = link_world[link_name] @ _origin_transform(collision)
    return collision_world


def _physical_joint_axis_root(path: Path, joint_name: str) -> np.ndarray:
    _, _, _, joints = _parse_model(path)
    joint = joints[joint_name]
    parent = joint.find("parent").get("link")
    parent_world = _link_world_transforms(path, {})[parent]
    axis = _axis(ET.parse(path).getroot(), joint_name)
    axis = axis / np.linalg.norm(axis)
    root_axis = (parent_world[:3, :3] @ _origin_transform(joint)[:3, :3]) @ axis
    return root_axis / np.linalg.norm(root_axis)


def _assert_pose_maps_close(test: unittest.TestCase, before: dict[str, np.ndarray], after: dict[str, np.ndarray]) -> None:
    test.assertEqual(set(before), set(after))
    for key in sorted(before):
        np.testing.assert_allclose(after[key], before[key], atol=1.0e-6, rtol=0.0, err_msg=key)


def _joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    raise AssertionError(f"missing joint {name}")


def _axis(root: ET.Element, name: str) -> np.ndarray:
    axis = _joint(root, name).find("axis")
    if axis is None:
        return np.zeros(3)
    return _vec(axis.get("xyz"), (0.0, 0.0, 0.0))


def _limit(root: ET.Element, name: str) -> tuple[float, float]:
    limit = _joint(root, name).find("limit")
    if limit is None:
        raise AssertionError(f"missing limit on {name}")
    return float(limit.get("lower")), float(limit.get("upper"))


def _mimic(root: ET.Element, name: str) -> tuple[str, float, float]:
    mimic = _joint(root, name).find("mimic")
    if mimic is None:
        raise AssertionError(f"missing mimic on {name}")
    return mimic.get("joint"), float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))


def _masses(root: ET.Element) -> dict[str, float]:
    out = {}
    for link in root.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            out[link.get("name")] = float(mass.get("value"))
    return out


def _total_mass(root: ET.Element) -> float:
    return float(sum(_masses(root).values()))


def _inertial_signature(root: ET.Element) -> dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    out = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        if inertia is None:
            continue
        xyz = tuple(_vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)))
        rpy = tuple(_vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)))
        tensor = tuple(float(inertia.get(name, "0")) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
        out[link.get("name")] = (xyz, rpy, tensor)
    return out


def _topology(root: ET.Element) -> list[tuple[str, str, str, str]]:
    topo = []
    for joint in root.findall("joint"):
        topo.append(
            (
                joint.get("name"),
                joint.get("type", ""),
                joint.find("parent").get("link"),
                joint.find("child").get("link"),
            )
        )
    return sorted(topo)


def _mesh_filenames(root: ET.Element) -> list[str]:
    names = []
    for mesh in root.findall(".//mesh"):
        names.append(mesh.get("filename"))
    return sorted(names)


def _visual_mesh_filenames(root: ET.Element) -> list[str]:
    names = []
    for mesh in root.findall(".//visual/geometry/mesh"):
        names.append(mesh.get("filename"))
    return sorted(names)


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_report_schema(test: unittest.TestCase, report: dict) -> None:
    test.assertEqual(report.get("schema_version"), 1)
    test.assertIsInstance(report.get("source"), str)
    test.assertIn(report.get("profile"), {"generic", "go2", "ri4438"})
    test.assertIsInstance(report.get("issues"), list)
    test.assertIsInstance(report.get("changes"), list)
    test.assertIsInstance(report.get("joint_coordinate_map"), dict)
    test.assertIsInstance(report.get("validation"), dict)
    summary = report.get("summary")
    test.assertIsInstance(summary, dict)
    for key in ("errors", "warnings", "infos"):
        test.assertIsInstance(summary.get(key), int)
    for issue in report["issues"]:
        test.assertIn(issue.get("severity"), {"error", "warning", "info"})
        test.assertIn(issue.get("status"), {"detected", "needs_reference"})
        test.assertIsInstance(issue.get("code"), str)
        test.assertIsInstance(issue.get("object"), str)
        test.assertIsInstance(issue.get("message"), str)


def _report_text(report: dict) -> str:
    return json.dumps(report.get("issues", []), sort_keys=True).lower()


def _stl_vertices(path: Path) -> np.ndarray:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("vertex "):
            continue
        points.append([float(value) for value in stripped.split()[1:4]])
    if not points:
        raise AssertionError(f"no vertices parsed from {path}")
    return np.array(points, dtype=np.float64)


def _resolve_mesh(filename: str, urdf_dir: Path) -> Path:
    if filename.startswith("package://"):
        tail_parts = filename[len("package://") :].split("/", 1)
        if len(tail_parts) == 2:
            candidates = list(ROOT.rglob(tail_parts[1]))
            if len(candidates) == 1:
                return candidates[0]
        raise AssertionError(f"cannot resolve package mesh {filename}")
    path = Path(filename)
    if path.is_absolute():
        return path
    return (urdf_dir / path).resolve()


def _collision_geometry(link: ET.Element) -> tuple[ET.Element, ET.Element]:
    collisions = link.findall("collision")
    if len(collisions) != 1:
        raise AssertionError(f"{link.get('name')} expected one collision, got {len(collisions)}")
    geometry = collisions[0].find("geometry")
    if geometry is None:
        raise AssertionError(f"{link.get('name')} missing collision geometry")
    return collisions[0], geometry


def _assert_collision_encloses_visual_mesh(test: unittest.TestCase, output_urdf: Path, link_name: str, primitive: str) -> None:
    _, _, links, _ = _parse_model(output_urdf)
    link = links[link_name]
    collision, geometry = _collision_geometry(link)
    prim = geometry.find(primitive)
    test.assertIsNotNone(prim, f"{link_name} should use {primitive} collision")
    visual = link.find("visual")
    test.assertIsNotNone(visual, f"{link_name} missing visual")
    mesh = visual.find("geometry/mesh")
    test.assertIsNotNone(mesh, f"{link_name} visual should still reference mesh")
    vertices = _stl_vertices(_resolve_mesh(mesh.get("filename"), output_urdf.parent))
    visual_tf = _origin_transform(visual)
    collision_tf = _origin_transform(collision)
    homog = np.c_[vertices, np.ones(len(vertices))]
    points_link = (visual_tf @ homog.T).T[:, :3]
    points_col = (np.linalg.inv(collision_tf) @ np.c_[points_link, np.ones(len(points_link))].T).T[:, :3]
    tol = 1.0e-6
    if primitive == "box":
        size = _vec(prim.get("size"), (0.0, 0.0, 0.0))
        test.assertTrue(np.all(size > 0), f"{link_name} box size must be positive")
        test.assertLessEqual(float(np.max(np.abs(points_col) - size / 2.0)), tol, link_name)
    elif primitive == "cylinder":
        radius = float(prim.get("radius"))
        length = float(prim.get("length"))
        test.assertGreater(radius, 0.0)
        test.assertGreater(length, 0.0)
        radial = np.linalg.norm(points_col[:, :2], axis=1)
        test.assertLessEqual(float(np.max(radial - radius)), tol, link_name)
        test.assertLessEqual(float(np.max(np.abs(points_col[:, 2]) - length / 2.0)), tol, link_name)
    elif primitive == "sphere":
        radius = float(prim.get("radius"))
        test.assertGreater(radius, 0.0)
        test.assertLessEqual(float(np.max(np.linalg.norm(points_col, axis=1) - radius)), tol, link_name)
    else:
        raise AssertionError(f"unsupported primitive {primitive}")


def _decode_viser_packet(raw: bytes) -> list[dict]:
    import msgspec.msgpack
    import zstandard

    decompressed_size = int.from_bytes(raw[:8], "little")
    compressed_size = int.from_bytes(raw[8:16], "little")
    compressed = raw[16 : 16 + compressed_size]
    inner = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=decompressed_size)
    decoded = msgspec.msgpack.decode(inner)
    return list(decoded.get("messages", ()))


def _encode_viser_message(message: object) -> bytes:
    import msgspec.msgpack

    return msgspec.msgpack.encode(message.as_serializable_dict())


async def _recv_viser_messages(websocket: object, timeout: float, predicate=None) -> list[dict]:
    messages: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.01, deadline - time.monotonic())
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if not isinstance(raw, bytes):
            continue
        messages.extend(_decode_viser_packet(raw))
        if predicate is not None and predicate(messages):
            break
    return messages


def _gui_text(message: dict) -> str:
    pieces = [str(message.get("value", ""))]
    props = message.get("props")
    if isinstance(props, dict):
        pieces.append(str(props.get("label", "")))
        options = props.get("options")
        if isinstance(options, (list, tuple)):
            pieces.extend(str(option) for option in options)
    return " ".join(pieces).lower()


def _find_gui(messages: list[dict], message_types: tuple[str, ...], keywords: tuple[str, ...]) -> dict | None:
    for message in messages:
        if message.get("type") not in message_types:
            continue
        text = _gui_text(message)
        if any(keyword in text for keyword in keywords):
            return message
    return None


def _is_scene_message(message: dict) -> bool:
    return message.get("type") in {
        "FrameMessage",
        "BatchedAxesMessage",
        "MeshMessage",
        "BoxMessage",
        "CylinderMessage",
        "IcosphereMessage",
        "SetPositionMessage",
        "SetOrientationMessage",
        "SceneNodeUpdateMessage",
        "RemoveSceneNodeMessage",
    }


def _has_scene_change(messages: list[dict]) -> bool:
    return any(_is_scene_message(message) for message in messages)


async def _exercise_preview_websocket(
    test: unittest.TestCase,
    port: int,
    *,
    require_controls: bool,
    initial_timeout: float = 8.0,
) -> dict[str, int]:
    import viser
    import websockets
    from viser import _messages

    async with websockets.connect(
        f"ws://127.0.0.1:{port}",
        subprotocols=[f"viser-v{viser.__version__}"],
        max_size=256 * 1024 * 1024,
        compression=None,
    ) as websocket:
        initial = await _recv_viser_messages(
            websocket,
            initial_timeout,
            lambda messages: any(message.get("type") == "GuiSliderMessage" for message in messages)
            and any(_is_scene_message(message) for message in messages)
            and any(
                message.get("type") == "MeshMessage" and "/visual/" in str(message.get("name", ""))
                for message in messages
            ),
        )
        scene_messages = [message for message in initial if _is_scene_message(message)]
        sliders = [message for message in initial if message.get("type") == "GuiSliderMessage"]
        test.assertTrue(scene_messages, "preview should publish scene nodes over Viser websocket")
        test.assertTrue(sliders, "preview should publish joint slider GUI controls")
        visual_meshes = [
            message
            for message in initial
            if message.get("type") == "MeshMessage" and "/visual/" in str(message.get("name", ""))
        ]
        visible_meshes = [
            message
            for message in visual_meshes
            if (
                not isinstance(message.get("props"), dict)
                or message["props"].get("opacity") is None
                or float(message["props"].get("opacity")) > 0.99
            )
        ]
        test.assertTrue(visual_meshes, "preview should publish real STL MeshMessage visuals")
        test.assertTrue(visible_meshes, "visual STL meshes should be visible by default")

        camera = _messages.ViewerCameraMessage(
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(1.5, 1.5, 1.0),
            fov=1.0,
            near=0.01,
            far=1000.0,
            image_height=720,
            image_width=1280,
            look_at=(0.0, 0.0, 0.0),
            up_direction=(0.0, 0.0, 1.0),
        )
        await websocket.send(_encode_viser_message(camera))
        await _recv_viser_messages(websocket, 0.25)

        slider = sliders[0]
        props = slider.get("props", {})
        current = float(slider.get("value", 0.0))
        step = float(props.get("step", 0.05)) if isinstance(props, dict) else 0.05
        lower = float(props.get("min", current - 1.0)) if isinstance(props, dict) else current - 1.0
        upper = float(props.get("max", current + 1.0)) if isinstance(props, dict) else current + 1.0
        new_value = min(upper, max(lower, current + max(step, 0.05)))
        if abs(new_value - current) < 1.0e-12:
            new_value = max(lower, current - max(step, 0.05))
        await websocket.send(_encode_viser_message(_messages.GuiUpdateMessage(slider["uuid"], {"value": new_value})))
        slider_update = await _recv_viser_messages(websocket, 5.0, _has_scene_change)
        test.assertTrue(_has_scene_change(slider_update), "updating a joint slider should change the preview scene")

        if require_controls:
            reset = _find_gui(initial, ("GuiButtonMessage",), ("reset", "zero", "home", "0", "复位", "零位"))
            step_button = _find_gui(initial, ("GuiButtonMessage",), ("step", "forward", "positive", "+", "步进", "正向"))
            compare = _find_gui(
                initial,
                ("GuiCheckboxMessage", "GuiDropdownMessage", "GuiButtonGroupMessage"),
                ("original", "candidate", "source", "原模型", "候选", "原始"),
            )
            axes = _find_gui(initial, ("GuiCheckboxMessage", "GuiDropdownMessage", "GuiButtonGroupMessage"), ("axis", "axes", "frame", "坐标"))
            com = _find_gui(initial, ("GuiCheckboxMessage", "GuiDropdownMessage", "GuiButtonGroupMessage"), ("com", "center of mass", "mass center", "质心"))
            collision = _find_gui(initial, ("GuiCheckboxMessage", "GuiDropdownMessage", "GuiButtonGroupMessage"), ("collision", "collider", "碰撞"))
            test.assertIsNotNone(reset, "preview should expose a zero/reset control")
            test.assertIsNotNone(step_button, "preview should expose a positive step control")
            test.assertIsNotNone(compare, "preview should expose an original/candidate toggle")
            test.assertIsNotNone(axes, "preview should expose coordinate-axis overlay control")
            test.assertIsNotNone(com, "preview should expose center-of-mass overlay control")
            test.assertIsNotNone(collision, "preview should expose collision overlay control")

            await websocket.send(_encode_viser_message(_messages.GuiUpdateMessage(step_button["uuid"], {"value": True})))
            step_update = await _recv_viser_messages(websocket, 5.0, _has_scene_change)
            test.assertTrue(_has_scene_change(step_update), "positive step button should update the preview scene")

            compare_payload = _toggle_gui_payload(compare)
            await websocket.send(_encode_viser_message(_messages.GuiUpdateMessage(compare["uuid"], compare_payload)))
            compare_update = await _recv_viser_messages(websocket, 5.0, _has_scene_change)
            test.assertTrue(_has_scene_change(compare_update), "original/candidate toggle should update the preview scene")

            collision_payload = _toggle_gui_payload(collision)
            await websocket.send(_encode_viser_message(_messages.GuiUpdateMessage(collision["uuid"], collision_payload)))
            collision_update = await _recv_viser_messages(websocket, 5.0, _has_scene_change)
            test.assertTrue(_has_scene_change(collision_update), "collision overlay toggle should update the preview scene")

    return {"scene_messages": len(scene_messages), "sliders": len(sliders), "visual_meshes": len(visual_meshes)}


def _toggle_gui_payload(message: dict) -> dict[str, object]:
    message_type = message.get("type")
    if message_type == "GuiCheckboxMessage":
        return {"value": not bool(message.get("value", False))}
    if message_type in {"GuiDropdownMessage", "GuiButtonGroupMessage"}:
        props = message.get("props", {})
        options = props.get("options", ()) if isinstance(props, dict) else ()
        for option in options:
            if option != message.get("value"):
                return {"value": option}
    if message_type == "GuiButtonMessage":
        return {"value": True}
    raise AssertionError(f"cannot toggle GUI message {message}")


class CheckUrdfCliContractTest(unittest.TestCase):
    def test_aggregates_mesh_and_inertia_errors_without_aux_inertial_false_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _bad_robot(tmp)
            report_path = tmp / "report.json"

            result = _run_check([urdf, "--report", report_path])

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertTrue(report_path.exists(), "error reports should still be written")
            report = _load_report(report_path)
            _assert_report_schema(self, report)
            self.assertGreaterEqual(report["summary"]["errors"], 2)
            issue_text = _report_text(report)
            self.assertIn("mesh", issue_text)
            self.assertIn("inertia", issue_text)
            self.assertNotIn("sensor_link", issue_text)

    def test_output_samefile_alias_is_rejected_and_inputs_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            alias = tmp / "alias.urdf"
            try:
                os.link(urdf, alias)
            except OSError:
                os.symlink(urdf, alias)
            report = tmp / "report.json"
            watched = [urdf, alias, *(urdf.parent / "meshes").glob("*.STL")]
            before = {path: _sha256(path) for path in watched}

            result = _run_check([urdf, "--output", alias, "--report", report])

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            after = {path: _sha256(path) for path in watched}
            self.assertEqual(before, after)

    def test_report_rules_output_and_asset_path_conflicts_are_rejected_without_damage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            rules = tmp / "rules.json"
            rules.write_text(json.dumps({"joints": {"FL_thigh_joint": {"axis": [0, 1, 0]}}}), encoding="utf-8")
            stl = urdf.parent / "meshes" / "base_link.STL"
            watched = [urdf, rules, stl]
            cases = [
                ("report_is_source", ["--report", urdf]),
                ("report_is_stl", ["--report", stl]),
                ("output_is_stl", ["--output", stl]),
                ("output_is_rules", ["--rules", rules, "--output", rules]),
                ("report_is_rules", ["--rules", rules, "--report", rules]),
                ("report_is_output", ["--output", tmp / "shared.xml", "--report", tmp / "shared.xml"]),
            ]

            for name, args in cases:
                with self.subTest(name=name):
                    before = {path: _sha256(path) for path in watched}
                    result = _run_check([urdf, *args])
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    after = {path: _sha256(path) for path in watched}
                    self.assertEqual(before, after)

    def test_ri4438_sign_flip_export_preserves_kinematics_limits_mimic_and_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            output = tmp / "candidate.urdf"
            output2 = tmp / "candidate_twice.urdf"
            report_path = tmp / "report.json"
            report2_path = tmp / "report2.json"
            source_root = _parse_model(urdf)[1]

            result = _run_check([urdf, "--profile", "ri4438", "--output", output, "--report", report_path])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists(), "successful export should create the requested output URDF")
            tree, root, _, _ = _parse_model(output)
            self.assertIsNotNone(tree)
            report = _load_report(report_path)
            _assert_report_schema(self, report)
            self.assertEqual(report["summary"]["errors"], 0)
            self.assertEqual(_topology(root), _topology(source_root))
            self.assertEqual(_masses(root), _masses(source_root))
            self.assertEqual(_visual_mesh_filenames(root), _visual_mesh_filenames(source_root))
            self.assertIsNotNone(root.find("checker_extension/child"))
            _assert_structural_zero_rotations_identity(self, output)

            np.testing.assert_allclose(_axis(root, "FL_hip_joint"), [1, 0, 0], atol=1.0e-9)
            np.testing.assert_allclose(_axis(root, "FL_thigh_joint"), [0, 1, 0], atol=1.0e-9)
            np.testing.assert_allclose(_axis(root, "FL_calf_joint"), [0, 1, 0], atol=1.0e-9)
            np.testing.assert_allclose(_axis(root, "shoulder_joint"), [0, -1, 0], atol=1.0e-9)
            np.testing.assert_allclose(_limit(root, "FL_thigh_joint"), [-1.10, 0.25], atol=1.0e-9)
            self.assertEqual(_mimic(root, "FL_calf_joint")[0], "FL_thigh_joint")
            np.testing.assert_allclose(_mimic(root, "FL_calf_joint")[1:], [2.0, -0.1], atol=1.0e-9)

            joint_map = report["joint_coordinate_map"]
            self.assertEqual(joint_map["FL_thigh_joint"]["scale"], -1)
            self.assertEqual(joint_map["FL_calf_joint"]["scale"], -1)
            self.assertTrue(joint_map["FL_thigh_joint"]["equivalent"])
            self.assertTrue(joint_map["FL_calf_joint"]["equivalent"])

            for q in (0.0, 0.23, -0.17):
                before = _visual_world_transforms(urdf, {"FL_hip_joint": 0.11, "FL_thigh_joint": q, "shoulder_joint": -0.19})
                after = _visual_world_transforms(output, {"FL_hip_joint": 0.11, "FL_thigh_joint": -q, "shoulder_joint": -0.19})
                _assert_pose_maps_close(self, before, after)

            second = _run_check([output, "--profile", "ri4438", "--output", output2, "--report", report2_path])
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            second_root = _parse_model(output2)[1]
            self.assertEqual(_topology(root), _topology(second_root))
            self.assertEqual(_masses(root), _masses(second_root))
            self.assertEqual(_visual_mesh_filenames(root), _visual_mesh_filenames(second_root))
            for name in ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "shoulder_joint"):
                np.testing.assert_allclose(_axis(root, name), _axis(second_root, name), atol=1.0e-9)
            self.assertEqual(_limit(root, "FL_thigh_joint"), _limit(second_root, "FL_thigh_joint"))
            self.assertEqual(_mimic(root, "FL_calf_joint"), _mimic(second_root, "FL_calf_joint"))

    def test_rule_validation_rejects_unknown_illegal_and_unsafe_noncollinear_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            cases = [
                ("unknown", {"joints": {"ghost_joint": {"axis": [0, 1, 0]}}}, "ghost_joint"),
                ("illegal_axis", {"joints": {"FL_thigh_joint": {"axis": [0, 0, 0]}}}, "axis"),
                ("noncollinear_without_limits", {"joints": {"FL_hip_joint": {"axis": [0, 1, 0]}}}, "limit"),
            ]
            for name, rules, needle in cases:
                with self.subTest(name=name):
                    rules_path = tmp / f"{name}.json"
                    rules_path.write_text(json.dumps(rules), encoding="utf-8")
                    report_path = tmp / f"{name}_report.json"
                    output = tmp / f"{name}_candidate.urdf"

                    result = _run_check([urdf, "--rules", rules_path, "--output", output, "--report", report_path])

                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    report = _load_report(report_path)
                    _assert_report_schema(self, report)
                    self.assertGreater(report["summary"]["errors"], 0)
                    self.assertIn(needle, _report_text(report))

    def test_mimic_cycles_are_errors_no_output_and_long_chain_is_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            cycle_cases = [
                ("self_loop", _mimic_chain_robot(tmp, "self_loop"), None, ("j1",)),
                ("indirect_loop", _mimic_chain_robot(tmp, "indirect_loop"), None, ("j1", "j2")),
            ]
            rules_robot = _mimic_chain_robot(tmp, "rules_loop")
            rules = tmp / "mimic_rules_loop.json"
            rules.write_text(
                json.dumps(
                    {
                        "joints": {
                            "j1": {"mimic": {"joint": "j2", "multiplier": 1.0, "offset": 0.1}},
                            "j2": {"mimic": {"joint": "j1", "multiplier": 1.0, "offset": 0.1}},
                        }
                    }
                ),
                encoding="utf-8",
            )
            cycle_cases.append(("rules_loop", rules_robot, rules, ("j1", "j2")))

            for name, urdf, rules_path, joint_names in cycle_cases:
                with self.subTest(name=name):
                    output = tmp / f"{name}_candidate.urdf"
                    report_path = tmp / f"{name}_report.json"
                    args: list[object] = [urdf, "--output", output, "--report", report_path]
                    if rules_path is not None:
                        args.extend(["--rules", rules_path])

                    result = _run_check(args)

                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertFalse(output.exists(), "mimic cycle validation must not export a candidate")
                    report = _load_report(report_path)
                    _assert_report_schema(self, report)
                    issue_text = _report_text(report)
                    self.assertIn("mimic", issue_text)
                    for joint_name in joint_names:
                        self.assertIn(joint_name, issue_text)
                    motion_validation = report.get("validation", {}).get("motion_equivalence")
                    self.assertNotEqual(motion_validation, "passed")
                    if isinstance(motion_validation, dict):
                        self.assertNotEqual(motion_validation.get("status"), "passed")

            long_chain = _mimic_chain_robot(tmp, "long_chain")
            output = tmp / "long_chain_candidate.urdf"
            report_path = tmp / "long_chain_report.json"
            result = _run_check([long_chain, "--output", output, "--report", report_path])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            _assert_structural_zero_rotations_identity(self, output)
            report = _load_report(report_path)
            _assert_report_schema(self, report)
            self.assertEqual(report["summary"]["errors"], 0)
            before = _visual_world_transforms(long_chain, {"j1": 0.2})
            after = _visual_world_transforms(output, {"j1": 0.2})
            _assert_pose_maps_close(self, before, after)

    def test_profile_noncollinear_axis_without_rules_is_preserved_and_marked_needs_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            _set_fixture_root_axes(urdf, {"FL_thigh_joint": (1.0, 0.0, 0.0)})
            before_axis = _physical_joint_axis_root(urdf, "FL_thigh_joint")
            output = tmp / "noncollinear_candidate.urdf"
            report_path = tmp / "noncollinear_report.json"

            result = _run_check([urdf, "--profile", "ri4438", "--output", output, "--report", report_path])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            report = _load_report(report_path)
            _assert_report_schema(self, report)
            matching = [
                issue
                for issue in report["issues"]
                if "FL_thigh_joint" in issue.get("object", "")
                and issue.get("status") == "needs_reference"
                and "axis" in (issue.get("code", "") + issue.get("message", "")).lower()
            ]
            self.assertTrue(matching, json.dumps(report["issues"], indent=2))
            after_axis = _physical_joint_axis_root(output, "FL_thigh_joint")
            np.testing.assert_allclose(after_axis, before_axis, atol=1.0e-6, rtol=0.0)

    def test_frame_normalization_preserves_existing_primitive_collision_world_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            tree, root, _, _ = _parse_model(urdf)
            sensor_joint = _joint(root, "sensor_joint")
            sensor_joint.find("origin").set("rpy", "0.22 -0.17 0.11")
            sensor_collision = {link.get("name"): link for link in root.findall("link")}["sensor_link"].find("collision")
            sensor_collision.find("origin").set("xyz", "0.01 -0.02 0.03")
            sensor_collision.find("origin").set("rpy", "0.2 0.1 -0.15")
            tree.write(urdf, encoding="utf-8", xml_declaration=True)
            before = _collision_world_transforms(urdf, {}, primitive_only=True)
            output = tmp / "primitive_collision_candidate.urdf"
            report = tmp / "primitive_collision_report.json"

            result = _run_check([urdf, "--profile", "generic", "--output", output, "--report", report])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            after = _collision_world_transforms(output, {}, primitive_only=True)
            key = "sensor_link:collision:sensor_collision"
            np.testing.assert_allclose(after[key], before[key], atol=1.0e-6, rtol=0.0)
            source_sensor = {link.get("name"): link for link in root.findall("link")}["sensor_link"]
            output_sensor = {link.get("name"): link for link in _parse_model(output)[1].findall("link")}["sensor_link"]
            self.assertEqual(
                source_sensor.find("collision/geometry/box").get("size"),
                output_sensor.find("collision/geometry/box").get("size"),
            )

    def test_collision_simplification_preserves_names_mass_inertia_and_encloses_visual_meshes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            fixture_tree, fixture_root, _, _ = _parse_model(urdf)
            _joint(fixture_root, "FL_hip_joint").find("axis").set("xyz", "1 0 0")
            _joint(fixture_root, "FL_thigh_joint").find("axis").set("xyz", "0 1 0")
            _joint(fixture_root, "FL_calf_joint").find("axis").set("xyz", "0 1 0")
            _joint(fixture_root, "shoulder_joint").find("axis").set("xyz", "0 -1 0")
            for joint in fixture_root.findall("joint"):
                origin = joint.find("origin")
                if origin is not None:
                    origin.set("rpy", "0 0 0")
            fixture_tree.write(urdf, encoding="utf-8", xml_declaration=True)
            output = tmp / "candidate.urdf"
            report = tmp / "report.json"
            before_root = _parse_model(urdf)[1]

            result = _run_check([urdf, "--profile", "ri4438", "--output", output, "--report", report])

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists(), "successful export should create the requested output URDF")
            root = _parse_model(output)[1]
            self.assertEqual(_masses(root), _masses(before_root))
            self.assertEqual(_inertial_signature(root), _inertial_signature(before_root))
            for link_name in ("base_link", "FL_hip", "FL_thigh", "FL_calf", "FL_foot", "mystery_link"):
                before_link = {link.get("name"): link for link in before_root.findall("link")}[link_name]
                after_link = {link.get("name"): link for link in root.findall("link")}[link_name]
                before_collision_name = before_link.find("collision").get("name")
                after_collision_name = after_link.find("collision").get("name")
                self.assertEqual(before_collision_name, after_collision_name)

            expected_primitives = {
                "base_link": "box",
                "FL_hip": "cylinder",
                "FL_thigh": "box",
                "FL_calf": "cylinder",
                "FL_foot": "sphere",
                "mystery_link": "box",
            }
            for link_name, primitive in expected_primitives.items():
                _assert_collision_encloses_visual_mesh(self, output, link_name, primitive)

            sensor = {link.get("name"): link for link in root.findall("link")}["sensor_link"]
            sensor_collision, sensor_geometry = _collision_geometry(sensor)
            self.assertEqual(sensor_collision.get("name"), "sensor_collision")
            self.assertIsNotNone(sensor_geometry.find("box"))
            self.assertEqual(sensor_geometry.find("box").get("size"), "0.02 0.03 0.04")

    @unittest.skipUnless(REAL_RI_PLUS_URDF.exists(), "RI-plus URDF asset is missing")
    def test_real_ri_plus_export_reload_mujoco_and_asset_hashes(self) -> None:
        asset_dir = REAL_RI_PLUS_URDF.parents[1] / "assets"
        watched_assets = [REAL_RI_PLUS_URDF, *sorted(asset_dir.glob("*.STL"))[:8]]
        before_hashes = {path: _sha256(path) for path in watched_assets}
        before_root = _parse_model(REAL_RI_PLUS_URDF)[1]
        self.assertEqual(len(before_root.findall("link")), 26)
        self.assertEqual(len(before_root.findall("joint")), 25)
        self.assertAlmostEqual(_total_mass(before_root), 8.41225125, places=8)
        np.testing.assert_allclose(_axis(before_root, "RL_thigh_joint"), [0, -1, 0], atol=1.0e-12)
        for name in ("FL_thigh_joint", "FR_thigh_joint", "RR_thigh_joint"):
            np.testing.assert_allclose(_axis(before_root, name), [0, 1, 0], atol=1.0e-12)
        for prefix in ("FL", "FR", "RL", "RR"):
            np.testing.assert_allclose(_axis(before_root, f"{prefix}_hip_joint"), [1, 0, 0], atol=1.0e-12)
            np.testing.assert_allclose(_axis(before_root, f"{prefix}_calf_joint"), [0, 1, 0], atol=1.0e-12)
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            output = tmp / "ri_plus_candidate.urdf"
            report_path = tmp / "ri_plus_report.json"

            result = _run_check(
                [
                    REAL_RI_PLUS_URDF,
                    "--profile",
                    "ri4438",
                    "--output",
                    output,
                    "--report",
                    report_path,
                    "--validate-mujoco",
                ],
                timeout=120.0,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.exists())
            report = _load_report(report_path)
            _assert_report_schema(self, report)
            self.assertEqual(report["profile"], "ri4438")
            self.assertEqual(report["summary"]["errors"], 0)

            after_root = _parse_model(output)[1]
            self.assertEqual(_topology(before_root), _topology(after_root))
            self.assertEqual(_masses(before_root), _masses(after_root))
            self.assertAlmostEqual(_total_mass(after_root), 8.41225125, places=8)
            np.testing.assert_allclose(_axis(after_root, "RL_thigh_joint"), [0, 1, 0], atol=1.0e-12)
            self.assertEqual(before_hashes, {path: _sha256(path) for path in watched_assets})
            for mesh in after_root.findall(".//mesh"):
                resolved = _resolve_mesh(mesh.get("filename"), output.parent)
                self.assertTrue(resolved.exists(), mesh.get("filename"))

            validation = report.get("validation", {})
            mujoco_validation = validation.get("mujoco", {})
            self.assertEqual(set(mujoco_validation), {"source", "candidate"})
            source_mj = mujoco_validation["source"]
            candidate_mj = mujoco_validation["candidate"]
            self.assertEqual(source_mj.get("status"), "needs_reference")
            self.assertEqual(source_mj.get("direct", {}).get("status"), "failed")
            self.assertRegex(
                (source_mj.get("details", "") + source_mj.get("direct", {}).get("details", "")).lower(),
                r"mesh|face|stl",
            )
            self.assertEqual(candidate_mj.get("status"), "needs_reference")
            self.assertIn("does not certify contact geometry", candidate_mj.get("details", ""))
            self.assertEqual(candidate_mj.get("direct", {}).get("status"), "failed")
            self.assertEqual(candidate_mj.get("auxiliary", {}).get("status"), "passed")
            self.assertGreater(candidate_mj.get("auxiliary", {}).get("temporary_visual_splits", 0), 0)
            self.assertNotIn("balanceinertia", json.dumps(mujoco_validation).lower())
            try:
                import mujoco  # type: ignore
            except BaseException:
                self.skipTest("MuJoCo is not importable")
            for direct_path in (REAL_RI_PLUS_URDF, output):
                with self.subTest(direct_path=str(direct_path)):
                    with self.assertRaises(Exception) as caught:
                        mujoco.MjModel.from_xml_path(str(direct_path))
                    self.assertRegex(str(caught.exception).lower(), r"mesh|face|stl")


class CheckUrdfPreviewSmokeTest(unittest.TestCase):
    def test_preview_websocket_controls_drive_scene_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _fixture_robot(tmp)
            report = tmp / "preview_report.json"
            port = _free_tcp_port()
            process = _start_preview_process(urdf, report, port, "--profile", "ri4438")
            try:
                _wait_for_http(port, process)
                stats = asyncio.run(_exercise_preview_websocket(self, port, require_controls=True))
                self.assertGreater(stats["scene_messages"], 0)
                self.assertGreater(stats["sliders"], 0)
            finally:
                _terminate_process(process)

    def test_preview_websocket_starts_despite_invalid_inertia(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            urdf = _invalid_inertia_preview_robot(tmp)
            report = tmp / "invalid_inertia_preview_report.json"
            port = _free_tcp_port()
            process = _start_preview_process(urdf, report, port, "--profile", "ri4438")
            try:
                _wait_for_http(port, process)
                stats = asyncio.run(_exercise_preview_websocket(self, port, require_controls=False))
                self.assertGreater(stats["scene_messages"], 0)
                self.assertGreater(stats["sliders"], 0)
            finally:
                _terminate_process(process)

    @unittest.skipUnless(REAL_RI_PLUS_URDF.exists(), "RI-plus URDF asset is missing")
    def test_real_ri_plus_preview_full_visual_mesh_websocket_smoke(self) -> None:
        asset_dir = REAL_RI_PLUS_URDF.parents[1] / "assets"
        watched_assets = [REAL_RI_PLUS_URDF, *sorted(asset_dir.glob("*.STL"))]
        before_hashes = {path: _sha256(path) for path in watched_assets}
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            report = tmp / "ri_plus_preview_report.json"
            port = _free_tcp_port()
            process = _start_preview_process(REAL_RI_PLUS_URDF, report, port, "--profile", "ri4438")
            try:
                _wait_for_http(port, process, timeout=30.0)
                stats = asyncio.run(
                    _exercise_preview_websocket(
                        self,
                        port,
                        require_controls=False,
                        initial_timeout=30.0,
                    )
                )
                self.assertGreater(stats["visual_meshes"], 0)
                self.assertGreater(stats["sliders"], 0)
            finally:
                _terminate_process(process)
        self.assertEqual(before_hashes, {path: _sha256(path) for path in watched_assets})
