#!/usr/bin/env python3
"""Convert a plain URDF file to a Unitree-style standalone MJCF model.

The MuJoCo URDF importer is used as a compatibility/preflight parser.  The
final XML is emitted from the URDF tree itself so fixed links (which MuJoCo
may collapse while serialising a ``MjSpec``) and URDF ``mimic`` constraints
remain explicit and auditable.

Examples:

    uv run python tools/urdf2mjcf.py robot.urdf
    uv run python tools/urdf2mjcf.py robot.urdf -o out/robot.xml \
        --collision-mode primitive --root-joint fixed
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class ConversionError(RuntimeError):
  """A user-facing conversion or validation error."""


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass
class Geometry:
  kind: str
  xyz: Vec3 = (0.0, 0.0, 0.0)
  rpy: Vec3 = (0.0, 0.0, 0.0)
  mesh_filename: str | None = None
  mesh_scale: Vec3 = (1.0, 1.0, 1.0)
  box_size: Vec3 | None = None
  radius: float | None = None
  length: float | None = None
  rgba: tuple[float, float, float, float] | None = None
  mesh_path: Path | None = None
  mesh_name: str | None = None
  name: str = ""


@dataclass
class Link:
  name: str
  mass: float | None = None
  inertial_xyz: Vec3 = (0.0, 0.0, 0.0)
  inertial_rpy: Vec3 = (0.0, 0.0, 0.0)
  inertia: tuple[float, float, float, float, float, float] | None = None
  visuals: list[Geometry] = field(default_factory=list)
  collisions: list[Geometry] = field(default_factory=list)


@dataclass
class Mimic:
  joint: str
  multiplier: float = 1.0
  offset: float = 0.0


@dataclass
class Joint:
  name: str
  kind: str
  parent: str
  child: str
  xyz: Vec3 = (0.0, 0.0, 0.0)
  rpy: Vec3 = (0.0, 0.0, 0.0)
  axis: Vec3 = (0.0, 0.0, 1.0)
  lower: float | None = None
  upper: float | None = None
  effort: float | None = None
  velocity: float | None = None
  damping: float | None = None
  friction: float | None = None
  mimic: Mimic | None = None


@dataclass
class Robot:
  name: str
  links: OrderedDict[str, Link]
  joints: OrderedDict[str, Joint]
  root: str
  materials: dict[str, tuple[float, float, float, float]]


def _float(value: str | None, default: float | None = None) -> float | None:
  if value is None:
    return default
  try:
    return float(value)
  except ValueError as exc:
    raise ConversionError(f"invalid numeric value: {value!r}") from exc


def _vec(value: str | None, n: int = 3, default: Iterable[float] | None = None) -> tuple[float, ...]:
  if value is None:
    return tuple(default if default is not None else (0.0,) * n)
  fields = value.replace(",", " ").split()
  if len(fields) != n:
    raise ConversionError(f"expected {n} numbers, got {value!r}")
  try:
    return tuple(float(x) for x in fields)
  except ValueError as exc:
    raise ConversionError(f"invalid vector value: {value!r}") from exc


def _fmt(value: float) -> str:
  if abs(value) < 5e-15:
    value = 0.0
  return f"{value:.12g}"


def _fmt_vec(values: Iterable[float]) -> str:
  return " ".join(_fmt(float(x)) for x in values)


def _rpy_matrix(rpy: Vec3):
  import numpy as np

  r, p, y = rpy
  cr, sr = math.cos(r), math.sin(r)
  cp, sp = math.cos(p), math.sin(p)
  cy, sy = math.cos(y), math.sin(y)
  return np.array(
    [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
    ],
    dtype=float,
  )


def _matrix_quat(rotation) -> Quat:
  """Convert a proper 3x3 rotation matrix to MuJoCo's wxyz quaternion."""
  import numpy as np

  m = np.asarray(rotation, dtype=float)
  trace = float(np.trace(m))
  if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    w = 0.25 * s
    x = (m[2, 1] - m[1, 2]) / s
    y = (m[0, 2] - m[2, 0]) / s
    z = (m[1, 0] - m[0, 1]) / s
  elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
    s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
    w = (m[2, 1] - m[1, 2]) / s
    x = 0.25 * s
    y = (m[0, 1] + m[1, 0]) / s
    z = (m[0, 2] + m[2, 0]) / s
  elif m[1, 1] > m[2, 2]:
    s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
    w = (m[0, 2] - m[2, 0]) / s
    x = (m[0, 1] + m[1, 0]) / s
    y = 0.25 * s
    z = (m[1, 2] + m[2, 1]) / s
  else:
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    w = (m[1, 0] - m[0, 1]) / s
    x = (m[0, 2] + m[2, 0]) / s
    y = (m[1, 2] + m[2, 1]) / s
    z = 0.25 * s
  q = np.asarray([w, x, y, z], dtype=float)
  q /= np.linalg.norm(q)
  if q[0] < 0.0:
    q *= -1.0
  return tuple(float(x) for x in q)


def _inertia_matrix(link: Link):
  import numpy as np

  if link.inertia is None:
    return None
  ixx, ixy, ixz, iyy, iyz, izz = link.inertia
  matrix = np.array(
    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float
  )
  # URDF inertia is expressed in the inertial frame. Rotate it to link frame.
  matrix = _rpy_matrix(link.inertial_rpy) @ matrix @ _rpy_matrix(link.inertial_rpy).T
  return matrix


def _inertial_principal(link: Link) -> tuple[Vec3, Quat, Vec3] | None:
  import numpy as np

  matrix = _inertia_matrix(link)
  if matrix is None or link.mass is None:
    return None
  eigenvalues, eigenvectors = np.linalg.eigh(matrix)
  scale = max(1.0, float(np.max(np.abs(eigenvalues))))
  if float(np.min(eigenvalues)) <= 1e-12 * scale:
    raise ConversionError(f"link {link.name!r} has a non-positive inertia matrix")
  # Principal moments must satisfy the triangle inequalities.
  if float(np.max(eigenvalues)) >= float(np.sum(eigenvalues) - np.max(eigenvalues)) + 1e-10 * scale:
    raise ConversionError(f"link {link.name!r} has a physically invalid inertia tensor")
  if np.linalg.det(eigenvectors) < 0.0:
    eigenvectors[:, 0] *= -1.0
  return link.inertial_xyz, _matrix_quat(eigenvectors), tuple(float(x) for x in eigenvalues)


def _rgba(element: ET.Element | None, materials: dict[str, tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
  if element is None:
    return None
  material = element.find("material")
  if material is None:
    return None
  color = material.find("color")
  if color is not None and color.get("rgba"):
    values = _vec(color.get("rgba"), 4)
    return tuple(float(x) for x in values)  # type: ignore[return-value]
  name = material.get("name")
  return materials.get(name) if name else None


def _resolve_mesh(filename: str, input_dir: Path, mesh_root: Path | None) -> Path:
  original = filename.strip()
  if original.startswith("file://"):
    original = original[7:]
  if original.startswith("package://"):
    if mesh_root is None:
      raise ConversionError(
        f"mesh uses package:// URI {filename!r}; pass --mesh-root to resolve it"
      )
    package_path = original[len("package://"):]
    parts = package_path.split("/", 1)
    candidate = mesh_root / package_path
    if len(parts) == 2:
      package_candidate = mesh_root / parts[0] / parts[1]
      if package_candidate.exists():
        candidate = package_candidate
  else:
    path = Path(original)
    candidate = path if path.is_absolute() else (mesh_root or input_dir) / path
  candidate = candidate.resolve()
  if not candidate.is_file():
    raise ConversionError(f"mesh file does not exist: {filename!r} -> {candidate}")
  return candidate


def _parse_geometry(
  element: ET.Element,
  *,
  rgba: tuple[float, float, float, float] | None,
  input_dir: Path,
  mesh_root: Path | None,
) -> Geometry:
  origin = element.find("origin")
  xyz = _vec(origin.get("xyz") if origin is not None else None, 3)
  rpy = _vec(origin.get("rpy") if origin is not None else None, 3)
  geometry = element.find("geometry")
  if geometry is None:
    raise ConversionError("visual/collision element is missing <geometry>")
  children = [child for child in geometry if child.tag in {"box", "cylinder", "sphere", "mesh"}]
  if len(children) != 1:
    raise ConversionError("each geometry must contain exactly one supported primitive")
  shape = children[0]
  if shape.tag == "box":
    return Geometry("box", xyz, rpy, box_size=tuple(_vec(shape.get("size"), 3)), rgba=rgba)
  if shape.tag == "cylinder":
    radius = _float(shape.get("radius"))
    length = _float(shape.get("length"))
    if radius is None or length is None or radius <= 0.0 or length <= 0.0:
      raise ConversionError("cylinder radius and length must be positive")
    return Geometry("cylinder", xyz, rpy, radius=radius, length=length, rgba=rgba)
  if shape.tag == "sphere":
    radius = _float(shape.get("radius"))
    if radius is None or radius <= 0.0:
      raise ConversionError("sphere radius must be positive")
    return Geometry("sphere", xyz, rpy, radius=radius, rgba=rgba)
  filename = shape.get("filename")
  if not filename:
    raise ConversionError("mesh geometry is missing filename")
  scale = tuple(_vec(shape.get("scale"), 3, (1.0, 1.0, 1.0)))
  return Geometry(
    "mesh",
    xyz,
    rpy,
    mesh_filename=filename,
    mesh_scale=scale,  # type: ignore[arg-type]
    rgba=rgba,
    mesh_path=_resolve_mesh(filename, input_dir, mesh_root),
  )


def parse_urdf(path: Path, mesh_root: Path | None = None) -> Robot:
  try:
    root = ET.parse(path).getroot()
  except (ET.ParseError, OSError) as exc:
    raise ConversionError(f"cannot parse URDF {path}: {exc}") from exc
  if root.tag != "robot":
    raise ConversionError(f"expected <robot> root, got <{root.tag}>")
  materials: dict[str, tuple[float, float, float, float]] = {}
  for material in root.findall("material"):
    name = material.get("name")
    color = material.find("color")
    if name and color is not None and color.get("rgba"):
      values = _vec(color.get("rgba"), 4)
      materials[name] = tuple(float(x) for x in values)  # type: ignore[assignment]

  links: OrderedDict[str, Link] = OrderedDict()
  for element in root.findall("link"):
    name = element.get("name")
    if not name:
      raise ConversionError("link without a name")
    if name in links:
      raise ConversionError(f"duplicate link name: {name}")
    link = Link(name)
    inertial = element.find("inertial")
    if inertial is not None:
      origin = inertial.find("origin")
      link.inertial_xyz = _vec(origin.get("xyz") if origin is not None else None, 3)  # type: ignore[assignment]
      link.inertial_rpy = _vec(origin.get("rpy") if origin is not None else None, 3)  # type: ignore[assignment]
      mass = inertial.find("mass")
      link.mass = _float(mass.get("value") if mass is not None else None)
      inertia = inertial.find("inertia")
      if inertia is not None:
        fields = [inertia.get(key) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")]
        if any(value is None for value in fields):
          raise ConversionError(f"link {name!r} has incomplete inertia")
        link.inertia = tuple(float(value) for value in fields)  # type: ignore[arg-type]
    for visual in element.findall("visual"):
      geometry = _parse_geometry(
        visual,
        rgba=_rgba(visual, materials),
        input_dir=path.parent,
        mesh_root=mesh_root,
      )
      link.visuals.append(geometry)
    for collision in element.findall("collision"):
      geometry = _parse_geometry(
        collision,
        rgba=None,
        input_dir=path.parent,
        mesh_root=mesh_root,
      )
      link.collisions.append(geometry)
    links[name] = link

  joints: OrderedDict[str, Joint] = OrderedDict()
  child_names: set[str] = set()
  supported = {"fixed", "revolute", "continuous", "prismatic"}
  for element in root.findall("joint"):
    name = element.get("name")
    kind = element.get("type")
    if not name or not kind:
      raise ConversionError("joint must have name and type")
    if kind not in supported:
      raise ConversionError(f"joint {name!r} has unsupported type {kind!r}")
    if name in joints:
      raise ConversionError(f"duplicate joint name: {name}")
    parent = element.find("parent")
    child = element.find("child")
    parent_name = parent.get("link") if parent is not None else None
    child_name = child.get("link") if child is not None else None
    if parent_name not in links or child_name not in links:
      raise ConversionError(f"joint {name!r} references an unknown parent or child link")
    origin = element.find("origin")
    axis = element.find("axis")
    limits = element.find("limit")
    dynamics = element.find("dynamics")
    joint = Joint(
      name=name,
      kind=kind,
      parent=parent_name,
      child=child_name,
      xyz=_vec(origin.get("xyz") if origin is not None else None, 3),  # type: ignore[arg-type]
      rpy=_vec(origin.get("rpy") if origin is not None else None, 3),  # type: ignore[arg-type]
      axis=_vec(axis.get("xyz") if axis is not None else None, 3, (0.0, 0.0, 1.0)),  # type: ignore[arg-type]
      lower=_float(limits.get("lower") if limits is not None else None),
      upper=_float(limits.get("upper") if limits is not None else None),
      effort=_float(limits.get("effort") if limits is not None else None),
      velocity=_float(limits.get("velocity") if limits is not None else None),
      damping=_float(dynamics.get("damping") if dynamics is not None else None),
      friction=_float(dynamics.get("friction") if dynamics is not None else None),
    )
    mimic = element.find("mimic")
    if mimic is not None:
      source = mimic.get("joint")
      if not source:
        raise ConversionError(f"mimic joint {name!r} is missing its source joint")
      joint.mimic = Mimic(
        source,
        multiplier=_float(mimic.get("multiplier"), 1.0) or 1.0,
        offset=_float(mimic.get("offset"), 0.0) or 0.0,
      )
    if child_name in child_names:
      raise ConversionError(f"link {child_name!r} has multiple parent joints")
    child_names.add(child_name)
    joints[name] = joint

  roots = [name for name in links if name not in child_names]
  if len(roots) != 1:
    raise ConversionError(f"URDF must have exactly one root link, found {roots}")
  for joint in joints.values():
    if joint.mimic is not None and joint.mimic.joint not in joints:
      raise ConversionError(
        f"mimic joint {joint.name!r} references unknown joint {joint.mimic.joint!r}"
      )
    if joint.kind != "fixed":
      if joint.effort is None:
        raise ConversionError(f"movable joint {joint.name!r} is missing <limit effort>")
      if joint.effort <= 0.0:
        raise ConversionError(f"movable joint {joint.name!r} has non-positive effort")
      axis_norm = math.sqrt(sum(x * x for x in joint.axis))
      if axis_norm <= 1e-12:
        raise ConversionError(f"movable joint {joint.name!r} has a zero axis")
      if joint.kind in {"revolute", "prismatic"} and (joint.lower is None or joint.upper is None):
        raise ConversionError(f"limited joint {joint.name!r} is missing lower/upper limits")
  for link in links.values():
    if link.mass is not None and link.mass <= 0.0:
      raise ConversionError(f"link {link.name!r} has non-positive mass")
    _inertial_principal(link)
  return Robot(root.get("name") or path.stem, links, joints, roots[0], materials)


def _safe_name(value: str) -> str:
  value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
  return value or "mesh"


def _quat_from_rpy(rpy: Vec3) -> Quat:
  return _matrix_quat(_rpy_matrix(rpy))


def _is_zero(values: Iterable[float], tol: float = 1e-12) -> bool:
  return all(abs(float(x)) <= tol for x in values)


def _set_transform(element: ET.Element, xyz: Vec3, rpy: Vec3) -> None:
  if not _is_zero(xyz):
    element.set("pos", _fmt_vec(xyz))
  if not _is_zero(rpy):
    element.set("quat", _fmt_vec(_quat_from_rpy(rpy)))


def _mesh_aabb(geometry: Geometry):
  import numpy as np
  import trimesh

  if geometry.mesh_path is None:
    raise ConversionError("mesh geometry has no resolved path")
  try:
    loaded = trimesh.load_mesh(geometry.mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
      loaded = loaded.dump(concatenate=True)
    vertices = np.asarray(loaded.vertices, dtype=float)
  except Exception as exc:
    raise ConversionError(f"cannot load mesh for primitive collision: {geometry.mesh_path}: {exc}") from exc
  if vertices.ndim != 2 or vertices.shape[0] == 0:
    raise ConversionError(f"mesh contains no vertices: {geometry.mesh_path}")
  vertices = vertices * np.asarray(geometry.mesh_scale, dtype=float)
  minimum = vertices.min(axis=0)
  maximum = vertices.max(axis=0)
  center = (minimum + maximum) / 2.0
  half = (maximum - minimum) / 2.0
  if np.any(half <= 1e-12):
    raise ConversionError(f"mesh has degenerate AABB: {geometry.mesh_path}")
  return tuple(float(x) for x in center), tuple(float(x) for x in half)


def _relative_path(path: Path, output: Path) -> str:
  return Path(os.path.relpath(path, output.parent)).as_posix()


class Emitter:
  def __init__(self, robot: Robot, output: Path, collision_mode: str):
    self.robot = robot
    self.output = output
    self.collision_mode = collision_mode
    self.mesh_assets: OrderedDict[tuple[Path, Vec3], str] = OrderedDict()
    self.materials: OrderedDict[tuple[float, float, float, float], str] = OrderedDict()
    self.children: dict[str, list[Joint]] = {name: [] for name in robot.links}
    for joint in robot.joints.values():
      self.children[joint.parent].append(joint)
    self._collect_assets()

  def _collect_assets(self) -> None:
    for link in self.robot.links.values():
      for geometry in [*link.visuals, *link.collisions]:
        if geometry.rgba is not None and geometry.rgba not in self.materials:
          self.materials[geometry.rgba] = f"mat_{len(self.materials):03d}"
        if geometry.kind == "mesh":
          assert geometry.mesh_path is not None
          key = (geometry.mesh_path, geometry.mesh_scale)
          if key not in self.mesh_assets:
            stem = _safe_name(geometry.mesh_path.stem)
            candidate = f"mesh_{len(self.mesh_assets):03d}_{stem}"
            self.mesh_assets[key] = candidate
          geometry.mesh_name = self.mesh_assets[key]

  def _emit_inertial(self, parent: ET.Element, link: Link) -> None:
    principal = _inertial_principal(link)
    if principal is None:
      return
    xyz, quat, diagonal = principal
    inertial = ET.SubElement(parent, "inertial", {
      "pos": _fmt_vec(xyz),
      "quat": _fmt_vec(quat),
      "mass": _fmt(link.mass or 0.0),
      "diaginertia": _fmt_vec(diagonal),
    })
    del inertial

  def _emit_geom(self, parent: ET.Element, link: Link, geometry: Geometry, role: str, index: int) -> None:
    name = f"{link.name}_{role}" if index == 0 else f"{link.name}_{role}_{index}"
    attributes = {"name": name, "class": role}
    kind = geometry.kind
    if role == "collision" and kind == "mesh" and self.collision_mode == "primitive":
      center, half = _mesh_aabb(geometry)
      rotation = _rpy_matrix(geometry.rpy)
      import numpy as np
      offset = rotation @ np.asarray(center)
      attributes["type"] = "box"
      attributes["size"] = _fmt_vec(half)
      if not _is_zero(offset):
        attributes["pos"] = _fmt_vec(offset)
      if not _is_zero(geometry.rpy):
        attributes["quat"] = _fmt_vec(_quat_from_rpy(geometry.rpy))
    elif kind == "box":
      attributes["type"] = "box"
      assert geometry.box_size is not None
      attributes["size"] = _fmt_vec(tuple(x / 2.0 for x in geometry.box_size))
      if not _is_zero(geometry.xyz):
        attributes["pos"] = _fmt_vec(geometry.xyz)
      if not _is_zero(geometry.rpy):
        attributes["quat"] = _fmt_vec(_quat_from_rpy(geometry.rpy))
    elif kind == "cylinder":
      attributes["type"] = "cylinder"
      attributes["size"] = _fmt_vec((geometry.radius or 0.0, (geometry.length or 0.0) / 2.0))
      if not _is_zero(geometry.xyz):
        attributes["pos"] = _fmt_vec(geometry.xyz)
      if not _is_zero(geometry.rpy):
        attributes["quat"] = _fmt_vec(_quat_from_rpy(geometry.rpy))
    elif kind == "sphere":
      attributes["type"] = "sphere"
      attributes["size"] = _fmt(geometry.radius or 0.0)
      if not _is_zero(geometry.xyz):
        attributes["pos"] = _fmt_vec(geometry.xyz)
      if not _is_zero(geometry.rpy):
        attributes["quat"] = _fmt_vec(_quat_from_rpy(geometry.rpy))
    elif kind == "mesh":
      attributes["type"] = "mesh"
      attributes["mesh"] = geometry.mesh_name or ""
      if not _is_zero(geometry.xyz):
        attributes["pos"] = _fmt_vec(geometry.xyz)
      if not _is_zero(geometry.rpy):
        attributes["quat"] = _fmt_vec(_quat_from_rpy(geometry.rpy))
    else:
      raise ConversionError(f"unsupported geometry kind {kind!r}")
    if role == "visual" and geometry.rgba is not None:
      attributes["material"] = self.materials[geometry.rgba]
    ET.SubElement(parent, "geom", attributes)

  def _emit_body(self, parent: ET.Element, link_name: str, root: bool, root_joint: str) -> None:
    link = self.robot.links[link_name]
    body_attributes = {"name": link.name}
    if root:
      body_attributes["childclass"] = "robot"
    else:
      attached = next(j for j in self.robot.joints.values() if j.child == link_name)
      if not _is_zero(attached.xyz):
        body_attributes["pos"] = _fmt_vec(attached.xyz)
      if not _is_zero(attached.rpy):
        body_attributes["quat"] = _fmt_vec(_quat_from_rpy(attached.rpy))
    body = ET.SubElement(parent, "body", body_attributes)
    self._emit_inertial(body, link)
    if root and root_joint == "free":
      ET.SubElement(body, "freejoint")
    if not root:
      attached = next(j for j in self.robot.joints.values() if j.child == link_name)
      if attached.kind != "fixed":
        joint_attributes = {"name": attached.name, "type": "hinge" if attached.kind in {"revolute", "continuous"} else "slide"}
        joint_attributes["axis"] = _fmt_vec(attached.axis)
        if attached.kind in {"revolute", "prismatic"}:
          joint_attributes["range"] = _fmt_vec((attached.lower or 0.0, attached.upper or 0.0))
        if attached.damping is not None:
          joint_attributes["damping"] = _fmt(attached.damping)
        if attached.friction is not None:
          joint_attributes["frictionloss"] = _fmt(attached.friction)
        ET.SubElement(body, "joint", joint_attributes)
    for index, geometry in enumerate(link.visuals):
      self._emit_geom(body, link, geometry, "visual", index)
    for index, geometry in enumerate(link.collisions):
      self._emit_geom(body, link, geometry, "collision", index)
    for child_joint in self.children[link_name]:
      self._emit_body(body, child_joint.child, False, root_joint)

  def emit(self, model_name: str, root_joint: str) -> ET.Element:
    root = ET.Element("mujoco", {"model": model_name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    default = ET.SubElement(root, "default")
    robot_default = ET.SubElement(default, "default", {"class": "robot"})
    ET.SubElement(robot_default, "geom", {"density": "0"})
    visual_default = ET.SubElement(robot_default, "default", {"class": "visual"})
    ET.SubElement(visual_default, "geom", {"contype": "0", "conaffinity": "0", "group": "2", "density": "0"})
    collision_default = ET.SubElement(robot_default, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", {"contype": "1", "conaffinity": "1", "group": "3", "density": "0"})

    asset = ET.SubElement(root, "asset")
    for rgba, name in self.materials.items():
      ET.SubElement(asset, "material", {"name": name, "rgba": _fmt_vec(rgba)})
    for (mesh_path, scale), name in self.mesh_assets.items():
      attributes = {"name": name, "file": _relative_path(mesh_path, self.output)}
      if not _is_zero(tuple(x - 1.0 for x in scale)):
        attributes["scale"] = _fmt_vec(scale)
      ET.SubElement(asset, "mesh", attributes)

    worldbody = ET.SubElement(root, "worldbody")
    self._emit_body(worldbody, self.robot.root, True, root_joint)

    actuator = ET.SubElement(root, "actuator")
    for joint in self.robot.joints.values():
      if joint.kind == "fixed":
        continue
      effort = joint.effort
      assert effort is not None
      ET.SubElement(actuator, "motor", {
        "name": f"motor_{joint.name}",
        "joint": joint.name,
        "gear": "1",
        "ctrllimited": "true",
        "ctrlrange": _fmt_vec((-effort, effort)),
        "forcelimited": "true",
        "forcerange": _fmt_vec((-effort, effort)),
      })

    mimic_joints = [joint for joint in self.robot.joints.values() if joint.mimic is not None]
    if mimic_joints:
      equality = ET.SubElement(root, "equality")
      for joint in mimic_joints:
        assert joint.mimic is not None
        ET.SubElement(equality, "joint", {
          "name": f"{joint.name}_mimic",
          "joint1": joint.name,
          "joint2": joint.mimic.joint,
          "polycoef": _fmt_vec((joint.mimic.offset, joint.mimic.multiplier, 0.0, 0.0, 0.0)),
        })
    return root


def _all_bodies(worldbody: ET.Element) -> dict[str, ET.Element]:
  result: dict[str, ET.Element] = {}
  for body in worldbody.findall(".//body"):
    name = body.get("name")
    if name:
      result[name] = body
  return result


def _validate_xml(xml_path: Path, robot: Robot, root_joint: str) -> None:
  try:
    tree = ET.parse(xml_path)
    root = tree.getroot()
  except (ET.ParseError, OSError) as exc:
    raise ConversionError(f"generated XML is not readable: {exc}") from exc
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise ConversionError("generated MJCF has no <worldbody>")
  bodies = _all_bodies(worldbody)
  missing = [name for name in robot.links if name not in bodies]
  if missing:
    raise ConversionError(f"generated MJCF is missing bodies: {missing}")
  joints = {
    j.get("name")
    for j in worldbody.findall(".//joint")
    if j.get("name")
  }
  expected_joints = {j.name for j in robot.joints.values() if j.kind != "fixed"}
  if joints != expected_joints:
    raise ConversionError(f"joint name mismatch: expected {sorted(expected_joints)}, got {sorted(joints)}")
  motors = {a.get("joint") for a in root.findall("./actuator/motor") if a.get("joint")}
  if motors != expected_joints:
    raise ConversionError(f"motor joint mismatch: expected {sorted(expected_joints)}, got {sorted(motors)}")
  freejoints = worldbody.findall(".//freejoint")
  if root_joint == "free" and len(freejoints) != 1:
    raise ConversionError("expected exactly one root <freejoint/>")
  if root_joint == "fixed" and freejoints:
    raise ConversionError("fixed root output unexpectedly contains a freejoint")
  for element in root.iter():
    mesh_name = element.get("mesh")
    if mesh_name:
      mesh = root.find(f"./asset/mesh[@name='{mesh_name}']")
      if mesh is None:
        raise ConversionError(f"geom references unknown mesh asset {mesh_name!r}")
      mesh_path = (xml_path.parent / mesh.get("file", "")).resolve()
      if not mesh_path.is_file():
        raise ConversionError(f"generated mesh path does not exist: {mesh_path}")
  # Reconstruct every emitted principal inertia and compare it with the URDF tensor.
  import numpy as np
  for name, link in robot.links.items():
    expected = _inertia_matrix(link)
    if expected is None:
      continue
    inertial = bodies[name].find("inertial")
    if inertial is None:
      raise ConversionError(f"generated body {name!r} is missing inertial data")
    diagonal = np.asarray(_vec(inertial.get("diaginertia"), 3), dtype=float)
    quat = np.asarray(_vec(inertial.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)), dtype=float)
    w, x, y, z = quat / np.linalg.norm(quat)
    rotation = np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    actual = rotation @ np.diag(diagonal) @ rotation.T
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-9):
      raise ConversionError(f"inertia mismatch for link {name!r}")
    position = np.asarray(_vec(inertial.get("pos"), 3), dtype=float)
    if not np.allclose(position, link.inertial_xyz, rtol=1e-6, atol=1e-9):
      raise ConversionError(f"inertial COM mismatch for link {name!r}")
  try:
    import mujoco
    mujoco.MjModel.from_xml_path(str(xml_path))
  except Exception as exc:
    raise ConversionError(f"MuJoCo failed to compile generated MJCF: {exc}") from exc


def _mujoco_preflight(path: Path, verbose: bool) -> None:
  try:
    import mujoco
  except ImportError as exc:
    raise ConversionError(
      "MuJoCo is required; run this command with the project environment, e.g. `uv run python ...`"
    ) from exc
  try:
    # MjSpec.from_file is intentionally used as the URDF compatibility parser.
    # We do not call to_xml() here because fixed-link collapsing in MuJoCo 3.11
    # would make the output non-auditable.
    mujoco.MjSpec.from_file(str(path))
  except Exception as exc:
    raise ConversionError(f"MuJoCo cannot import URDF {path}: {exc}") from exc
  if verbose:
    print("MuJoCo URDF preflight: passed", file=sys.stderr)


def convert(args: argparse.Namespace) -> Path:
  input_path = Path(args.input).expanduser().resolve()
  if not input_path.is_file():
    raise ConversionError(f"input URDF does not exist: {input_path}")
  output = Path(args.output).expanduser() if args.output else input_path.with_suffix(".xml")
  output = output.resolve()
  if output.exists() and not args.overwrite:
    raise ConversionError(f"output already exists (use --overwrite): {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  mesh_root = Path(args.mesh_root).expanduser().resolve() if args.mesh_root else None

  _mujoco_preflight(input_path, args.verbose)
  robot = parse_urdf(input_path, mesh_root)
  emitter = Emitter(robot, output, args.collision_mode)
  xml_root = emitter.emit(args.model_name or robot.name, args.root_joint)
  ET.indent(xml_root, space="  ")
  xml_text = ET.tostring(xml_root, encoding="unicode", xml_declaration=True) + "\n"

  with tempfile.NamedTemporaryFile(
    mode="w", encoding="utf-8", suffix=".xml", prefix=f".{output.stem}.", dir=output.parent, delete=False
  ) as handle:
    temporary = Path(handle.name)
    handle.write(xml_text)
  try:
    if args.no_validate:
      # Still check XML and all mesh references without compiling the model.
      tree = ET.parse(temporary)
      for mesh in tree.getroot().findall("./asset/mesh"):
        mesh_path = (temporary.parent / mesh.get("file", "")).resolve()
        if not mesh_path.is_file():
          raise ConversionError(f"generated mesh path does not exist: {mesh_path}")
    else:
      _validate_xml(temporary, robot, args.root_joint)
    os.replace(temporary, output)
  except Exception:
    temporary.unlink(missing_ok=True)
    raise

  movable = sum(j.kind != "fixed" for j in robot.joints.values())
  fixed = len(robot.joints) - movable
  visual_count = sum(len(link.visuals) for link in robot.links.values())
  collision_count = sum(len(link.collisions) for link in robot.links.values())
  mimic_count = sum(j.mimic is not None for j in robot.joints.values())
  print(f"model: {args.model_name or robot.name}", file=sys.stderr)
  print(f"links: {len(robot.links)}", file=sys.stderr)
  print(f"movable joints: {movable}", file=sys.stderr)
  print(f"fixed joints: {fixed}", file=sys.stderr)
  print(f"visual geoms: {visual_count}", file=sys.stderr)
  print(f"collision geoms: {collision_count}", file=sys.stderr)
  print(f"motors: {movable}", file=sys.stderr)
  print(f"mimic constraints: {mimic_count}", file=sys.stderr)
  print(f"validation: {'skipped' if args.no_validate else 'passed'}", file=sys.stderr)
  print(f"output: {output}", file=sys.stderr)
  return output


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("input", metavar="INPUT", help="input plain URDF file")
  parser.add_argument("-o", "--output", help="output MJCF XML (default: INPUT with .xml suffix)")
  parser.add_argument(
    "--collision-mode", choices=("preserve", "primitive"), default="preserve",
    help="preserve collision geometry or replace mesh collision with local AABB boxes",
  )
  parser.add_argument(
    "--root-joint", choices=("free", "fixed"), default="free",
    help="root body mode (default: free)",
  )
  parser.add_argument("--mesh-root", help="root used to resolve relative/package mesh paths")
  parser.add_argument("--model-name", help="override the MJCF model name")
  parser.add_argument("--overwrite", action="store_true", help="overwrite an existing output file")
  parser.add_argument("--no-validate", action="store_true", help="skip MuJoCo compilation validation")
  parser.add_argument("-v", "--verbose", action="store_true", help="print detailed conversion information")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    convert(args)
  except ConversionError as exc:
    print(f"urdf2mjcf: error: {exc}", file=sys.stderr)
    return 2
  except KeyboardInterrupt:
    print("urdf2mjcf: interrupted", file=sys.stderr)
    return 130
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
