#!/usr/bin/env python3
"""Check, normalize, simplify, and preview a URDF without trusting inertia.

The tool treats STL meshes, link geometry, link masses, and joint mounting
locations as the fixed baseline.  It reports questionable inertia, center of
mass, coordinate, axis, and collision modeling choices, then optionally writes
a candidate URDF that preserves the trusted geometry/topology.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
  import numpy as np
except ImportError:  # pragma: no cover - exercised only in broken environments.
  np = None  # type: ignore[assignment]


SUPPORTED_JOINT_TYPES = {"fixed", "revolute", "continuous", "prismatic"}
STANDARD_ROOT_TAGS = {"link", "joint", "material"}
LEG_PREFIXES = ("FL", "FR", "RL", "RR")
AXIS_TOL = 1e-6
POSE_TOL = 1e-6


class CheckUrdfError(RuntimeError):
  """Base class for user-facing failures."""


class UsageError(CheckUrdfError):
  """CLI usage, path conflict, or environment error."""


def require_numpy() -> Any:
  if np is None:
    raise UsageError("numpy is required for URDF checks")
  return np


@dataclass
class Issue:
  severity: str
  code: str
  object: str
  message: str
  status: str = "detected"
  original: Any | None = None
  candidate: Any | None = None
  id: str = ""

  def to_json(self) -> dict[str, Any]:
    data = {
      "severity": self.severity,
      "code": self.code,
      "object": self.object,
      "message": self.message,
      "status": self.status,
      "id": self.id,
    }
    if self.original is not None:
      data["original"] = sanitize_json(self.original)
    if self.candidate is not None:
      data["candidate"] = sanitize_json(self.candidate)
    return data


@dataclass
class Change:
  kind: str
  object: str
  before: Any
  after: Any

  def to_json(self) -> dict[str, Any]:
    return {
      "kind": self.kind,
      "object": self.object,
      "before": sanitize_json(self.before),
      "after": sanitize_json(self.after),
    }


class ReportBuilder:
  def __init__(self) -> None:
    self.issues: list[Issue] = []
    self.changes: list[Change] = []
    self._issue_counts: dict[tuple[str, str], int] = defaultdict(int)

  def issue(
    self,
    severity: str,
    code: str,
    obj: str,
    message: str,
    *,
    status: str = "detected",
    original: Any | None = None,
    candidate: Any | None = None,
  ) -> None:
    key = (code, obj)
    self._issue_counts[key] += 1
    stable = f"{code}:{obj}"
    if self._issue_counts[key] > 1:
      stable = f"{stable}:{self._issue_counts[key]}"
    self.issues.append(
      Issue(
        severity=severity,
        code=code,
        object=obj,
        message=message,
        status=status,
        original=original,
        candidate=candidate,
        id=stable,
      )
    )

  def change(self, kind: str, obj: str, before: Any, after: Any) -> None:
    self.changes.append(Change(kind=kind, object=obj, before=before, after=after))

  def summary(self) -> dict[str, int]:
    return {
      "errors": sum(1 for issue in self.issues if issue.severity == "error"),
      "warnings": sum(1 for issue in self.issues if issue.severity == "warning"),
      "infos": sum(1 for issue in self.issues if issue.severity == "info"),
    }

  def has_error_code(self, *codes: str) -> bool:
    wanted = set(codes)
    return any(issue.severity == "error" and issue.code in wanted for issue in self.issues)


@dataclass
class Origin:
  xyz: Any
  rpy: Any
  element: ET.Element | None = None


@dataclass
class Geometry:
  role: str
  link: str
  index: int
  element: ET.Element
  geometry_element: ET.Element | None
  origin: Origin
  kind: str = "invalid"
  shape_element: ET.Element | None = None
  mesh_filename: str | None = None
  mesh_path: Path | None = None
  mesh_scale: Any = None
  box_size: Any = None
  radius: float | None = None
  length: float | None = None
  valid: bool = True


@dataclass
class Mimic:
  joint: str
  multiplier: float = 1.0
  offset: float = 0.0


@dataclass
class JointLimit:
  lower: float | None = None
  upper: float | None = None
  effort: float | None = None
  velocity: float | None = None


@dataclass
class Link:
  name: str
  element: ET.Element
  inertial_element: ET.Element | None = None
  inertial_origin: Origin | None = None
  mass: float | None = None
  inertia: tuple[float, float, float, float, float, float] | None = None
  visuals: list[Geometry] = field(default_factory=list)
  collisions: list[Geometry] = field(default_factory=list)


@dataclass
class Joint:
  name: str
  kind: str
  element: ET.Element
  parent: str | None
  child: str | None
  origin: Origin
  axis: Any
  limit: JointLimit
  mimic: Mimic | None = None
  supported: bool = True


@dataclass
class RobotModel:
  name: str
  path: Path
  tree: ET.ElementTree
  root_element: ET.Element
  links: OrderedDict[str, Link]
  joints: OrderedDict[str, Joint]
  root_link: str | None
  children: dict[str, list[Joint]]
  parent_joint: dict[str, Joint]
  mesh_roots: list[Path]


@dataclass
class JointRule:
  axis: Any | None = None
  limits: dict[str, float] | None = None
  mimic: Mimic | None = None


@dataclass
class Rules:
  joints: dict[str, JointRule] = field(default_factory=dict)


def nparray(values: Iterable[float]) -> Any:
  module = require_numpy()
  return module.asarray(tuple(float(x) for x in values), dtype=float)


def zero3() -> Any:
  return nparray((0.0, 0.0, 0.0))


def eye3() -> Any:
  module = require_numpy()
  return module.eye(3, dtype=float)


def sanitize_json(value: Any) -> Any:
  if np is not None and isinstance(value, np.ndarray):
    return sanitize_json(value.tolist())
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(key): sanitize_json(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [sanitize_json(item) for item in value]
  if isinstance(value, float):
    if not math.isfinite(value):
      return None
    return float(value)
  if isinstance(value, (str, int, bool)) or value is None:
    return value
  return str(value)


def fmt_float(value: float) -> str:
  value = float(value)
  if abs(value) < 5e-15:
    value = 0.0
  return f"{value:.12g}"


def fmt_vec(values: Iterable[float]) -> str:
  return " ".join(fmt_float(float(x)) for x in values)


def vec_json(values: Any) -> list[float]:
  return [float(x) for x in values]


def finite(value: float | None) -> bool:
  return value is not None and math.isfinite(float(value))


def parse_float(
  text: str | None,
  *,
  default: float | None = None,
  required: bool = False,
  reporter: ReportBuilder,
  obj: str,
  code: str,
  field_name: str,
) -> float | None:
  if text is None:
    if required:
      reporter.issue("error", code, obj, f"missing numeric attribute {field_name}")
    return default
  try:
    value = float(text)
  except ValueError:
    reporter.issue("error", code, obj, f"invalid numeric value for {field_name}: {text!r}", original=text)
    return default
  if not math.isfinite(value):
    reporter.issue("error", code, obj, f"non-finite numeric value for {field_name}", original=text)
    return default
  return value


def parse_vec(
  text: str | None,
  *,
  size: int = 3,
  default: Iterable[float] | None = None,
  reporter: ReportBuilder,
  obj: str,
  code: str,
  field_name: str,
  required: bool = False,
) -> Any:
  if default is None:
    default = (0.0,) * size
  if text is None:
    if required:
      reporter.issue("error", code, obj, f"missing vector attribute {field_name}")
    return nparray(default)
  parts = text.replace(",", " ").split()
  if len(parts) != size:
    reporter.issue(
      "error",
      code,
      obj,
      f"expected {size} values for {field_name}, got {len(parts)}",
      original=text,
    )
    return nparray(default)
  values: list[float] = []
  for part in parts:
    try:
      value = float(part)
    except ValueError:
      reporter.issue("error", code, obj, f"invalid vector value for {field_name}: {text!r}", original=text)
      return nparray(default)
    if not math.isfinite(value):
      reporter.issue("error", code, obj, f"non-finite vector value for {field_name}", original=text)
      return nparray(default)
    values.append(value)
  return nparray(values)


def normalize(vec: Any, *, tol: float = 1e-12) -> Any | None:
  module = require_numpy()
  arr = module.asarray(vec, dtype=float)
  norm = float(module.linalg.norm(arr))
  if not math.isfinite(norm) or norm <= tol:
    return None
  return arr / norm


def is_zero(values: Any, tol: float = 1e-12) -> bool:
  module = require_numpy()
  return bool(module.all(module.abs(module.asarray(values, dtype=float)) <= tol))


def rpy_to_matrix(rpy: Any) -> Any:
  module = require_numpy()
  roll, pitch, yaw = [float(x) for x in rpy]
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)
  return module.asarray(
    [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
    ],
    dtype=float,
  )


def matrix_to_rpy(rotation: Any) -> Any:
  module = require_numpy()
  r = module.asarray(rotation, dtype=float)
  sy = -float(r[2, 0])
  sy = max(-1.0, min(1.0, sy))
  pitch = math.asin(sy)
  cp = math.cos(pitch)
  if abs(cp) > 1e-9:
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
  else:
    roll = 0.0
    yaw = math.atan2(-float(r[0, 1]), float(r[1, 1]))
  return module.asarray([roll, pitch, yaw], dtype=float)


def quat_wxyz_from_matrix(rotation: Any) -> tuple[float, float, float, float]:
  module = require_numpy()
  m = module.asarray(rotation, dtype=float)
  trace = float(module.trace(m))
  if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    w = 0.25 * s
    x = (float(m[2, 1]) - float(m[1, 2])) / s
    y = (float(m[0, 2]) - float(m[2, 0])) / s
    z = (float(m[1, 0]) - float(m[0, 1])) / s
  elif float(m[0, 0]) > float(m[1, 1]) and float(m[0, 0]) > float(m[2, 2]):
    s = math.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) * 2.0
    w = (float(m[2, 1]) - float(m[1, 2])) / s
    x = 0.25 * s
    y = (float(m[0, 1]) + float(m[1, 0])) / s
    z = (float(m[0, 2]) + float(m[2, 0])) / s
  elif float(m[1, 1]) > float(m[2, 2]):
    s = math.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) * 2.0
    w = (float(m[0, 2]) - float(m[2, 0])) / s
    x = (float(m[0, 1]) + float(m[1, 0])) / s
    y = 0.25 * s
    z = (float(m[1, 2]) + float(m[2, 1])) / s
  else:
    s = math.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) * 2.0
    w = (float(m[1, 0]) - float(m[0, 1])) / s
    x = (float(m[0, 2]) + float(m[2, 0])) / s
    y = (float(m[1, 2]) + float(m[2, 1])) / s
    z = 0.25 * s
  quat = module.asarray([w, x, y, z], dtype=float)
  quat /= module.linalg.norm(quat)
  if float(quat[0]) < 0.0:
    quat *= -1.0
  return tuple(float(x) for x in quat)


def transform_from_origin(origin: Origin) -> Any:
  module = require_numpy()
  transform = module.eye(4, dtype=float)
  transform[:3, :3] = rpy_to_matrix(origin.rpy)
  transform[:3, 3] = origin.xyz
  return transform


def transform_from_xyz_rpy(xyz: Any, rpy: Any) -> Any:
  return transform_from_origin(Origin(xyz=xyz, rpy=rpy))


def decompose_transform(transform: Any) -> tuple[Any, Any]:
  return transform[:3, 3].copy(), matrix_to_rpy(transform[:3, :3])


def rotation_motion(axis: Any, q: float) -> Any:
  module = require_numpy()
  unit = normalize(axis)
  if unit is None:
    return module.eye(3, dtype=float)
  x, y, z = [float(v) for v in unit]
  c = math.cos(float(q))
  s = math.sin(float(q))
  c1 = 1.0 - c
  return module.asarray(
    [
      [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
      [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
      [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
    ],
    dtype=float,
  )


def motion_transform(joint: Joint, q: float) -> Any:
  module = require_numpy()
  transform = module.eye(4, dtype=float)
  if joint.kind in {"revolute", "continuous"}:
    transform[:3, :3] = rotation_motion(joint.axis, q)
  elif joint.kind == "prismatic":
    axis = normalize(joint.axis)
    if axis is not None:
      transform[:3, 3] = axis * float(q)
  return transform


def set_origin(origin: Origin, xyz: Any, rpy: Any, parent: ET.Element | None = None) -> None:
  if origin.element is None:
    if parent is None:
      return
    origin.element = ET.Element("origin")
    parent.insert(0, origin.element)
  origin.xyz = nparray(xyz)
  origin.rpy = nparray(rpy)
  origin.element.set("xyz", fmt_vec(origin.xyz))
  origin.element.set("rpy", fmt_vec(origin.rpy))


def set_axis(joint: Joint, axis: Any) -> None:
  axis = normalize(axis)
  if axis is None:
    return
  if joint.element.find("axis") is None:
    joint.element.insert(1, ET.Element("axis"))
  axis_element = joint.element.find("axis")
  assert axis_element is not None
  joint.axis = axis
  axis_element.set("xyz", fmt_vec(axis))


def set_limit_attr(joint: Joint, key: str, value: float) -> None:
  limit = joint.element.find("limit")
  if limit is None:
    limit = ET.SubElement(joint.element, "limit")
  limit.set(key, fmt_float(value))
  setattr(joint.limit, key, value)


def set_mimic(joint: Joint, mimic: Mimic) -> None:
  element = joint.element.find("mimic")
  if element is None:
    element = ET.SubElement(joint.element, "mimic")
  element.set("joint", mimic.joint)
  element.set("multiplier", fmt_float(mimic.multiplier))
  element.set("offset", fmt_float(mimic.offset))
  joint.mimic = mimic


def parse_origin(
  parent: ET.Element,
  *,
  reporter: ReportBuilder,
  obj: str,
  code: str,
) -> Origin:
  origin = parent.find("origin")
  xyz = parse_vec(
    origin.get("xyz") if origin is not None else None,
    reporter=reporter,
    obj=obj,
    code=code,
    field_name="origin.xyz",
  )
  rpy = parse_vec(
    origin.get("rpy") if origin is not None else None,
    reporter=reporter,
    obj=obj,
    code=code,
    field_name="origin.rpy",
  )
  return Origin(xyz=xyz, rpy=rpy, element=origin)


def parse_xml(path: Path) -> ET.ElementTree:
  parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
  return ET.parse(path, parser=parser)


def mesh_root_candidates(input_dir: Path, mesh_roots: Iterable[Path]) -> list[Path]:
  roots = [input_dir]
  for root in mesh_roots:
    resolved = root.expanduser().resolve()
    if resolved not in roots:
      roots.append(resolved)
  return roots


def resolve_mesh(filename: str, input_dir: Path, mesh_roots: Iterable[Path]) -> tuple[Path | None, list[Path]]:
  text = filename.strip()
  candidates: list[Path] = []
  if text.startswith("file://"):
    text = text[7:]
  roots = mesh_root_candidates(input_dir, mesh_roots)
  if text.startswith("package://"):
    package_path = text[len("package://") :]
    parts = package_path.split("/", 1)
    for root in roots:
      candidates.append((root / package_path).resolve())
      if len(parts) == 2:
        candidates.append((root / parts[0] / parts[1]).resolve())
        candidates.append((root / parts[1]).resolve())
  else:
    path = Path(text)
    if path.is_absolute():
      candidates.append(path.resolve())
    else:
      for root in roots:
        candidates.append((root / path).resolve())
  for candidate in candidates:
    if candidate.is_file():
      return candidate, candidates
  return None, candidates


def parse_geometry(
  element: ET.Element,
  *,
  role: str,
  link_name: str,
  index: int,
  input_dir: Path,
  mesh_roots: list[Path],
  reporter: ReportBuilder,
) -> Geometry:
  obj = f"{link_name}.{role}[{index}]"
  origin = parse_origin(element, reporter=reporter, obj=obj, code="geometry_origin_invalid")
  geometry_element = element.find("geometry")
  geometry = Geometry(
    role=role,
    link=link_name,
    index=index,
    element=element,
    geometry_element=geometry_element,
    origin=origin,
    mesh_scale=nparray((1.0, 1.0, 1.0)),
  )
  if geometry_element is None:
    geometry.valid = False
    reporter.issue("error", "geometry_missing", obj, "visual/collision is missing <geometry>")
    return geometry
  shapes = [child for child in list(geometry_element) if isinstance(child.tag, str) and child.tag in {"mesh", "box", "cylinder", "sphere"}]
  if len(shapes) != 1:
    geometry.valid = False
    reporter.issue(
      "error",
      "geometry_shape_count",
      obj,
      "geometry must contain exactly one supported mesh/box/cylinder/sphere shape",
      original=[child.tag for child in list(geometry_element) if isinstance(child.tag, str)],
    )
    return geometry
  shape = shapes[0]
  geometry.kind = shape.tag
  geometry.shape_element = shape
  if shape.tag == "mesh":
    filename = shape.get("filename")
    if not filename:
      geometry.valid = False
      reporter.issue("error", "mesh_filename_missing", obj, "mesh geometry is missing filename")
    else:
      mesh_path, tried = resolve_mesh(filename, input_dir, mesh_roots)
      geometry.mesh_filename = filename
      geometry.mesh_path = mesh_path
      if mesh_path is None:
        geometry.valid = False
        reporter.issue(
          "error",
          "mesh_missing",
          obj,
          "mesh file could not be resolved",
          original={"filename": filename, "tried": [str(path) for path in tried[:8]]},
        )
    scale = parse_vec(
      shape.get("scale"),
      default=(1.0, 1.0, 1.0),
      reporter=reporter,
      obj=obj,
      code="mesh_scale_invalid",
      field_name="mesh.scale",
    )
    if any(abs(float(v)) <= 1e-12 for v in scale):
      geometry.valid = False
      reporter.issue("error", "mesh_scale_invalid", obj, "mesh scale contains a zero component", original=vec_json(scale))
    if any(float(v) < 0.0 for v in scale):
      reporter.issue("warning", "mesh_scale_negative", obj, "negative mesh scale mirrors geometry; verify downstream import support", original=vec_json(scale))
    geometry.mesh_scale = scale
  elif shape.tag == "box":
    size = parse_vec(shape.get("size"), reporter=reporter, obj=obj, code="box_size_invalid", field_name="box.size", required=True)
    geometry.box_size = size
    if any(float(v) <= 0.0 for v in size):
      geometry.valid = False
      reporter.issue("error", "box_size_invalid", obj, "box size must be positive", original=vec_json(size))
  elif shape.tag == "cylinder":
    radius = parse_float(shape.get("radius"), reporter=reporter, obj=obj, code="cylinder_invalid", field_name="cylinder.radius", required=True)
    length = parse_float(shape.get("length"), reporter=reporter, obj=obj, code="cylinder_invalid", field_name="cylinder.length", required=True)
    geometry.radius = radius
    geometry.length = length
    if not finite(radius) or not finite(length) or float(radius or 0.0) <= 0.0 or float(length or 0.0) <= 0.0:
      geometry.valid = False
      reporter.issue("error", "cylinder_invalid", obj, "cylinder radius and length must be positive")
  elif shape.tag == "sphere":
    radius = parse_float(shape.get("radius"), reporter=reporter, obj=obj, code="sphere_invalid", field_name="sphere.radius", required=True)
    geometry.radius = radius
    if not finite(radius) or float(radius or 0.0) <= 0.0:
      geometry.valid = False
      reporter.issue("error", "sphere_invalid", obj, "sphere radius must be positive")
  return geometry


def parse_robot(
  tree: ET.ElementTree,
  path: Path,
  mesh_roots: list[Path],
  reporter: ReportBuilder,
  *,
  context: str = "source",
) -> RobotModel | None:
  root = tree.getroot()
  if root.tag != "robot":
    reporter.issue("error", "xml_root_invalid", context, f"expected <robot> root, got <{root.tag}>")
    return None

  for child in list(root):
    if not isinstance(child.tag, str):
      continue
    if child.tag not in STANDARD_ROOT_TAGS:
      reporter.issue(
        "info",
        "xml_extension_needs_reference",
        child.tag,
        "non-standard URDF root extension is preserved but its control semantics are not validated",
        status="needs_reference",
      )

  links: OrderedDict[str, Link] = OrderedDict()
  for link_element in root.findall("link"):
    name = link_element.get("name")
    if not name:
      reporter.issue("error", "link_name_missing", context, "link element is missing name")
      continue
    if name in links:
      reporter.issue("error", "link_duplicate", name, "duplicate link name", original=name)
      continue
    link = Link(name=name, element=link_element)
    inertial = link_element.find("inertial")
    link.inertial_element = inertial
    if inertial is not None:
      link.inertial_origin = parse_origin(inertial, reporter=reporter, obj=f"{name}.inertial", code="inertial_origin_invalid")
      mass_element = inertial.find("mass")
      link.mass = parse_float(
        mass_element.get("value") if mass_element is not None else None,
        reporter=reporter,
        obj=f"{name}.mass",
        code="mass_invalid",
        field_name="mass.value",
        required=True,
      )
      inertia_element = inertial.find("inertia")
      if inertia_element is None:
        reporter.issue("error", "inertia_missing", name, "inertial block is missing <inertia>")
      else:
        fields = []
        missing = False
        for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
          value = parse_float(
            inertia_element.get(key),
            reporter=reporter,
            obj=f"{name}.inertia",
            code="inertia_invalid",
            field_name=f"inertia.{key}",
            required=True,
          )
          if value is None:
            missing = True
          fields.append(float(value or 0.0))
        if not missing:
          link.inertia = tuple(fields)  # type: ignore[assignment]
    for index, visual in enumerate(link_element.findall("visual")):
      link.visuals.append(
        parse_geometry(
          visual,
          role="visual",
          link_name=name,
          index=index,
          input_dir=path.parent,
          mesh_roots=mesh_roots,
          reporter=reporter,
        )
      )
    for index, collision in enumerate(link_element.findall("collision")):
      link.collisions.append(
        parse_geometry(
          collision,
          role="collision",
          link_name=name,
          index=index,
          input_dir=path.parent,
          mesh_roots=mesh_roots,
          reporter=reporter,
        )
      )
    links[name] = link

  joints: OrderedDict[str, Joint] = OrderedDict()
  child_seen: dict[str, str] = {}
  for joint_element in root.findall("joint"):
    name = joint_element.get("name")
    kind = joint_element.get("type")
    if not name:
      reporter.issue("error", "joint_name_missing", context, "joint element is missing name")
      continue
    if not kind:
      reporter.issue("error", "joint_type_missing", name, "joint is missing type")
      kind = "invalid"
    if name in joints:
      reporter.issue("error", "joint_duplicate", name, "duplicate joint name", original=name)
      continue
    supported = kind in SUPPORTED_JOINT_TYPES
    if not supported:
      reporter.issue("error", "joint_type_unsupported", name, f"unsupported joint type {kind!r}", original=kind)
    parent_element = joint_element.find("parent")
    child_element = joint_element.find("child")
    parent = parent_element.get("link") if parent_element is not None else None
    child = child_element.get("link") if child_element is not None else None
    if parent not in links:
      reporter.issue("error", "joint_parent_unknown", name, "joint references an unknown parent link", original=parent)
    if child not in links:
      reporter.issue("error", "joint_child_unknown", name, "joint references an unknown child link", original=child)
    if child is not None and child in child_seen:
      reporter.issue("error", "link_multiple_parents", child, "link has multiple parent joints", original=[child_seen[child], name])
    if child is not None:
      child_seen[child] = name
    origin = parse_origin(joint_element, reporter=reporter, obj=f"{name}.origin", code="joint_origin_invalid")
    axis_element = joint_element.find("axis")
    axis_default = (0.0, 0.0, 1.0) if kind != "fixed" else (0.0, 0.0, 0.0)
    axis = parse_vec(
      axis_element.get("xyz") if axis_element is not None else None,
      default=axis_default,
      reporter=reporter,
      obj=f"{name}.axis",
      code="joint_axis_invalid",
      field_name="axis.xyz",
    )
    limit_element = joint_element.find("limit")
    limit = JointLimit(
      lower=parse_float(limit_element.get("lower") if limit_element is not None else None, reporter=reporter, obj=f"{name}.limit", code="joint_limit_invalid", field_name="limit.lower"),
      upper=parse_float(limit_element.get("upper") if limit_element is not None else None, reporter=reporter, obj=f"{name}.limit", code="joint_limit_invalid", field_name="limit.upper"),
      effort=parse_float(limit_element.get("effort") if limit_element is not None else None, reporter=reporter, obj=f"{name}.limit", code="joint_limit_invalid", field_name="limit.effort"),
      velocity=parse_float(limit_element.get("velocity") if limit_element is not None else None, reporter=reporter, obj=f"{name}.limit", code="joint_limit_invalid", field_name="limit.velocity"),
    )
    mimic_element = joint_element.find("mimic")
    mimic: Mimic | None = None
    if mimic_element is not None:
      source = mimic_element.get("joint")
      if not source:
        reporter.issue("error", "mimic_joint_missing", name, "mimic is missing source joint")
      else:
        multiplier = parse_float(
          mimic_element.get("multiplier"),
          default=1.0,
          reporter=reporter,
          obj=f"{name}.mimic",
          code="mimic_invalid",
          field_name="mimic.multiplier",
        )
        offset = parse_float(
          mimic_element.get("offset"),
          default=0.0,
          reporter=reporter,
          obj=f"{name}.mimic",
          code="mimic_invalid",
          field_name="mimic.offset",
        )
        mimic = Mimic(source, float(multiplier if multiplier is not None else 1.0), float(offset if offset is not None else 0.0))
    joints[name] = Joint(
      name=name,
      kind=kind or "invalid",
      element=joint_element,
      parent=parent,
      child=child,
      origin=origin,
      axis=axis,
      limit=limit,
      mimic=mimic,
      supported=supported,
    )

  children: dict[str, list[Joint]] = {name: [] for name in links}
  parent_joint: dict[str, Joint] = {}
  for joint in joints.values():
    if joint.parent in children and joint.child in links:
      children[joint.parent].append(joint)  # type: ignore[index]
      parent_joint[joint.child] = joint  # type: ignore[index]

  roots = [name for name in links if name not in parent_joint]
  root_link = roots[0] if len(roots) == 1 else None
  if len(roots) != 1:
    reporter.issue("error", "topology_root_count", context, "URDF must have exactly one root link", original=roots)

  model = RobotModel(
    name=root.get("name") or path.stem,
    path=path,
    tree=tree,
    root_element=root,
    links=links,
    joints=joints,
    root_link=root_link,
    children=children,
    parent_joint=parent_joint,
    mesh_roots=mesh_roots,
  )
  return model


def inertia_matrix(link: Link) -> Any | None:
  if link.inertia is None:
    return None
  module = require_numpy()
  ixx, ixy, ixz, iyy, iyz, izz = link.inertia
  matrix = module.asarray([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=float)
  if link.inertial_origin is not None:
    rotation = rpy_to_matrix(link.inertial_origin.rpy)
    matrix = rotation @ matrix @ rotation.T
  return matrix


def geometry_vertices(geometry: Geometry) -> Any | None:
  module = require_numpy()
  if geometry.kind == "mesh":
    if geometry.mesh_path is None:
      return None
    vertices, _faces = load_stl(geometry.mesh_path)
    if vertices.size == 0:
      return None
    return vertices * module.asarray(geometry.mesh_scale, dtype=float)
  if geometry.kind == "box" and geometry.box_size is not None:
    half = module.asarray(geometry.box_size, dtype=float) / 2.0
    return module.asarray(
      [[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
      dtype=float,
    )
  if geometry.kind == "cylinder" and geometry.radius is not None and geometry.length is not None:
    samples = []
    for z in (-geometry.length / 2.0, geometry.length / 2.0):
      for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        samples.append([geometry.radius * math.cos(angle), geometry.radius * math.sin(angle), z])
    return module.asarray(samples, dtype=float)
  if geometry.kind == "sphere" and geometry.radius is not None:
    radius = geometry.radius
    return module.asarray(
      [
        [radius, 0, 0],
        [-radius, 0, 0],
        [0, radius, 0],
        [0, -radius, 0],
        [0, 0, radius],
        [0, 0, -radius],
      ],
      dtype=float,
    )
  return None


def geometry_vertices_in_link(geometry: Geometry) -> Any | None:
  vertices = geometry_vertices(geometry)
  if vertices is None:
    return None
  rotation = rpy_to_matrix(geometry.origin.rpy)
  return vertices @ rotation.T + geometry.origin.xyz


def combined_visual_bounds(link: Link) -> tuple[Any, Any] | None:
  module = require_numpy()
  arrays = []
  for visual in link.visuals:
    vertices = geometry_vertices_in_link(visual)
    if vertices is not None and len(vertices) > 0:
      arrays.append(vertices)
  if not arrays:
    return None
  points = module.vstack(arrays)
  return points.min(axis=0), points.max(axis=0)


def check_structure(model: RobotModel, reporter: ReportBuilder) -> None:
  if model.root_link is not None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(link_name: str) -> None:
      if link_name in visiting:
        reporter.issue("error", "topology_cycle", link_name, "cycle detected in link/joint graph")
        return
      if link_name in visited:
        return
      visiting.add(link_name)
      for joint in model.children.get(link_name, []):
        if joint.child is not None:
          dfs(joint.child)
      visiting.remove(link_name)
      visited.add(link_name)

    dfs(model.root_link)
    missing = [name for name in model.links if name not in visited]
    if missing:
      reporter.issue("error", "topology_disconnected", model.name, "some links are disconnected from the root", original=missing)

  for joint in model.joints.values():
    if joint.mimic is not None and joint.mimic.joint not in model.joints:
      reporter.issue("error", "mimic_source_unknown", joint.name, "mimic references an unknown joint", original=joint.mimic.joint)
    if joint.kind != "fixed" and joint.supported:
      axis = normalize(joint.axis)
      if axis is None:
        reporter.issue("error", "joint_axis_zero", joint.name, "movable joint has a zero axis", original=vec_json(joint.axis))
      elif not is_zero(axis - joint.axis, 1e-7):
        reporter.issue("warning", "joint_axis_not_unit", joint.name, "joint axis is not unit length; candidate will normalize it", original=vec_json(joint.axis), candidate=vec_json(axis))
      if joint.kind in {"revolute", "prismatic"}:
        if joint.limit.lower is None or joint.limit.upper is None:
          reporter.issue("error", "joint_limit_missing", joint.name, "limited joint is missing lower or upper limit")
        elif joint.limit.lower > joint.limit.upper:
          reporter.issue("error", "joint_limit_invalid", joint.name, "joint lower limit is greater than upper limit", original={"lower": joint.limit.lower, "upper": joint.limit.upper})
        elif not (joint.limit.lower <= 0.0 <= joint.limit.upper):
          reporter.issue(
            "warning",
            "joint_zero_outside_limit",
            joint.name,
            "zero displacement is outside the declared limit; q=0 is still used for structural frame checks",
            original={"lower": joint.limit.lower, "upper": joint.limit.upper},
          )
      for attr_name in ("effort", "velocity"):
        value = getattr(joint.limit, attr_name)
        if value is None:
          reporter.issue("warning", "joint_limit_missing", joint.name, f"movable joint is missing limit {attr_name}")
        elif value <= 0.0:
          reporter.issue("error", "joint_limit_invalid", joint.name, f"limit {attr_name} must be positive", original=value)

  for link in model.links.values():
    parent_joint = model.parent_joint.get(link.name)
    fixed_aux = parent_joint is not None and parent_joint.kind == "fixed" and not model.children.get(link.name)
    if link.inertial_element is None:
      if not fixed_aux:
        reporter.issue("warning", "inertial_missing", link.name, "link has no inertial block; CAD reference is required", status="needs_reference")
      continue
    if link.mass is None or link.mass <= 0.0:
      reporter.issue("error", "mass_invalid", link.name, "mass must be finite and positive", original=link.mass)
    if link.inertia is None:
      continue
    matrix = inertia_matrix(link)
    if matrix is None:
      continue
    eigenvalues = require_numpy().linalg.eigvalsh(matrix)
    scale = max(1.0, float(require_numpy().max(require_numpy().abs(eigenvalues))))
    if float(eigenvalues.min()) <= 1e-12 * scale:
      reporter.issue("error", "inertia_not_spd", link.name, "inertia matrix is not positive definite", original=link.inertia, candidate="needs CAD reference")
    largest = float(eigenvalues.max())
    if largest > float(eigenvalues.sum() - largest) + 1e-10 * scale:
      reporter.issue("error", "inertia_triangle_invalid", link.name, "principal moments violate the triangle inequality", original=[float(x) for x in eigenvalues])
    if link.inertial_origin is not None:
      bounds = combined_visual_bounds(link)
      if bounds is not None:
        low, high = bounds
        extent = high - low
        margin = max(1e-4, 0.05 * float(require_numpy().max(extent)))
        com = link.inertial_origin.xyz
        if bool(require_numpy().any(com < low - margin) or require_numpy().any(com > high + margin)):
          reporter.issue(
            "warning",
            "com_outside_visual_bounds",
            link.name,
            "center of mass lies outside visual mesh bounds; keep original value until CAD inertia is available",
            status="needs_reference",
            original=vec_json(com),
            candidate={"visual_min": vec_json(low), "visual_max": vec_json(high)},
          )

  reporter.issue(
    "info",
    "inertial_truth_unavailable",
    model.name,
    "no independent CAD mass-property reference was provided; COM and inertia are checked for consistency only",
    status="needs_reference",
  )
  check_mimic_cycles(model, reporter)
  check_mirror_consistency(model, reporter)


def check_mimic_cycles(model: RobotModel, reporter: ReportBuilder) -> None:
  visited: set[str] = set()
  active: list[str] = []

  def visit(name: str) -> None:
    if name in active:
      cycle = [*active[active.index(name) :], name]
      reporter.issue("error", "mimic_cycle", name, "mimic chain contains a cycle", original=cycle)
      return
    if name in visited:
      return
    visited.add(name)
    joint = model.joints.get(name)
    if joint is None or joint.mimic is None:
      return
    active.append(name)
    visit(joint.mimic.joint)
    active.pop()

  for joint_name in model.joints:
    visit(joint_name)


def check_mirror_consistency(model: RobotModel, reporter: ReportBuilder) -> None:
  checked: set[tuple[str, str]] = set()
  pairs = (("FL_", "FR_"), ("RL_", "RR_"))
  for left_prefix, right_prefix in pairs:
    for name, left in model.links.items():
      if not name.startswith(left_prefix):
        continue
      right_name = right_prefix + name[len(left_prefix) :]
      if right_name not in model.links or (name, right_name) in checked:
        continue
      checked.add((name, right_name))
      right = model.links[right_name]
      left_matrix = inertia_matrix(left)
      right_matrix = inertia_matrix(right)
      if left_matrix is None or right_matrix is None:
        continue
      left_eig = require_numpy().sort(require_numpy().linalg.eigvalsh(left_matrix))
      right_eig = require_numpy().sort(require_numpy().linalg.eigvalsh(right_matrix))
      denom = max(1e-12, float(require_numpy().max(require_numpy().abs(left_eig))))
      rel = float(require_numpy().max(require_numpy().abs(left_eig - right_eig))) / denom
      if rel > 0.10:
        reporter.issue(
          "warning",
          "mirror_inertia_inconsistent",
          f"{name}<->{right_name}",
          "mirrored leg links have noticeably different principal inertia values",
          status="needs_reference",
          original=vec_json(left_eig),
          candidate=vec_json(right_eig),
        )


def expected_profile_axis(profile: str, joint_name: str) -> Any | None:
  if profile not in {"go2", "ri4438"}:
    return None
  parts = joint_name.split("_")
  if len(parts) < 3 or parts[0] not in LEG_PREFIXES or parts[-1] != "joint":
    return None
  role = parts[1]
  if role == "hip":
    return nparray((1.0, 0.0, 0.0))
  if role in {"thigh", "calf"}:
    return nparray((0.0, 1.0, 0.0))
  return None


def check_quadruped_mounts(model: RobotModel, profile: str, reporter: ReportBuilder) -> None:
  if profile not in {"go2", "ri4438"}:
    return
  expected = {
    "FL": (1.0, 1.0),
    "FR": (1.0, -1.0),
    "RL": (-1.0, 1.0),
    "RR": (-1.0, -1.0),
  }
  for joint in model.joints.values():
    parts = joint.name.split("_")
    if len(parts) >= 3 and parts[0] in expected and parts[1] == "hip" and parts[-1] == "joint":
      x_sign, y_sign = expected[parts[0]]
      xyz = joint.origin.xyz
      wrong = []
      if float(xyz[0]) * x_sign <= 0.0:
        wrong.append("x")
      if float(xyz[1]) * y_sign <= 0.0:
        wrong.append("y")
      if wrong:
        reporter.issue(
          "warning",
          "leg_mount_quadrant_unexpected",
          joint.name,
          "hip joint mount is outside the expected front/back and left/right quadrant",
          original=vec_json(xyz),
          candidate={"expected_prefix": parts[0], "axes": wrong},
        )


def load_rules(path: Path | None, model: RobotModel | None, reporter: ReportBuilder) -> Rules:
  if path is None:
    return Rules()
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except OSError as exc:
    raise UsageError(f"cannot read rules file {path}: {exc}") from exc
  except json.JSONDecodeError as exc:
    raise UsageError(f"cannot parse rules file {path}: {exc}") from exc
  if not isinstance(data, dict):
    reporter.issue("error", "rules_invalid", str(path), "rules file must contain a JSON object")
    return Rules()
  unknown_top = sorted(set(data) - {"joints"})
  for key in unknown_top:
    reporter.issue("error", "rules_unknown_key", key, "unknown top-level key in rules file")
  joints_data = data.get("joints", {})
  if not isinstance(joints_data, dict):
    reporter.issue("error", "rules_invalid", str(path), "rules.joints must be an object")
    return Rules()
  rules = Rules()
  known_joints = set(model.joints) if model is not None else set()
  for joint_name, value in joints_data.items():
    if not isinstance(joint_name, str) or not isinstance(value, dict):
      reporter.issue("error", "rules_invalid", str(joint_name), "each rule entry must be an object keyed by joint name")
      continue
    if known_joints and joint_name not in known_joints:
      reporter.issue("error", "rules_unknown_joint", joint_name, "rules mention a joint not present in the URDF")
      continue
    unknown = sorted(set(value) - {"axis", "limits", "mimic"})
    for key in unknown:
      reporter.issue("error", "rules_unknown_key", f"{joint_name}.{key}", "unknown key in joint rule")
    rule = JointRule()
    if "axis" in value:
      axis_value = value["axis"]
      if not isinstance(axis_value, list) or len(axis_value) != 3:
        reporter.issue("error", "rules_axis_invalid", joint_name, "rule axis must be a three-value list")
      else:
        try:
          axis = nparray(axis_value)
          if normalize(axis) is None:
            raise ValueError("zero axis")
          rule.axis = normalize(axis)
        except (TypeError, ValueError):
          reporter.issue("error", "rules_axis_invalid", joint_name, "rule axis must contain finite non-zero numbers", original=axis_value)
    if "limits" in value:
      limits_value = value["limits"]
      if not isinstance(limits_value, dict) or set(limits_value) - {"lower", "upper"}:
        reporter.issue("error", "rules_limits_invalid", joint_name, "rule limits must contain only lower and upper")
      elif "lower" not in limits_value or "upper" not in limits_value:
        reporter.issue("error", "rules_limits_invalid", joint_name, "rule limits must include lower and upper")
      else:
        try:
          lower = float(limits_value["lower"])
          upper = float(limits_value["upper"])
          if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise ValueError("invalid limits")
          rule.limits = {"lower": lower, "upper": upper}
        except (TypeError, ValueError):
          reporter.issue("error", "rules_limits_invalid", joint_name, "rule limits must be finite and lower <= upper", original=limits_value)
    if "mimic" in value:
      mimic_value = value["mimic"]
      if not isinstance(mimic_value, dict) or set(mimic_value) - {"joint", "multiplier", "offset"}:
        reporter.issue("error", "rules_mimic_invalid", joint_name, "rule mimic must contain joint, multiplier, and offset")
      else:
        source = mimic_value.get("joint")
        try:
          multiplier = float(mimic_value.get("multiplier"))
          offset = float(mimic_value.get("offset"))
          if not isinstance(source, str) or not math.isfinite(multiplier) or not math.isfinite(offset):
            raise ValueError("invalid mimic")
          if known_joints and source not in known_joints:
            reporter.issue("error", "rules_mimic_invalid", joint_name, "rule mimic references unknown joint", original=source)
          else:
            rule.mimic = Mimic(source, multiplier, offset)
        except (TypeError, ValueError):
          reporter.issue("error", "rules_mimic_invalid", joint_name, "rule mimic values must be finite", original=mimic_value)
    rules.joints[joint_name] = rule
  return rules


def resolve_mimic_values(model: RobotModel, values: dict[str, float], *, ignore_mimic: bool = False) -> dict[str, float]:
  resolved = {name: float(values.get(name, 0.0)) for name, joint in model.joints.items() if joint.kind != "fixed"}
  if ignore_mimic:
    return resolved
  memo: dict[str, float] = {}

  def value_for(name: str, stack: list[str]) -> float:
    if name in memo:
      return memo[name]
    joint = model.joints.get(name)
    if joint is None or joint.kind == "fixed" or joint.mimic is None:
      memo[name] = resolved.get(name, 0.0)
      return memo[name]
    if name in stack:
      path = " -> ".join([*stack, name])
      raise CheckUrdfError(f"mimic cycle detected while evaluating FK: {path}")
    source_value = value_for(joint.mimic.joint, [*stack, name])
    memo[name] = joint.mimic.multiplier * source_value + joint.mimic.offset
    return memo[name]

  for name, joint in model.joints.items():
    if joint.kind != "fixed":
      resolved[name] = value_for(name, [])
  return resolved


def compute_fk(model: RobotModel, values: dict[str, float] | None = None, *, ignore_mimic: bool = False) -> dict[str, Any]:
  module = require_numpy()
  if model.root_link is None:
    return {}
  joint_values = resolve_mimic_values(model, values or {}, ignore_mimic=ignore_mimic)
  transforms: dict[str, Any] = {model.root_link: module.eye(4, dtype=float)}

  def visit(link_name: str) -> None:
    parent_transform = transforms[link_name]
    for joint in model.children.get(link_name, []):
      if joint.child is None:
        continue
      origin_transform = transform_from_origin(joint.origin)
      q = joint_values.get(joint.name, 0.0)
      transforms[joint.child] = parent_transform @ origin_transform @ motion_transform(joint, q)
      visit(joint.child)

  visit(model.root_link)
  return transforms


def joint_install_transforms(model: RobotModel, link_transforms: dict[str, Any]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for joint in model.joints.values():
    if joint.parent not in link_transforms:
      continue
    result[joint.name] = link_transforms[joint.parent] @ transform_from_origin(joint.origin)
  return result


def visual_world_transforms(model: RobotModel, link_transforms: dict[str, Any]) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for link_name, link in model.links.items():
    if link_name not in link_transforms:
      continue
    for geom in link.visuals:
      result[f"{link_name}.visual[{geom.index}]"] = link_transforms[link_name] @ transform_from_origin(geom.origin)
  return result


def transform_error(a: Any, b: Any) -> tuple[float, float]:
  module = require_numpy()
  pos_error = float(module.linalg.norm(a[:3, 3] - b[:3, 3]))
  rot = a[:3, :3].T @ b[:3, :3]
  trace = max(-1.0, min(1.0, (float(module.trace(rot)) - 1.0) / 2.0))
  angle = abs(math.acos(trace))
  return pos_error, angle


def standardize_candidate_frames(source: RobotModel, candidate: RobotModel, reporter: ReportBuilder) -> None:
  zero = compute_fk(source, {}, ignore_mimic=True)
  if not zero:
    return
  link_rotations = {name: transform[:3, :3].copy() for name, transform in zero.items()}

  for link_name, link in candidate.links.items():
    if link_name == candidate.root_link:
      continue
    rotation = link_rotations.get(link_name)
    if rotation is None:
      continue
    for geometry in [*link.visuals, *link.collisions]:
      before = {"xyz": vec_json(geometry.origin.xyz), "rpy": vec_json(geometry.origin.rpy)}
      new_xyz = rotation @ geometry.origin.xyz
      new_rpy = matrix_to_rpy(rotation @ rpy_to_matrix(geometry.origin.rpy))
      if not is_zero(new_xyz - geometry.origin.xyz, 1e-12) or not is_zero(new_rpy - geometry.origin.rpy, 1e-12):
        set_origin(geometry.origin, new_xyz, new_rpy, geometry.element)
        reporter.change("frame_origin", f"{link_name}.{geometry.role}[{geometry.index}]", before, {"xyz": vec_json(new_xyz), "rpy": vec_json(new_rpy)})
    if link.inertial_origin is not None and link.inertial_element is not None:
      before = {"xyz": vec_json(link.inertial_origin.xyz), "rpy": vec_json(link.inertial_origin.rpy)}
      new_xyz = rotation @ link.inertial_origin.xyz
      new_rpy = matrix_to_rpy(rotation @ rpy_to_matrix(link.inertial_origin.rpy))
      if not is_zero(new_xyz - link.inertial_origin.xyz, 1e-12) or not is_zero(new_rpy - link.inertial_origin.rpy, 1e-12):
        set_origin(link.inertial_origin, new_xyz, new_rpy, link.inertial_element)
        reporter.change("frame_origin", f"{link_name}.inertial", before, {"xyz": vec_json(new_xyz), "rpy": vec_json(new_rpy)})

  for joint_name, joint in candidate.joints.items():
    if joint.parent is None or joint.child is None:
      continue
    source_joint = source.joints.get(joint_name)
    if source_joint is None:
      continue
    parent_rotation = link_rotations.get(joint.parent)
    child_rotation = link_rotations.get(joint.child)
    if parent_rotation is None or child_rotation is None:
      continue
    before_origin = {"xyz": vec_json(joint.origin.xyz), "rpy": vec_json(joint.origin.rpy)}
    old_origin_rotation = rpy_to_matrix(source_joint.origin.rpy)
    new_xyz = parent_rotation @ source_joint.origin.xyz
    new_rpy = matrix_to_rpy(parent_rotation @ old_origin_rotation @ child_rotation.T)
    if not is_zero(new_xyz - joint.origin.xyz, 1e-12) or not is_zero(new_rpy - joint.origin.rpy, 1e-12):
      set_origin(joint.origin, new_xyz, new_rpy, joint.element)
      reporter.change("joint_origin", joint.name, before_origin, {"xyz": vec_json(new_xyz), "rpy": vec_json(new_rpy)})
    if joint.kind != "fixed":
      old_axis_world = parent_rotation @ old_origin_rotation @ source_joint.axis
      new_origin_rotation = rpy_to_matrix(new_rpy)
      new_axis = new_origin_rotation.T @ old_axis_world
      axis_unit = normalize(new_axis)
      if axis_unit is not None:
        before_axis = vec_json(joint.axis)
        if not is_zero(axis_unit - joint.axis, 1e-12):
          set_axis(joint, axis_unit)
          reporter.change("joint_axis_frame", joint.name, before_axis, vec_json(axis_unit))


def profile_axis_checks_and_fixes(
  model: RobotModel,
  profile: str,
  rules: Rules,
  reporter: ReportBuilder,
) -> dict[str, dict[str, Any]]:
  coordinate_map: dict[str, dict[str, Any]] = {}
  for joint in model.joints.values():
    if joint.kind == "fixed":
      continue
    coordinate_map[joint.name] = {"scale": 1.0, "offset": 0.0, "equivalent": True}
    axis_unit = normalize(joint.axis)
    if axis_unit is not None and not is_zero(axis_unit - joint.axis, 1e-12):
      before_axis = vec_json(joint.axis)
      set_axis(joint, axis_unit)
      reporter.change("joint_axis_normalized", joint.name, before_axis, vec_json(axis_unit))

  for joint in model.joints.values():
    if joint.kind == "fixed":
      continue
    current = normalize(joint.axis)
    if current is None:
      continue
    rule = rules.joints.get(joint.name)
    target = rule.axis if rule is not None and rule.axis is not None else expected_profile_axis(profile, joint.name)
    if target is None:
      continue
    target = normalize(target)
    if target is None:
      continue
    dot = float(require_numpy().dot(current, target))
    if dot > 1.0 - AXIS_TOL:
      if rule is not None and rule.axis is not None and not is_zero(current - target, 1e-12):
        before = vec_json(joint.axis)
        set_axis(joint, target)
        reporter.change("joint_axis_rule", joint.name, before, vec_json(target))
      continue
    if dot < -1.0 + AXIS_TOL:
      before_axis = vec_json(joint.axis)
      set_axis(joint, target)
      coordinate_map[joint.name]["scale"] = -1.0
      reporter.change("joint_axis_sign", joint.name, before_axis, vec_json(target))
      if joint.kind in {"revolute", "prismatic"} and joint.limit.lower is not None and joint.limit.upper is not None:
        before_limits = {"lower": joint.limit.lower, "upper": joint.limit.upper}
        new_lower = -float(joint.limit.upper)
        new_upper = -float(joint.limit.lower)
        set_limit_attr(joint, "lower", new_lower)
        set_limit_attr(joint, "upper", new_upper)
        reporter.change("joint_limit_sign", joint.name, before_limits, {"lower": new_lower, "upper": new_upper})
      continue
    if rule is not None and rule.axis is not None:
      if joint.kind in {"revolute", "prismatic"} and rule.limits is None:
        reporter.issue(
          "error",
          "rules_limits_required",
          joint.name,
          "non-collinear axis rule for a limited joint must provide explicit lower and upper limits",
          original=vec_json(current),
          candidate=vec_json(target),
        )
        continue
      before_axis = vec_json(joint.axis)
      set_axis(joint, target)
      coordinate_map[joint.name]["equivalent"] = False
      reporter.change("joint_axis_rule_non_equivalent", joint.name, before_axis, vec_json(target))
      reporter.issue(
        "warning",
        "joint_axis_rule_non_equivalent",
        joint.name,
        "explicit rule changes a non-collinear axis; kinematic equivalence is not claimed",
        status="needs_reference",
        original=vec_json(current),
        candidate=vec_json(target),
      )
    else:
      reporter.issue(
        "warning",
        "joint_axis_non_collinear",
        joint.name,
        "joint axis in the standardized frame is not collinear with the profile expectation; the physical axis is retained and requires reference data before changing it",
        status="needs_reference",
        original=vec_json(current),
        candidate=vec_json(target),
      )

  for joint_name, rule in rules.joints.items():
    joint = model.joints.get(joint_name)
    if joint is None:
      continue
    if rule.limits is not None:
      if joint.kind == "continuous":
        reporter.issue("warning", "rules_limits_ignored", joint_name, "continuous joint does not require lower/upper limits")
      elif joint.kind in {"revolute", "prismatic"}:
        before = {"lower": joint.limit.lower, "upper": joint.limit.upper}
        set_limit_attr(joint, "lower", float(rule.limits["lower"]))
        set_limit_attr(joint, "upper", float(rule.limits["upper"]))
        reporter.change("joint_limit_rule", joint_name, before, rule.limits)
    if rule.mimic is not None:
      before = None if joint.mimic is None else {"joint": joint.mimic.joint, "multiplier": joint.mimic.multiplier, "offset": joint.mimic.offset}
      set_mimic(joint, rule.mimic)
      reporter.change("joint_mimic_rule", joint_name, before, {"joint": rule.mimic.joint, "multiplier": rule.mimic.multiplier, "offset": rule.mimic.offset})

  update_mimic_for_coordinate_map(model, coordinate_map, reporter)
  return coordinate_map


def update_mimic_for_coordinate_map(
  model: RobotModel,
  coordinate_map: dict[str, dict[str, Any]],
  reporter: ReportBuilder,
) -> None:
  for joint in model.joints.values():
    if joint.kind == "fixed" or joint.mimic is None:
      continue
    if joint.mimic.joint not in model.joints:
      continue
    child_scale = float(coordinate_map.get(joint.name, {}).get("scale", 1.0))
    parent_scale = float(coordinate_map.get(joint.mimic.joint, {}).get("scale", 1.0))
    before = {"joint": joint.mimic.joint, "multiplier": joint.mimic.multiplier, "offset": joint.mimic.offset}
    new_mimic = Mimic(
      joint=joint.mimic.joint,
      multiplier=child_scale * joint.mimic.multiplier / parent_scale,
      offset=child_scale * joint.mimic.offset,
    )
    if abs(new_mimic.multiplier - joint.mimic.multiplier) > 1e-14 or abs(new_mimic.offset - joint.mimic.offset) > 1e-14:
      set_mimic(joint, new_mimic)
      reporter.change("mimic_coordinate_map", joint.name, before, {"joint": new_mimic.joint, "multiplier": new_mimic.multiplier, "offset": new_mimic.offset})


def load_stl(path: Path) -> tuple[Any, Any]:
  module = require_numpy()
  data = path.read_bytes()
  vertices: list[tuple[float, float, float]] = []
  faces: list[tuple[int, int, int]] = []
  if len(data) >= 84:
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + count * 50
    if expected == len(data):
      offset = 84
      for _ in range(count):
        offset += 12
        face = []
        for _vertex in range(3):
          vertex = struct.unpack("<fff", data[offset : offset + 12])
          face.append(len(vertices))
          vertices.append((float(vertex[0]), float(vertex[1]), float(vertex[2])))
          offset += 12
        faces.append((face[0], face[1], face[2]))
        offset += 2
      return module.asarray(vertices, dtype=float), module.asarray(faces, dtype=np.int64)
  try:
    text = data.decode("utf-8", errors="ignore")
  except UnicodeDecodeError:
    text = ""
  current: list[int] = []
  for line in text.splitlines():
    parts = line.strip().split()
    if len(parts) == 4 and parts[0].lower() == "vertex":
      try:
        vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
      except ValueError:
        continue
      current.append(len(vertices))
      vertices.append(vertex)
      if len(current) == 3:
        faces.append((current[0], current[1], current[2]))
        current = []
  return module.asarray(vertices, dtype=float), module.asarray(faces, dtype=np.int64)


def replace_geometry_shape(geometry: Geometry, tag: str, attributes: dict[str, str]) -> None:
  if geometry.geometry_element is None:
    geometry.geometry_element = ET.SubElement(geometry.element, "geometry")
  for child in list(geometry.geometry_element):
    if isinstance(child.tag, str) and child.tag in {"mesh", "box", "cylinder", "sphere"}:
      geometry.geometry_element.remove(child)
  shape = ET.SubElement(geometry.geometry_element, tag, attributes)
  geometry.kind = tag
  geometry.shape_element = shape
  geometry.mesh_filename = None
  geometry.mesh_path = None
  if tag == "box":
    geometry.box_size = parse_vec(attributes["size"], reporter=ReportBuilder(), obj="", code="", field_name="")
  elif tag == "sphere":
    geometry.radius = float(attributes["radius"])
  elif tag == "cylinder":
    geometry.radius = float(attributes["radius"])
    geometry.length = float(attributes["length"])


def classify_collision_link(link_name: str) -> str:
  lower = link_name.lower()
  if lower in {"base", "base_link", "trunk"} or "base" in lower or "body" in lower:
    return "body"
  if lower.endswith("_foot") or "foot" in lower:
    return "foot"
  if lower.endswith("_hip") or "_hip" in lower:
    return "hip"
  if lower.endswith("_thigh") or "_thigh" in lower:
    return "thigh"
  if lower.endswith("_calf") or "_calf" in lower or "shank" in lower:
    return "calf"
  return "generic"


def make_basis_from_axis(axis: Any) -> tuple[Any, Any, Any]:
  module = require_numpy()
  w = normalize(axis)
  if w is None:
    w = nparray((0.0, 0.0, 1.0))
  seed = nparray((1.0, 0.0, 0.0))
  if abs(float(module.dot(w, seed))) > 0.9:
    seed = nparray((0.0, 1.0, 0.0))
  u = normalize(module.cross(seed, w))
  if u is None:
    u = nparray((0.0, 1.0, 0.0))
  v = module.cross(w, u)
  return u, v, w


def rotation_from_z_axis(axis: Any) -> Any:
  module = require_numpy()
  u, v, w = make_basis_from_axis(axis)
  rotation = module.column_stack((u, v, w))
  if float(module.linalg.det(rotation)) < 0.0:
    rotation[:, 0] *= -1.0
  return rotation


def fit_box(vertices: Any) -> tuple[Any, Any]:
  low = vertices.min(axis=0)
  high = vertices.max(axis=0)
  center = (low + high) / 2.0
  size = high - low
  size = require_numpy().maximum(size, 1e-9)
  return center, size


def fit_sphere(vertices: Any) -> tuple[Any, float]:
  center, _size = fit_box(vertices)
  radius = float(require_numpy().max(require_numpy().linalg.norm(vertices - center, axis=1)))
  return center, max(radius, 1e-9)


def fit_cylinder(vertices: Any, axis: Any) -> tuple[Any, float, float, Any]:
  module = require_numpy()
  u, v, w = make_basis_from_axis(axis)
  s = vertices @ w
  uu = vertices @ u
  vv = vertices @ v
  s_mid = (float(s.min()) + float(s.max())) / 2.0
  u_mid = (float(uu.min()) + float(uu.max())) / 2.0
  v_mid = (float(vv.min()) + float(vv.max())) / 2.0
  center = w * s_mid + u * u_mid + v * v_mid
  radius = float(module.max(module.sqrt((uu - u_mid) ** 2 + (vv - v_mid) ** 2)))
  length = float(s.max() - s.min())
  return center, max(radius, 1e-9), max(length, 1e-9), rotation_from_z_axis(w)


def trusted_vertices_for_collision(link: Link, fallback: Geometry) -> Any | None:
  module = require_numpy()
  visual_vertices = []
  for visual in link.visuals:
    vertices = geometry_vertices_in_link(visual)
    if vertices is not None and len(vertices) > 0:
      visual_vertices.append(vertices)
  if visual_vertices:
    return module.vstack(visual_vertices)
  return geometry_vertices_in_link(fallback)


def incoming_joint_axis(model: RobotModel, link_name: str) -> Any | None:
  joint = model.parent_joint.get(link_name)
  if joint is None:
    return None
  return normalize(joint.axis)


def calf_to_foot_axis(model: RobotModel, link_name: str) -> Any | None:
  for joint in model.children.get(link_name, []):
    child = joint.child or ""
    if joint.kind == "fixed" and "foot" in child.lower():
      return normalize(joint.origin.xyz)
  for joint in model.children.get(link_name, []):
    axis = normalize(joint.origin.xyz)
    if axis is not None:
      return axis
  return None


def ensure_collision_name(link: Link, geometry: Geometry, used: set[str]) -> None:
  existing = geometry.element.get("name")
  if existing:
    used.add(existing)
    return
  base = f"{link.name}_collision"
  name = base
  index = 1
  while name in used:
    index += 1
    name = f"{base}_{index}"
  geometry.element.set("name", name)
  used.add(name)


def simplify_collisions(model: RobotModel, reporter: ReportBuilder) -> None:
  used_names: set[str] = set()
  for link in model.links.values():
    for collision in link.collisions:
      ensure_collision_name(link, collision, used_names)
      if collision.kind != "mesh":
        continue
      vertices = trusted_vertices_for_collision(link, collision)
      if vertices is None or len(vertices) == 0:
        reporter.issue("error", "collision_fit_failed", f"{link.name}.collision[{collision.index}]", "mesh collision has no vertices for primitive fitting")
        continue
      role = classify_collision_link(link.name)
      before = {"kind": "mesh", "filename": collision.mesh_filename, "origin": {"xyz": vec_json(collision.origin.xyz), "rpy": vec_json(collision.origin.rpy)}}
      if role in {"body", "thigh", "generic"}:
        center, size = fit_box(vertices)
        set_origin(collision.origin, center, zero3(), collision.element)
        replace_geometry_shape(collision, "box", {"size": fmt_vec(size)})
        after = {"kind": "box", "size": vec_json(size), "origin": {"xyz": vec_json(center), "rpy": [0.0, 0.0, 0.0]}}
      elif role == "foot":
        center, radius = fit_sphere(vertices)
        set_origin(collision.origin, center, zero3(), collision.element)
        replace_geometry_shape(collision, "sphere", {"radius": fmt_float(radius)})
        after = {"kind": "sphere", "radius": radius, "origin": {"xyz": vec_json(center), "rpy": [0.0, 0.0, 0.0]}}
      elif role == "hip":
        axis = incoming_joint_axis(model, link.name)
        if axis is None:
          center, size = fit_box(vertices)
          set_origin(collision.origin, center, zero3(), collision.element)
          replace_geometry_shape(collision, "box", {"size": fmt_vec(size)})
          after = {"kind": "box", "size": vec_json(size), "origin": {"xyz": vec_json(center), "rpy": [0.0, 0.0, 0.0]}, "fallback": "missing incoming joint axis"}
        else:
          center, radius, length, rotation = fit_cylinder(vertices, axis)
          rpy = matrix_to_rpy(rotation)
          set_origin(collision.origin, center, rpy, collision.element)
          replace_geometry_shape(collision, "cylinder", {"radius": fmt_float(radius), "length": fmt_float(length)})
          after = {"kind": "cylinder", "radius": radius, "length": length, "axis": vec_json(axis), "origin": {"xyz": vec_json(center), "rpy": vec_json(rpy)}}
      else:
        axis = calf_to_foot_axis(model, link.name)
        if axis is None:
          center, size = fit_box(vertices)
          set_origin(collision.origin, center, zero3(), collision.element)
          replace_geometry_shape(collision, "box", {"size": fmt_vec(size)})
          after = {"kind": "box", "size": vec_json(size), "origin": {"xyz": vec_json(center), "rpy": [0.0, 0.0, 0.0]}, "fallback": "missing foot endpoint"}
        else:
          center, radius, length, rotation = fit_cylinder(vertices, axis)
          rpy = matrix_to_rpy(rotation)
          set_origin(collision.origin, center, rpy, collision.element)
          replace_geometry_shape(collision, "cylinder", {"radius": fmt_float(radius), "length": fmt_float(length)})
          after = {"kind": "cylinder", "radius": radius, "length": length, "axis": vec_json(axis), "origin": {"xyz": vec_json(center), "rpy": vec_json(rpy)}}
      reporter.change("collision_simplified", geometry_object_name(collision), before, after)


def geometry_object_name(geometry: Geometry) -> str:
  name = geometry.element.get("name")
  if name:
    return f"{geometry.link}.{geometry.role}.{name}"
  return f"{geometry.link}.{geometry.role}[{geometry.index}]"


def apply_mesh_paths_for_output(model: RobotModel, output: Path, reporter: ReportBuilder) -> None:
  for link in model.links.values():
    for geometry in [*link.visuals, *link.collisions]:
      if geometry.kind != "mesh" or geometry.shape_element is None or geometry.mesh_path is None:
        continue
      old = geometry.shape_element.get("filename")
      new = os.path.relpath(geometry.mesh_path, output.parent)
      new = Path(new).as_posix()
      if old != new:
        geometry.shape_element.set("filename", new)
        reporter.change("mesh_reference", geometry_object_name(geometry), old, new)


def validation_zero_pose(source: RobotModel, candidate: RobotModel) -> dict[str, Any]:
  module = require_numpy()
  source_fk = compute_fk(source, {}, ignore_mimic=True)
  candidate_fk = compute_fk(candidate, {}, ignore_mimic=True)
  max_pos = 0.0
  max_rot = 0.0
  max_link_pos = 0.0
  max_candidate_link_rot_from_identity = 0.0
  checked = 0
  for link_name, source_transform in source_fk.items():
    candidate_transform = candidate_fk.get(link_name)
    if candidate_transform is None:
      continue
    max_link_pos = max(max_link_pos, float(module.linalg.norm(source_transform[:3, 3] - candidate_transform[:3, 3])))
    _pos_error, rot_from_identity = transform_error(module.eye(4, dtype=float), candidate_transform)
    max_candidate_link_rot_from_identity = max(max_candidate_link_rot_from_identity, rot_from_identity)
    checked += 1
  source_visuals = visual_world_transforms(source, source_fk)
  candidate_visuals = visual_world_transforms(candidate, candidate_fk)
  for name, source_transform in source_visuals.items():
    if name not in candidate_visuals:
      continue
    pos_error, rot_error = transform_error(source_transform, candidate_visuals[name])
    max_pos = max(max_pos, pos_error)
    max_rot = max(max_rot, rot_error)
    checked += 1
  source_joints = joint_install_transforms(source, source_fk)
  candidate_joints = joint_install_transforms(candidate, candidate_fk)
  for name, source_transform in source_joints.items():
    if name not in candidate_joints:
      continue
    pos_error = float(require_numpy().linalg.norm(source_transform[:3, 3] - candidate_joints[name][:3, 3]))
    max_pos = max(max_pos, pos_error)
    checked += 1
  status = "passed" if max_pos <= POSE_TOL and max_rot <= POSE_TOL and max_link_pos <= POSE_TOL and max_candidate_link_rot_from_identity <= POSE_TOL else "failed"
  return {
    "status": status,
    "details": {
      "max_position_error": max_pos,
      "max_rotation_error": max_rot,
      "max_link_position_error": max_link_pos,
      "max_candidate_link_rotation_from_identity": max_candidate_link_rot_from_identity,
      "checked_items": checked,
    },
  }


def sample_joint_values(model: RobotModel) -> list[dict[str, float]]:
  samples: list[dict[str, float]] = []
  movable = [joint for joint in model.joints.values() if joint.kind != "fixed" and joint.mimic is None]
  zero = {joint.name: 0.0 for joint in movable}
  samples.append(zero)
  positive: dict[str, float] = {}
  negative: dict[str, float] = {}
  for joint in movable:
    if joint.kind == "continuous":
      positive[joint.name] = 0.25
      negative[joint.name] = -0.25
    elif joint.kind == "prismatic":
      lo = joint.limit.lower if joint.limit.lower is not None else -0.02
      hi = joint.limit.upper if joint.limit.upper is not None else 0.02
      positive[joint.name] = min(max(0.01, lo), hi)
      negative[joint.name] = min(max(-0.01, lo), hi)
    else:
      lo = joint.limit.lower if joint.limit.lower is not None else -0.25
      hi = joint.limit.upper if joint.limit.upper is not None else 0.25
      positive[joint.name] = min(max(0.25, lo), hi)
      negative[joint.name] = min(max(-0.25, lo), hi)
  samples.append(positive)
  samples.append(negative)
  return samples


def map_values_to_candidate(values: dict[str, float], coordinate_map: dict[str, dict[str, Any]]) -> dict[str, float]:
  mapped: dict[str, float] = {}
  for name, value in values.items():
    entry = coordinate_map.get(name, {"scale": 1.0, "offset": 0.0})
    mapped[name] = float(entry.get("scale", 1.0)) * float(value) + float(entry.get("offset", 0.0))
  return mapped


def validation_motion_equivalence(
  source: RobotModel,
  candidate: RobotModel,
  coordinate_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
  if any(not bool(entry.get("equivalent", True)) for entry in coordinate_map.values()):
    return {"status": "needs_reference", "details": "non-collinear rule changes were applied; equivalence was intentionally not claimed"}
  max_pos = 0.0
  max_rot = 0.0
  checked = 0
  for source_values in sample_joint_values(source):
    candidate_values = map_values_to_candidate(source_values, coordinate_map)
    source_fk = compute_fk(source, source_values)
    candidate_fk = compute_fk(candidate, candidate_values)
    source_visuals = visual_world_transforms(source, source_fk)
    candidate_visuals = visual_world_transforms(candidate, candidate_fk)
    for name, source_transform in source_visuals.items():
      if name not in candidate_visuals:
        continue
      pos_error, rot_error = transform_error(source_transform, candidate_visuals[name])
      max_pos = max(max_pos, pos_error)
      max_rot = max(max_rot, rot_error)
      checked += 1
  status = "passed" if max_pos <= POSE_TOL and max_rot <= POSE_TOL else "failed"
  return {"status": status, "details": {"max_position_error": max_pos, "max_rotation_error": max_rot, "checked_items": checked}}


def disable_mujoco_autofix(root: ET.Element) -> None:
  mujoco = root.find("mujoco")
  if mujoco is None:
    mujoco = ET.SubElement(root, "mujoco")
  compiler = mujoco.find("compiler")
  if compiler is None:
    compiler = ET.SubElement(mujoco, "compiler")
  compiler.set("balanceinertia", "false")
  compiler.set("inertiafromgeom", "false")
  compiler.set("boundmass", "0")
  compiler.set("boundinertia", "0")
  compiler.set("settotalmass", "0")
  compiler.set("discardvisual", "false")
  compiler.set("fusestatic", "false")


def rewrite_meshes_absolute(root: ET.Element, model: RobotModel) -> None:
  for mesh in root.findall(".//mesh"):
    filename = mesh.get("filename")
    if not filename:
      continue
    path, _tried = resolve_mesh(filename, model.path.parent, model.mesh_roots)
    if path is not None:
      mesh.set("filename", str(path))


def write_binary_stl(path: Path, vertices: Any, faces: Any) -> None:
  module = require_numpy()
  with path.open("wb") as handle:
    handle.write(b"check_urdf temporary split STL".ljust(80, b" "))
    handle.write(struct.pack("<I", int(len(faces))))
    for face in faces:
      tri = module.asarray([vertices[int(face[0])], vertices[int(face[1])], vertices[int(face[2])]], dtype=float)
      normal = module.cross(tri[1] - tri[0], tri[2] - tri[0])
      norm = float(module.linalg.norm(normal))
      if norm > 1e-12:
        normal = normal / norm
      else:
        normal = module.zeros(3, dtype=float)
      handle.write(struct.pack("<fff", *[float(x) for x in normal]))
      for vertex in tri:
        handle.write(struct.pack("<fff", *[float(x) for x in vertex]))
      handle.write(struct.pack("<H", 0))


def split_stl_for_mujoco(path: Path, temp_dir: Path, *, max_faces: int = 200000) -> list[Path]:
  if path.suffix.lower() != ".stl":
    return [path]
  vertices, faces = load_stl(path)
  face_count = int(len(faces))
  if face_count <= max_faces:
    return [path]
  split_dir = temp_dir / "mesh_splits"
  split_dir.mkdir(exist_ok=True)
  result: list[Path] = []
  for start in range(0, face_count, max_faces):
    stop = min(start + max_faces, face_count)
    chunk_path = split_dir / f"{path.stem}_part_{len(result) + 1}.STL"
    write_binary_stl(chunk_path, vertices, faces[start:stop])
    result.append(chunk_path)
  return result


def split_large_mesh_elements_for_mujoco(
  root: ET.Element,
  model: RobotModel,
  temp_dir: Path,
  *,
  roles: set[str],
) -> int:
  split_count = 0
  for parent in root.iter():
    children = list(parent)
    index = 0
    while index < len(children):
      child = children[index]
      if not isinstance(child.tag, str) or child.tag not in {"visual", "collision"} or child.tag not in roles:
        index += 1
        continue
      mesh = child.find("./geometry/mesh")
      if mesh is None:
        index += 1
        continue
      filename = mesh.get("filename")
      if not filename:
        index += 1
        continue
      path, _tried = resolve_mesh(filename, model.path.parent, model.mesh_roots)
      if path is None:
        index += 1
        continue
      pieces = split_stl_for_mujoco(path, temp_dir)
      if len(pieces) <= 1:
        mesh.set("filename", str(pieces[0]))
        index += 1
        continue
      insertion_index = list(parent).index(child)
      parent.remove(child)
      for piece_index, piece in enumerate(pieces):
        duplicate = copy.deepcopy(child)
        duplicate_mesh = duplicate.find("./geometry/mesh")
        assert duplicate_mesh is not None
        duplicate_mesh.set("filename", str(piece))
        name = duplicate.get("name")
        if name:
          duplicate.set("name", f"{name}_part_{piece_index + 1}")
        parent.insert(insertion_index + piece_index, duplicate)
      split_count += 1
      children = list(parent)
      index = insertion_index + len(pieces)
  return split_count


def compile_mujoco_temp_model(model: RobotModel, mujoco_module: Any, *, split_visuals: bool) -> dict[str, Any]:
  temp_tree = copy.deepcopy(model.tree)
  temp_root = temp_tree.getroot()
  disable_mujoco_autofix(temp_root)
  rewrite_meshes_absolute(temp_root, model)
  with tempfile.TemporaryDirectory(prefix="check_urdf_mujoco_") as tmp:
    temp_dir = Path(tmp)
    split_count = split_large_mesh_elements_for_mujoco(temp_root, model, temp_dir, roles={"visual"} if split_visuals else set())
    temp_path = Path(tmp) / "model.urdf"
    ET.indent(temp_tree, space="  ")
    temp_tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    try:
      mujoco_module.MjModel.from_xml_path(str(temp_path))
    except Exception as exc:  # pragma: no cover - depends on optional dependency.
      return {"status": "failed", "details": str(exc), "temporary_visual_splits": split_count}
  details = "compiled with explicit inertial parameters preserved"
  if split_count:
    details = f"{details}; split {split_count} visual STL reference(s) in the temporary validation copy"
  return {"status": "passed", "details": details, "temporary_visual_splits": split_count}


def mujoco_failure_is_mesh_limit(result: dict[str, Any]) -> bool:
  details = str(result.get("details", "")).lower()
  return "number of faces should be between" in details or "too many faces" in details or "stl_decoder" in details


def validate_mujoco_model(model: RobotModel, *, enabled: bool) -> dict[str, Any]:
  if not enabled:
    return {"status": "skipped", "details": "pass --validate-mujoco to compile with MuJoCo"}
  try:
    import mujoco  # type: ignore[import-not-found]
  except ImportError as exc:
    raise UsageError("MuJoCo is required for --validate-mujoco") from exc
  direct = compile_mujoco_temp_model(model, mujoco, split_visuals=False)
  if direct["status"] == "passed":
    return {"status": "passed", "details": "strict direct compile passed", "direct": direct}
  auxiliary = compile_mujoco_temp_model(model, mujoco, split_visuals=True)
  if auxiliary["status"] == "passed":
    return {
      "status": "needs_reference",
      "details": "strict direct compile failed, but a temporary visual-only STL split compiled; this does not certify contact geometry",
      "direct": direct,
      "auxiliary": auxiliary,
    }
  if mujoco_failure_is_mesh_limit(direct) or mujoco_failure_is_mesh_limit(auxiliary):
    return {
      "status": "needs_reference",
      "details": "MuJoCo compile is limited by mesh face-count support; visual-only split did not fully remove the importer limit, often because collision mesh still exceeds the limit; this does not certify contact geometry",
      "direct": direct,
      "auxiliary": auxiliary,
    }
  return {"status": "failed", "details": "MuJoCo compile failed without inertial auto-fixes", "direct": direct, "auxiliary": auxiliary}


def build_candidate(
  source: RobotModel,
  profile: str,
  rules: Rules,
  reporter: ReportBuilder,
) -> tuple[RobotModel | None, dict[str, dict[str, Any]]]:
  candidate_tree = copy.deepcopy(source.tree)
  candidate = parse_robot(candidate_tree, source.path, source.mesh_roots, ReportBuilder(), context="candidate")
  if candidate is None:
    reporter.issue("error", "candidate_parse_failed", source.name, "internal candidate parse failed before modifications")
    return None, {}
  standardize_candidate_frames(source, candidate, reporter)
  candidate = parse_robot(candidate_tree, source.path, source.mesh_roots, ReportBuilder(), context="candidate")
  if candidate is None:
    reporter.issue("error", "candidate_parse_failed", source.name, "internal candidate parse failed after frame standardization")
    return None, {}
  coordinate_map = profile_axis_checks_and_fixes(candidate, profile, rules, reporter)
  check_mimic_cycles(candidate, reporter)
  simplify_collisions(candidate, reporter)
  candidate = parse_robot(candidate_tree, source.path, source.mesh_roots, ReportBuilder(), context="candidate")
  if candidate is None:
    reporter.issue("error", "candidate_parse_failed", source.name, "internal candidate parse failed after collision simplification")
    return None, coordinate_map
  for name, entry in coordinate_map.items():
    if name not in candidate.joints:
      continue
    entry.setdefault("scale", 1.0)
    entry.setdefault("offset", 0.0)
    entry.setdefault("equivalent", True)
  return candidate, coordinate_map


def build_report(
  *,
  source_path: Path,
  profile: str,
  reporter: ReportBuilder,
  coordinate_map: dict[str, dict[str, Any]],
  validation: dict[str, Any],
) -> dict[str, Any]:
  summary = reporter.summary()
  return {
    "schema_version": 1,
    "source": str(source_path.resolve()),
    "profile": profile,
    "issues": [issue.to_json() for issue in reporter.issues],
    "changes": [change.to_json() for change in reporter.changes],
    "joint_coordinate_map": sanitize_json(coordinate_map),
    "validation": sanitize_json(validation),
    "summary": summary,
  }


def validate_output_paths(input_path: Path, output: Path | None, report: Path | None) -> None:
  resolved = {"input": input_path.resolve()}
  if output is not None:
    resolved["output"] = output.resolve()
  if report is not None:
    resolved["report"] = report.resolve()
  items = list(resolved.items())
  for index, (left_name, left_path) in enumerate(items):
    for right_name, right_path in items[index + 1 :]:
      if left_path == right_path:
        raise UsageError(f"{left_name} and {right_name} paths must be distinct: {left_path}")
  if output is not None:
    if output.resolve() == input_path.resolve():
      raise UsageError("output must not overwrite the input URDF")
    if output.exists():
      raise UsageError(f"output already exists; choose a new path: {output}")
  if report is not None and report.exists():
    raise UsageError(f"report already exists; choose a new path: {report}")


def write_report(path: Path, report: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
  with tempfile.NamedTemporaryFile(
    "w",
    encoding="utf-8",
    suffix=".json",
    prefix=f".{path.stem}.",
    dir=path.parent,
    delete=False,
  ) as handle:
    temp_path = Path(handle.name)
    handle.write(text)
  try:
    os.replace(temp_path, path)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise


def write_candidate(output: Path, candidate: RobotModel, reporter: ReportBuilder) -> None:
  apply_mesh_paths_for_output(candidate, output, reporter)
  output.parent.mkdir(parents=True, exist_ok=True)
  ET.indent(candidate.tree, space="  ")
  with tempfile.NamedTemporaryFile(
    "wb",
    suffix=".urdf",
    prefix=f".{output.stem}.",
    dir=output.parent,
    delete=False,
  ) as handle:
    temp_path = Path(handle.name)
    candidate.tree.write(handle, encoding="utf-8", xml_declaration=True)
  try:
    os.replace(temp_path, output)
  except Exception:
    temp_path.unlink(missing_ok=True)
    raise


def run_analysis(args: argparse.Namespace) -> tuple[int, dict[str, Any], RobotModel | None, RobotModel | None]:
  require_numpy()
  source_path = Path(args.model).expanduser().resolve()
  if not source_path.is_file():
    raise UsageError(f"input URDF does not exist: {source_path}")
  output_path = Path(args.output).expanduser().resolve() if args.output else None
  report_path = Path(args.report).expanduser().resolve() if args.report else None
  validate_output_paths(source_path, output_path, report_path)
  mesh_roots = [Path(path).expanduser().resolve() for path in (args.mesh_root or [])]

  reporter = ReportBuilder()
  try:
    tree = parse_xml(source_path)
  except ET.ParseError as exc:
    reporter.issue("error", "xml_parse_error", str(source_path), str(exc))
    report = build_report(source_path=source_path, profile=args.profile, reporter=reporter, coordinate_map={}, validation=empty_validation())
    if report_path is not None:
      write_report(report_path, report)
    return 1, report, None, None

  source = parse_robot(tree, source_path, mesh_roots, reporter, context="source")
  rules = load_rules(Path(args.rules).expanduser().resolve() if args.rules else None, source, reporter)
  validation = empty_validation()
  candidate: RobotModel | None = None
  coordinate_map: dict[str, dict[str, Any]] = {}

  if source is not None:
    check_structure(source, reporter)
    check_quadruped_mounts(source, args.profile, reporter)
    candidate, coordinate_map = build_candidate(source, args.profile, rules, reporter)
    if candidate is not None:
      if reporter.has_error_code("mimic_cycle"):
        zero_pose = {"status": "not_run", "details": "mimic cycle prevents reliable FK validation"}
        motion = {"status": "not_run", "details": "mimic cycle prevents reliable FK validation"}
      else:
        zero_pose = validation_zero_pose(source, candidate)
        motion = validation_motion_equivalence(source, candidate, coordinate_map)
      validation["zero_pose"] = zero_pose
      validation["motion_equivalence"] = motion
      if zero_pose["status"] == "failed":
        reporter.issue("error", "zero_pose_invariant_failed", source.name, "candidate does not preserve zero-pose visual geometry or joint mounts", candidate=zero_pose["details"])
      if motion["status"] == "failed":
        reporter.issue("error", "motion_equivalence_failed", source.name, "candidate coordinate/sign changes are not kinematically equivalent", candidate=motion["details"])
      try:
        source_mj = validate_mujoco_model(source, enabled=args.validate_mujoco)
        candidate_mj = validate_mujoco_model(candidate, enabled=args.validate_mujoco)
      except UsageError:
        raise
      validation["mujoco"] = {"source": source_mj, "candidate": candidate_mj}
      if args.validate_mujoco:
        if source_mj["status"] == "failed":
          reporter.issue("error", "mujoco_compile_failed", "source", "MuJoCo failed to compile source URDF without inertia auto-fixes", candidate=source_mj["details"])
        if candidate_mj["status"] == "failed":
          reporter.issue("error", "mujoco_compile_failed", "candidate", "MuJoCo failed to compile candidate URDF without inertia auto-fixes", candidate=candidate_mj["details"])

  if output_path is not None:
    if candidate is None:
      reporter.issue("error", "output_blocked", str(output_path), "candidate could not be built")
    elif reporter.summary()["errors"] > 0:
      reporter.issue("error", "output_blocked", str(output_path), "candidate export is blocked because errors were detected")
    else:
      write_candidate(output_path, candidate, reporter)

  report = build_report(
    source_path=source_path,
    profile=args.profile,
    reporter=reporter,
    coordinate_map=coordinate_map,
    validation=validation,
  )
  if report_path is not None:
    write_report(report_path, report)
  exit_code = 1 if report["summary"]["errors"] else 0
  return exit_code, report, source, candidate


def empty_validation() -> dict[str, Any]:
  return {
    "zero_pose": {"status": "not_run", "details": "candidate was not available"},
    "motion_equivalence": {"status": "not_run", "details": "candidate was not available"},
    "mujoco": {"status": "skipped", "details": "pass --validate-mujoco to compile with MuJoCo"},
  }


def preview_joint_order(model: RobotModel) -> list[Joint]:
  return [joint for joint in model.joints.values() if joint.kind != "fixed" and joint.mimic is None]


def preview_mesh(
  geometry: Geometry,
  mesh_cache: dict[tuple[str, tuple[float, float, float]], tuple[Any, Any]],
) -> tuple[Any, Any]:
  if geometry.mesh_path is None:
    return require_numpy().empty((0, 3)), require_numpy().empty((0, 3), dtype=np.int64)
  key = (str(geometry.mesh_path), tuple(float(x) for x in geometry.mesh_scale))
  if key in mesh_cache:
    return mesh_cache[key]
  vertices, faces = load_stl(geometry.mesh_path)
  vertices = vertices * geometry.mesh_scale
  mesh_cache[key] = (vertices, faces)
  return vertices, faces


def draw_preview_model(
  server: Any,
  model: RobotModel,
  *,
  namespace: str,
  values: dict[str, float],
  visible: bool,
  show_frames: bool,
  show_axes: bool,
  show_com: bool,
  show_collision: bool,
  show_visuals: bool = True,
  mesh_cache: dict[tuple[str, tuple[float, float, float]], tuple[Any, Any]] | None = None,
) -> list[Any]:
  handles: list[Any] = []
  if mesh_cache is None:
    mesh_cache = {}
  transforms = compute_fk(model, values)
  for link_name, link in model.links.items():
    link_transform = transforms.get(link_name)
    if link_transform is None:
      continue
    if show_frames:
      handles.append(add_frame(server, f"/{namespace}/frames/{link_name}", link_transform, visible=visible))
    if show_com and link.inertial_origin is not None:
      com_transform = link_transform @ transform_from_origin(link.inertial_origin)
      handles.append(add_sphere(server, f"/{namespace}/com/{link_name}", com_transform[:3, 3], 0.008, (255, 80, 80), visible=visible))
    if show_visuals:
      for visual in link.visuals:
        if visual.kind != "mesh" or visual.mesh_path is None:
          continue
        vertices, faces = preview_mesh(visual, mesh_cache)
        if len(vertices) == 0 or len(faces) == 0:
          continue
        local = transform_from_origin(visual.origin)
        world = link_transform @ local
        points = vertices @ world[:3, :3].T + world[:3, 3]
        try:
          handle = server.scene.add_mesh_simple(
            f"/{namespace}/visual/{link_name}/{visual.index}",
            vertices=points,
            faces=faces,
            color=(170, 180, 190),
            opacity=1.0 if visible else 0.0,
          )
          handles.append(handle)
        except TypeError:
          handle = server.scene.add_mesh_simple(
            name=f"/{namespace}/visual/{link_name}/{visual.index}",
            vertices=points,
            faces=faces,
            color=(170, 180, 190),
          )
          if hasattr(handle, "visible"):
            handle.visible = visible
          handles.append(handle)
    if show_collision:
      for collision in link.collisions:
        collision_transform = link_transform @ transform_from_origin(collision.origin)
        handles.extend(add_collision_shape(server, f"/{namespace}/collision/{link_name}/{collision.index}", collision, collision_transform, visible=visible))
    if show_axes:
      for joint in model.children.get(link_name, []):
        joint_transform = link_transform @ transform_from_origin(joint.origin)
        handles.append(add_axis(server, f"/{namespace}/axis/{joint.name}", joint_transform, joint.axis, visible=visible))
  return handles


def add_frame(server: Any, name: str, transform: Any, *, visible: bool) -> Any:
  quat = quat_wxyz_from_matrix(transform[:3, :3])
  handle = server.scene.add_frame(name, position=transform[:3, 3], wxyz=quat, axes_length=0.06, axes_radius=0.003)
  if hasattr(handle, "visible"):
    handle.visible = visible
  return handle


def add_sphere(server: Any, name: str, center: Any, radius: float, color: tuple[int, int, int], *, visible: bool) -> Any:
  try:
    handle = server.scene.add_icosphere(name, radius=radius, position=center, color=color)
  except AttributeError:
    handle = server.scene.add_frame(name, position=center, axes_length=radius, axes_radius=radius * 0.15)
  if hasattr(handle, "visible"):
    handle.visible = visible
  return handle


def add_axis(server: Any, name: str, transform: Any, axis: Any, *, visible: bool) -> Any:
  module = require_numpy()
  axis = normalize(axis)
  if axis is None:
    axis = nparray((0.0, 0.0, 1.0))
  start = transform[:3, 3]
  end = start + transform[:3, :3] @ axis * 0.08
  try:
    handle = server.scene.add_line_segments(name, points=module.asarray([[start, end]], dtype=float), colors=module.asarray([[[255, 180, 60], [255, 180, 60]]], dtype=np.uint8))
  except AttributeError:
    handle = add_frame(server, name, transform, visible=visible)
  if hasattr(handle, "visible"):
    handle.visible = visible
  return handle


def add_collision_shape(server: Any, name: str, geometry: Geometry, transform: Any, *, visible: bool) -> list[Any]:
  handles: list[Any] = []
  quat = quat_wxyz_from_matrix(transform[:3, :3])
  color = (240, 160, 40)
  try:
    if geometry.kind == "box" and geometry.box_size is not None:
      handle = server.scene.add_box(name, dimensions=geometry.box_size, position=transform[:3, 3], wxyz=quat, color=color, opacity=0.35)
      handles.append(handle)
    elif geometry.kind == "sphere" and geometry.radius is not None:
      handles.append(server.scene.add_icosphere(name, radius=geometry.radius, position=transform[:3, 3], color=color))
    elif geometry.kind == "cylinder" and geometry.radius is not None and geometry.length is not None:
      handles.append(server.scene.add_cylinder(name, radius=geometry.radius, height=geometry.length, position=transform[:3, 3], wxyz=quat, color=color, opacity=0.35))
  except Exception:
    handles.append(add_frame(server, name, transform, visible=visible))
  for handle in handles:
    if hasattr(handle, "visible"):
      handle.visible = visible
  return handles


def build_preview(
  source: RobotModel,
  candidate: RobotModel | None,
  report: dict[str, Any],
  *,
  host: str = "127.0.0.1",
  port: int | None = None,
  load_visuals: bool = True,
) -> Any:
  try:
    import viser  # type: ignore[import-not-found]
  except ImportError as exc:
    raise UsageError("viser is required for --preview") from exc
  kwargs: dict[str, Any] = {"host": host}
  if port is not None:
    kwargs["port"] = port
  server = construct_viser_server(viser, kwargs)
  gui = server.gui
  state = {
    "show_candidate": False,
    "show_frames": True,
    "show_axes": True,
    "show_com": True,
    "show_collision": True,
    "show_visuals": load_visuals,
    "handles": [],
    "values": {joint.name: 0.0 for joint in preview_joint_order(source)},
    "mesh_cache": {},
  }
  sliders = {}
  for joint in preview_joint_order(source):
    lower = joint.limit.lower if joint.limit.lower is not None else -math.pi
    upper = joint.limit.upper if joint.limit.upper is not None else math.pi
    lower = min(float(lower), 0.0)
    upper = max(float(upper), 0.0)
    if lower == upper:
      upper = lower + 1.0
    slider = gui.add_slider(joint.name, min=lower, max=upper, step=max((upper - lower) / 500.0, 1e-4), initial_value=0.0)
    sliders[joint.name] = slider

  show_candidate = gui.add_checkbox("candidate", initial_value=state["show_candidate"])
  show_frames = gui.add_checkbox("frames", initial_value=True)
  show_axes = gui.add_checkbox("axes", initial_value=True)
  show_com = gui.add_checkbox("com", initial_value=True)
  show_collision = gui.add_checkbox("collision", initial_value=True)
  show_visuals = gui.add_checkbox("visual STL", initial_value=load_visuals)
  reset_button = gui.add_button("reset zero")
  jog_button = gui.add_button("positive step")
  status = gui.add_text("status", initial_value=f"errors={report['summary']['errors']} warnings={report['summary']['warnings']}")

  def clear() -> None:
    for handle in state["handles"]:
      try:
        handle.remove()
      except Exception:
        pass
    state["handles"] = []

  def redraw() -> None:
    clear()
    values = {name: float(slider.value) for name, slider in sliders.items()}
    state["values"] = values
    if bool(show_candidate.value) and candidate is not None:
      mapped = map_values_to_candidate(values, report.get("joint_coordinate_map", {}))
      state["handles"] = draw_preview_model(
        server,
        candidate,
        namespace="candidate",
        values=mapped,
        visible=True,
        show_frames=bool(show_frames.value),
        show_axes=bool(show_axes.value),
        show_com=bool(show_com.value),
        show_collision=bool(show_collision.value),
        show_visuals=bool(show_visuals.value),
        mesh_cache=state["mesh_cache"],
      )
    else:
      state["handles"] = draw_preview_model(
        server,
        source,
        namespace="source",
        values=values,
        visible=True,
        show_frames=bool(show_frames.value),
        show_axes=bool(show_axes.value),
        show_com=bool(show_com.value),
        show_collision=bool(show_collision.value),
        show_visuals=bool(show_visuals.value),
        mesh_cache=state["mesh_cache"],
      )
    loaded_faces = sum(int(len(faces)) for _vertices, faces in state["mesh_cache"].values()) if bool(show_visuals.value) else 0
    status.value = (
      f"mode={'candidate' if bool(show_candidate.value) and candidate is not None else 'source'} "
      f"errors={report['summary']['errors']} warnings={report['summary']['warnings']} "
      f"visual_faces={loaded_faces}"
    )

  for slider in sliders.values():
    slider.on_update(lambda _event: redraw())
  for checkbox in (show_candidate, show_frames, show_axes, show_com, show_collision, show_visuals):
    checkbox.on_update(lambda _event: redraw())

  @reset_button.on_click
  def _reset(_event: Any) -> None:
    for slider in sliders.values():
      slider.value = 0.0
    redraw()

  @jog_button.on_click
  def _jog(_event: Any) -> None:
    for joint in preview_joint_order(source):
      slider = sliders[joint.name]
      slider.value = min(float(slider.max), float(slider.value) + float(slider.step))
    redraw()

  redraw()
  return server


def construct_viser_server(viser_module: Any, kwargs: dict[str, Any], *, timeout_s: float = 5.0) -> Any:
  result: dict[str, Any] = {}

  def worker() -> None:
    try:
      result["server"] = viser_module.ViserServer(**kwargs)
    except BaseException as exc:  # pragma: no cover - depends on local socket stack.
      result["error"] = exc

  thread = threading.Thread(target=worker, daemon=True)
  thread.start()
  thread.join(timeout_s)
  if thread.is_alive():
    raise UsageError(
      "Viser server did not start within "
      f"{timeout_s:.1f}s; local socket binding may be unavailable in this environment"
    )
  if "error" in result:
    raise UsageError(f"Viser server failed to start: {result['error']}") from result["error"]
  return result["server"]


def start_preview(
  source: RobotModel,
  candidate: RobotModel | None,
  report: dict[str, Any],
  *,
  host: str = "127.0.0.1",
  port: int | None = None,
  stop_event: Any | None = None,
  load_visuals: bool = True,
) -> Any:
  server = build_preview(source, candidate, report, host=host, port=port, load_visuals=load_visuals)
  print(f"check_urdf preview running on {host}:{port or 'auto'}", file=sys.stderr)
  try:
    while stop_event is None or not stop_event.is_set():
      time.sleep(0.1)
  except KeyboardInterrupt:
    pass
  return server


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("model", metavar="MODEL.urdf", help="URDF model to check")
  parser.add_argument("--profile", choices=("generic", "go2", "ri4438"), default="generic", help="rule profile (default: generic)")
  parser.add_argument("--mesh-root", action="append", help="additional root for resolving relative or package:// meshes; may be repeated")
  parser.add_argument("--rules", help="JSON file with explicit joint axis/limit/mimic overrides")
  parser.add_argument("--preview", action="store_true", help="start a local Viser preview after checks")
  parser.add_argument("--report", help="write JSON report to this new path")
  parser.add_argument("--output", help="write candidate URDF to this new path; never overwrites the input")
  parser.add_argument("--validate-mujoco", action="store_true", help="compile source and candidate with MuJoCo without inertia auto-fixes")
  parser.add_argument("--host", default="127.0.0.1", help="Viser host for --preview (default: 127.0.0.1)")
  parser.add_argument("--port", type=int, help="Viser port for --preview")
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)
  try:
    exit_code, report, source, candidate = run_analysis(args)
    summary = report["summary"]
    if args.output and summary["errors"] == 0:
      print(f"check_urdf: wrote candidate {Path(args.output).expanduser().resolve()}", file=sys.stderr)
    if args.report:
      print(f"check_urdf: wrote report {Path(args.report).expanduser().resolve()}", file=sys.stderr)
    print(
      f"check_urdf: errors={summary['errors']} warnings={summary['warnings']} infos={summary['infos']}",
      file=sys.stderr,
    )
    if args.preview:
      if source is None:
        raise UsageError("cannot preview because the URDF did not parse")
      if any(issue.get("severity") == "error" and issue.get("code") == "mimic_cycle" for issue in report.get("issues", [])):
        raise UsageError("cannot preview because a mimic cycle prevents reliable FK evaluation")
      start_preview(source, candidate, report, host=args.host, port=args.port)
    return exit_code
  except UsageError as exc:
    print(f"check_urdf: error: {exc}", file=sys.stderr)
    return 2
  except KeyboardInterrupt:
    print("check_urdf: interrupted", file=sys.stderr)
    return 130


if __name__ == "__main__":
  raise SystemExit(main())