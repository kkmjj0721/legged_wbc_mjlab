"""Convert a portable URDF tree to readable, project-style MuJoCo XML.

This is deliberately an explicit translation rather than MuJoCo's URDF loader:
fixed bodies, visual/collision separation and the original object names remain
available to the migration workbench. Physical parameters are never calibrated
or replaced with those of the template robots.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import struct
import xml.etree.ElementTree as ET

MAX_BYTES = 2 * 1024**3
MAX_FILES = 10000


class UrdfError(ValueError):
    """A conversion error that can be displayed directly to the user."""


def _numbers(value, count, context, default=None):
    if value is None:
        if default is None:
            raise UrdfError(f"{context} 缺少数值")
        return list(default)
    try:
        result = [float(v) for v in value.split()]
    except (ValueError, AttributeError) as exc:
        raise UrdfError(f"{context} 需要 {count} 个有限数值") from exc
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise UrdfError(f"{context} 需要 {count} 个有限数值")
    return result


def _number(value, context, default=None):
    return _numbers(value, 1, context, None if default is None else [default])[0]


def _fmt(values):
    return " ".join(format(float(v), ".12g") for v in values)


def _rotation(rpy):
    """URDF uses fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    import numpy as np

    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _quaternion(rpy):
    roll, pitch, yaw = (v / 2 for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]


def _origin(element, context):
    origin = element.find("origin")
    if origin is None:
        return {"pos": "0 0 0"}
    xyz = _numbers(origin.get("xyz"), 3, f"{context}.origin.xyz", [0, 0, 0])
    rpy = _numbers(origin.get("rpy"), 3, f"{context}.origin.rpy", [0, 0, 0])
    result = {"pos": _fmt(xyz)}
    if any(rpy):
        result["quat"] = _fmt(_quaternion(rpy))
    return result


def _safe_file(root, candidate):
    candidate = Path(candidate)
    # Resolve only after checking every component, including linked directories.
    absolute = candidate.absolute()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise UrdfError(f"资源引用超出模型根目录：{candidate}") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise UrdfError(f"URDF 不接受符号链接资源：{cursor}")
    resolved = absolute.resolve()
    if not resolved.is_relative_to(root):
        raise UrdfError(f"资源引用超出模型根目录：{candidate}")
    if not resolved.is_file():
        raise UrdfError(f"缺失 URDF 资源：{candidate}")
    return resolved


class _Resources:
    def __init__(self, root, entry, destination, asset, report):
        self.root, self.entry, self.destination = root, entry, destination
        self.asset, self.report = asset, report
        self.files, self.meshes, self.textures = {}, {}, {}

    def resolve(self, reference):
        if not reference or "\x00" in reference or "\\" in reference:
            raise UrdfError(f"无效 URDF 资源路径：{reference!r}")
        if reference.startswith("package://"):
            package_path = reference[len("package://"):]
            parts = package_path.split("/", 1)
            if len(parts) != 2 or not parts[0] or ".." in Path(package_path).parts:
                raise UrdfError(f"无效 package:// 资源路径：{reference}")
            package, rel = parts
            bases = [self.entry.parent, *self.entry.parent.parents]
            bases = [base for base in bases if base.is_relative_to(self.root)]
            candidates = [self.root / package / rel, self.root / rel]
            for base in bases:
                candidates.extend([base / package / rel, base / rel])
        elif "://" in reference:
            raise UrdfError(f"不支持网络或其他 URI 资源：{reference}；请将资源放入模型目录")
        else:
            path = Path(reference)
            candidates = [path] if path.is_absolute() else [self.entry.parent / path, self.root / path]
        errors = []
        for candidate in dict.fromkeys(candidates):
            try:
                result = _safe_file(self.root, candidate)
            except UrdfError as exc:
                errors.append(str(exc))
                continue
            if result not in self.files:
                self.files[result] = result.stat().st_size
                if len(self.files) > MAX_FILES or sum(self.files.values()) > MAX_BYTES:
                    raise UrdfError("URDF 资源超过 10000 个文件或 2 GiB")
            return result
        raise UrdfError(f"无法解析资源 {reference}；请选择覆盖 URDF 与全部资源的根目录。{'；'.join(errors[:2])}")

    def asset_name(self, path):
        relative = path.relative_to(self.root).as_posix()
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "asset"
        return f"{stem}_{hashlib.sha256(relative.encode()).hexdigest()[:10]}"

    def texture(self, reference):
        path = self.resolve(reference)
        if path not in self.textures:
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                raise UrdfError(f"纹理仅支持 PNG/JPEG：{reference}")
            name = "texture_" + self.asset_name(path)
            relative = f"assets/{self.asset_name(path)}{path.suffix.lower()}"
            (self.destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, self.destination / relative)
            ET.SubElement(self.asset, "texture", {"name": name, "type": "2d", "file": relative})
            self.textures[path] = name
        return self.textures[path]

    def mesh(self, reference, scale, collision=False):
        path = self.resolve(reference)
        if collision:
            warning = "碰撞 mesh 在 MuJoCo 中按凸包参与接触；请确认源网格适合该语义，复杂凹形需人工提供碰撞代理。"
            if warning not in self.report["warnings"]:
                self.report["warnings"].append(warning)
        key = (path, tuple(scale))
        if key in self.meshes:
            return self.meshes[key]
        name = self.asset_name(path)
        if any(p == path for p, _ in self.meshes):
            name += "_" + hashlib.sha256(_fmt(scale).encode()).hexdigest()[:8]
        extension = path.suffix.lower()
        data = path.read_bytes()
        converted, encoding = extension != ".obj", {}
        relative = f"assets/{name}.obj"
        target = self.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if extension == ".stl":
            face_count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
            if face_count >= 0 and 84 + face_count * 50 == len(data):
                if not face_count:
                    raise UrdfError(f"STL 没有三角面：{reference}")
                encoding = self._stl_to_obj(data, target, reference)
            else:
                encoding = self._mesh_to_obj(path, target)
        elif extension == ".obj":
            # MuJoCo reads mesh geometry. URDF material/color remains authoritative.
            if re.search(rb"(?m)^\s*(mtllib|usemtl)\s", data):
                self.report["warnings"].append(f"{reference} 的 OBJ 材质不参与转换；使用 URDF material/color。")
            target.write_bytes(data)
            encoding = {"encoding": "obj_copy", "geometry_preserved": True}
        elif extension in (".dae", ".ply", ".off"):
            if extension == ".dae":
                if importlib.util.find_spec("collada") is None:
                    raise UrdfError(f"DAE 网格 {reference} 需要可选 pycollada；请先在建模工具中导出 STL/OBJ，或配置该依赖。")
                # Check DAE external assets before letting a mesh reader see them.
                tree = _parse_xml(data, path.name)
                for node in tree.iter():
                    if node.tag.rsplit("}", 1)[-1] == "init_from" and node.text and not node.text.startswith("#"):
                        dependency = _safe_file(self.root, path.parent / node.text)
                        self.files[dependency] = dependency.stat().st_size
                        if len(self.files) > MAX_FILES or sum(self.files.values()) > MAX_BYTES:
                            raise UrdfError("URDF 资源超过 10000 个文件或 2 GiB")
            encoding = self._mesh_to_obj(path, target)
        else:
            raise UrdfError(f"不支持网格格式 {path.suffix}：{reference}；请导出 STL/OBJ。")
        outputs = [(name, relative)]
        if extension in (".dae", ".ply", ".off"):
            self.report["warnings"].append(f"{reference} 已直接编码为 OBJ，保留解码后的三角几何与可用 UV；材质/颜色使用 URDF material/color。")
        for mesh_name, relative in outputs:
            ET.SubElement(self.asset, "mesh", {"name": mesh_name, "file": relative, "scale": _fmt(scale)})
        self.report["meshes"].append({"source": path.relative_to(self.root).as_posix(), "scale": scale,
                                      "outputs": [relative for _, relative in outputs], "converted": converted,
                                      **encoding})
        self.meshes[key] = [mesh_name for mesh_name, _ in outputs]
        return self.meshes[key]

    def _stl_to_obj(self, data, target, reference):
        """Change STL storage without splitting its geometry or collision hull."""
        import numpy as np

        # Binary STL contains float32 vertices in the order of the original
        # triangles. Deduplicate equal coordinates only; keep face order and
        # winding. Seventeen decimal digits round-trip every float32 value.
        record_type = np.dtype([("normal", "<f4", (3,)),
                                ("vertices", "<f4", (3, 3)),
                                ("attribute", "<u2")])
        records = np.frombuffer(data, dtype=record_type, offset=84)
        triangles = records["vertices"]
        if not np.isfinite(triangles).all():
            raise UrdfError(f"STL 顶点含有非有限数值：{reference}")
        vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        self._write_obj(target, vertices, inverse.reshape(-1, 3))
        return {"encoding": "stl_to_obj", "triangle_count": len(triangles),
                "vertex_count": len(vertices), "coordinate_precision": "float32_exact",
                "triangle_order_preserved": True, "winding_preserved": True,
                "collision_hull_preserved": True, "simplified": False,
                "split": False}

    @staticmethod
    def _write_obj(target, vertices, faces, uv=None):
        import numpy as np

        if not len(faces) or not np.isfinite(vertices).all():
            raise UrdfError("网格没有有效三角面或顶点含有非有限数值")
        with target.open("w", encoding="ascii", newline="\n") as output:
            output.write("# Original decoded triangle geometry; no simplification or hull splitting.\n")
            np.savetxt(output, vertices, fmt="v %.17g %.17g %.17g")
            if uv is not None:
                np.savetxt(output, uv, fmt="vt %.17g %.17g")
                indices = np.repeat(faces + 1, 2, axis=1)
                np.savetxt(output, indices, fmt="f %d/%d %d/%d %d/%d")
            else:
                np.savetxt(output, faces + 1, fmt="f %d %d %d")

    def _mesh_to_obj(self, path, target):
        try:
            import numpy as np
            import trimesh

            geometry = trimesh.load(path, process=False)
            if isinstance(geometry, trimesh.Scene):
                geometry = geometry.to_geometry()
            if not isinstance(geometry, trimesh.Trimesh) or not len(geometry.faces):
                raise ValueError("没有可用三角几何")
            uv = getattr(geometry.visual, "uv", None)
            if uv is not None:
                uv = np.asarray(uv)
                if uv.shape != (len(geometry.vertices), 2) or not np.isfinite(uv).all():
                    raise ValueError("纹理坐标无效")
            self._write_obj(target, geometry.vertices, geometry.faces, uv)
            return {"encoding": path.suffix.lower().lstrip(".") + "_to_obj",
                    "triangle_count": len(geometry.faces), "vertex_count": len(geometry.vertices),
                    "coordinate_precision": "float64_roundtrip", "texcoords_preserved": uv is not None,
                    "geometry_preserved": True, "simplified": False, "split": False}
        except Exception as exc:
            raise UrdfError(f"无法转换网格 {path.name}：{exc}；请在建模工具中导出 OBJ。") from exc


def _parse_xml(content, context):
    try:
        content = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UrdfError(f"{context} 请保存为 UTF-8 XML") from exc
    if "<!DOCTYPE" in content.upper() or "<!ENTITY" in content.upper():
        raise UrdfError(f"{context} 不允许 DTD 或外部实体")
    try:
        return ET.fromstring(content)
    except ET.ParseError as exc:
        raise UrdfError(f"URDF/XML 解析失败 {context}：{exc}") from exc


def _named(elements, context):
    result = {}
    for element in elements:
        name = element.get("name", "")
        if not name or "/" in name or name in result:
            raise UrdfError(f"{context} 名称必须非空、唯一且不含 /：{name!r}")
        result[name] = element
    return result


def _inertial(link, body):
    import numpy as np

    inertial = link.find("inertial")
    if inertial is None:
        return False
    mass, inertia = inertial.find("mass"), inertial.find("inertia")
    if mass is None or inertia is None:
        raise UrdfError(f"{link.get('name')} 的 inertial 必须包含 mass 和 inertia")
    m = _number(mass.get("value"), f"{link.get('name')}.mass")
    values = [_number(inertia.get(key), f"{link.get('name')}.{key}") for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
    ixx, iyy, izz, ixy, ixz, iyz = values
    matrix = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    if m <= 0 or eigenvalues[0] <= 0 or eigenvalues[2] > eigenvalues[0] + eigenvalues[1] + 1e-12:
        raise UrdfError(f"{link.get('name')} 的质量/惯量不满足正质量、正定及三角不等式；请核对源 URDF，工具不会自动平衡惯量。")
    origin = inertial.find("origin")
    rpy = _numbers(origin.get("rpy") if origin is not None else None, 3, "inertial.rpy", [0, 0, 0])
    rotation = _rotation(rpy)
    matrix = rotation @ matrix @ rotation.T
    attributes = _origin(inertial, f"{link.get('name')}.inertial")
    attributes.pop("quat", None)  # fullinertia is expressed in the body frame.
    attributes.update(mass=_fmt([m]), fullinertia=_fmt([matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[0, 1], matrix[0, 2], matrix[1, 2]]))
    ET.SubElement(body, "inertial", attributes)
    return True


def convert_urdf(root: Path, entry: Path, destination: Path) -> dict:
    """Write an isolated MJCF bundle and conversion report into ``destination``.

    ``root`` must cover the entry and every referenced resource. The caller owns
    the destination transaction and removes it if conversion or inspection fails.
    Only fixed/revolute trees are accepted; the importing migration contract then
    enforces a floating base plus exactly twelve driven hinges.
    """
    root = Path(root).resolve()
    entry = Path(entry)
    entry = _safe_file(root, entry if entry.is_absolute() else root / entry)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise UrdfError("URDF 转换输出目录必须为空，避免覆盖已有模型")
    destination.mkdir(parents=True, exist_ok=True)
    if entry.stat().st_size > MAX_BYTES:
        raise UrdfError("URDF 超过 2 GiB")
    source = entry.read_bytes()
    robot = _parse_xml(source, entry.name)
    if robot.tag != "robot":
        raise UrdfError("URDF 入口根元素必须为 <robot>；xacro 请先展开为 URDF")
    if any("xacro" in str(node.tag) or "${" in str(value) or "$(" in str(value)
           for node in robot.iter() for value in [node.tag, *node.attrib.values()]):
        raise UrdfError("不直接执行 xacro；请先展开宏并导出 URDF")
    links = _named(robot.findall("link"), "link")
    joints = _named(robot.findall("joint"), "joint")
    mimics = [(name, joint.find("mimic")) for name, joint in joints.items()
              if joint.find("mimic") is not None]
    if mimics:
        counts = {kind: sum((joint.get("type") or "未指定") == kind for joint in joints.values())
                  for kind in sorted({joint.get("type") or "未指定" for joint in joints.values()})}
        relationships = [f"{name} 跟随 {node.get('joint', '未指定')}"
                         f"（倍率 {node.get('multiplier', '1')}，偏移 {node.get('offset', '0')}）"
                         for name, node in mimics]
        raise UrdfError("当前迁移工具尚不支持 URDF mimic 联动关节，未改变源模型。\n"
                        f"实际读取文件：{entry}\n机器人：{robot.get('name') or entry.stem}\n"
                        f"关节数量：{', '.join(f'{kind}={count}' for kind, count in counts.items())}\n"
                        f"联动关系：{'；'.join(relationships)}\n"
                        "当前任务模板要求四条腿共 12 个独立驱动关节；含机械臂、夹爪或额外被动联动的模型需要专门适配。"
                        "若文件与预期不同，请检查两个路径输入框或 ZIP 入口；不要直接删除 mimic 标签来绕过约束。")
    if robot.find(".//loop_joint") is not None or robot.find(".//loop") is not None:
        raise UrdfError("第一版不支持 URDF 闭链约束")
    if not links:
        raise UrdfError("URDF 没有 link")
    children, parents = {name: [] for name in links}, {}
    joint_metadata = []
    for name, joint in joints.items():
        kind = joint.get("type")
        if kind not in ("fixed", "revolute"):
            raise UrdfError(f"关节 {name} 类型 {kind!r} 不受支持；第一版支持 fixed 和有有限限位的 revolute")
        parent, child = joint.find("parent"), joint.find("child")
        p, c = parent.get("link") if parent is not None else None, child.get("link") if child is not None else None
        if p not in links or c not in links or p == c or c in parents:
            raise UrdfError(f"关节 {name} 的 parent/child 无效、重复或形成闭链")
        parents[c] = p
        children[p].append((c, joint))
    roots = [name for name in links if name not in parents]
    if len(roots) != 1:
        raise UrdfError(f"URDF 必须具有唯一根 link；当前根节点：{roots}")
    visited = set()
    def visit(name):
        if name in visited:
            raise UrdfError("URDF 存在闭链或重复子节点")
        visited.add(name)
        for child, _ in children[name]:
            visit(child)
    visit(roots[0])
    if visited != set(links):
        raise UrdfError("URDF 存在不连通子树或闭链")
    for transmission in robot.findall("transmission"):
        types = [node.text or "" for node in transmission.findall("type")]
        if len(transmission.findall("joint")) != 1 or any("SimpleTransmission" not in item for item in types):
            raise UrdfError("第一版不支持差速或多关节 transmission；请提供独立驱动关节 URDF")
    report = {"version": 1, "source_format": "urdf", "mesh_output_format": "obj", "source_entry": entry.relative_to(root).as_posix(),
              "source_file": entry.relative_to(root).as_posix(),
              "source_sha256": hashlib.sha256(source).hexdigest(), "root_link": roots[0],
              "style_references": ["go2/xmls/go2.xml", "ri_4438/xmls/ri_4438.xml"],
              "preserved": ["link/joint names", "fixed body hierarchy", "joint origins and axes", "mass and full inertia", "mesh scale", "visual color", "original collision geometry"],
              "warnings": ["浮动基座初始位置为原点；请在工作台确认离地高度与站姿。", "未从其他机器人复制 PD、惯量、摩擦、延迟或碰撞代理；控制参数需要人工确认。"],
              "meshes": [], "joints": joint_metadata, "generated_sites": []}
    if robot.find("mujoco") is not None:
        report["warnings"].append("源 URDF 的 <mujoco> 扩展不直接执行；按显式关节/几何/惯量转换，不启用 balanceinertia 或自动折叠。")
    report["warnings"].append("limit.effort 用于初始化任务力矩上限；limit.velocity 仅记录在报告，不自动施加速度控制。")
    for tag in ("gazebo", "ros2_control", "transmission"):
        if robot.find(tag) is not None:
            report["warnings"].append(f"{tag} 的控制器/插件定义未移植；请在工作台确认控制参数。")
    model_name = robot.get("name") or entry.stem
    mjcf = ET.Element("mujoco", {"model": model_name})
    mjcf.append(ET.Comment(" Compiler: radians, explicit source inertias, preserved fixed bodies. "))
    ET.SubElement(mjcf, "compiler", {"angle": "radian", "autolimits": "true", "fusestatic": "false", "inertiafromgeom": "false"})
    mjcf.append(ET.Comment(" Visual and collision defaults, following the Go2 / RI-4438 layout. "))
    defaults = ET.SubElement(mjcf, "default")
    visual = ET.SubElement(defaults, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", {"contype": "0", "conaffinity": "0", "density": "0", "group": "1"})
    collision = ET.SubElement(defaults, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", {"group": "3", "density": "0"})
    ET.SubElement(defaults, "site", {"size": "0.01", "group": "5"})
    mjcf.append(ET.Comment(" Materials and portable mesh / texture assets. "))
    asset = ET.SubElement(mjcf, "asset")
    resources = _Resources(root, entry, destination, asset, report)
    global_materials = {node.get("name"): node for node in robot.findall("material") if node.get("name")}
    used_geom_names, material_cache, bodies = set(), {}, {}

    def unique_name(preferred, used):
        name, index = preferred, 2
        while name in used:
            name, index = f"{preferred}_{index}", index + 1
        used.add(name)
        return name

    def material_for(visual):
        material = visual.find("material")
        if material is None:
            return None
        if material.find("color") is None and material.find("texture") is None:
            material = global_materials.get(material.get("name"))
            if material is None:
                raise UrdfError(f"visual 引用了未定义材质：{visual.find('material').get('name')}")
        signature = ET.tostring(material)
        if signature not in material_cache:
            attributes = {"name": f"urdf_material_{len(material_cache) + 1}"}
            color, texture = material.find("color"), material.find("texture")
            if color is not None:
                rgba = _numbers(color.get("rgba"), 4, "material.rgba")
                if not all(0 <= v <= 1 for v in rgba):
                    raise UrdfError("材质 rgba 必须位于 [0,1]")
                attributes["rgba"] = _fmt(rgba)
            if texture is not None:
                attributes["texture"] = resources.texture(texture.get("filename"))
            ET.SubElement(asset, "material", attributes)
            material_cache[signature] = attributes["name"]
        return material_cache[signature]

    def geometry_for(element, context):
        geometry = element.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise UrdfError(f"{context} 必须具有唯一 geometry")
        primitive = geometry[0]
        if primitive.tag == "mesh":
            scale = _numbers(primitive.get("scale"), 3, f"{context}.mesh.scale", [1, 1, 1])
            if any(v <= 0 for v in scale):
                raise UrdfError(f"{context} 的 mesh scale 必须为正数；镜像网格请在建模工具中烘焙")
            return [{"type": "mesh", "mesh": name} for name in resources.mesh(primitive.get("filename"), scale, element.tag == "collision")]
        if primitive.tag == "box":
            size = _numbers(primitive.get("size"), 3, f"{context}.box.size")
            attributes = {"type": "box", "size": _fmt([v / 2 for v in size])}
        elif primitive.tag == "sphere":
            size = [_number(primitive.get("radius"), f"{context}.sphere.radius")]
            attributes = {"type": "sphere", "size": _fmt(size)}
        elif primitive.tag == "cylinder":
            size = [_number(primitive.get("radius"), f"{context}.cylinder.radius"), _number(primitive.get("length"), f"{context}.cylinder.length") / 2]
            attributes = {"type": "cylinder", "size": _fmt(size)}
        else:
            raise UrdfError(f"{context} 不支持 geometry 类型 {primitive.tag}")
        if any(v <= 0 for v in size):
            raise UrdfError(f"{context} 的几何尺寸必须大于零")
        return [attributes]

    free_name = unique_name("floating_base_joint", set(joints))
    def build(name, parent, joint=None):
        link = links[name]
        attributes = {"name": name}
        attributes.update(_origin(joint, f"joint {joint.get('name')}") if joint is not None else {"pos": "0 0 0"})
        body = ET.SubElement(parent, "body", attributes)
        bodies[name] = body
        _inertial(link, body)
        if joint is None:
            ET.SubElement(body, "freejoint", {"name": free_name})
        elif joint.get("type") == "revolute":
            context = f"joint {joint.get('name')}"
            limit, axis, dynamics = joint.find("limit"), joint.find("axis"), joint.find("dynamics")
            if limit is None:
                raise UrdfError(f"{context} 缺少有限 limit")
            lo, hi = _number(limit.get("lower"), context + ".lower"), _number(limit.get("upper"), context + ".upper")
            if lo >= hi:
                raise UrdfError(f"{context} 需要 lower < upper")
            xyz = _numbers(axis.get("xyz") if axis is not None else None, 3, context + ".axis", [1, 0, 0])
            length = math.sqrt(sum(v * v for v in xyz))
            if length < 1e-12:
                raise UrdfError(f"{context} 的轴不能为零")
            attributes = {"name": joint.get("name"), "type": "hinge", "pos": "0 0 0", "axis": _fmt([v / length for v in xyz]), "range": _fmt([lo, hi])}
            metadata = {"name": joint.get("name"), "range": [lo, hi]}
            for field in ("effort", "velocity"):
                if limit.get(field) is not None:
                    value = _number(limit.get(field), context + "." + field)
                    if value <= 0:
                        raise UrdfError(f"{context}.{field} 必须为正数")
                    metadata[field] = value
            if dynamics is not None:
                for source_name, target in (("damping", "damping"), ("friction", "frictionloss")):
                    value = _number(dynamics.get(source_name), context + "." + source_name, 0)
                    if value < 0:
                        raise UrdfError(f"{context}.{source_name} 不能为负")
                    attributes[target] = _fmt([value])
            ET.SubElement(body, "joint", attributes)
            joint_metadata.append(metadata)
        for kind in ("visual", "collision"):
            for i, element in enumerate(link.findall(kind)):
                context = f"{name}.{kind}[{i}]"
                preferred = element.get("name") or f"{name}_{kind}" + (f"_{i + 1}" if i else "")
                if "/" in preferred:
                    raise UrdfError(f"geom 名称不能包含 /：{preferred}")
                common = {"class": kind, **_origin(element, context)}
                material = material_for(element) if kind == "visual" else None
                if material:
                    common["material"] = material
                for index, geometry in enumerate(geometry_for(element, context)):
                    geom_name = unique_name(preferred + (f"_part_{index + 1}" if index else ""), used_geom_names)
                    ET.SubElement(body, "geom", {"name": geom_name, **common, **geometry})
        for child, child_joint in children[name]:
            build(child, body, child_joint)

    mjcf.append(ET.Comment(" Robot bodies: inertial, joint, visual, collision, then child bodies. "))
    worldbody = ET.SubElement(mjcf, "worldbody")
    worldbody.append(ET.Comment(" Original URDF link frames and fixed body hierarchy are preserved. "))
    build(roots[0], worldbody)
    # Sites are candidates, not physical sensor calibration. Their source is
    # recorded and they remain editable in the workbench before task generation.
    imu_candidates = [name for name in links if re.search(r"(?:^|[_-])imu(?:[_-]|$)", name, re.I)]
    imu_body = imu_candidates[0] if len(imu_candidates) == 1 else roots[0]
    ET.SubElement(bodies[imu_body], "site", {"name": "imu", "pos": "0 0 0"})
    report["generated_sites"].append({"name": "imu", "body": imu_body, "position": [0, 0, 0], "source": "named_imu_link" if len(imu_candidates) == 1 else "base_origin_candidate"})
    report["warnings"].append("IMU site 与足端 site 为可编辑候选；请确认实机传感器安装位姿和接触中心。")
    for leg in ("FL", "FR", "RL", "RR"):
        candidates = [name for name in links if "foot" in name.lower() and re.search(rf"(?:^|[_-]){leg}(?:[_-]|$)", name, re.I)]
        if len(candidates) == 1:
            foot = candidates[0]
            collisions = bodies[foot].findall("geom[@class='collision']")
            position = collisions[0].get("pos", "0 0 0") if len(collisions) == 1 else "0 0 0"
            ET.SubElement(bodies[foot], "site", {"name": leg, "pos": position})
            report["generated_sites"].append({"name": leg, "body": foot, "position": [float(v) for v in position.split()], "source": "named_foot_link_candidate"})
    mjcf.append(ET.Comment(" Candidate IMU definitions; review their placement in the workbench. "))
    sensors = ET.SubElement(mjcf, "sensor")
    ET.SubElement(sensors, "gyro", {"name": "imu_ang_vel", "site": "imu"})
    ET.SubElement(sensors, "velocimeter", {"name": "imu_lin_vel", "site": "imu"})
    ET.SubElement(sensors, "accelerometer", {"name": "imu_lin_acc", "site": "imu"})
    ET.SubElement(sensors, "subtreeangmom", {"name": "root_angmom", "body": roots[0]})
    filename = re.sub(r"[^A-Za-z0-9_-]+", "_", entry.stem).strip("_") or "robot"
    xml = filename + ".xml"
    ET.indent(mjcf, space="  ")
    (destination / xml).write_text(ET.tostring(mjcf, encoding="unicode") + "\n", encoding="utf-8")
    archived = destination / "source" / entry.relative_to(root)
    archived.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(entry, archived)
    for dependency in resources.files:
        target = destination / "source" / dependency.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dependency, target)
    report["source_archive"] = archived.relative_to(destination).as_posix()
    report["source_archive_purpose"] = "原始输入与资源来源快照；绝对路径/package URI保持原文，独立运行使用转换后的相对路径MJCF。"
    report["xml"] = xml
    report["counts"] = {"links": len(links), "fixed_joints": sum(j.get("type") == "fixed" for j in joints.values()), "revolute_joints": len(joint_metadata)}
    report["dependencies"] = [{"source": path.relative_to(root).as_posix(), "size": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path, size in sorted(resources.files.items())]
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(destination / xml))
        report["compiled"] = {"nbody": model.nbody, "njnt": model.njnt, "ngeom": model.ngeom, "nsite": model.nsite}
    except Exception as exc:
        raise UrdfError(f"转换后的 MuJoCo XML 编译失败：{exc}；请核对源模型质量、惯量和网格。") from exc
    (destination / "conversion_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"xml": xml, "report": report, "files": sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file())}
