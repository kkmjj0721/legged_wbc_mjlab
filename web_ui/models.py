"""Import MJCF or URDF bundles and validate the migration contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / ".artifacts" / "robot-migration"
LEGS = ("FL", "FR", "RL", "RR")
ROLES = ("hip", "thigh", "calf")
ALGORITHMS = ("ppo", "him", "him_symmetry", "him_lpacrl")
MAX_BYTES = 2 * 1024**3
MAX_FILES = 10000


class ModelError(ValueError):
    """An actionable model or specification error."""


def _workspace(path=None):
    result = Path(path or WORKSPACE).resolve()
    if not result.is_relative_to(ROOT):
        raise ModelError("工作目录必须位于项目根目录内")
    result.mkdir(parents=True, exist_ok=True)
    return result


def _inside(root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise ModelError(f"模型不接受符号链接：{path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ModelError(f"资源引用超出模型根目录：{path}")
    if not resolved.is_file():
        raise ModelError(f"缺失模型资源：{path}")
    return resolved


def _xml_root(path: Path):
    content = path.read_text(encoding="utf-8-sig")
    if "<!DOCTYPE" in content.upper() or "<!ENTITY" in content.upper():
        raise ModelError("机器人描述不允许 DTD 或外部实体")
    try:
        return ET.fromstring(content)
    except ET.ParseError as exc:
        raise ModelError(f"XML/URDF 解析失败 {path.name}: {exc}") from exc


def _dependencies(root: Path, entry: Path) -> list[Path]:
    """Resolve includes/assets before handing any XML to MuJoCo."""
    found, visited = set(), set()
    main_dir = entry.parent
    directories = {"mesh": "", "texture": "", "hfield": "", "skin": ""}

    def visit(path):
        path = _inside(root, path)
        if path in visited:
            return
        visited.add(path)
        content = path.read_text(encoding="utf-8")
        if "<!DOCTYPE" in content.upper() or "<!ENTITY" in content.upper():
            raise ModelError("MJCF 不允许 DTD 或外部实体")
        try:
            tree = ET.fromstring(content)
        except ET.ParseError as exc:
            raise ModelError(f"XML 解析失败 {path.name}: {exc}") from exc
        if tree.find(".//extension") is not None:
            raise ModelError("第一版不支持需要外部插件的 MJCF 模型")
        compiler = tree.find("compiler")
        if compiler is not None:
            assetdir = compiler.get("assetdir", "")
            if any(Path(compiler.get(key, "")).is_absolute() for key in ("assetdir", "meshdir", "texturedir")):
                raise ModelError("compiler 的资源目录必须使用相对路径，以保证生成模型可迁移")
            directories["mesh"] = compiler.get("meshdir", assetdir)
            directories["texture"] = compiler.get("texturedir", assetdir)
            directories["hfield"] = directories["skin"] = assetdir
            if compiler.get("strippath", "false").lower() == "true":
                raise ModelError("请先关闭 compiler.strippath，保留可移植的相对资源路径")
        found.add(path)
        for element in tree.iter():
            reference = element.get("file")
            if not reference:
                continue
            reference_path = Path(reference)
            if reference_path.is_absolute():
                raise ModelError(f"请将资源路径改为模型目录内的相对路径：{reference}")
            if element.tag == "include":
                candidates = [path.parent / reference_path, main_dir / reference_path]
            else:
                directory = directories.get(element.tag, "")
                candidates = [main_dir / directory / reference_path, path.parent / directory / reference_path]
            candidate = next((p for p in candidates if p.is_file()), candidates[0])
            candidate = _inside(root, candidate)
            if element.tag == "include":
                visit(candidate)
            else:
                found.add(candidate)
        if len(found) > MAX_FILES:
            raise ModelError("模型资源文件过多（上限 10000）")

    visit(entry)
    if sum(path.stat().st_size for path in found) > MAX_BYTES:
        raise ModelError("模型资源超过 2 GiB")
    return sorted(found)


def inspect_model(entry: Path) -> dict:
    import mujoco
    import numpy as np

    try:
        model = mujoco.MjModel.from_xml_path(str(entry))
    except Exception as exc:
        raise ModelError(f"MuJoCo 编译失败：{exc}") from exc
    name = lambda kind, index: mujoco.mj_id2name(model, kind, index) or ""
    free_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    hinge_ids = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(free_ids) != 1 or len(hinge_ids) != 12 or model.njnt != 13:
        raise ModelError(f"需要 1 个浮动基座和 12 个转动关节；当前 njnt={model.njnt}，转动关节={len(hinge_ids)}")
    base_id = int(model.jnt_bodyid[free_ids[0]])
    if int(model.body_parentid[base_id]) != 0 or any(int(p) == 0 and i != base_id for i, p in enumerate(model.body_parentid) if i):
        raise ModelError("请选择独立机器人 XML；场景中只能有一个浮动机器人根 body")
    if any(int(body) == 0 for body in model.geom_bodyid):
        raise ModelError("入口包含地面或其他世界几何，请选择机器人 XML，而非场景 XML")
    body_names = [name(mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
    if not all(body_names):
        raise ModelError("所有机器人 body 必须具有唯一名称")
    # A URDF can have a massless root wrapper welded to the actual trunk.
    # Keep the floating frame, but bind mass/COM randomization to a massive body.
    base_ids = {base_id}
    for i in range(base_id + 1, model.nbody):
        if int(model.body_parentid[i]) in base_ids and model.body_jntnum[i] == 0:
            base_ids.add(i)
    physical_base_id = base_id if model.body_mass[base_id] > 0 else max(base_ids, key=lambda i: float(model.body_mass[i]))
    if model.body_mass[physical_base_id] <= 0:
        raise ModelError("浮动机体及其固定子体缺少正质量，请检查 URDF/MJCF 惯量定义")
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                        (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
                        (mujoco.mjtObj.mjOBJ_SITE, model.nsite)):
        if any("/" in name(kind, i) for i in range(count)):
            raise ModelError("MjLab 将 / 用作实体命名空间，请先移除模型 body/joint/geom/site 名称中的 /")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = []
    counts = {leg: 0 for leg in LEGS}
    base_rotation = data.xmat[base_id].reshape(3, 3)
    for i in hinge_ids:
        joint_name = name(mujoco.mjtObj.mjOBJ_JOINT, i)
        if not joint_name:
            raise ModelError("12 个驱动关节必须具有唯一名称")
        body_id = int(model.jnt_bodyid[i])
        leg_body = body_id
        while int(model.body_parentid[leg_body]) not in ({0} | base_ids):
            leg_body = int(model.body_parentid[leg_body])
        offset = base_rotation.T @ (data.xpos[leg_body] - data.xpos[base_id])
        geometric_leg = ("F" if offset[0] >= 0 else "R") + ("L" if offset[1] >= 0 else "R")
        matched_leg = re.search(r"(?:^|[_/\-])(FL|FR|RL|RR)(?:[_/\-]|$)", joint_name, re.I)
        leg = matched_leg.group(1).upper() if matched_leg else geometric_leg
        matched_role = next((r for r in ROLES if r in joint_name.lower()), None)
        role = matched_role or ROLES[min(counts[leg], 2)]
        counts[leg] += 1
        bounds = model.jnt_range[i].tolist() if model.jnt_limited[i] else [-math.pi, math.pi]
        initial = float(model.qpos0[model.jnt_qposadr[i]])
        if model.nkey:
            initial = float(model.key_qpos[0, model.jnt_qposadr[i]])
        elif role == "thigh":
            initial = 0.9
        elif role == "calf":
            initial = -1.8
        initial = float(np.clip(initial, *bounds))
        joints.append({"name": joint_name, "body": body_names[body_id - 1], "range": bounds,
                       "limited": bool(model.jnt_limited[i]), "axis": model.jnt_axis[i].tolist(),
                       "type": "hinge", "leg": leg, "role": role, "default": initial,
                       "qpos_address": int(model.jnt_qposadr[i]),
                       "armature": float(model.dof_armature[model.jnt_dofadr[i]]),
                       "friction": float(model.dof_frictionloss[model.jnt_dofadr[i]]),
                       "passive_damping": float(model.dof_damping[model.jnt_dofadr[i]])})
    geoms = [{"name": name(mujoco.mjtObj.mjOBJ_GEOM, i), "body": body_names[int(model.geom_bodyid[i]) - 1],
              "position": model.geom_pos[i].tolist(),
              "group": int(model.geom_group[i]), "collision": bool(model.geom_contype[i] or model.geom_conaffinity[i])}
             for i in range(model.ngeom)]
    sites = [name(mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    site_details = [{"name": sites[i], "body": body_names[int(model.site_bodyid[i]) - 1], "position": model.site_pos[i].tolist()}
                    for i in range(model.nsite)]
    qadr = int(model.jnt_qposadr[free_ids[0]])
    meshes = [{"name": name(mujoco.mjtObj.mjOBJ_MESH, i),
               "faces": int(model.mesh_facenum[i]), "vertices": int(model.mesh_vertnum[i])}
              for i in range(model.nmesh)]
    pos = model.qpos0[qadr:qadr + 3].tolist()
    rot = model.qpos0[qadr + 3:qadr + 7].tolist()
    if model.nkey:
        pos = model.key_qpos[0, qadr:qadr + 3].tolist()
        rot = model.key_qpos[0, qadr + 3:qadr + 7].tolist()
    return {"name": entry.stem, "joints": joints, "bodies": body_names,
            "meshes": meshes, "mesh_max_faces": max((m["faces"] for m in meshes), default=0),
            "body_parents": {body_names[i - 1]: name(mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[i]))
                             for i in range(1, model.nbody)}, "sites": sites,
            "site_details": site_details, "geoms": geoms, "base_body": body_names[physical_base_id - 1],
            "floating_body": body_names[base_id - 1],
            "base_bodies": [body_names[i - 1] for i in sorted(base_ids)],
            "body_masses": {body_names[i - 1]: float(model.body_mass[i]) for i in range(1, model.nbody)},
            "sensors": [{"name": name(mujoco.mjtObj.mjOBJ_SENSOR, i), "type": int(model.sensor_type[i])} for i in range(model.nsensor)],
            "initial_pos": pos, "initial_rot": rot, "initial_source": "keyframe" if model.nkey else "template",
            "default_joint": {j["name"]: j["default"] for j in joints},
            "issues": [{"level": "warning", "message": "角色、关节镜像符号和足端是推测值，请逐项确认；PD 等模板值不代表实机标定参数。"}]}


def _infer_model_root(entry: Path) -> Path:
    """Infer a portable bundle from an explicitly supplied model file path."""
    directory = entry.parent.resolve()
    package = next((parent for parent in (directory, *directory.parents)
                    if (parent / "package.xml").is_file()), None)
    root = package or (directory.parent if directory.name.lower() in ("urdf", "xml", "xmls") else directory)
    document = _xml_root(entry)
    parents = [root]
    if document.tag == "robot":
        # Only the files referenced by this URDF are copied, even if relative
        # resources need a common ancestor above the conventional package root.
        for node in document.iter():
            reference = node.get("filename") if node.tag in ("mesh", "texture") else None
            if reference and "://" not in reference:
                path = Path(reference).expanduser()
                parents.append((path if path.is_absolute() else directory / path).resolve().parent)
    elif document.tag == "mujoco":
        compiler = document.find("compiler")
        assetdir = compiler.get("assetdir", "") if compiler is not None else ""
        for node in document.iter():
            reference = node.get("file")
            if reference and not Path(reference).is_absolute():
                key = "meshdir" if node.tag == "mesh" else "texturedir" if node.tag == "texture" else "assetdir"
                prefix = compiler.get(key, assetdir) if compiler is not None and node.tag != "include" else ""
                parents.append((directory / prefix / reference).resolve().parent)
    # Common ancestors cover resources; no directory tree is imported wholesale.
    return Path(os.path.commonpath(parents))


def _import_location(root, xml, *, strict_root=False):
    raw_root, raw_xml = str(root or "").strip(), str(xml or "").strip()
    if not raw_root and not raw_xml:
        raise ModelError("请填写模型目录或完整的 XML / URDF 文件路径")
    selected = Path(raw_root).expanduser() if raw_root else None
    requested = Path(raw_xml).expanduser() if raw_xml else None
    if requested is not None and requested.is_absolute():
        if selected is not None and selected.is_file() and selected.resolve() != requested.resolve():
            raise ModelError(f"两个输入框指向不同的模型文件，未执行导入。\n模型路径：{selected}\n"
                             f"入口路径：{requested}\n请清空其中一个，或将两处填写为同一个文件。")
        entry = requested
        directory = (selected.resolve() if selected is not None and selected.is_dir()
                     and entry.resolve().is_relative_to(selected.resolve()) else None)
    elif selected is not None and selected.is_file():
        entry, directory = selected, None
    elif selected is not None and selected.is_dir():
        directory = selected.resolve()
        if requested is not None:
            entry = directory / requested
        else:
            candidates = []
            for folder in (directory, directory / "urdf", directory / "xml", directory / "xmls"):
                if not folder.is_dir():
                    continue
                for path in sorted(folder.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in (".xml", ".urdf"):
                        continue
                    try:
                        node = _xml_root(path)
                    except (ModelError, UnicodeError):
                        continue
                    if node.tag == "robot" or (node.tag == "mujoco" and
                        (node.find(".//freejoint") is not None or node.find(".//joint[@type='free']") is not None)):
                        candidates.append(path)
            if len(candidates) != 1:
                raise ModelError("模型目录中未找到唯一的机器人入口；请在任一输入框填写完整 XML / URDF 路径，或指定相对入口")
            entry = candidates[0]
    elif requested is not None and not raw_root:
        entry, directory = requested, None
    else:
        raise ModelError(f"模型路径不存在：{selected}；请填写模型目录或完整 XML / URDF 文件路径")
    if entry.is_symlink():
        raise ModelError(f"模型不接受符号链接：{entry}")
    if not entry.is_file():
        raise ModelError(f"入口模型文件不存在：{entry}")
    if not strict_root:
        inferred = _infer_model_root(entry)
        directory = (Path(os.path.commonpath([directory, inferred]))
                     if directory is not None and entry.resolve().is_relative_to(directory) else inferred)
    if directory is None:
        raise ModelError("ZIP 模型缺少有效的包内根目录")
    try:
        return directory.resolve(), _inside(directory.resolve(), entry)
    except ModelError as exc:
        raise ModelError(f"{exc}\n当前模型目录：{directory}\n可直接填写完整入口路径，由工具确定资源目录。") from exc


def check_mesh_budget(limit, meshes):
    """A generation budget, separate from MuJoCo's fixed STL decoder limit."""
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ModelError("mesh_face_limit 需要正整数（三角面 / 网格）")
    exceeded = [m for m in meshes if m["faces"] > limit]
    if exceeded:
        maximum = max(m["faces"] for m in exceeded)
        raise ModelError(f"网格面数超过配置预算：{exceeded[0]['name']} 等 {len(exceeded)} 个网格，"
                         f"最大 {maximum} 面，预算 {limit} 面；请在配置页将模型资源面数预算调至至少 {maximum}。")


def import_model(root, xml=None, workspace=None, *, _strict_root=False):
    workspace = _workspace(workspace)
    root, entry = _import_location(root, xml, strict_root=_strict_root)
    document = _xml_root(entry)
    if document.tag not in ("robot", "mujoco"):
        raise ModelError("入口需要是 URDF <robot> 或 MJCF <mujoco>；xacro 请先展开为 URDF")
    source_format = "urdf" if document.tag == "robot" else "mjcf"
    # Converted bundles may be wrapped in a top-level ZIP directory. Restore
    # their canonical bundle root so the report and source archive travel too.
    if source_format == "mjcf":
        for parent in entry.parents:
            if not parent.is_relative_to(root):
                break
            report_path = parent / "conversion_report.json"
            if report_path.is_file():
                report = json.loads(_inside(root, report_path).read_text())
                if report.get("source_format") == "urdf" and report.get("xml") == entry.relative_to(parent).as_posix():
                    root = parent
                    break
    source_file = entry.relative_to(root).as_posix()
    model_id = uuid.uuid4().hex
    destination = workspace / "imports" / model_id
    model_root = destination / "model"
    model_root.mkdir(parents=True)
    try:
        conversion = None
        if source_format == "urdf":
            from .urdf import convert_urdf
            converted = convert_urdf(root, entry, model_root)
            conversion = converted["report"]
            xml_path = converted["xml"]
        else:
            files = _dependencies(root, entry)
            xml_path = entry.relative_to(root).as_posix()
            report_path = root / "conversion_report.json"
            if report_path.is_file():
                report = json.loads(_inside(root, report_path).read_text())
                if report.get("source_format") == "urdf" and report.get("xml") == xml_path:
                    conversion = report
                    files.append(report_path)
                    archive = root / "source"
                    if archive.is_dir():
                        files.extend(_inside(root, path) for path in archive.rglob("*") if path.is_file())
            for path in sorted(set(files)):
                target = model_root / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        manifest, digest = [], hashlib.sha256()
        files = sorted(path for path in model_root.rglob("*") if path.is_file())
        if len(files) > MAX_FILES or sum(path.stat().st_size for path in files) > MAX_BYTES:
            raise ModelError("模型资源超出 10000 个文件或 2 GiB 大小限制")
        for target in files:
            rel = target.relative_to(model_root)
            file_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            digest.update(str(rel).encode() + file_hash.encode())
            manifest.append({"path": rel.as_posix(), "size": target.stat().st_size, "sha256": file_hash})
        inspection = inspect_model(model_root / xml_path)
        inspection["files"] = manifest
        model = {"id": model_id, "root": str(model_root), "xml": xml_path,
                 "sha256": digest.hexdigest(), "name": inspection["name"],
                 "source_format": "urdf" if conversion else source_format,
                 "source_file": conversion.get("source_file", source_file) if conversion else source_file,
                 "import_format": source_format,
                 "mesh_max_faces": inspection["mesh_max_faces"],
                 "mesh_face_limit": max(200000, inspection["mesh_max_faces"])}
        if conversion:
            model["conversion"] = conversion
            inspection["conversion"] = conversion
            inspection["issues"].extend({"level": "warning", "message": item} for item in conversion.get("warnings", []))
            if any("effort" in j for j in conversion.get("joints", [])):
                inspection["parameter_sources"] = {"control.effort_limit": "model"}
        (destination / "inspection.json").write_text(json.dumps(inspection, indent=2, ensure_ascii=False))
        (destination / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False))
        return {"model": model, "inspection": inspection, "spec": make_spec(model, inspection),
                "import_input": {"root": str(root), "xml": entry.relative_to(root).as_posix()}}
    except BaseException:
        shutil.rmtree(destination)
        raise


def import_zip(file, workspace=None, xml=None):
    workspace = _workspace(workspace)
    upload = workspace / "uploads" / uuid.uuid4().hex
    upload.mkdir(parents=True)
    archive_path = upload / "model.zip"
    try:
        if hasattr(file, "save"):
            file.save(archive_path)
        else:
            shutil.copyfile(file, archive_path)
        extraction = upload / "contents"
        extraction.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES or sum(item.file_size for item in infos) > MAX_BYTES:
                raise ModelError("ZIP 超出 10000 个文件或 2 GiB 解压大小限制")
            for info in infos:
                rel = PurePosixPath(info.filename)
                if rel.is_absolute() or ".." in rel.parts or "\\" in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                    raise ModelError(f"ZIP 中有无效路径或符号链接：{info.filename}")
                if info.is_dir():
                    continue
                destination = extraction.joinpath(*rel.parts)
                if not destination.resolve().is_relative_to(extraction):
                    raise ModelError("ZIP 资源路径越界")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if xml:
            entry = _inside(extraction, extraction / xml)
            result = import_model(extraction, entry.relative_to(extraction), workspace, _strict_root=True)
            result["import_input"] = {"root": result["model"]["root"], "xml": result["model"]["xml"]}
            return result
        # Prefer standalone robot files to scene wrappers and include fragments.
        candidates = []
        for path in extraction.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".xml", ".urdf"):
                continue
            if any((parent / "conversion_report.json").is_file() and path.is_relative_to(parent / "source")
                   for parent in path.parents if parent.is_relative_to(extraction)):
                continue
            try:
                node = _xml_root(path)
                if node.tag == "robot" or (node.tag == "mujoco" and (node.find(".//freejoint") is not None or node.find(".//joint[@type='free']") is not None)):
                    candidates.append(path)
            except (ModelError, UnicodeError):
                continue
        if len(candidates) != 1:
            raise ModelError("ZIP 必须含唯一的机器人入口 XML/URDF；多个模型请填写包内入口路径")
        result = import_model(extraction, candidates[0].relative_to(extraction), workspace, _strict_root=True)
        result["import_input"] = {"root": result["model"]["root"], "xml": result["model"]["xml"]}
        return result
    except zipfile.BadZipFile as exc:
        raise ModelError("上传文件不是有效的 ZIP") from exc
    finally:
        shutil.rmtree(upload)


def resolve_model(spec):
    model = spec.get("model", spec)
    root = Path(model.get("root", "")).resolve()
    if not root.is_relative_to(ROOT / ".artifacts"):
        raise ModelError("模型必须先通过本工具导入")
    stored = root.parent / "model.json"
    if not stored.is_file():
        raise ModelError("导入模型已不存在，请重新导入")
    original = json.loads(stored.read_text())
    if original.get("id") != model.get("id") or original.get("xml") != model.get("xml") or original.get("root") != str(root):
        raise ModelError("导入模型身份与记录不一致")
    return _inside(root, root / model["xml"])


def make_spec(model, inspection=None):
    if inspection is None:
        inspection = json.loads((Path(model["root"]).parent / "inspection.json").read_text())
    slug = re.sub(r"[^a-z0-9]+", "_", model["name"].lower()).strip("_")
    slug = slug if slug and slug[0].isalpha() else "robot_" + slug
    bindings = {"base_body": inspection["base_body"], "joints": [
        {"name": j["name"], "leg": j["leg"], "role": j["role"], "sign": -1 if j["role"] == "hip" else 1}
        for j in inspection["joints"]], "feet": [], "base_geoms": [], "leg_geoms": []}
    for leg in LEGS:
        calf = next((j for j in inspection["joints"] if j["leg"] == leg and j["role"] == "calf"), None)
        body = calf["body"] if calf else ""
        descendants = {body}
        for _ in inspection["bodies"]:
            descendants.update(child for child, parent in inspection.get("body_parents", {}).items()
                               if parent in descendants)
        site = next((s for s in inspection["site_details"] if s["name"].upper() == leg), None)
        if site is None:
            site = next((s for s in inspection["site_details"] if s["body"] in descendants), None)
        foot_geoms = [g for g in inspection["geoms"] if g["body"] in descendants and g["collision"] and g["name"]]
        geom = next((g for g in foot_geoms if "foot" in g["name"].lower()), foot_geoms[-1] if foot_geoms else None)
        body = site["body"] if site else geom["body"] if geom else body
        bindings["feet"].append({"leg": leg, "site": site["name"] if site else "", "body": body,
                                 "geom": geom["name"] if geom else "",
                                 "position": site["position"] if site else geom["position"] if geom else [0, 0, 0]})
    feet = {f["geom"] for f in bindings["feet"]}
    for geom in inspection["geoms"]:
        if not geom["collision"] or not geom["name"] or geom["name"] in feet:
            continue
        base_bodies = inspection.get("base_bodies", [bindings["base_body"]])
        bindings["base_geoms" if geom["body"] in base_bodies else "leg_geoms"].append(geom["name"])
    # A new robot has no confirmed gains.  Explicit nulls also prevent the
    # template's RI-4438 gains from appearing through the ordinary merge path.
    control = {"stiffness": None, "damping": None,
               "action_clip": {j["name"]: j["range"] for j in inspection["joints"]}}
    effort = {j["name"]: j["effort"] for j in model.get("conversion", {}).get("joints", []) if "effort" in j}
    if effort:
        control["effort_limit"] = effort
    return {"version": 1, "model": deepcopy(model), "robot": {"slug": slug, "prefix": slug.replace("_", "-").title()},
            "bindings": bindings, "parameters": {"init_state": {"pos": inspection["initial_pos"], "rot": inspection["initial_rot"],
                           "default_joint": inspection["default_joint"]},
                           "control": control},
            "motor_inertia": {"version": 1, "scopes": {"common": {
                joint["name"]: {"rotor_inertia": None, "gear_ratio": 1} for joint in bindings["joints"]}}},
            "tasks": {"him": {"terrains": ["Flat", "Rough"], "parameters": {}}}, "confirmed": False}


def _merge(left, right):
    result = deepcopy(left)
    for key, value in right.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def _validate_spec(spec):
    issues = []
    def error(message):
        issues.append({"level": "error", "message": message})
    if spec.get("version") != 1:
        error("不支持的草稿版本，需要 version=1")
    slug, prefix = spec.get("robot", {}).get("slug", ""), spec.get("robot", {}).get("prefix", "")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", slug):
        error("机器人模块名需要以小写字母开头，仅含小写字母、数字、下划线，最长 64 字符")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", prefix):
        error("任务前缀需要以英文字母开头，仅含字母、数字、下划线或连字符")
    if not spec.get("confirmed"):
        error("请确认关节、足端和控制参数后勾选人工确认")
    try:
        entry = resolve_model(spec)
        inspection = json.loads((Path(spec["model"]["root"]).parent / "inspection.json").read_text())
    except (ValueError, KeyError, OSError) as exc:
        error(str(exc))
        return issues
    bindings = spec.get("bindings", {})
    try:
        check_mesh_budget(spec.get("model", {}).get("mesh_face_limit"), inspection.get("meshes", []))
    except ModelError as exc:
        error(str(exc))
    joints = bindings.get("joints", [])
    try:
        from .motor_inertia import validate as validate_motor_inertia
        validate_motor_inertia(spec, bindings)
    except ValueError as exc:
        error(str(exc))
    if len(joints) != 12 or {j.get("name") for j in joints} != {j["name"] for j in inspection["joints"]}:
        error("动作映射必须恰好包含模型的 12 个不同关节")
    expected = {(leg, role) for leg in LEGS for role in ROLES}
    if len(joints) != 12 or {(j.get("leg"), j.get("role")) for j in joints} != expected:
        error("每条腿必须分别绑定 hip、thigh、calf 三个关节，不能重复")
    if any(j.get("sign") not in (-1, 1) for j in joints):
        error("镜像符号只能是 -1 或 +1")
    base_body = bindings.get("base_body")
    if (base_body not in inspection.get("base_bodies", [inspection["base_body"]])
            or inspection.get("body_masses", {}).get(base_body, 1) <= 0):
        error("机体必须选择浮动基座或其固定连接的正质量 body")
    feet = bindings.get("feet", [])
    if len(feet) != 4 or {f.get("leg") for f in feet} != set(LEGS):
        error("必须为 FL、FR、RL、RR 各配置一个足端")
    geom_names = {g["name"] for g in inspection["geoms"] if g["name"]}
    for foot in feet:
        if foot.get("geom") not in geom_names:
            error(f"{foot.get('leg')} 的足端碰撞几何未选择或不存在")
        if foot.get("body") not in inspection["bodies"]:
            error(f"{foot.get('leg')} 的足端 body 不存在")
        if "/" in foot.get("site", ""):
            error("新增足端 site 的名称不能包含 /")
        position = foot.get("position", [])
        if len(position) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position):
            error("足端位置需要三个有限数值")
    if len({f.get("geom") for f in feet}) != 4:
        error("四个足端必须使用不同的碰撞几何")
    site_names = [foot["site"] for foot in feet if foot.get("site")]
    if len(set(site_names)) != len(site_names):
        error("四个足端必须绑定不同的 site")
    for group in ("base_geoms", "leg_geoms"):
        if any(g not in geom_names for g in bindings.get(group, [])):
            error(f"{group} 包含不存在的几何名称")
    if not bindings.get("base_geoms"):
        error("至少选择一个机体碰撞几何")
    tasks = spec.get("tasks", {})
    if not tasks or set(tasks) - set(ALGORITHMS):
        error("请选择至少一个受支持的任务模板")
    for key, task in tasks.items():
        if not task.get("terrains") or set(task["terrains"]) - {"Flat", "Rough"}:
            error(f"{key} 需要选择 Flat 或 Rough")
        params = _merge(spec.get("parameters", {}), task.get("parameters", {}))
        try:
            from .terrain_parameters import validate as validate_terrain
            validate_terrain(params.get("terrain", {}), f"{key}: terrain")
        except ValueError as exc:
            error(str(exc))
        initial = params.get("init_state", {})
        default_joint = initial.get("default_joint", {})
        for joint in inspection["joints"]:
            value = default_joint.get(joint["name"])
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or not joint["range"][0] <= value <= joint["range"][1]:
                error(f"{key}: {joint['name']} 初始角度缺失或超过模型限位")
        for field, width in (("pos", 3), ("rot", 4)):
            value = initial.get(field)
            if value is not None and (len(value) != width or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
                error(f"{key}: 初始 {field} 格式不正确")
        rot = initial.get("rot", [1, 0, 0, 0])
        if len(rot) == 4 and abs(sum(v*v for v in rot) - 1) > 1e-4:
            error(f"{key}: 初始四元数必须为单位四元数（wxyz）")
        control = params.get("control", {})
        reduction = control.get("hip_reduction", 1)
        if (isinstance(reduction, bool) or not isinstance(reduction, (int, float))
                or not math.isfinite(reduction) or not 0 < reduction <= 1):
            error(f"{key}: hip_reduction 必须为 (0, 1] 内的有限数值")
        scale = control.get("action_scale")
        scales = scale.values() if isinstance(scale, dict) else ([] if scale is None else [scale])
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in scales):
            error(f"{key}: action_scale 必须为有限正数或逐关节有限正数映射")
        for group in ("control", "observations"):
            delay = params.get(group, {})
            lo, hi = delay.get("delay_min_lag", 0), delay.get("delay_max_lag", 0)
            if not all(isinstance(v, int) and v >= 0 for v in (lo, hi)) or lo > hi:
                error(f"{key}: {group} 延迟步数需要非负整数且 min ≤ max")
            if not 0 <= delay.get("delay_hold_prob", 0) <= 1:
                error(f"{key}: {group} 延迟保持概率应在 [0,1]")
        for field in ("stiffness", "damping", "effort_limit", "armature", "friction", "passive_damping"):
            values = control.get(field)
            if values is None:
                continue
            values = values.values() if isinstance(values, dict) else [values]
            if any((v is None and field not in ("armature", "friction", "passive_damping")) or
                   (v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0))
                   for v in values):
                error(f"{key}: {field} 需要非负有限数值")
        env = params.get("environment", {})
        for field in ("num_envs", "env_spacing", "episode_length_s", "sim_dt"):
            if field in env and (not isinstance(env[field], (int, float)) or not math.isfinite(env[field]) or env[field] <= 0):
                error(f"{key}: {field} 必须为正数")
    try:
        json.dumps(spec, allow_nan=False)
    except (TypeError, ValueError):
        error("配置必须是 JSON 数据，不能包含 NaN 或 Infinity")
    return issues


def validate_spec(spec):
    """Return field errors for malformed input as well as ordinary invalid values."""
    try:
        return _validate_spec(spec)
    except (TypeError, ValueError, AttributeError, KeyError, OSError) as exc:
        return [{"level": "error", "message": f"配置结构或数值不正确：{exc}"}]
