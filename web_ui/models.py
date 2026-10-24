"""Import self-contained MJCF bundles and validate the migration contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
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
        while int(model.body_parentid[leg_body]) not in (0, base_id):
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
    pos = model.qpos0[qadr:qadr + 3].tolist()
    rot = model.qpos0[qadr + 3:qadr + 7].tolist()
    if model.nkey:
        pos = model.key_qpos[0, qadr:qadr + 3].tolist()
        rot = model.key_qpos[0, qadr + 3:qadr + 7].tolist()
    return {"name": entry.stem, "joints": joints, "bodies": body_names, "sites": sites,
            "site_details": site_details, "geoms": geoms, "base_body": body_names[base_id - 1],
            "sensors": [{"name": name(mujoco.mjtObj.mjOBJ_SENSOR, i), "type": int(model.sensor_type[i])} for i in range(model.nsensor)],
            "initial_pos": pos, "initial_rot": rot, "initial_source": "keyframe" if model.nkey else "template",
            "default_joint": {j["name"]: j["default"] for j in joints},
            "issues": [{"level": "warning", "message": "角色、关节镜像符号和足端是推测值，请逐项确认；PD 等模板值不代表实机标定参数。"}]}


def import_model(root, xml=None, workspace=None):
    workspace = _workspace(workspace)
    root = Path(root).expanduser().resolve()
    if root.is_file() and xml is None:
        root, xml = root.parent, root.name
    if not root.is_dir():
        raise ModelError("模型根目录不存在")
    if not xml:
        candidates = sorted(root.glob("*.xml"))
        if len(candidates) != 1:
            raise ModelError("请指定入口 XML 文件名")
        xml = candidates[0].name
    entry = _inside(root, root / xml)
    files = _dependencies(root, entry)
    model_id = uuid.uuid4().hex
    destination = workspace / "imports" / model_id
    model_root = destination / "model"
    model_root.mkdir(parents=True)
    try:
        manifest, digest = [], hashlib.sha256()
        for path in files:
            rel = path.relative_to(root)
            target = model_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            file_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            digest.update(str(rel).encode() + file_hash.encode())
            manifest.append({"path": rel.as_posix(), "size": target.stat().st_size, "sha256": file_hash})
        inspection = inspect_model(model_root / entry.relative_to(root))
        inspection["files"] = manifest
        model = {"id": model_id, "root": str(model_root), "xml": entry.relative_to(root).as_posix(),
                 "sha256": digest.hexdigest(), "name": inspection["name"]}
        (destination / "inspection.json").write_text(json.dumps(inspection, indent=2, ensure_ascii=False))
        (destination / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False))
        return {"model": model, "inspection": inspection, "spec": make_spec(model, inspection)}
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
            return import_model(extraction, entry.relative_to(extraction), workspace)
        # Prefer a standalone robot file to scene wrappers and include fragments.
        candidates = []
        for path in extraction.rglob("*.xml"):
            try:
                node = ET.parse(path).getroot()
                if node.tag == "mujoco" and (node.find(".//freejoint") is not None or node.find(".//joint[@type='free']") is not None):
                    candidates.append(path)
            except ET.ParseError:
                continue
        if len(candidates) != 1:
            raise ModelError("ZIP 必须含唯一的机器人入口 XML；多个模型请使用本地路径指定入口")
        return import_model(extraction, candidates[0].relative_to(extraction), workspace)
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
    slug += "_new"
    bindings = {"base_body": inspection["base_body"], "joints": [
        {"name": j["name"], "leg": j["leg"], "role": j["role"], "sign": -1 if j["role"] == "hip" else 1}
        for j in inspection["joints"]], "feet": [], "base_geoms": [], "leg_geoms": []}
    for leg in LEGS:
        calf = next((j for j in inspection["joints"] if j["leg"] == leg and j["role"] == "calf"), None)
        body = calf["body"] if calf else ""
        site = next((s for s in inspection["site_details"] if s["name"].upper() == leg), None)
        if site is None:
            site = next((s for s in inspection["site_details"] if s["body"] == body), None)
        foot_geoms = [g for g in inspection["geoms"] if g["body"] == body and g["collision"] and g["name"]]
        geom = next((g for g in foot_geoms if "foot" in g["name"].lower()), foot_geoms[-1] if foot_geoms else None)
        bindings["feet"].append({"leg": leg, "site": site["name"] if site else "", "body": body,
                                 "geom": geom["name"] if geom else "",
                                 "position": site["position"] if site else geom["position"] if geom else [0, 0, 0]})
    feet = {f["geom"] for f in bindings["feet"]}
    for geom in inspection["geoms"]:
        if not geom["collision"] or not geom["name"] or geom["name"] in feet:
            continue
        bindings["base_geoms" if geom["body"] == bindings["base_body"] else "leg_geoms"].append(geom["name"])
    return {"version": 1, "model": deepcopy(model), "robot": {"slug": slug, "prefix": slug.replace("_", "-").title()},
            "bindings": bindings, "parameters": {"init_state": {"pos": inspection["initial_pos"], "rot": inspection["initial_rot"],
                           "default_joint": inspection["default_joint"]},
                           "control": {"action_clip": {j["name"]: j["range"] for j in inspection["joints"]}}},
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
    joints = bindings.get("joints", [])
    if len(joints) != 12 or {j.get("name") for j in joints} != {j["name"] for j in inspection["joints"]}:
        error("动作映射必须恰好包含模型的 12 个不同关节")
    expected = {(leg, role) for leg in LEGS for role in ROLES}
    if len(joints) != 12 or {(j.get("leg"), j.get("role")) for j in joints} != expected:
        error("每条腿必须分别绑定 hip、thigh、calf 三个关节，不能重复")
    if any(j.get("sign") not in (-1, 1) for j in joints):
        error("镜像符号只能是 -1 或 +1")
    if bindings.get("base_body") != inspection["base_body"]:
        error("机体必须选择浮动基座所在的 body")
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
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
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
