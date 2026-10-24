"""Local motor evidence and explicitly confirmed torque/speed envelopes."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 20 * 1024**2
FORMATS = {"PNG": ("png", "image/png"), "JPEG": ("jpg", "image/jpeg"), "WEBP": ("webp", "image/webp")}


def _number(value, label, low=0, high=1e9, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high or (positive and value <= low):
        raise ValueError(f"{label} 必须是范围内的有限数值")
    return float(value)


def _safe_path(path):
    path = Path(path).absolute()
    if not path.is_relative_to(ROOT / ".artifacts"):
        raise ValueError("电机证据必须位于本项目 .artifacts 内")
    for parent in (path, *path.parents):
        if parent == ROOT:
            break
        if parent.is_symlink():
            raise ValueError("电机证据路径不接受符号链接")
    if not path.resolve().is_relative_to(ROOT / ".artifacts"):
        raise ValueError("电机证据路径越界")
    return path


def store_image(upload, workspace, *, voltage_v, label=""):
    """Retain the original raster image; never infer numeric data from pixels."""
    from PIL import Image

    voltage = _number(voltage_v, "曲线电压 V", 0, 10000, positive=True)
    data = upload.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("T–N 曲线图片需要非空且不超过 20 MiB")
    try:
        with Image.open(io.BytesIO(data)) as picture:
            kind, width, height = picture.format, *picture.size
            if kind not in FORMATS or width * height > 32_000_000 or max(width, height) > 16000:
                raise ValueError("请使用尺寸合理的 PNG、JPEG 或 WebP 曲线图片")
            picture.verify()
    except Exception as exc:
        raise ValueError(f"无法读取 T–N 曲线图片：{exc}") from exc
    asset_id = uuid.uuid4().hex
    extension, mime = FORMATS[kind]
    directory = _safe_path(Path(workspace) / "motor-curves" / asset_id)
    directory.mkdir(parents=True, exist_ok=False)
    asset = {"id": asset_id, "root": str(directory), "filename": f"image.{extension}",
             "url": f"/api/motor-curves/{asset_id}/image", "mime": mime, "width": width, "height": height,
             "sha256": hashlib.sha256(data).hexdigest(), "voltage_v": voltage,
             "label": str(label)[:128], "original_name": Path(upload.filename or "curve").name}
    try:
        (directory / asset["filename"]).write_bytes(data)
        (directory / "asset.json").write_text(json.dumps(asset, ensure_ascii=False, indent=2), encoding="utf-8")
    except BaseException:
        shutil.rmtree(directory)
        raise
    return asset


def image_path(asset):
    if not isinstance(asset, dict) or not re.fullmatch(r"[a-f0-9]{32}", str(asset.get("id", ""))):
        raise ValueError("无效的电机图片 ID")
    directory = _safe_path(asset["root"])
    if directory.name != asset["id"] or directory.parent.name != "motor-curves":
        raise ValueError("电机图片目录与 ID 不一致")
    saved = json.loads((directory / "asset.json").read_text(encoding="utf-8"))
    if any(asset.get(key) != saved.get(key) for key in ("id", "root", "filename", "sha256")):
        raise ValueError("电机图片记录与上传快照不一致")
    path = _safe_path(directory / saved["filename"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != saved["sha256"]:
        raise ValueError("电机图片已被修改，请重新上传")
    return path


def compile_envelopes(spec, joint_names):
    """Compile only confirmed points, converted once to the joint output shaft.

    Curves describe an explicitly symmetric |speed| / |torque| approximation;
    they are not a four-quadrant drive, thermal model, or OCR interpretation.
    """
    curves = spec.get("motor_curves", [])
    if not isinstance(curves, list) or len(curves) > 64:
        raise ValueError("motor_curves 必须是最多 64 项的列表")
    known, result = set(joint_names), {}
    for curve in curves:
        if not isinstance(curve, dict):
            raise ValueError("电机曲线必须为对象")
        if not curve.get("enabled", curve.get("confirmed", False)):
            continue
        if curve.get("confirmed") is not True:
            raise ValueError("用于测试的 T–N 曲线数值必须先核对确认")
        _number(curve.get("voltage_v"), "曲线电压 V", 0, 10000, positive=True)
        shaft = curve.get("shaft", "joint")
        if shaft not in ("joint", "module"):
            raise ValueError("曲线参数需要标明关节端或电机模组输出端")
        ratio = _number(curve.get("extra_reduction", 1), "额外传动比", 0, 1e6, positive=True)
        efficiency = _number(curve.get("efficiency", 1), "额外传动效率", 0, 1, positive=True)
        if shaft == "joint" and (ratio != 1 or efficiency != 1):
            raise ValueError("已是关节端的曲线不再折算传动比或效率；请设为 1")
        selected = curve.get("joint_names", [])
        if (not isinstance(selected, list) or not selected or any(not isinstance(name, str) for name in selected)
                or len(selected) != len(set(selected)) or set(selected) - known):
            raise ValueError("启用的 T–N 曲线必须绑定不重复的模型关节")
        points = curve.get("points", [])
        if not isinstance(points, list) or not 2 <= len(points) <= 1000:
            raise ValueError("启用的 T–N 曲线需要 2–1000 个已确认的转速/扭矩数值点")
        parsed = []
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("T–N 数值点必须包含 speed_rad_s 和 torque_nm")
            speed = _number(point.get("speed_rad_s"), "曲线转速 rad/s", 0, 1e6)
            torque = _number(point.get("torque_nm"), "曲线扭矩 N·m", 0, 1e9)
            parsed.append((speed, torque))
        parsed.sort()
        if parsed[0][0] != 0 or parsed[-1][1] != 0 or parsed[0][1] <= 0:
            raise ValueError("测试包络需从零转速正扭矩开始，至最高转速零扭矩结束；请核对曲线端点")
        if any(b[0] <= a[0] or b[1] > a[1] for a, b in zip(parsed, parsed[1:])):
            raise ValueError("测试包络的转速需严格递增、可用扭矩需非递增")
        effective = [(speed / ratio, torque * ratio * efficiency) for speed, torque in parsed]
        for name in selected:
            if name in result:
                raise ValueError(f"{name} 同时绑定多条启用的 T–N 曲线；请选择本次电压对应的一条")
            result[name] = effective
    return result


def torque_limit(points, velocity):
    speed = abs(float(velocity))
    if not math.isfinite(speed):
        raise ValueError("曲线查询转速必须有限")
    if speed >= points[-1][0]:
        return 0.0
    index = max(0, bisect_right([point[0] for point in points], speed) - 1)
    a, b = points[index], points[index + 1]
    fraction = (speed - a[0]) / (b[0] - a[0])
    return max(0.0, a[1] + fraction * (b[1] - a[1]))


def export_files(spec):
    """Archive confirmed motor inputs; temporary PD videos are never source assets."""
    compile_envelopes(spec, [joint["name"] for joint in spec["bindings"]["joints"]])
    prefix = f"src/config/{spec['robot']['slug']}"
    files, index = {}, []
    for curve in spec.get("motor_curves", []):
        entry = deepcopy(curve)
        asset = curve.get("image")
        if asset:
            source = image_path(asset)
            relative = f"motor_curves/{asset['id']}{source.suffix}"
            files[f"{prefix}/{relative}"] = source.read_bytes()
            entry["image_path"] = relative
            entry.pop("image", None)
            entry.pop("image_url", None)
        index.append(entry)
    if index:
        files[f"{prefix}/motor_curves/index.json"] = json.dumps(
            {"version": 1, "curves": index, "training_envelope_applied": False,
             "note": "图片与已确认曲线作为输入证据归档；当前训练执行器仍使用固定限矩。"},
            ensure_ascii=False, indent=2, allow_nan=False)
    return files
