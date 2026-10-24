"""Resolve motor-shaft inertia into the joint coordinates used by all consumers."""

from copy import deepcopy
import math

SCOPES = {"common", "ppo", "him", "him_symmetry", "him_lpacrl"}
FIELDS = {"rotor_inertia", "gear_ratio"}


def _number(value, label, *, ratio=False):
    if value is None:
        return 1.0 if ratio else None
    high = 10000 if ratio else 100
    if (type(value) not in (int, float) or value < 0 or value > high
            or not math.isfinite(value) or (ratio and value == 0)):
        interval = f"(0, {high}]" if ratio else f"[0, {high}]"
        raise ValueError(f"{label} 必须是 {interval} 内的有限数值")
    return float(value)


def validate(spec, bindings=None):
    """Validate canonical metadata, including inactive but supported task scopes."""
    if "motor_inertia" not in spec:
        return
    root = spec["motor_inertia"]
    if (not isinstance(root, dict) or set(root) - {"version", "scopes"}
            or type(root.get("version")) is not int or root["version"] != 1):
        raise ValueError("motor_inertia 必须是 version=1 的对象，且仅包含 version / scopes")
    scopes = root.get("scopes")
    if not isinstance(scopes, dict) or set(scopes) - SCOPES:
        raise ValueError("motor_inertia.scopes 必须是以 common 或受支持任务模板为键的对象")
    names = [joint["name"] for joint in (bindings or spec.get("bindings", {})).get("joints", [])]
    for scope, joints in scopes.items():
        if not isinstance(joints, dict) or set(joints) - set(names):
            raise ValueError(f"motor_inertia.scopes.{scope} 必须是以已绑定关节名称为键的对象")
        for name, values in joints.items():
            if not isinstance(values, dict) or set(values) - FIELDS:
                raise ValueError(f"{scope}.{name} 的电机惯量必须是仅含 rotor_inertia / gear_ratio 的对象")
            for field, value in values.items():
                _number(value, f"motor_inertia.{scope}.{name}.{field}", ratio=field == "gear_ratio")
    # Check inherited combinations too: a task can provide only one field.
    for scope in ("common", *sorted(set(scopes) - {"common"})):
        _resolve(spec, scope, names)


def _resolve(spec, scope, names):
    canonical = "motor_inertia" in spec
    if canonical:
        scopes = spec["motor_inertia"]["scopes"]
        source = "motor_inertia"
    else:
        legacy = spec.get("pd_analysis", {})
        scopes = legacy.get("scopes", {}) if isinstance(legacy, dict) else {}
        scopes = scopes if isinstance(scopes, dict) else {}
        source = "pd_analysis"
    result = {}
    for name in names:
        raw, field_sources = {}, {}
        for current in ("common", scope) if scope != "common" else ("common",):
            joints = scopes.get(current, {})
            if not canonical:
                options = joints.get("options", {}) if isinstance(joints, dict) else {}
                joints = options.get("joints", {}) if isinstance(options, dict) else {}
            values = joints.get(name, {}) if isinstance(joints, dict) else {}
            if not isinstance(values, dict):
                raise ValueError(f"{source}.{current}.{name} 的电机参数必须为对象")
            for field in ("rotor_inertia", "gear_ratio"):
                if field not in values:
                    continue
                raw[field] = values[field]
                field_sources[field] = f"{source}.scopes.{current}" + (".options.joints" if not canonical else "")
        rotor = _number(raw.get("rotor_inertia"), f"{source}.{scope}.{name}.rotor_inertia")
        ratio = _number(raw.get("gear_ratio"), f"{source}.{scope}.{name}.gear_ratio", ratio=True)
        armature = None if rotor is None else rotor * ratio**2
        if armature is not None and (not math.isfinite(armature) or armature > 1e10):
            raise ValueError(f"{name} 的折算关节惯量必须位于 [0, 1e10] kg·m²")
        result[name] = {"rotor_inertia": rotor, "gear_ratio": ratio,
                        "armature": armature, "field_sources": field_sources}
    return {"version": 1, "source": source, "scope": scope, "joints": result}


def resolve(spec, scope, bindings=None):
    """Resolve common → task field inheritance; unknown rotor inertia stays null."""
    if scope not in SCOPES:
        raise ValueError("电机惯量范围必须为 common 或受支持的任务模板")
    bindings = bindings or spec.get("bindings", {})
    validate(spec, bindings)
    return _resolve(spec, scope, [joint["name"] for joint in bindings.get("joints", [])])


def apply(spec, scope, bindings, control):
    """Replace effective armature only where a rotor inertia is known; return provenance."""
    metadata = resolve(spec, scope, bindings)
    if any(values["armature"] is not None for values in metadata["joints"].values()):
        previous = control.get("armature")
        values = {}
        for joint in bindings["joints"]:
            name = joint["name"]
            prior = previous.get(name, previous.get(joint["role"])) if isinstance(previous, dict) else previous
            derived = metadata["joints"][name]["armature"]
            values[name] = deepcopy(prior) if derived is None else derived
        control["armature"] = values
    return metadata
