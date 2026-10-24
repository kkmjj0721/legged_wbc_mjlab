"""One URDF Studio canvas backed by authoritative CPU MuJoCo kinematics."""

from __future__ import annotations

import threading
from pathlib import Path


class PosePreview:
    def __init__(self):
        self._lock = threading.RLock()
        self.server = None
        self.scene = None
        self.model = None
        self.data = None
        self.spec = None
        self.bindings = None
        self.foot_markers = []
        self.joint_markers = []
        self.token, self.viewer_source, self.assets = None, None, {}
        self.pose, self.revision = None, 0

    def start(self, xml: Path, spec: dict) -> dict:
        import secrets
        import mujoco
        import xml.etree.ElementTree as ET
        from .generator import _model_details
        from .template_runtime import model_spec
        from .studio_helper import STUDIO_COMMIT

        with self._lock:
            self.stop()
            root, _, bindings, _, _ = _model_details(spec)
            prepared = model_spec(xml, bindings)
            self.model = prepared.compile()
            try:
                self.data, self.spec, self.bindings = mujoco.MjData(self.model), spec, bindings
                self.token = secrets.token_urlsafe(24)
                tree = ET.fromstring(prepared.to_xml())
                compiler = tree.find("compiler")
                directories = {}
                if compiler is not None:
                    default_dir = compiler.get("assetdir", "")
                    directories = {"mesh": compiler.get("meshdir", default_dir),
                                   "texture": compiler.get("texturedir", default_dir)}
                    for attribute in ("assetdir", "meshdir", "texturedir"):
                        compiler.attrib.pop(attribute, None)
                self.assets = {}
                for element in tree.iter():
                    reference = element.get("file")
                    if not reference:
                        continue
                    asset = (xml.parent / directories.get(element.tag, "") / reference).resolve()
                    if not asset.is_relative_to(root.resolve()) or not asset.is_file() or asset.is_symlink():
                        raise ValueError(f"预览资源不属于当前模型：{reference}")
                    relative = asset.relative_to(root).as_posix()
                    self.assets[relative] = asset
                    element.set("file", relative)
                # Base placement comes from CPU FK state, rather than the
                # viewer's automatic ground alignment or an MJCF keyframe.
                floating_body = tree.find("worldbody/body")
                if floating_body is not None:
                    floating_body.set("pos", "0 0 0")
                    floating_body.set("quat", "1 0 0 0")
                    for attribute in ("euler", "axisangle", "xyaxes", "zaxis"):
                        floating_body.attrib.pop(attribute, None)
                for keyframe in list(tree.findall("keyframe")):
                    tree.remove(keyframe)
                self.viewer_source = ET.tostring(tree, encoding="unicode")
                initial = spec.get("parameters", {}).get("init_state", {})
                initial = spec.get("initial_state", initial)
                pose = self.update(joint_positions=initial.get("default_joint", initial.get("joint_pos", {})),
                                   pos=initial.get("pos"), rot=initial.get("rot"))
                return {"url": f"/studio/?token={self.token}", "pose": pose,
                        "renderer": "urdf-studio", "renderer_version": STUDIO_COMMIT}
            except Exception:
                self.stop()
                raise

    def _check_token(self, token):
        import secrets
        if not self.token or not secrets.compare_digest(str(token), self.token):
            raise KeyError("预览已关闭或模型已切换，请重新打开 3D 预览")

    def studio_model(self, token):
        from urllib.parse import quote
        from .studio_helper import STUDIO_COMMIT
        with self._lock:
            self._check_token(token)
            return {"source": {"format": "mjcf", "content": self.viewer_source,
                               "sourceFilePath": "preview.xml"},
                    "assets": {name: f"/api/preview/studio/{self.token}/asset/{quote(name, safe='/')}"
                               for name in self.assets},
                    "renderer": "urdf-studio", "commit": STUDIO_COMMIT}

    def studio_state(self, token):
        from copy import deepcopy
        with self._lock:
            self._check_token(token)
            return {"pose": deepcopy(self.pose), "revision": self.revision}

    def studio_asset(self, token, filename):
        with self._lock:
            self._check_token(token)
            path = self.assets.get(filename)
            if path is None or not path.is_file() or path.is_symlink():
                raise KeyError("资源不属于当前预览模型")
            return path

    def update(self, joint_positions=None, pos=None, rot=None, **_ignored) -> dict:
        import mujoco
        import numpy as np

        with self._lock:
            if self.model is None:
                raise ValueError("请先打开机器人 3D 预览")
            model, data = self.model, self.data
            updated = data.qpos.copy()
            for name, value in (joint_positions or {}).items():
                joint = model.joint(name)
                if int(joint.type[0]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    raise ValueError(f"关节 {name} 不是单自由度关节")
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"关节 {name} 的位置必须为有限数值")
                if joint.limited[0] and not joint.range[0] <= value <= joint.range[1]:
                    raise ValueError(f"关节 {name} 的位置超出模型限位")
                updated[int(joint.qposadr[0])] = value
            free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            if len(free):
                address = int(model.jnt_qposadr[free[0]])
                if pos is not None:
                    values = np.asarray(pos, dtype=float)
                    if values.shape != (3,) or not np.isfinite(values).all():
                        raise ValueError("初始位置需要 3 个有限数值")
                    updated[address:address + 3] = values
                if rot is not None:
                    values = np.asarray(rot, dtype=float)
                    if values.shape != (4,) or not np.isfinite(values).all() or np.linalg.norm(values) < 1e-10:
                        raise ValueError("初始四元数需要 4 个有限数值，顺序为 wxyz")
                    updated[address + 3:address + 7] = values / np.linalg.norm(values)
            data.qpos[:] = updated
            mujoco.mj_forward(model, data)
            joints = {
                model.joint(i).name: float(data.qpos[model.jnt_qposadr[i]])
                for i in range(model.njnt)
                if model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            }
            feet = {item["site"]: data.site_xpos[model.site(item["site"]).id].tolist()
                    for item in self.bindings["feet"]}
            pose = {"joint_positions": joints, "feet": feet}
            if len(free):
                pose.update(pos=data.qpos[address:address + 3].tolist(), rot=data.qpos[address + 3:address + 7].tolist())
            self.pose = pose
            self.revision += 1
            return pose

    def stop(self):
        with self._lock:
            self.server = self.scene = self.model = self.data = self.spec = self.bindings = None
            self.foot_markers, self.joint_markers = [], []
            self.token, self.viewer_source, self.assets = None, None, {}
            self.pose, self.revision = None, 0
