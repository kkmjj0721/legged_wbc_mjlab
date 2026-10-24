"""One local, CPU-only MuJoCo pose preview per workbench."""

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

    def start(self, xml: Path, spec: dict) -> dict:
        import mujoco
        import numpy as np
        import viser
        from mjviser import ViserMujocoScene
        from .generator import _model_details
        from .template_runtime import model_spec

        with self._lock:
            self.stop()
            # Match generated foot sites and sensor frames, including models that
            # do not supply named feet. This path never advances simulation time.
            _, _, bindings, _, _ = _model_details(spec)
            model = model_spec(xml, bindings).compile()
            server = viser.ViserServer(host="127.0.0.1", port=0, verbose=False)
            try:
                server.gui.configure_theme(show_logo=False, show_share_button=False, brand_color=(153, 101, 61))
                server.gui.main_panel.dock_right()
                server.gui.main_panel.minimize()
                self.scene = ViserMujocoScene(server, model, num_envs=1)
                self.scene.camera_tracking_enabled = False
                self.server, self.model = server, model
                self.data, self.spec, self.bindings = mujoco.MjData(model), spec, bindings
                server.scene.set_background_image(np.full((2, 2, 3), (246, 242, 234), dtype=np.uint8), format="png")
                server.scene.add_grid("/migration/grid", width=3, height=3, cell_size=0.1, section_size=0.5,
                                      cell_color=(220, 209, 193), section_color=(185, 164, 140),
                                      plane_color=(246, 242, 234), plane_opacity=0.1, fade_distance=3)
                colors = ((177, 99, 58), (91, 128, 103), (89, 118, 154), (178, 140, 55))
                for foot, color in zip(bindings["feet"], colors):
                    name = f"/migration/feet/{foot['leg']}"
                    marker = server.scene.add_icosphere(name + "/point", radius=0.008, color=color, subdivisions=1)
                    label = server.scene.add_label(name + "/label", text=foot["leg"], font_screen_scale=0.8)
                    self.foot_markers.append((foot["site"], marker, label))
                for index, joint in enumerate(bindings["joints"]):
                    name = f"/migration/joints/{index}"
                    marker = server.scene.add_frame(name + "/axis", axes_length=0.045, axes_radius=0.0015,
                                                    origin_radius=0.004, visible=False)
                    label = server.scene.add_label(name + "/label", text=joint["name"], font_screen_scale=0.7, visible=False)
                    self.joint_markers.append((joint["name"], marker, label))
                show_feet = server.gui.add_checkbox("显示足端名称与位置", initial_value=True)
                show_joints = server.gui.add_checkbox("显示关节与转轴", initial_value=False)
                server.gui.add_markdown("关节标记的蓝色 Z 轴表示模型转轴；拖动场景旋转，滚轮缩放。")

                @show_feet.on_update
                def feet_visibility(_):
                    for _, marker, label in self.foot_markers:
                        marker.visible = label.visible = show_feet.value

                @show_joints.on_update
                def joint_visibility(_):
                    for _, marker, label in self.joint_markers:
                        marker.visible = label.visible = show_joints.value
                initial = spec.get("parameters", {}).get("init_state", {})
                # Older saved drafts used the top-level initial-state group.
                initial = spec.get("initial_state", initial)
                pose = self.update(
                    joint_positions=initial.get("default_joint", initial.get("joint_pos", {})),
                    pos=initial.get("pos"), rot=initial.get("rot"),
                )
                points = np.concatenate((self.data.xpos[1:], self.data.site_xpos), axis=0)
                lower, upper = points.min(0), points.max(0)
                center = (lower + upper) / 2
                size = max(float(np.max(upper - lower)), 0.2)
                server.initial_camera.fov = float(np.deg2rad(50))
                server.initial_camera.position = tuple(center + size * np.array([1.15, -1.15, 0.8]))
                server.initial_camera.look_at = tuple(center)
                return {"url": f"http://127.0.0.1:{server.get_port()}/", "pose": pose}
            except Exception:
                server.stop()
                self.server = self.scene = self.model = self.data = self.spec = self.bindings = None
                self.foot_markers, self.joint_markers = [], []
                raise

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
            self.scene.update_from_mjdata(data)
            for name, marker, label in self.foot_markers:
                point = data.site_xpos[model.site(name).id]
                marker.position = tuple(point)
                label.position = tuple(point + np.array([0, 0, 0.018]))
            for name, marker, label in self.joint_markers:
                joint_id = model.joint(name).id
                point, axis = data.xanchor[joint_id], data.xaxis[joint_id]
                reference = np.array([0, 1, 0]) if abs(axis[1]) < 0.9 else np.array([1, 0, 0])
                x = np.cross(reference, axis)
                x /= np.linalg.norm(x)
                rotation = np.column_stack((x, np.cross(axis, x), axis))
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, rotation.ravel())
                marker.position, marker.wxyz = tuple(point), tuple(quaternion)
                label.position = tuple(point + np.array([0, 0, 0.025]))
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
            return pose

    def stop(self):
        with self._lock:
            if self.server is not None:
                self.server.stop()
            self.server = self.scene = self.model = self.data = self.spec = self.bindings = None
            self.foot_markers, self.joint_markers = [], []
