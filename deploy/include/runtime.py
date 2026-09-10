from __future__ import annotations

import time

from .config import SimConfig
from .fsm import LocomotionFSM
from .inputs import CommandBus, InputManager
from .mujoco_bridge import MujocoBackend
from .policy import OnnxPolicy


class DeploymentRuntime:
    """Generic control-loop shell; backend and task policy are injected."""

    def __init__(self, cfg: SimConfig, policy: OnnxPolicy):
        self.cfg = cfg
        self.backend = MujocoBackend(cfg)
        self.commands = CommandBus(cfg.command_limit, cfg.command_deadzone)
        self.inputs = InputManager(cfg.input_backend, self.commands, cfg.joystick_axes)
        self.fsm = LocomotionFSM(cfg, self.backend, policy, self.commands)

    def run(self, headless: bool, max_steps: int | None = None) -> None:
        self.inputs.start(headless=headless)
        print(
            "[INFO] Controls: U=stand, P/Space=passive, R=reset, Esc=quit; "
            "W/S=forward/back, A/D=left/right, Q/E=yaw, F=follow/free camera. "
            "In the viewer, motion keys are toggles: press again to stop."
        )
        if self.cfg.input_backend in ("joystick", "both"):
            print("[INFO] Gamepad: left stick=vx/vy, right-X=yaw, A=stand, B/Circle=passive, Start=reset, Select=quit.")
        try:
            if headless:
                steps = 0
                while max_steps is None or steps < max_steps:
                    if self.commands.consume("quit"):
                        break
                    self.fsm.step()
                    steps += 1
            else:
                import mujoco
                import mujoco.viewer
                with mujoco.viewer.launch_passive(self.backend.model, self.backend.data, key_callback=self.inputs.key_callback) as viewer:
                    follow_camera = self.cfg.camera_follow
                    self._set_camera(viewer, follow_camera, mujoco)
                    last = time.perf_counter()
                    while viewer.is_running():
                        now = time.perf_counter()
                        elapsed = min(0.05, max(0.0, now - last))
                        last = now
                        for _ in range(max(1, int(round(elapsed / self.cfg.timestep)))):
                            self.fsm.step()
                        viewer.sync()
                        if self.commands.consume("camera_toggle"):
                            follow_camera = not follow_camera
                            self._set_camera(viewer, follow_camera, mujoco)
                            print(f"[INFO] camera={'follow' if follow_camera else 'free'}")
                        if self.commands.consume("quit"):
                            break
                        time.sleep(self.cfg.timestep)
        finally:
            self.inputs.close()

    def close(self) -> None:
        self.backend.close()

    def _set_camera(self, viewer, follow: bool, mujoco_module) -> None:
        """Configure the native MuJoCo camera to track the configured body."""
        with viewer.lock():
            if follow:
                body_id = self.backend.model.body(self.cfg.camera_track_body).id
                viewer.cam.type = mujoco_module.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = body_id
                viewer.cam.distance = self.cfg.camera_distance
                viewer.cam.azimuth = self.cfg.camera_azimuth
                viewer.cam.elevation = self.cfg.camera_elevation
            else:
                viewer.cam.type = mujoco_module.mjtCamera.mjCAMERA_FREE
