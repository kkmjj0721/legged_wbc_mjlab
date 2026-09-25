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
        self.inputs = InputManager(
            cfg.input_backend,
            self.commands,
            cfg.joystick_axes,
            **getattr(cfg, "gamepad_options", {}),
        )
        self.fsm = LocomotionFSM(cfg, self.backend, policy, self.commands)
        self._shutdown_sent = False

    def run(self, headless: bool, max_steps: int | None = None) -> None:
        print(
            "[INFO] Controls: U=stand, L=enable RL, B=hold STAND, "
            "P/G/Space=getdown, R=reset, Esc=estop+quit; "
            "W/S=forward/back, A/D=left/right, Q/E=yaw, F=follow/free camera. "
            "Motion keys adjust velocity by 0.1 per press; Z/X=zeros velocity."
        )
        if self.cfg.input_backend in ("joystick", "both"):
            print(
                "[INFO] Gamepad: left stick=vx/vy, right-X=yaw, "
                "A=stand, X=enable RL, B=hold stand, Y=getdown, "
                "Start=reset, Back/Select=estop+quit."
            )
            print(
                "[INFO] Leave both sticks centered while the gamepad calibrates; "
                "velocity remains zero until calibration and the neutral hold complete."
            )
        normal_exit = False
        try:
            # SDL event pumping and gamepad device access must stay on this
            # runtime thread.  InputManager.poll() below is therefore part of
            # both the headless and viewer control loops.
            self.inputs.start(headless=headless)
            if headless:
                steps = 0
                while max_steps is None or steps < max_steps:
                    self.inputs.poll()
                    if self.commands.consume("quit"):
                        break
                    self.fsm.step()
                    steps += 1
                normal_exit = True
            else:
                import mujoco
                import mujoco.viewer
                with mujoco.viewer.launch_passive(self.backend.model, self.backend.data, key_callback=self.inputs.key_callback) as viewer:
                    follow_camera = self.cfg.camera_follow
                    self._set_camera(viewer, follow_camera, mujoco)
                    # Use a monotonic deadline to avoid accumulating scheduler
                    # drift.  Physics remains fixed-step while rendering may
                    # run at a lower rate on a busy desktop.
                    next_deadline = time.monotonic()
                    while viewer.is_running():
                        self.inputs.poll()
                        if self.commands.consume("quit"):
                            break
                        now = time.monotonic()
                        elapsed = min(0.05, max(0.0, now - next_deadline + self.cfg.timestep))
                        physics_steps = max(1, int(round(elapsed / self.cfg.timestep)))
                        for _ in range(physics_steps):
                            self.fsm.step()
                        next_deadline += physics_steps * self.cfg.timestep
                        viewer.sync()
                        if self.commands.consume("camera_toggle"):
                            follow_camera = not follow_camera
                            self._set_camera(viewer, follow_camera, mujoco)
                            print(f"[INFO] camera={'follow' if follow_camera else 'free'}")
                        sleep_for = next_deadline - time.monotonic()
                        if sleep_for > 0.0:
                            time.sleep(min(sleep_for, self.cfg.timestep))
                normal_exit = True
        except KeyboardInterrupt:
            # Ctrl-C is an operator request to leave; still perform the same
            # safe actuator shutdown as an emergency event.
            print("[INFO] interrupt received; stopping safely")
            normal_exit = True
        finally:
            self.inputs.close()
            self._safe_shutdown("runtime shutdown" if normal_exit else "runtime failure")

    def close(self) -> None:
        self._safe_shutdown("runtime close")
        self.backend.close()

    def _safe_shutdown(self, reason: str) -> None:
        """Best-effort torque-off on every runtime exit path.

        Backends used by tests or legacy integrations may not implement the
        emergency API, so shutdown deliberately degrades to a no-op there.
        """
        if self._shutdown_sent:
            return
        self._shutdown_sent = True
        emergency_stop = getattr(self.backend, "emergency_stop", None)
        if callable(emergency_stop):
            try:
                emergency_stop(reason)
            except Exception:
                pass

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
