from __future__ import annotations

import select
import os
import sys
import termios
import threading
import time
import tty
import logging
import math
from typing import Any

import numpy as np


_LOG = logging.getLogger(__name__)


class CommandBus:
    """Thread-safe command/event bus shared by keyboard and gamepad inputs."""

    _EVENT_ALIASES = {"passive": "get_down", "camera_toggle": "camera"}

    def __init__(self, limits: np.ndarray, deadzone: float):
        self._limits = limits.copy()
        self._deadzone = deadzone
        self._lock = threading.Lock()
        self._command = np.zeros(3, dtype=np.float64)
        self._events = {
            name: False
            for name in (
                "zero_velocity", "stand", "get_down", "reset", "camera", "quit",
                "enable_rl", "disable_rl", "estop",
            )
        }

    def set_axes(self, vx: float, vy: float, wz: float) -> None:
        def shape(value: float) -> float:
            return 0.0 if abs(value) < self._deadzone else float(np.clip(value, -1.0, 1.0))
        with self._lock:
            self._command[:] = [shape(vx) * self._limits[0], shape(vy) * self._limits[1], shape(wz) * self._limits[2]]

    def command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Set a physical velocity command, clipping to configured limits."""
        with self._lock:
            self._command[:] = np.clip((vx, vy, wz), -self._limits, self._limits)

    def request(self, name: str) -> None:
        name = self._EVENT_ALIASES.get(name, name)
        if name not in self._events:
            raise KeyError(name)
        with self._lock:
            self._events[name] = True

    def consume(self, name: str) -> bool:
        name = self._EVENT_ALIASES.get(name, name)
        with self._lock:
            value = self._events[name]
            self._events[name] = False
            return value

    def stop(self) -> None:
        self.set_axes(0.0, 0.0, 0.0)

    def emergency_stop(self) -> None:
        """Atomically zero velocity and latch an emergency-stop event."""
        with self._lock:
            self._command.fill(0.0)
            self._events["estop"] = True


class KeyboardInput:
    _KEYS = {
        ord("w"): (1.0, 0.0, 0.0), ord("s"): (-1.0, 0.0, 0.0),
        ord("a"): (0.0, 1.0, 0.0), ord("d"): (0.0, -1.0, 0.0),
        ord("q"): (0.0, 0.0, 1.0), ord("e"): (0.0, 0.0, -1.0),
    }

    _EVENT_KEYS = {
        ord("z"): "zero_velocity", ord("x"): "zero_velocity",
        ord("u"): "stand", 265: "stand",
        ord("g"): "get_down", ord("p"): "get_down", 32: "get_down", 264: "get_down",
        # Manual policy lifecycle controls.  ``l`` enables RL once the
        # scripted stand/settle sequence is complete; ``b`` disables RL and
        # holds the standing pose (the FSM handles both events safely in any
        # other state).
        ord("l"): "enable_rl", ord("b"): "disable_rl",
        ord("r"): "reset", ord("f"): "camera", 27: "quit", 256: "quit",
    }

    def __init__(self, bus: CommandBus, keyboard_step: float = 0.1):
        self.bus = bus
        keyboard_step = float(keyboard_step)
        if not np.isfinite(keyboard_step) or keyboard_step < 0.0:
            raise ValueError("keyboard_step must be a finite non-negative number")
        self.keyboard_step = keyboard_step
        self._velocity = np.zeros(3, dtype=np.float64)
        self._pressed: set[int] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._old_terminal: list[Any] | None = None

    def handle_key(self, key: int, action: int = 1) -> None:
        key = key + 32 if 65 <= key <= 90 else key
        with self._lock:
            if action == 0:
                self._pressed.discard(key)
                return
            if action not in (1, 2) or key in self._pressed:
                return
            self._pressed.add(key)
        self._dispatch_key(key)

    def _dispatch_key(self, key: int) -> None:
        event = self._EVENT_KEYS.get(key)
        if event is not None:
            self.bus.request(event)
            if event == "zero_velocity":
                with self._lock:
                    self._velocity.fill(0.0)
                self.bus.stop()
            return
        direction = self._KEYS.get(key)
        if direction is None:
            return
        with self._lock:
            self._velocity += self.keyboard_step * np.asarray(direction)
            self._velocity = np.clip(self._velocity, -self.bus._limits, self.bus._limits)
            velocity = self._velocity.copy()
        self.bus.set_command(*velocity)

    def _recompute(self) -> None:
        with self._lock:
            velocity = self._velocity.copy()
        self.bus.set_command(*velocity)

    def handle_viewer_key(self, key: int) -> None:
        """Handle MuJoCo's key-only callback.

        Each callback is a discrete key press; motion is incremented by one
        keyboard step and is never latched by the viewer callback.
        """
        normalized = key + 32 if 65 <= key <= 90 else key
        self._dispatch_key(normalized)

    def start_headless(self) -> None:
        if not sys.stdin.isatty() or self._thread is not None:
            return
        try:
            self._old_terminal = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except (termios.error, OSError):
            self._old_terminal = None
        self._running = True
        self._thread = threading.Thread(target=self._read, name="sim2sim-keyboard", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while self._running:
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            char = sys.stdin.read(1)
            if char == "\x1b":
                sequence = ""
                for _ in range(2):
                    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if not ready:
                        break
                    sequence += sys.stdin.read(1)
                key = {"[A": 265, "[B": 264, "[C": 262, "[D": 263}.get(sequence)
                if key is not None:
                    self.handle_key(key, 1)
                    self.handle_key(key, 0)
                else:
                    self.bus.request("quit")
                continue
            if char:
                self.handle_key(ord(char), 1)
                self.handle_key(ord(char), 0)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._old_terminal is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_terminal)
            except (termios.error, OSError):
                pass
            self._old_terminal = None


class JoystickInput:
    """SDL gamepad/joystick input with hot-plug and safe reconnect handling.

    Pygame's semantic SDL2 ``Controller`` API is preferred when available.  A
    raw ``Joystick`` is used as a fallback, which is useful for controllers
    (including wireless dongles) that do not advertise an SDL game-controller
    mapping.  No VID/PID or device index is assumed: selection can be made by
    ``device_name``, ``device_guid`` and/or ``device_index``.
    """

    _DEFAULT_BUTTONS = {
        "stand": 0,       # A / Cross
        "enable_rl": 2,   # X / Square
        "disable_rl": 1,  # B / Circle
        "get_down": 3,    # Y / Triangle
        "reset": (7, 9),  # Start (SDL mappings commonly use either index)
        "back": (6, 8),   # Back / Select / Guide on a few raw devices
    }
    _CONTROLLER_DEFAULT_BUTTONS = {
        "stand": "A",
        "enable_rl": "X",
        "disable_rl": "B",
        "get_down": "Y",
        "reset": "START",
        "back": "BACK",
    }
    _CONTROLLER_BUTTON_NAMES = {
        "a": "A", "stand": "A", "cross": "A",
        "b": "B", "disable_rl": "B", "circle": "B",
        "x": "X", "enable_rl": "X", "square": "X",
        "y": "Y", "get_down": "Y", "triangle": "Y",
        "start": "START", "reset": "START",
        "back": "BACK", "select": "BACK",
    }
    # SDL_GameControllerButton / SDL_GameControllerAxis enum values are stable
    # and intentionally mirrored here because pygame does not expose the enum
    # classes on every supported release.
    _CONTROLLER_BUTTON_IDS = {
        "A": 0, "B": 1, "X": 2, "Y": 3, "BACK": 4, "GUIDE": 5,
        "START": 6, "LEFTSTICK": 7, "RIGHTSTICK": 8,
        "LEFTSHOULDER": 9, "RIGHTSHOULDER": 10,
        "DPAD_UP": 11, "DPAD_DOWN": 12, "DPAD_LEFT": 13, "DPAD_RIGHT": 14,
    }
    # Raw pygame.Joystick devices do not expose SDL's semantic button enum;
    # their physical indices vary by firmware and wireless dongle.  Keep the
    # common A/B/X/Y indices stable and accept the usual aliases for the
    # lifecycle buttons.  The START/BACK candidates intentionally include the
    # indices used by common XInput/DInput mappings (including the Asura
    # 2 Pro/星闪 variants).
    _RAW_BUTTON_ALIASES = {
        "A": 0,
        "CROSS": 0,
        "B": 1,
        "CIRCLE": 1,
        "X": 2,
        "SQUARE": 2,
        "Y": 3,
        "TRIANGLE": 3,
        "BACK": (4, 6, 8),
        "SELECT": (4, 6, 8),
        "GUIDE": 5,
        "START": (6, 7, 9),
    }
    _CONTROLLER_AXIS_IDS = {
        "LEFTX": 0, "LEFTY": 1, "RIGHTX": 2, "RIGHTY": 3,
        "TRIGGERLEFT": 4, "TRIGGERRIGHT": 5,
    }
    _DEFAULT_DEADZONE = 0.12

    def __init__(
        self,
        bus: CommandBus,
        axes: tuple[int, int, int] = (0, 1, 2),
        *,
        device_name: str | None = None,
        device_guid: str | None = None,
        device_index: int | None = None,
        axis_config: dict[str, Any] | None = None,
        button_config: dict[str, Any] | None = None,
        deadzone: float | None = None,
        yaw_rescale: float = 1.0,
        reconnect_interval: float = 1.0,
        liveness_timeout: float = 2.0,
        prefer_controller: bool = True,
    ):
        self.bus = bus
        self.axes = tuple(int(axis) for axis in axes)  # legacy raw-joystick API
        if len(self.axes) != 3:
            raise ValueError("axes must contain (left_x, left_y, yaw)")
        self.device_name = str(device_name).strip() if device_name else None
        self.device_guid = str(device_guid).strip().lower() if device_guid else None
        self.device_index = None if device_index is None else int(device_index)
        self.axis_config = dict(axis_config or {})
        self.button_config = dict(button_config or {})
        self.deadzone = self._validate_deadzone(self._DEFAULT_DEADZONE if deadzone is None else deadzone)
        self.yaw_rescale = self._validate_rescale(yaw_rescale)
        self.reconnect_interval = max(0.05, float(reconnect_interval))
        self.liveness_timeout = max(0.1, float(liveness_timeout))
        self.prefer_controller = bool(prefer_controller)
        self.available = False  # True only while a selected device is open
        self.device_info: str | None = None
        self._pygame: Any = None
        self._device: Any = None
        self._controller = False
        self._instance_id: int | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_seen = 0.0
        self._last_reconnect = 0.0
        self._warned_no_device = False
        self._lock = threading.RLock()

    @staticmethod
    def _validate_deadzone(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("gamepad deadzone must be in [0, 1)")
        return value

    @staticmethod
    def _validate_rescale(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("gamepad yaw_rescale must be a finite positive number")
        return value

    @staticmethod
    def _event_type(pygame: Any, name: str) -> int | None:
        value = getattr(pygame, name, None)
        return int(value) if value is not None else None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            self._pygame = pygame
            # ``pygame.init`` is intentionally avoided: it may open a window.
            # Joystick/event subsystems work with just these two initializers.
            if not pygame.get_init():
                pygame.joystick.init()
            else:
                pygame.joystick.init()
            try:
                import pygame._sdl2.controller as controller_mod
                controller_mod.init()
            except Exception:
                pass
            try:
                pygame.event.pump()
            except Exception:
                pass
        except Exception as exc:
            _LOG.warning("gamepad disabled: pygame/SDL unavailable (%s)", exc)
            self._pygame = None
            return
        self._running = True
        self._thread = threading.Thread(target=self._read, name="sim2sim-gamepad", daemon=True)
        self._thread.start()
        _LOG.info("gamepad input started (name=%r guid=%r index=%r)", self.device_name, self.device_guid, self.device_index)

    def _guid(self, device: Any) -> str:
        try:
            getter = getattr(device, "get_guid", None)
            value = getter() if callable(getter) else getattr(device, "guid", "")
            if not value and self._controller:
                joystick = getattr(device, "as_joystick", lambda: None)()
                value = joystick.get_guid() if joystick is not None else ""
            return str(value).strip().lower()
        except Exception:
            return ""

    def _name(self, device: Any) -> str:
        try:
            getter = getattr(device, "get_name", None)
            value = getter() if callable(getter) else getattr(device, "name", "")
            return str(value)
        except Exception:
            return ""

    def _matches(self, device: Any, index: int | None = None) -> bool:
        if self.device_index is not None and index is not None and index != self.device_index:
            return False
        name = self._name(device)
        guid = self._guid(device)
        if self.device_name and self.device_name.casefold() not in name.casefold():
            return False
        if self.device_guid and self.device_guid not in guid:
            return False
        return True

    def _close_device(self, *, disconnected: bool = False) -> None:
        with self._lock:
            device, is_controller = self._device, self._controller
            self._device = None
            self._controller = False
            self._instance_id = None
            had_device = self.available
            self.available = False
            self.device_info = None
        if device is not None and is_controller:
            try:
                device.quit()
            except Exception:
                pass
        if had_device or disconnected:
            # Never leave a stale command behind, including during orderly
            # shutdown.  Only an actual disconnect latches the estop event.
            self.bus.stop()
            if disconnected:
                self.bus.request("estop")
                _LOG.warning("gamepad disconnected; velocity command zeroed")

    def _open_controller(self, index: int) -> bool:
        pygame = self._pygame
        if pygame is None:
            return False
        controller_mod = getattr(getattr(pygame, "_sdl2", None), "controller", None)
        if controller_mod is None or not self.prefer_controller:
            return False
        try:
            is_controller = getattr(controller_mod, "is_controller", None)
            if is_controller is not None and not is_controller(index):
                return False
            device = controller_mod.Controller(index)
            device.init()
            if not self._matches(device, index):
                device.quit()
                return False
            instance = getattr(device, "get_instance_id", lambda: getattr(device, "id", None))()
            with self._lock:
                self._device, self._controller, self._instance_id = device, True, instance
                self.available, self.device_info = True, self._name(device) or f"controller#{index}"
                self._last_seen = time.monotonic()
            _LOG.info("gamepad connected via SDL Controller: %s guid=%s", self.device_info, self._guid(device) or "unknown")
            return True
        except Exception:
            return False

    def _open_joystick(self, index: int) -> bool:
        pygame = self._pygame
        try:
            device = pygame.joystick.Joystick(index)
            device.init()
            if not self._matches(device, index):
                try:
                    device.quit()
                except Exception:
                    pass
                return False
            instance = getattr(device, "get_instance_id", lambda: getattr(device, "id", None))()
            with self._lock:
                self._device, self._controller, self._instance_id = device, False, instance
                self.available, self.device_info = True, self._name(device) or f"joystick#{index}"
                self._last_seen = time.monotonic()
            _LOG.info("gamepad connected via raw Joystick: %s guid=%s", self.device_info, self._guid(device) or "unknown")
            return True
        except Exception as exc:
            _LOG.debug("cannot open joystick index %s: %s", index, exc)
            return False

    def _try_connect(self) -> bool:
        pygame = self._pygame
        if pygame is None:
            return False
        try:
            count = int(pygame.joystick.get_count())
        except Exception:
            return False
        indices = [self.device_index] if self.device_index is not None else list(range(count))
        for index in indices:
            if index is None or index < 0 or index >= count:
                continue
            if self._open_controller(index) or self._open_joystick(index):
                self._warned_no_device = False
                return True
        if not self._warned_no_device:
            devices = []
            for candidate in range(count):
                try:
                    probe = pygame.joystick.Joystick(candidate)
                    devices.append(f"{candidate}:{self._name(probe) or 'unnamed'}:{self._guid(probe) or 'unknown'}")
                except Exception:
                    devices.append(f"{candidate}:<unreadable>")
            _LOG.warning(
                "no matching gamepad found (count=%d, name=%r, guid=%r, index=%r; devices=%s)",
                count, self.device_name, self.device_guid, self.device_index, ", ".join(devices) or "none",
            )
            self._warned_no_device = True
        return False

    def _axis_value(self, name: str) -> float:
        device = self._device
        if device is None:
            return 0.0
        configured = self.axis_config.get(name)
        if self._controller:
            # Semantic controller axes are stable across SDL mappings.  Accept
            # either an enum value or a string (e.g. ``"LEFTX"``).
            axis_name = configured if configured is not None else {"left_x": "LEFTX", "left_y": "LEFTY", "yaw": "RIGHTX"}.get(name)
            controller_mod = getattr(getattr(self._pygame, "_sdl2", None), "controller", None)
            enum = getattr(controller_mod, "CONTROLLER_AXIS", None) if controller_mod is not None else None
            if isinstance(axis_name, str) and enum is not None:
                axis_name = getattr(enum, axis_name.upper(), axis_name)
            if isinstance(axis_name, str):
                axis_name = self._CONTROLLER_AXIS_IDS.get(axis_name.upper(), axis_name)
            try:
                value = float(device.get_axis(axis_name))
                # SDL's semantic API returns signed int16 values, while raw
                # Joystick returns normalized floats.
                return value / 32767.0 if abs(value) > 1.5 else value
            except Exception:
                return 0.0
        axis = configured if configured is not None else self.axes[{"left_x": 0, "left_y": 1, "yaw": 2}[name]]
        if isinstance(axis, str):
            aliases = {"leftx": 0, "left_x": 0, "lefty": 1, "left_y": 1, "rightx": 2, "right_x": 2}
            axis = aliases.get(axis.casefold(), axis)
        try:
            axis = int(axis)
            return float(device.get_axis(axis)) if 0 <= axis < device.get_numaxes() else 0.0
        except Exception:
            return 0.0

    def _button_value(self, event_button: Any, action: str) -> bool:
        configured = self.button_config.get(
            action,
            self._CONTROLLER_DEFAULT_BUTTONS[action]
            if self._controller else self._DEFAULT_BUTTONS[action],
        )
        if self._controller:
            controller_mod = getattr(getattr(self._pygame, "_sdl2", None), "controller", None)
            enum = getattr(controller_mod, "CONTROLLER_BUTTON", None) if controller_mod is not None else None
            if isinstance(configured, str):
                configured = self._CONTROLLER_BUTTON_NAMES.get(configured.casefold(), configured.upper())
                if enum is not None:
                    configured = getattr(enum, configured, configured)
                if isinstance(configured, str):
                    configured = self._CONTROLLER_BUTTON_IDS.get(configured.upper(), configured)
        elif isinstance(configured, str):
            # Raw Joystick button events carry an integer index.  Translate
            # only semantic strings here; explicit numeric values and
            # tuple/list/set configurations remain untouched so callers can
            # provide device-specific mappings without being rewritten.
            semantic = configured.strip().upper()
            # Reuse the action/button spellings accepted by the SDL
            # Controller path (for example ``enable_rl`` or ``cross``).
            semantic = self._CONTROLLER_BUTTON_NAMES.get(semantic.casefold(), semantic)
            configured = self._RAW_BUTTON_ALIASES.get(semantic, configured)
        try:
            if isinstance(configured, (tuple, list, set, frozenset)):
                return event_button in configured
            return event_button == configured or str(event_button).casefold() == str(configured).casefold()
        except Exception:
            return False

    def _set_axes_from_device(self) -> None:
        device = self._device
        if device is None:
            raise RuntimeError("gamepad device is not open")
        try:
            attached = getattr(device, "attached", None)
            if callable(attached):
                attached = attached()
            if attached is False:
                raise RuntimeError("gamepad reports detached")
            getter = getattr(device, "get_attached", None)
            if callable(getter) and not getter():
                raise RuntimeError("gamepad reports detached")
        except RuntimeError:
            raise
        except Exception:
            # Older pygame versions do not expose an attachment query.
            pass
        lx, ly, yaw = self._axis_value("left_x"), self._axis_value("left_y"), self._axis_value("yaw")
        # Radial deadzone on the translation stick, with linear rescaling so
        # full travel still reaches the configured command limits.
        magnitude = math.hypot(lx, ly)
        if magnitude <= self.deadzone:
            lx = ly = 0.0
        else:
            scale = (magnitude - self.deadzone) / (max(1.0e-9, magnitude) * (1.0 - self.deadzone))
            lx, ly = lx * scale, ly * scale
        # SDL reports positive Y downwards; forward command is -left_y.
        yaw = float(np.clip(yaw * self.yaw_rescale, -1.0, 1.0))
        if abs(yaw) < self.deadzone:
            yaw = 0.0
        limits = np.asarray(self.bus._limits, dtype=np.float64)
        # SDL/XInput gamepads expose the left-stick X axis as lateral motion
        # and Y as fore/aft motion.  SDL's positive Y points down, therefore
        # pushing the stick forward (negative Y) produces a positive vx.
        self.bus.set_command(float(-ly * limits[0]), float(lx * limits[1]), float(yaw * limits[2]))
        self._last_seen = time.monotonic()

    def _dispatch_button(self, button: Any) -> None:
        if self._button_value(button, "stand"):
            self.bus.request("stand")
        elif self._button_value(button, "enable_rl"):
            self.bus.request("enable_rl")
        elif self._button_value(button, "disable_rl"):
            self.bus.request("disable_rl")
        elif self._button_value(button, "get_down"):
            self.bus.request("get_down")
        elif self._button_value(button, "reset"):
            self.bus.request("reset")
        elif self._button_value(button, "back"):
            self.bus.emergency_stop()
            self.bus.request("quit")

    def _handle_event(self, event: Any) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        typ = getattr(event, "type", None)
        joy_added = self._event_type(pygame, "JOYDEVICEADDED")
        joy_removed = self._event_type(pygame, "JOYDEVICEREMOVED")
        ctrl_added = self._event_type(pygame, "CONTROLLERDEVICEADDED")
        ctrl_removed = self._event_type(pygame, "CONTROLLERDEVICEREMOVED")
        if typ in (joy_removed, ctrl_removed):
            instance = getattr(event, "instance_id", getattr(event, "which", None))
            if self._instance_id is None or instance is None or instance == self._instance_id:
                self._close_device(disconnected=True)
            return
        if typ in (joy_added, ctrl_added):
            if not self.available:
                self._try_connect()
            return
        axis_types = {self._event_type(pygame, "JOYAXISMOTION"), self._event_type(pygame, "CONTROLLERAXISMOTION")}
        button_types = {self._event_type(pygame, "JOYBUTTONDOWN"), self._event_type(pygame, "CONTROLLERBUTTONDOWN")}
        if typ in axis_types and typ is not None:
            if self.available:
                self._set_axes_from_device()
        elif typ in button_types and typ is not None and self.available:
            self._dispatch_button(getattr(event, "button", None))

    def _read(self) -> None:
        while self._running:
            pygame = self._pygame
            now = time.monotonic()
            if not self.available and now - self._last_reconnect >= self.reconnect_interval:
                self._last_reconnect = now
                self._try_connect()
            if pygame is not None:
                try:
                    for event in pygame.event.get():
                        self._handle_event(event)
                except Exception as exc:
                    _LOG.debug("gamepad event pump failed: %s", exc)
                if self.available:
                    try:
                        # Poll axes even without MOTION events.  This provides a
                        # liveness heartbeat and catches dongle loss promptly.
                        self._set_axes_from_device()
                    except Exception as exc:
                        _LOG.warning("gamepad read failed; reconnecting (%s)", exc)
                        self._close_device(disconnected=True)
                    if self.available and now - self._last_seen > self.liveness_timeout:
                        self._close_device(disconnected=True)
            time.sleep(0.01)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_device()
        # Do not call pygame.quit(): keyboard/viewer integrations may share
        # SDL in the same process.  Quitting only the joystick subsystem is
        # sufficient and is unavailable on older pygame releases anyway.
        if self._pygame is not None:
            try:
                self._pygame.joystick.quit()
            except Exception:
                pass


class InputManager:
    def __init__(self, backend: str, bus: CommandBus, joystick_axes: tuple[int, int, int], **gamepad_options: Any):
        self.backend = backend
        self.keyboard = KeyboardInput(bus)
        self.joystick = JoystickInput(bus, joystick_axes, **gamepad_options)

    def start(self, headless: bool) -> None:
        if self.backend in ("keyboard", "both"):
            if headless:
                self.keyboard.start_headless()
        if self.backend in ("joystick", "both"):
            self.joystick.start()

    def key_callback(self, key: int) -> None:
        if self.backend not in ("keyboard", "both"):
            return
        self.keyboard.handle_viewer_key(key)

    def close(self) -> None:
        self.keyboard.close()
        self.joystick.close()
