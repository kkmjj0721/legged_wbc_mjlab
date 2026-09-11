from __future__ import annotations

import select
import os
import sys
import termios
import threading
import time
import tty
from typing import Any

import numpy as np


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
            for name in ("zero_velocity", "stand", "get_down", "reset", "camera", "quit")
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
    def __init__(self, bus: CommandBus, axes: tuple[int, int, int]):
        self.bus = bus
        self.axes = axes
        self.available = False
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.joystick.init()
            if not pygame.get_init():
                pygame.display.init()
            if pygame.joystick.get_count() == 0:
                pygame.quit()
                return
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
        except Exception:
            return
        self.available = True
        self._running = True
        self._thread = threading.Thread(target=self._read, args=(pygame, joystick), name="sim2sim-joystick", daemon=True)
        self._thread.start()

    def _read(self, pygame: Any, joystick: Any) -> None:
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    values = [joystick.get_axis(axis) if 0 <= axis < joystick.get_numaxes() else 0.0 for axis in self.axes]
                    self.bus.set_axes(-values[1], values[0], values[2])
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:
                        self.bus.request("stand")
                    elif event.button in (1, 2):
                        self.bus.request("passive")
                    elif event.button in (7, 9):
                        self.bus.request("reset")
                    elif event.button == 6:
                        self.bus.request("quit")
                elif event.type == pygame.JOYDEVICEREMOVED:
                    self.bus.stop()
                    self._running = False
            time.sleep(0.005)

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.available:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass


class InputManager:
    def __init__(self, backend: str, bus: CommandBus, joystick_axes: tuple[int, int, int]):
        self.backend = backend
        self.keyboard = KeyboardInput(bus)
        self.joystick = JoystickInput(bus, joystick_axes)

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
