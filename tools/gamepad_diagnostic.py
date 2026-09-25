#!/usr/bin/env python3
"""Inspect pygame/SDL gamepad input without starting MuJoCo or the FSM.

SDL is initialized and polled on the process main thread.  Axis calibration
and shaping reuse the runtime's pure input components, so the displayed
command is the command that sim2sim would derive from the same stick values.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence


# Running ``python3 tools/gamepad_diagnostic.py`` puts only ``tools/`` at the
# front of sys.path.  Add the repository root so the diagnostic can reuse the
# exact, simulator-independent axis processing used by sim2sim.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from deploy.include.inputs import (  # noqa: E402
    AxisCalibrator,
    AxisProcessingSnapshot,
    CALIBRATION_RELEASE_MARGIN,
    CalibrationUpdate,
    GamepadAxisProcessor,
)


EXIT_OK = 0
EXIT_UNAVAILABLE = 2
POLL_HZ = 50.0
_CONTROLLER_AXIS_IDS = {
    "LEFTX": 0,
    "LEFTY": 1,
    "RIGHTX": 2,
    "RIGHTY": 3,
    "TRIGGERLEFT": 4,
    "TRIGGERRIGHT": 5,
}
_PERMISSION_HINT = (
    "Check that the controller is connected and that this user can read "
    "/dev/input/event* and /dev/input/js*. On Linux, inspect `ls -l "
    "/dev/input` and add the user to the appropriate input group or install "
    "a suitable udev rule, then log in again."
)


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    guid: str
    mode: str
    axes: int
    buttons: int
    hats: int

    def line(self) -> str:
        return (
            f"index={self.index} name={self.name!r} guid={self.guid or 'unknown'} "
            f"mode={self.mode} axes={self.axes} buttons={self.buttons} hats={self.hats}"
        )


@dataclass
class OpenDevice:
    info: DeviceInfo
    raw: Any
    controller: Any = None
    axis_state: dict[int, float] = field(default_factory=dict)
    button_state: dict[int, bool] = field(default_factory=dict)
    hat_state: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def instance_id(self) -> int | None:
        getter = getattr(self.raw, "get_instance_id", None)
        try:
            return int(getter()) if callable(getter) else None
        except Exception:
            return None

    def close(self) -> None:
        for device in (self.controller, self.raw):
            if device is None:
                continue
            try:
                device.quit()
            except Exception:
                pass


@dataclass(frozen=True)
class AxisBindings:
    left_x: int | str
    left_y: int | str
    yaw: int | str

    def values(self) -> tuple[int | str, int | str, int | str]:
        return self.left_x, self.left_y, self.yaw

    def line(self) -> str:
        return f"left_x={self.left_x} left_y={self.left_y} yaw={self.yaw}"


@dataclass(frozen=True)
class PhysicalSnapshot:
    axes: tuple[float, ...]
    buttons: tuple[int, ...]
    hats: tuple[tuple[int, int], ...]

    def line(self, *, prefix: str = "snapshot") -> str:
        return (
            f"{prefix} axes={self.axes} buttons={self.buttons} hats={self.hats}"
        )


@dataclass(frozen=True)
class DiagnosticSample:
    """One fixed-rate view of calibration and processed motion state."""

    mode: str
    bindings: AxisBindings
    raw: tuple[float, float, float]
    center: tuple[float, float, float] | None
    noise_mad: tuple[float, float, float] | None
    noise_peak_to_peak: tuple[float, float, float] | None
    calibrated: tuple[float, float, float] | None
    command: tuple[float, float, float]
    neutral: bool
    calibration_ready: bool
    calibration_reason: str | None
    enter_deadzone: float
    exit_deadzone: float
    center_limit: float


class AxisDiagnosticPipeline:
    """Small adapter around the runtime's calibration and shaping classes."""

    def __init__(
        self,
        *,
        mode: str,
        bindings: AxisBindings,
        deadzone: float,
        release_deadzone: float,
        calibration_seconds: float,
        center_limit: float = 0.15,
        now: float | None = None,
    ) -> None:
        self.mode = mode
        self.bindings = bindings
        self.enter_deadzone = float(deadzone)
        self.exit_deadzone = float(release_deadzone)
        self.center_limit = float(center_limit)
        self.calibrator = AxisCalibrator(
            min_duration=calibration_seconds,
            center_limit=self.center_limit,
            exit_deadzone=self.exit_deadzone,
        )
        self.processor = GamepadAxisProcessor(
            enter_deadzone=self.enter_deadzone,
            exit_deadzone=self.exit_deadzone,
        )
        self.calibrator.reset(time.monotonic() if now is None else now)
        self._center: tuple[float, float, float] | None = None
        self._calibration: CalibrationUpdate | None = None

    def update(
        self,
        raw: Sequence[float],
        *,
        now: float | None = None,
    ) -> DiagnosticSample:
        timestamp = time.monotonic() if now is None else float(now)
        raw_tuple = tuple(float(value) for value in raw)
        if len(raw_tuple) != 3 or not all(math.isfinite(value) for value in raw_tuple):
            raise ValueError("selected gamepad axes must contain three finite values")

        if self._center is None:
            self._calibration = self.calibrator.add_sample(raw_tuple, now=timestamp)
            if self._calibration.ready:
                self._center = self._calibration.center

        update = self._calibration
        if self._center is None:
            return DiagnosticSample(
                mode=self.mode,
                bindings=self.bindings,
                raw=raw_tuple,
                center=None if update is None else update.center,
                noise_mad=None if update is None else update.mad,
                noise_peak_to_peak=None if update is None else update.peak_to_peak,
                calibrated=None,
                command=(0.0, 0.0, 0.0),
                neutral=True,
                calibration_ready=False,
                calibration_reason=None if update is None else update.reason,
                enter_deadzone=self.enter_deadzone,
                exit_deadzone=self.exit_deadzone,
                center_limit=self.center_limit,
            )

        processed = self.processor.process(raw_tuple, self._center)
        return diagnostic_sample_from_processed(
            mode=self.mode,
            bindings=self.bindings,
            calibration=update,
            processed=processed,
            enter_deadzone=self.enter_deadzone,
            exit_deadzone=self.exit_deadzone,
            center_limit=self.center_limit,
        )


@dataclass
class MonitoredDevice:
    device: OpenDevice
    bindings: AxisBindings
    pipeline: AxisDiagnosticPipeline
    last_line: str | None = None


def _vector_text(value: Sequence[float] | None) -> str:
    if value is None:
        return "n/a"
    return "[" + ",".join(f"{float(component):+.5f}" for component in value) + "]"


def diagnostic_sample_from_processed(
    *,
    mode: str,
    bindings: AxisBindings,
    calibration: CalibrationUpdate | None,
    processed: AxisProcessingSnapshot,
    enter_deadzone: float,
    exit_deadzone: float,
    center_limit: float,
) -> DiagnosticSample:
    """Map processed stick axes to the runtime body-command sign convention."""

    lx, ly, yaw = processed.processed
    return DiagnosticSample(
        mode=mode,
        bindings=bindings,
        raw=processed.raw,
        center=processed.center,
        noise_mad=None if calibration is None else calibration.mad,
        noise_peak_to_peak=(
            None if calibration is None else calibration.peak_to_peak
        ),
        calibrated=processed.calibrated,
        command=(-ly, -lx, -yaw),
        neutral=processed.neutral,
        calibration_ready=True,
        calibration_reason=None,
        enter_deadzone=enter_deadzone,
        exit_deadzone=exit_deadzone,
        center_limit=center_limit,
    )


def format_diagnostic_sample(sample: DiagnosticSample) -> str:
    vx, vy, wz = sample.command
    reason = sample.calibration_reason or "none"
    return (
        f"poll mode={sample.mode} bindings=({sample.bindings.line()}) "
        f"enter={sample.enter_deadzone:.5f} exit={sample.exit_deadzone:.5f} "
        f"center_limit={sample.center_limit:.5f} "
        f"raw(lx,ly,yaw)={_vector_text(sample.raw)} "
        f"center={_vector_text(sample.center)} "
        f"noise_mad={_vector_text(sample.noise_mad)} "
        f"noise_peak_to_peak={_vector_text(sample.noise_peak_to_peak)} "
        f"calibrated={_vector_text(sample.calibrated)} "
        f"processed vx={vx:+.5f} vy={vy:+.5f} wz={wz:+.5f} "
        f"neutral={int(sample.neutral)} ready={int(sample.calibration_ready)} "
        f"reason={reason}"
    )


def _safe_call(device: Any, name: str, default: Any) -> Any:
    try:
        method = getattr(device, name, None)
        return method() if callable(method) else default
    except Exception:
        return default


def _controller_module() -> Any:
    try:
        import pygame._sdl2.controller as controller_mod
    except Exception:
        return None
    return controller_mod


def initialize_sdl() -> tuple[Any, Any]:
    """Initialize display/event, joystick and controller in required order."""

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # SDL reads this hint while its joystick subsystem is initialized.  Set it
    # before importing pygame so button/axis events keep flowing when the tiny
    # hidden diagnostics window does not have desktop focus.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    try:
        import pygame
    except Exception as exc:
        raise RuntimeError(f"pygame is unavailable: {exc}") from exc

    try:
        pygame.display.init()
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1), flags=int(getattr(pygame, "HIDDEN", 0)))
        pygame.joystick.init()
        controller_mod = _controller_module()
        if controller_mod is not None:
            get_init = getattr(controller_mod, "get_init", lambda: False)
            if not bool(get_init()):
                controller_mod.init()
        # Pump only after a display/event backend exists.  A successful pump
        # is the actual event-subsystem readiness check.
        pygame.event.pump()
    except Exception as exc:
        try:
            pygame.quit()
        except Exception:
            pass
        raise RuntimeError(f"pygame/SDL event initialization failed: {exc}") from exc
    return pygame, controller_mod


def _is_controller(controller_mod: Any, index: int) -> bool:
    if controller_mod is None:
        return False
    try:
        return bool(controller_mod.is_controller(index))
    except Exception:
        return False


def _device_info(pygame: Any, controller_mod: Any, index: int) -> DeviceInfo:
    raw = pygame.joystick.Joystick(index)
    try:
        raw.init()
        return DeviceInfo(
            index=index,
            name=str(_safe_call(raw, "get_name", "unnamed")),
            guid=str(_safe_call(raw, "get_guid", "")).lower(),
            mode="controller" if _is_controller(controller_mod, index) else "raw",
            axes=int(_safe_call(raw, "get_numaxes", 0)),
            buttons=int(_safe_call(raw, "get_numbuttons", 0)),
            hats=int(_safe_call(raw, "get_numhats", 0)),
        )
    finally:
        try:
            raw.quit()
        except Exception:
            pass


def enumerate_devices(pygame: Any, controller_mod: Any) -> list[DeviceInfo]:
    try:
        count = int(pygame.joystick.get_count())
    except Exception as exc:
        raise RuntimeError(f"cannot enumerate SDL joysticks: {exc}") from exc

    devices: list[DeviceInfo] = []
    errors: list[str] = []
    for index in range(count):
        try:
            devices.append(_device_info(pygame, controller_mod, index))
        except Exception as exc:
            errors.append(f"index {index}: {exc}")
    if count and not devices:
        raise RuntimeError("detected joystick slots but none could be opened: " + "; ".join(errors))
    return devices


def select_device(
    devices: Sequence[DeviceInfo],
    *,
    index: int | None,
    name: str | None,
    guid: str | None,
) -> DeviceInfo | None:
    name_filter = name.casefold() if name else None
    guid_filter = guid.casefold() if guid else None
    for device in devices:
        if index is not None and device.index != index:
            continue
        if name_filter and name_filter not in device.name.casefold():
            continue
        if guid_filter and guid_filter not in device.guid.casefold():
            continue
        return device
    return None


def _axis_identifier(value: str) -> int | str:
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        normalized = value.upper().replace("_", "")
        if normalized not in _CONTROLLER_AXIS_IDS:
            raise argparse.ArgumentTypeError(
                f"unknown controller axis {value!r}; use an integer or one of "
                + ", ".join(_CONTROLLER_AXIS_IDS)
            )
        return normalized


def resolve_axis_bindings(
    mode: str,
    axis_count: int,
    *,
    left_x: int | str | None = None,
    left_y: int | str | None = None,
    yaw: int | str | None = None,
) -> AxisBindings:
    """Resolve and validate the three independent motion axes."""

    if mode not in ("raw", "controller"):
        raise ValueError(f"unsupported gamepad mode: {mode!r}")
    defaults = (
        AxisBindings("LEFTX", "LEFTY", "RIGHTX")
        if mode == "controller"
        else AxisBindings(0, 1, 3)
    )
    requested = AxisBindings(
        defaults.left_x if left_x is None else left_x,
        defaults.left_y if left_y is None else left_y,
        defaults.yaw if yaw is None else yaw,
    )

    resolved: list[int] = []
    display: list[int | str] = []
    for value in requested.values():
        if mode == "raw":
            if not isinstance(value, int):
                raise ValueError(f"raw joystick axes must be integers, got {value!r}")
            axis_id = value
            shown: int | str = value
        else:
            if isinstance(value, str):
                semantic = value.upper().replace("_", "")
                if semantic not in _CONTROLLER_AXIS_IDS:
                    raise ValueError(f"unknown SDL controller axis: {value!r}")
                axis_id = _CONTROLLER_AXIS_IDS[semantic]
                shown = semantic
            else:
                axis_id = int(value)
                shown = axis_id
        if not 0 <= axis_id < axis_count:
            raise ValueError(
                f"axis {shown!r} resolves to {axis_id}, outside device range "
                f"[0, {axis_count})"
            )
        resolved.append(axis_id)
        display.append(shown)
    if len(set(resolved)) != 3:
        raise ValueError(
            "left-x, left-y and yaw axes must be distinct; resolved indices="
            + repr(tuple(resolved))
        )
    return AxisBindings(*display)


def select_mode(info: DeviceInfo, *, force_raw: bool, force_controller: bool) -> DeviceInfo:
    if force_raw and force_controller:
        raise ValueError("--raw and --controller are mutually exclusive")
    if force_controller and info.mode != "controller":
        raise ValueError(
            f"device {info.name!r} has no SDL Controller mapping; use --raw"
        )
    if force_raw:
        return replace(info, mode="raw")
    if force_controller:
        return replace(info, mode="controller")
    return info


def open_device(pygame: Any, controller_mod: Any, info: DeviceInfo) -> OpenDevice:
    controller = None
    raw = None
    try:
        if info.mode == "controller" and controller_mod is not None:
            controller = controller_mod.Controller(info.index)
            controller.init()
            raw = controller.as_joystick()
            if raw is None:
                raise RuntimeError("SDL Controller did not expose its joystick state")
        else:
            raw = pygame.joystick.Joystick(info.index)
            raw.init()
        return OpenDevice(info=info, raw=raw, controller=controller)
    except Exception:
        for device in (controller, raw):
            if device is not None:
                try:
                    device.quit()
                except Exception:
                    pass
        raise


def read_physical_snapshot(device: OpenDevice) -> PhysicalSnapshot:
    """Read every physical control, including values that emitted no event."""

    raw = device.raw
    try:
        axes = tuple(
            _normalized_axis(raw.get_axis(index))
            for index in range(int(raw.get_numaxes()))
        )
        buttons = tuple(
            int(bool(raw.get_button(index)))
            for index in range(int(raw.get_numbuttons()))
        )
        hats = tuple(
            tuple(int(value) for value in raw.get_hat(index))
            for index in range(int(raw.get_numhats()))
        )
    except Exception as exc:
        raise RuntimeError(f"cannot read complete gamepad snapshot: {exc}") from exc
    if not all(math.isfinite(value) for value in axes):
        raise RuntimeError("gamepad snapshot contains NaN or infinity")
    return PhysicalSnapshot(axes=axes, buttons=buttons, hats=hats)


def _controller_axis_token(controller_mod: Any, binding: int | str) -> Any:
    if isinstance(binding, int):
        return binding
    semantic = binding.upper().replace("_", "")
    enum = getattr(controller_mod, "CONTROLLER_AXIS", None) if controller_mod else None
    if enum is not None:
        resolved = getattr(enum, semantic, None)
        if resolved is not None:
            return resolved
    return _CONTROLLER_AXIS_IDS[semantic]


def read_selected_axes(
    device: OpenDevice,
    controller_mod: Any,
    bindings: AxisBindings,
) -> tuple[float, float, float]:
    source = device.controller if device.info.mode == "controller" else device.raw
    if source is None:
        raise RuntimeError(f"{device.info.mode} input handle is unavailable")
    values: list[float] = []
    for binding in bindings.values():
        token = (
            _controller_axis_token(controller_mod, binding)
            if device.info.mode == "controller"
            else int(binding)
        )
        try:
            value = _normalized_axis(source.get_axis(token))
        except Exception as exc:
            raise RuntimeError(f"cannot read selected axis {binding!r}: {exc}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"selected axis {binding!r} is non-finite")
        values.append(value)
    return values[0], values[1], values[2]


def print_initial_snapshot(device: OpenDevice, bindings: AxisBindings) -> PhysicalSnapshot:
    snapshot = read_physical_snapshot(device)
    print(
        f"initial mode={device.info.mode} bindings=({bindings.line()}) "
        f"axes={snapshot.axes} buttons={snapshot.buttons} hats={snapshot.hats}",
        flush=True,
    )
    return snapshot


def _event_type(pygame: Any, name: str) -> int | None:
    value = getattr(pygame, name, None)
    return int(value) if value is not None else None


def _event_instance(event: Any) -> int | None:
    value = getattr(event, "instance_id", getattr(event, "joy", None))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _belongs_to(event: Any, device: OpenDevice) -> bool:
    event_instance = _event_instance(event)
    return event_instance is None or device.instance_id is None or event_instance == device.instance_id


def _normalized_axis(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("axis value is NaN or infinity")
    if abs(value) > 1.5:
        value /= 32767.0 if value >= 0.0 else 32768.0
    return max(-1.0, min(1.0, value))


def print_state_change(pygame: Any, event: Any, device: OpenDevice) -> bool:
    """Print one selected-device state transition; return True if handled."""

    typ = getattr(event, "type", None)
    if not _belongs_to(event, device):
        return False

    if device.info.mode == "controller":
        axis_types = {_event_type(pygame, "CONTROLLERAXISMOTION")}
        button_down_types = {_event_type(pygame, "CONTROLLERBUTTONDOWN")}
        button_up_types = {_event_type(pygame, "CONTROLLERBUTTONUP")}
    else:
        axis_types = {_event_type(pygame, "JOYAXISMOTION")}
        button_down_types = {_event_type(pygame, "JOYBUTTONDOWN")}
        button_up_types = {_event_type(pygame, "JOYBUTTONUP")}
    hat_type = _event_type(pygame, "JOYHATMOTION")

    if typ in axis_types and typ is not None:
        index = int(event.axis)
        value = _normalized_axis(event.value)
        if device.axis_state.get(index) == value:
            return False
        device.axis_state[index] = value
        print(f"axis index={index} value={value:+.5f}", flush=True)
        return True
    if typ in button_down_types and typ is not None:
        index = int(event.button)
        if device.button_state.get(index) is True:
            return False
        device.button_state[index] = True
        print(f"button index={index} pressed=1", flush=True)
        return True
    if typ in button_up_types and typ is not None:
        index = int(event.button)
        if device.button_state.get(index) is False:
            return False
        device.button_state[index] = False
        print(f"button index={index} pressed=0", flush=True)
        return True
    if typ == hat_type and typ is not None:
        index = int(event.hat)
        value = tuple(int(component) for component in event.value)
        if device.hat_state.get(index) == value:
            return False
        device.hat_state[index] = value
        print(f"hat index={index} value={value}", flush=True)
        return True
    return False


def _open_selected(
    pygame: Any,
    controller_mod: Any,
    args: argparse.Namespace,
) -> MonitoredDevice | None:
    devices = enumerate_devices(pygame, controller_mod)
    info = select_device(
        devices,
        index=args.index,
        name=args.name,
        guid=args.guid,
    )
    if info is None:
        return None
    info = select_mode(
        info,
        force_raw=bool(args.raw),
        force_controller=bool(args.controller),
    )
    bindings = resolve_axis_bindings(
        info.mode,
        info.axes,
        left_x=args.left_x_axis,
        left_y=args.left_y_axis,
        yaw=args.yaw_axis,
    )
    device = open_device(pygame, controller_mod, info)
    try:
        print("connected " + info.line(), flush=True)
        print_initial_snapshot(device, bindings)
        now = time.monotonic()
        pipeline = AxisDiagnosticPipeline(
            mode=info.mode,
            bindings=bindings,
            deadzone=args.deadzone,
            release_deadzone=args.release_deadzone,
            calibration_seconds=args.calibration_seconds,
            center_limit=args.center_limit,
            now=now,
        )
        monitored = MonitoredDevice(device, bindings, pipeline)
        _poll_selected(monitored, controller_mod, now=now, force=True)
        return monitored
    except Exception:
        device.close()
        raise


def _poll_selected(
    selected: MonitoredDevice,
    controller_mod: Any,
    *,
    now: float,
    force: bool = False,
) -> DiagnosticSample:
    raw = read_selected_axes(selected.device, controller_mod, selected.bindings)
    sample = selected.pipeline.update(raw, now=now)
    line = format_diagnostic_sample(sample)
    if force or line != selected.last_line:
        print(line, flush=True)
        selected.last_line = line
    return sample


def monitor(pygame: Any, controller_mod: Any, args: argparse.Namespace) -> int:
    selected = _open_selected(pygame, controller_mod, args)
    if selected is None:
        print("No matching gamepad could be opened. " + _PERMISSION_HINT, file=sys.stderr)
        return EXIT_UNAVAILABLE

    known_devices = {device.index: device for device in enumerate_devices(pygame, controller_mod)}
    start_time = time.monotonic()
    next_poll = start_time
    poll_period = 1.0 / POLL_HZ
    added_types = {
        value
        for value in (
            _event_type(pygame, "JOYDEVICEADDED"),
            _event_type(pygame, "CONTROLLERDEVICEADDED"),
        )
        if value is not None
    }
    removed_types = {
        value
        for value in (
            _event_type(pygame, "JOYDEVICEREMOVED"),
            _event_type(pygame, "CONTROLLERDEVICEREMOVED"),
        )
        if value is not None
    }
    try:
        while args.duration == 0.0 or time.monotonic() - start_time < args.duration:
            now = time.monotonic()
            if now < next_poll:
                time.sleep(next_poll - now)
                now = time.monotonic()
            for event in pygame.event.get():
                typ = getattr(event, "type", None)
                if typ in removed_types or typ in added_types:
                    current_devices = {
                        device.index: device
                        for device in enumerate_devices(pygame, controller_mod)
                    }
                    for index in sorted(known_devices.keys() - current_devices.keys()):
                        print("hotplug disconnected " + known_devices[index].line(), flush=True)
                    for index in sorted(current_devices.keys() - known_devices.keys()):
                        print("hotplug added " + current_devices[index].line(), flush=True)

                    selected_removed = (
                        selected is not None
                        and typ in removed_types
                        and _belongs_to(event, selected.device)
                    )
                    if selected_removed:
                        selected.device.close()
                        selected = None
                    known_devices = current_devices
                    if selected is None:
                        try:
                            selected = _open_selected(pygame, controller_mod, args)
                        except Exception as exc:
                            print(f"hotplug open failed: {exc}", file=sys.stderr, flush=True)
                    continue
                if selected is not None:
                    print_state_change(pygame, event, selected.device)
            if selected is not None:
                _poll_selected(selected, controller_mod, now=now)
            next_poll += poll_period
            if next_poll <= now:
                next_poll = now + poll_period
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        if selected is not None:
            selected.device.close()
    return EXIT_OK


def _non_negative_duration(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            "duration must be finite and non-negative (0 means run until Ctrl-C)"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or monitor pygame/SDL gamepad input without launching sim2sim.",
    )
    parser.add_argument("--list", action="store_true", help="list visible devices and exit")
    parser.add_argument("--index", type=int, help="select one SDL joystick index")
    parser.add_argument("--name", help="select by case-insensitive name substring")
    parser.add_argument("--guid", help="select by case-insensitive GUID substring")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--raw", action="store_true", help="force pygame raw Joystick mode")
    mode.add_argument(
        "--controller",
        action="store_true",
        help="require SDL semantic Controller mode",
    )
    parser.add_argument(
        "--left-x-axis",
        type=_axis_identifier,
        help="raw index or SDL semantic axis for lateral stick motion",
    )
    parser.add_argument(
        "--left-y-axis",
        type=_axis_identifier,
        help="raw index or SDL semantic axis for fore/aft stick motion",
    )
    parser.add_argument(
        "--yaw-axis",
        type=_axis_identifier,
        help="raw index or SDL semantic axis for yaw",
    )
    parser.add_argument(
        "--deadzone",
        "--enter-deadzone",
        dest="deadzone",
        type=float,
        default=0.18,
        help="motion-entry deadzone after center calibration (default: 0.18)",
    )
    parser.add_argument(
        "--release-deadzone",
        "--exit-deadzone",
        dest="release_deadzone",
        type=float,
        default=0.15,
        help="neutral/release exit deadzone for hysteresis (default: 0.15)",
    )
    parser.add_argument(
        "--center-limit",
        type=float,
        default=0.15,
        help="largest calibratable absolute axis center (default: 0.15)",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=_non_negative_duration,
        default=0.5,
        help="minimum hands-off center-calibration time (default: 0.5)",
    )
    parser.add_argument(
        "--duration",
        type=_non_negative_duration,
        default=0.0,
        metavar="SECONDS",
        help="monitor duration; 0 (default) runs until Ctrl-C",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.index is not None and args.index < 0:
        print("--index must be non-negative", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if not (
        math.isfinite(args.deadzone)
        and math.isfinite(args.release_deadzone)
        and 0.0 <= args.release_deadzone < args.deadzone < 1.0
    ):
        print(
            "deadzone values must satisfy 0 <= release-deadzone < deadzone < 1",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE
    if not math.isfinite(args.center_limit) or not 0.0 < args.center_limit < 1.0:
        print("center-limit must be finite and in (0, 1)", file=sys.stderr)
        return EXIT_UNAVAILABLE
    release_at_limit = args.center_limit / (1.0 + args.center_limit)
    if not release_at_limit + CALIBRATION_RELEASE_MARGIN < args.release_deadzone:
        print(
            "unsafe center-limit/exit-deadzone combination: "
            "center_limit/(1+center_limit) + release_margin must be below "
            "exit-deadzone",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE

    try:
        pygame, controller_mod = initialize_sdl()
    except RuntimeError as exc:
        print(f"ERROR: {exc}. {_PERMISSION_HINT}", file=sys.stderr)
        return EXIT_UNAVAILABLE

    try:
        if args.list:
            devices = enumerate_devices(pygame, controller_mod)
            if not devices:
                print("No gamepad detected. " + _PERMISSION_HINT, file=sys.stderr)
                return EXIT_UNAVAILABLE
            for device in devices:
                print(device.line())
            return EXIT_OK
        return monitor(pygame, controller_mod, args)
    except Exception as exc:
        print(f"ERROR: cannot access gamepad: {exc}. {_PERMISSION_HINT}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    finally:
        try:
            pygame.joystick.quit()
            pygame.display.quit()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
