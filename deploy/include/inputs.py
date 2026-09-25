from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable, Sequence

import numpy as np


_LOG = logging.getLogger(__name__)


AxisVector = tuple[float, float, float]


# Leave a measurable gap between the largest release residual that calibration
# may accept and the Schmitt trigger's exit threshold.  This ensures a stick
# released to physical raw zero can leave an already-active command despite
# quantization and noise.
CALIBRATION_RELEASE_MARGIN = 0.005


def _axis_vector(values: Sequence[float], name: str) -> np.ndarray:
    """Return one finite three-axis vector without clipping invalid data."""

    try:
        vector = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain three finite numeric values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite numeric values")
    return vector


def _tuple3(values: Sequence[float]) -> AxisVector:
    vector = _axis_vector(values, "axis vector")
    return float(vector[0]), float(vector[1]), float(vector[2])


@dataclass(frozen=True)
class CommandSnapshot:
    """Atomic view of the velocity command and its arbitration state."""

    command: np.ndarray
    owner: str | None
    stop_pending: bool
    timestamp: float


@dataclass(frozen=True)
class CalibrationUpdate:
    """Result of adding one raw sample to an :class:`AxisCalibrator`."""

    ready: bool
    stable: bool
    sample_count: int
    center: AxisVector | None
    mad: AxisVector | None
    peak_to_peak: AxisVector | None
    reason: str | None


class AxisCalibrator:
    """Robust, time-gated center calibration for three gamepad axes.

    A rolling window must contain at least 25 stable samples *and* the
    calibration interval must have elapsed.  Stable but implausible centers
    are rejected rather than normalized away; this catches a trigger exposed
    as RIGHTX and a stick held near an end stop during connection.
    """

    def __init__(
        self,
        min_samples: int = 25,
        min_duration: float = 0.25,
        max_mad: float = 0.01,
        max_peak_to_peak: float = 0.04,
        center_limit: float = 0.15,
        extreme_limit: float = 0.90,
        clock: Callable[[], float] | None = None,
        enter_deadzone: float | None = None,
        release_margin: float = CALIBRATION_RELEASE_MARGIN,
        exit_deadzone: float | None = None,
    ):
        self.min_samples = int(min_samples)
        self.min_duration = float(min_duration)
        self.max_mad = float(max_mad)
        self.max_peak_to_peak = float(max_peak_to_peak)
        self.center_limit = float(center_limit)
        self.extreme_limit = float(extreme_limit)
        if exit_deadzone is not None and enter_deadzone is not None:
            if not math.isclose(
                float(exit_deadzone),
                float(enter_deadzone),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError(
                    "calibration exit_deadzone conflicts with deprecated "
                    "enter_deadzone alias"
                )
        resolved_exit = (
            0.15
            if exit_deadzone is None and enter_deadzone is None
            else enter_deadzone if exit_deadzone is None
            else exit_deadzone
        )
        self.exit_deadzone = float(resolved_exit)
        # Compatibility attribute for callers from the first calibration
        # implementation.  It now denotes the release/exit safety threshold.
        self.enter_deadzone = self.exit_deadzone
        self.release_margin = float(release_margin)
        if self.min_samples < 25:
            raise ValueError("gamepad calibration requires at least 25 samples")
        for value, name in (
            (self.min_duration, "min_duration"),
            (self.max_mad, "max_mad"),
            (self.max_peak_to_peak, "max_peak_to_peak"),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"calibration {name} must be finite and non-negative")
        if not np.isfinite(self.center_limit) or not 0.0 < self.center_limit < 1.0:
            raise ValueError("calibration center_limit must be in (0, 1)")
        if (
            not np.isfinite(self.extreme_limit)
            or not self.center_limit < self.extreme_limit <= 1.0
        ):
            raise ValueError(
                "calibration extreme_limit must be greater than center_limit and at most 1"
            )
        if not np.isfinite(self.exit_deadzone) or not 0.0 < self.exit_deadzone < 1.0:
            raise ValueError("calibration exit_deadzone must be in (0, 1)")
        if not np.isfinite(self.release_margin) or self.release_margin <= 0.0:
            raise ValueError("calibration release_margin must be finite and positive")
        # ``center_limit`` is a per-axis/yaw bound.  Its worst scalar release
        # must be below the exit threshold; the actual two-axis translation
        # center is checked radially in add_sample() because rejecting every
        # possible diagonal at configuration time would unnecessarily shrink
        # the usable single-axis drift range.
        release_at_limit = self.center_limit / (1.0 + self.center_limit)
        if not release_at_limit + self.release_margin < self.exit_deadzone:
            raise ValueError(
                "unsafe calibration center_limit/deadzone combination: "
                "center_limit/(1+center_limit) + release_margin must be "
                "strictly below exit_deadzone "
                f"({release_at_limit:.6f} + {self.release_margin:.6f} "
                f"!< {self.exit_deadzone:.6f}); diagnose the physical center, "
                "then lower center_limit or explicitly increase both deadzones"
            )
        self._clock = clock or time.monotonic
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._started_at = 0.0
        self._center: np.ndarray | None = None
        self.reset()

    @property
    def center(self) -> AxisVector | None:
        return None if self._center is None else _tuple3(self._center)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def reset(self, now: float | None = None) -> None:
        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("calibration timestamp must be finite")
        self._samples.clear()
        self._started_at = timestamp
        self._center = None

    def add_sample(
        self,
        raw: Sequence[float],
        now: float | None = None,
    ) -> CalibrationUpdate:
        sample = _axis_vector(raw, "raw gamepad axes")
        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("calibration timestamp must be finite")
        if self._samples and timestamp < self._samples[-1][0]:
            raise ValueError("calibration timestamps must be monotonic")
        if not self._samples:
            self._started_at = timestamp
        self._samples.append((timestamp, sample.copy()))
        # Keep the smallest recent suffix that can still satisfy both gates.
        # At a fast poll rate this retains more than ``min_samples`` so the
        # stable window itself spans ``min_duration``; at a slow rate it keeps
        # at least ``min_samples`` even when that spans longer than requested.
        while (
            len(self._samples) > self.min_samples
            and timestamp - self._samples[1][0] >= self.min_duration
        ):
            self._samples.popleft()
        samples = np.stack(tuple(sample for _, sample in self._samples), axis=0)
        center = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - center), axis=0)
        peak_to_peak = np.ptp(samples, axis=0)
        enough_samples = len(self._samples) >= self.min_samples
        window_started_at = self._samples[0][0]
        enough_time = timestamp - window_started_at >= self.min_duration
        stable = bool(
            enough_samples
            and np.all(mad <= self.max_mad)
            and np.all(peak_to_peak <= self.max_peak_to_peak)
        )
        center_tuple = _tuple3(center)
        mad_tuple = _tuple3(mad)
        peak_tuple = _tuple3(peak_to_peak)

        if not enough_samples:
            return CalibrationUpdate(
                False,
                False,
                len(self._samples),
                center_tuple,
                mad_tuple,
                peak_tuple,
                f"collecting stable samples ({len(self._samples)}/{self.min_samples})",
            )
        if not enough_time:
            remaining = max(0.0, self.min_duration - (timestamp - window_started_at))
            return CalibrationUpdate(
                False,
                stable,
                len(self._samples),
                center_tuple,
                mad_tuple,
                peak_tuple,
                f"waiting for calibration interval ({remaining:.3f}s remaining)",
            )
        if not stable:
            return CalibrationUpdate(
                False,
                False,
                len(self._samples),
                center_tuple,
                mad_tuple,
                peak_tuple,
                "axes are moving or noisy; release the sticks and keep them still",
            )

        largest_center = float(np.max(np.abs(center)))
        if largest_center >= self.extreme_limit:
            reason = (
                "stable axis center is near +/-1; suspected trigger or wrong axis mapping"
            )
        elif largest_center > self.center_limit:
            reason = (
                "stable axis center exceeds the safe limit; release the stick "
                "(held-at-connect suspected) or check axis mapping; diagnose "
                "before explicitly increasing both center_limit and deadzone"
            )
        else:
            # If a user held a stick during calibration, the held position can
            # look perfectly stable.  Check what the normalized value would be
            # after the physical stick is released to raw zero.  Translation
            # uses the same radial metric as GamepadAxisProcessor, while yaw is
            # independent.  The configured scalar invariant above is a cheap
            # constructor-time bound; this vector check is intentionally
            # stronger for diagonal translation offsets.
            release_residual = np.abs(center) / (1.0 + np.abs(center))
            translation_release = math.hypot(
                float(release_residual[0]), float(release_residual[1])
            )
            yaw_release = float(release_residual[2])
            if (
                translation_release + self.release_margin >= self.exit_deadzone
                or yaw_release + self.release_margin >= self.exit_deadzone
            ):
                reason = (
                    "stable center is unsafe after release; held-at-connect suspected "
                    f"(release residual translation={translation_release:.3f}, "
                    f"yaw={yaw_release:.3f}, exit_deadzone={self.exit_deadzone:.3f})"
                )
            else:
                self._center = center.copy()
                return CalibrationUpdate(
                    True,
                    True,
                    len(self._samples),
                    center_tuple,
                    mad_tuple,
                    peak_tuple,
                    None,
                )

        # A rejected stable window must not become control input.  Start a new
        # full window so releasing a trigger/stick can recover without restart.
        rejected_count = len(self._samples)
        self.reset(timestamp)
        return CalibrationUpdate(
            False,
            True,
            rejected_count,
            center_tuple,
            mad_tuple,
            peak_tuple,
            reason,
        )


@dataclass(frozen=True)
class AxisProcessingSnapshot:
    """Raw, center-corrected, and hysteresis-shaped axis state."""

    raw: AxisVector
    center: AxisVector
    calibrated: AxisVector
    processed: AxisVector
    translation_active: bool
    yaw_active: bool
    neutral: bool


class GamepadAxisProcessor:
    """Asymmetric center normalization plus independent Schmitt deadzones."""

    def __init__(
        self,
        enter_deadzone: float = 0.18,
        exit_deadzone: float = 0.15,
        yaw_rescale: float = 1.0,
    ):
        self.enter_deadzone = float(enter_deadzone)
        self.exit_deadzone = float(exit_deadzone)
        self.yaw_rescale = float(yaw_rescale)
        if (
            not np.isfinite(self.exit_deadzone)
            or not np.isfinite(self.enter_deadzone)
            or not 0.0 <= self.exit_deadzone <= self.enter_deadzone < 1.0
        ):
            raise ValueError(
                "gamepad deadzones must satisfy 0 <= exit <= enter < 1"
            )
        if not np.isfinite(self.yaw_rescale) or self.yaw_rescale <= 0.0:
            raise ValueError("gamepad yaw_rescale must be a finite positive number")
        self.translation_active = False
        self.yaw_active = False

    def reset(self) -> None:
        self.translation_active = False
        self.yaw_active = False

    @staticmethod
    def normalize(raw: Sequence[float], center: Sequence[float]) -> np.ndarray:
        raw_vector = _axis_vector(raw, "raw gamepad axes")
        center_vector = _axis_vector(center, "gamepad axis center")
        if np.any(np.abs(center_vector) >= 1.0):
            raise ValueError("gamepad axis center must lie strictly inside (-1, 1)")
        delta = raw_vector - center_vector
        positive_range = 1.0 - center_vector
        negative_range = 1.0 + center_vector
        normalized = np.where(delta >= 0.0, delta / positive_range, delta / negative_range)
        return np.clip(normalized, -1.0, 1.0)

    def is_neutral(self, calibrated: Sequence[float]) -> bool:
        values = _axis_vector(calibrated, "calibrated gamepad axes")
        return bool(
            math.hypot(float(values[0]), float(values[1])) <= self.exit_deadzone
            and abs(float(values[2])) <= self.exit_deadzone
        )

    @staticmethod
    def _shape(magnitude: float, threshold: float) -> float:
        return float(np.clip((magnitude - threshold) / (1.0 - threshold), 0.0, 1.0))

    def process(
        self,
        raw: Sequence[float],
        center: Sequence[float],
    ) -> AxisProcessingSnapshot:
        raw_vector = _axis_vector(raw, "raw gamepad axes")
        center_vector = _axis_vector(center, "gamepad axis center")
        calibrated = self.normalize(raw_vector, center_vector)

        lx, ly, yaw = (float(value) for value in calibrated)
        translation_magnitude = math.hypot(lx, ly)
        if self.translation_active:
            if translation_magnitude <= self.exit_deadzone:
                self.translation_active = False
        elif translation_magnitude > self.enter_deadzone or (
            self.enter_deadzone > 0.0
            and translation_magnitude == self.enter_deadzone
        ):
            self.translation_active = True

        if self.translation_active:
            clipped_magnitude = min(translation_magnitude, 1.0)
            shaped_magnitude = self._shape(clipped_magnitude, self.exit_deadzone)
            scale = shaped_magnitude / max(translation_magnitude, 1.0e-12)
            processed_lx, processed_ly = lx * scale, ly * scale
        else:
            processed_lx = processed_ly = 0.0

        yaw_magnitude = abs(yaw)
        if self.yaw_active:
            if yaw_magnitude <= self.exit_deadzone:
                self.yaw_active = False
        elif yaw_magnitude > self.enter_deadzone or (
            self.enter_deadzone > 0.0 and yaw_magnitude == self.enter_deadzone
        ):
            self.yaw_active = True
        if self.yaw_active:
            processed_yaw = math.copysign(
                self._shape(min(yaw_magnitude, 1.0), self.exit_deadzone),
                yaw,
            )
            processed_yaw = float(
                np.clip(processed_yaw * self.yaw_rescale, -1.0, 1.0)
            )
        else:
            processed_yaw = 0.0

        processed = (float(processed_lx), float(processed_ly), processed_yaw)
        return AxisProcessingSnapshot(
            raw=_tuple3(raw_vector),
            center=_tuple3(center_vector),
            calibrated=_tuple3(calibrated),
            processed=processed,
            translation_active=self.translation_active,
            yaw_active=self.yaw_active,
            neutral=self.is_neutral(calibrated),
        )


@dataclass(frozen=True)
class GamepadDebugSnapshot:
    timestamp: float
    mode: str
    axis_bindings: tuple[int, int, int] | None
    raw: AxisVector | None
    center: AxisVector | None
    calibrated: AxisVector | None
    processed: AxisVector | None
    neutral: bool
    calibration_ready: bool
    calibration_reason: str | None
    calibration_mad: AxisVector | None
    calibration_peak_to_peak: AxisVector | None
    last_successful_poll: float | None
    owner: str | None
    command: AxisVector
    stop_pending: bool


class CommandBus:
    """Thread-safe command/event bus shared by keyboard and gamepad inputs."""

    _EVENT_ALIASES = {"passive": "get_down", "camera_toggle": "camera"}
    _STOP_EVENTS = {
        "zero_velocity",
        "get_down",
        "reset",
        "quit",
        "disable_rl",
        "estop",
    }
    _NEUTRAL_REARM_SOURCES = frozenset({"gamepad"})

    def __init__(self, limits: np.ndarray, deadzone: float):
        limits = np.asarray(limits, dtype=np.float64)
        if limits.shape != (3,) or not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
            raise ValueError("command limits must contain three finite non-negative values")
        deadzone = float(deadzone)
        if not np.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
            raise ValueError("command deadzone must be in [0, 1)")
        self._limits = limits.copy()
        self._deadzone = deadzone
        self._lock = threading.Lock()
        self._command = np.zeros(3, dtype=np.float64)
        self._active_source: str | None = None
        self._stop_pending = False
        self._stop_generation = 0
        self._neutral_rearm_required: set[str] = set()
        self._timestamp = time.monotonic()
        self._events = {
            name: False
            for name in (
                "zero_velocity", "stand", "get_down", "reset", "camera", "quit",
                "enable_rl", "disable_rl", "estop",
            )
        }

    def _clear_command_locked(self, *, require_gamepad_neutral: bool) -> None:
        self._command.fill(0.0)
        self._active_source = None
        if require_gamepad_neutral:
            self._neutral_rearm_required.update(self._NEUTRAL_REARM_SOURCES)
        self._timestamp = time.monotonic()

    def _begin_stop_locked(self) -> int:
        self._clear_command_locked(require_gamepad_neutral=True)
        self._stop_generation += 1
        self._stop_pending = True
        return self._stop_generation

    def _reject_nonfinite(self, message: str) -> None:
        # A malformed local API call is rejected rather than converted to a
        # saturated command.  Clearing an existing command is the fail-safe
        # behavior, but unlike a physical device failure this does not invent
        # an estop/quit protocol event for the caller.
        with self._lock:
            self._clear_command_locked(require_gamepad_neutral=True)
        raise ValueError(message)

    def _acknowledge_stop_locked(self, generation: int | None = None) -> bool:
        if not self._stop_pending:
            return False
        if generation is None or generation != self._stop_generation:
            return False
        if any(self._events[name] for name in self._STOP_EVENTS):
            return False
        self._stop_pending = False
        self._timestamp = time.monotonic()
        return True

    def set_axes(self, vx: float, vy: float, wz: float, *, source: str | None = None) -> None:
        """Set normalized axes after a continuous, rescaled deadzone.

        ``source`` is optional for compatibility with the original public API.
        Source-aware callers gain input arbitration: a neutral source does not
        overwrite another source, while returning the current source to neutral
        emits exactly one zero command and releases ownership.
        """

        try:
            requested = np.asarray((vx, vy, wz), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            self._reject_nonfinite("normalized axes must contain finite numeric values")
        if not np.all(np.isfinite(requested)):
            self._reject_nonfinite("normalized axes must contain finite values")

        def shape(value: float) -> float:
            value = float(np.clip(value, -1.0, 1.0))
            magnitude = abs(value)
            if magnitude <= self._deadzone:
                return 0.0
            return math.copysign((magnitude - self._deadzone) / (1.0 - self._deadzone), value)

        shaped = np.asarray(tuple(shape(value) for value in requested), dtype=np.float64)
        self.set_command(
            *(shaped * self._limits),
            source=source,
            active=bool(np.any(shaped != 0.0)),
        )

    def command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def snapshot(self) -> CommandSnapshot:
        """Return command, owner, STOP state, and update time under one lock."""

        with self._lock:
            return CommandSnapshot(
                command=self._command.copy(),
                owner=self._active_source,
                stop_pending=self._stop_pending,
                timestamp=self._timestamp,
            )

    @property
    def active_source(self) -> str | None:
        with self._lock:
            return self._active_source

    @property
    def stop_pending(self) -> bool:
        with self._lock:
            return self._stop_pending

    @property
    def stop_generation(self) -> int:
        with self._lock:
            return self._stop_generation

    def set_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        source: str | None = None,
        active: bool | None = None,
    ) -> None:
        """Set a clipped physical command, optionally with source arbitration."""

        try:
            requested = np.asarray((vx, vy, wz), dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            self._reject_nonfinite("velocity command must contain finite numeric values")
        if not np.all(np.isfinite(requested)):
            self._reject_nonfinite("velocity command must contain finite values")
        command = np.clip(requested, -self._limits, self._limits)
        if active is None:
            active = bool(np.any(command != 0.0))
        with self._lock:
            if self._stop_pending:
                return
            if source is not None and source in self._neutral_rearm_required:
                return
            if source is None:
                self._command[:] = command
                self._active_source = None
                self._timestamp = time.monotonic()
                return
            if active:
                self._command[:] = command
                self._active_source = source
                self._timestamp = time.monotonic()
            elif self._active_source == source:
                # Releasing the active source must zero once.  Clearing source
                # ownership prevents subsequent neutral polls from repeatedly
                # overwriting a keyboard command.
                self._command.fill(0.0)
                self._active_source = None
                self._timestamp = time.monotonic()

    def increment_command(self, delta: np.ndarray, *, source: str) -> np.ndarray:
        """Atomically increment the current command and transfer ownership."""

        try:
            delta = np.asarray(delta, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            self._reject_nonfinite("command increment must contain three finite numeric values")
        if delta.shape != (3,) or not np.all(np.isfinite(delta)):
            self._reject_nonfinite("command increment must contain three finite values")
        with self._lock:
            if self._stop_pending:
                return self._command.copy()
            # A discrete keyboard action may take ownership, but must never
            # inherit lateral/yaw components left by a gamepad (or vice versa).
            baseline = self._command if self._active_source == source else np.zeros(3)
            self._command[:] = np.clip(baseline + delta, -self._limits, self._limits)
            self._active_source = source
            self._timestamp = time.monotonic()
            return self._command.copy()

    def require_neutral(self, source: str) -> None:
        """Block a source until its input adapter confirms a neutral hold."""

        with self._lock:
            self._neutral_rearm_required.add(str(source))
            if self._active_source == source:
                self._command.fill(0.0)
                self._active_source = None
            self._timestamp = time.monotonic()

    def neutral_required(self, source: str) -> bool:
        with self._lock:
            return str(source) in self._neutral_rearm_required

    def acknowledge_neutral(self, source: str) -> bool:
        """Release a source after its adapter verifies a continuous neutral hold."""

        with self._lock:
            source = str(source)
            if self._stop_pending or source not in self._neutral_rearm_required:
                return False
            self._neutral_rearm_required.discard(source)
            self._timestamp = time.monotonic()
            return True

    def request(self, name: str) -> None:
        name = self._EVENT_ALIASES.get(name, name)
        if name not in self._events:
            raise KeyError(name)
        with self._lock:
            if name in self._STOP_EVENTS:
                self._begin_stop_locked()
            self._events[name] = True

    def consume(self, name: str, *, acknowledge: bool = True) -> bool:
        name = self._EVENT_ALIASES.get(name, name)
        with self._lock:
            value = self._events[name]
            self._events[name] = False
            if value and acknowledge and name in self._STOP_EVENTS:
                self._acknowledge_stop_locked(self._stop_generation)
            return value

    def consume_many(self, *names: str) -> tuple[dict[str, bool], int | None]:
        """Atomically consume an event snapshot without releasing STOP yet.

        The returned generation must be passed to :meth:`acknowledge_stop`
        after the corresponding state transition has completed.  A newer STOP
        request cannot be accidentally unlocked by an older control tick.
        """

        canonical = tuple(self._EVENT_ALIASES.get(name, name) for name in names)
        with self._lock:
            values = {name: bool(self._events[name]) for name in canonical}
            for name in canonical:
                self._events[name] = False
            generation = self._stop_generation if self._stop_pending else None
            return values, generation

    def acknowledge_stop(self, generation: int | None) -> bool:
        with self._lock:
            return self._acknowledge_stop_locked(generation)

    def begin_stop(self) -> int:
        """Hold all command writers until the returned generation is acked."""

        with self._lock:
            return self._begin_stop_locked()

    def pending(self, *names: str) -> bool:
        """Return whether any named event is pending without consuming it."""

        with self._lock:
            return any(self._events[self._EVENT_ALIASES.get(name, name)] for name in names)

    def stop(self, *, source: str | None = None) -> None:
        with self._lock:
            if source is None or self._active_source == source:
                self._clear_command_locked(require_gamepad_neutral=source is None)

    def emergency_stop(self, *, quit: bool = False) -> None:
        """Atomically zero velocity and latch emergency-stop/quit events."""
        with self._lock:
            self._begin_stop_locked()
            self._events["estop"] = True
            if quit:
                self._events["quit"] = True

    def emergency_quit(self) -> None:
        self.emergency_stop(quit=True)


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
            if event == "quit":
                self.bus.emergency_quit()
            else:
                self.bus.request(event)
            if event in CommandBus._STOP_EVENTS:
                with self._lock:
                    self._velocity.fill(0.0)
            return
        direction = self._KEYS.get(key)
        if direction is None:
            return
        velocity = self.bus.increment_command(
            self.keyboard_step * np.asarray(direction, dtype=np.float64),
            source="keyboard",
        )
        with self._lock:
            self._velocity[:] = velocity

    def _recompute(self) -> None:
        """Synchronize the legacy cache without reviving a stale command."""

        velocity = self.bus.command()
        with self._lock:
            self._velocity[:] = velocity

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
                    self.bus.emergency_quit()
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
        "reset": 7,       # Start on the supported raw Xbox/XInput layout
        "back": 6,        # Back / Select on the supported raw layout
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
    # their physical indices vary by firmware and wireless dongle.  The safe
    # fallback contract is deliberately one unambiguous Xbox/XInput layout;
    # device-specific layouts must be supplied explicitly in YAML.
    _RAW_BUTTON_ALIASES = {
        "A": 0,
        "CROSS": 0,
        "B": 1,
        "CIRCLE": 1,
        "X": 2,
        "SQUARE": 2,
        "Y": 3,
        "TRIANGLE": 3,
        "BACK": 6,
        "SELECT": 6,
        "GUIDE": 5,
        "START": 7,
    }
    _CONTROLLER_AXIS_IDS = {
        "LEFTX": 0, "LEFTY": 1, "RIGHTX": 2, "RIGHTY": 3,
        "TRIGGERLEFT": 4, "TRIGGERRIGHT": 5,
    }
    # Linux/raw Xbox devices expose triggers before the right stick.  SDL's
    # semantic Controller API uses RIGHTX=2, but raw XInput uses RIGHTX=3.
    _RAW_AXIS_IDS = {
        "LEFTX": 0,
        "LEFTY": 1,
        "TRIGGERLEFT": 2,
        "RIGHTX": 3,
        "RIGHTY": 4,
        "TRIGGERRIGHT": 5,
    }
    _DEFAULT_DEADZONE = 0.18
    _DEFAULT_EXIT_DEADZONE = 0.15

    def __init__(
        self,
        bus: CommandBus,
        axes: tuple[int, int, int] = (0, 1, 3),
        *,
        device_name: str | None = None,
        device_guid: str | None = None,
        device_index: int | None = None,
        axis_config: dict[str, Any] | None = None,
        button_config: dict[str, Any] | None = None,
        deadzone: float | None = None,
        deadzone_enter: float | None = None,
        deadzone_exit: float | None = None,
        yaw_rescale: float = 1.0,
        reconnect_interval: float = 1.0,
        liveness_timeout: float = 2.0,
        prefer_controller: bool = True,
        calibration_samples: int = 25,
        calibration_duration: float = 0.25,
        calibration_max_mad: float = 0.01,
        calibration_max_peak_to_peak: float = 0.04,
        calibration_center_limit: float = 0.15,
        calibration_extreme_limit: float = 0.90,
        calibration_release_margin: float = CALIBRATION_RELEASE_MARGIN,
        neutral_rearm_samples: int = 5,
        neutral_rearm_duration: float = 0.10,
        input_debug: bool = False,
        debug_interval: float = 0.25,
        clock: Callable[[], float] | None = None,
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
        self._validate_button_bindings()
        self._validate_axis_bindings_config()
        enter = (
            self._DEFAULT_DEADZONE
            if deadzone is None and deadzone_enter is None
            else deadzone if deadzone_enter is None
            else deadzone_enter
        )
        enter = float(enter)
        exit_value = (
            min(self._DEFAULT_EXIT_DEADZONE, enter)
            if deadzone_exit is None
            else float(deadzone_exit)
        )
        self.deadzone = self._validate_deadzone(enter)
        self.deadzone_enter = self.deadzone
        self.deadzone_exit = self._validate_deadzone(exit_value)
        self.yaw_rescale = self._validate_rescale(yaw_rescale)
        self._axis_processor = GamepadAxisProcessor(
            self.deadzone_enter,
            self.deadzone_exit,
            self.yaw_rescale,
        )
        self._clock = clock or time.monotonic
        self._calibrator = AxisCalibrator(
            min_samples=calibration_samples,
            min_duration=calibration_duration,
            max_mad=calibration_max_mad,
            max_peak_to_peak=calibration_max_peak_to_peak,
            center_limit=calibration_center_limit,
            extreme_limit=calibration_extreme_limit,
            clock=self._clock,
            exit_deadzone=self.deadzone_exit,
            release_margin=calibration_release_margin,
        )
        self.reconnect_interval = self._validate_positive_time(
            reconnect_interval,
            "gamepad reconnect_interval",
        )
        self.liveness_timeout = self._validate_positive_time(
            liveness_timeout,
            "gamepad liveness_timeout",
        )
        self.neutral_rearm_samples = int(neutral_rearm_samples)
        self.neutral_rearm_duration = float(neutral_rearm_duration)
        if self.neutral_rearm_samples < 2:
            raise ValueError("gamepad neutral_rearm_samples must be at least 2")
        if (
            not np.isfinite(self.neutral_rearm_duration)
            or self.neutral_rearm_duration < 0.0
        ):
            raise ValueError(
                "gamepad neutral_rearm_duration must be finite and non-negative"
            )
        self.input_debug = bool(input_debug)
        self.debug_interval = self._validate_positive_time(
            debug_interval,
            "gamepad debug_interval",
        )
        self.prefer_controller = bool(prefer_controller)
        self.available = False  # True only while a selected device is open
        self.device_info: str | None = None
        self._pygame: Any = None
        self._controller_mod: Any = None
        self._device: Any = None
        self._controller = False
        self._instance_id: int | None = None
        self._running = False
        # Kept as a compatibility attribute for callers that inspected the old
        # implementation.  SDL is now polled exclusively by the runtime thread.
        self._thread: threading.Thread | None = None
        self._owner_thread: int | None = None
        self._display_owned = False
        self._display_surface_owned = False
        self._joystick_owned = False
        self._controller_owned = False
        self._last_seen = 0.0
        self._last_successful_poll: float | None = None
        self._last_reconnect = -math.inf
        self._warned_no_device = False
        self._resolved_axis_bindings: tuple[int, int, int] | None = None
        # Directly injected legacy/test devices retain the historical
        # zero-center behavior.  Every real open path calls begin_calibration.
        self._calibration_ready = True
        self._calibration_reason: str | None = None
        self._center = np.zeros(3, dtype=np.float64)
        self._last_calibration_update: CalibrationUpdate | None = None
        self._last_raw: AxisVector | None = None
        self._last_axis_snapshot: AxisProcessingSnapshot | None = None
        self._neutral_rearm_required = False
        self._neutral_rearm_reason: str | None = None
        self._neutral_since: float | None = None
        self._neutral_samples = 0
        self._observed_stop_generation = self.bus.stop_generation
        self._last_debug_at = -math.inf
        self._last_debug_signature: tuple[Any, ...] | None = None
        self._last_calibration_warning_at = -math.inf
        self._last_calibration_warning: str | None = None
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
    def _validate_positive_time(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive number")
        return value

    @staticmethod
    def _event_type(pygame: Any, name: str) -> int | None:
        value = getattr(pygame, name, None)
        return int(value) if value is not None else None

    def _assert_owner(self, operation: str) -> None:
        if self._owner_thread is not None and threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                f"gamepad {operation} must run on the runtime/SDL owner thread"
            )

    @staticmethod
    def _flatten_button_values(value: Any) -> tuple[Any, ...]:
        if isinstance(value, (tuple, list, set, frozenset)):
            flattened: list[Any] = []
            for item in value:
                flattened.extend(JoystickInput._flatten_button_values(item))
            return tuple(flattened)
        return (value,)

    def _button_candidates(self, action: str, *, controller: bool) -> tuple[Any, ...]:
        defaults = self._CONTROLLER_DEFAULT_BUTTONS if controller else self._DEFAULT_BUTTONS
        configured = self.button_config.get(action, defaults[action])
        candidates: list[Any] = []
        for value in self._flatten_button_values(configured):
            if isinstance(value, str):
                semantic = self._CONTROLLER_BUTTON_NAMES.get(
                    value.strip().casefold(), value.strip().upper()
                )
                resolved = (
                    self._CONTROLLER_BUTTON_IDS.get(semantic, semantic)
                    if controller
                    else self._RAW_BUTTON_ALIASES.get(semantic, value)
                )
                candidates.extend(self._flatten_button_values(resolved))
            else:
                candidates.append(value)
        return tuple(candidates)

    @staticmethod
    def _buttons_overlap(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
        for lhs in left:
            for rhs in right:
                try:
                    if lhs == rhs or str(lhs).casefold() == str(rhs).casefold():
                        return True
                except Exception:
                    continue
        return False

    def _validate_button_bindings(self) -> None:
        for controller in (False, True):
            reset = self._button_candidates("reset", controller=controller)
            back = self._button_candidates("back", controller=controller)
            if self._buttons_overlap(reset, back):
                api = "SDL Controller" if controller else "raw joystick"
                raise ValueError(
                    f"unsafe {api} button mapping: reset and back/estop overlap "
                    f"({reset!r} vs {back!r})"
                )

    @staticmethod
    def _axis_semantic(value: str) -> str:
        return value.strip().replace("_", "").replace("-", "").upper()

    def _resolve_axis_bindings(
        self,
        *,
        controller: bool,
        axis_count: int | None = None,
    ) -> tuple[int, int, int]:
        names = ("left_x", "left_y", "yaw")
        semantic_defaults = ("LEFTX", "LEFTY", "RIGHTX")
        mapping = self._CONTROLLER_AXIS_IDS if controller else self._RAW_AXIS_IDS
        mode_name = "controller" if controller else "raw"
        mode_config = self.axis_config.get(mode_name, {})
        if mode_config is None:
            mode_config = {}
        if not isinstance(mode_config, dict):
            raise ValueError(f"axis_config.{mode_name} must be a mapping")
        resolved: list[int] = []
        for offset, name in enumerate(names):
            configured = mode_config.get(name, self.axis_config.get(name))
            if configured is None:
                configured = semantic_defaults[offset] if controller else self.axes[offset]
            if isinstance(configured, str):
                stripped = configured.strip()
                try:
                    configured = int(stripped, 10)
                except ValueError:
                    semantic = self._axis_semantic(stripped)
                    if semantic not in mapping:
                        api = "SDL Controller" if controller else "raw joystick"
                        raise ValueError(
                            f"unknown {api} axis {configured!r} for {name}"
                        )
                    configured = mapping[semantic]
            try:
                axis = int(configured)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"axis {name} must resolve to an integer index") from exc
            if axis < 0:
                raise ValueError(f"axis {name} index must be non-negative, got {axis}")
            if controller and axis not in self._CONTROLLER_AXIS_IDS.values():
                raise ValueError(
                    f"SDL Controller axis {name} index {axis} is not a valid semantic axis"
                )
            if axis_count is not None and axis >= axis_count:
                api = "SDL Controller" if controller else "raw joystick"
                raise ValueError(
                    f"{api} axis {name} index {axis} is outside [0, {axis_count})"
                )
            resolved.append(axis)
        if len(set(resolved)) != 3:
            api = "SDL Controller" if controller else "raw joystick"
            raise ValueError(
                f"{api} left_x, left_y, and yaw axes must be distinct, got {tuple(resolved)}"
            )
        return int(resolved[0]), int(resolved[1]), int(resolved[2])

    def _validate_axis_bindings_config(self) -> None:
        # Validate both interpretations up front.  Device-specific bounds are
        # checked again after opening, when get_numaxes() is available.
        self._resolve_axis_bindings(controller=False)
        self._resolve_axis_bindings(controller=True)

    def begin_calibration(self, now: float | None = None) -> None:
        """Start a safe calibration window for a newly opened device."""

        self._assert_owner("calibration")
        timestamp = self._clock() if now is None else float(now)
        if not np.isfinite(timestamp):
            raise ValueError("calibration timestamp must be finite")
        self.bus.stop()
        self.bus.require_neutral("gamepad")
        self._calibrator.reset(timestamp)
        self._axis_processor.reset()
        self._calibration_ready = False
        self._calibration_reason = "waiting for stable centered axes"
        self._center = np.zeros(3, dtype=np.float64)
        self._last_calibration_update = None
        self._last_raw = None
        self._last_axis_snapshot = None
        self._neutral_rearm_required = False
        self._neutral_rearm_reason = None
        self._neutral_since = None
        self._neutral_samples = 0
        self._observed_stop_generation = self.bus.stop_generation
        self._debug_emit("calibration-start", force=True, now=timestamp)

    def _require_neutral_hold(
        self,
        reason: str,
        now: float,
        *,
        reset: bool = False,
    ) -> None:
        self.bus.require_neutral("gamepad")
        if reset or not self._neutral_rearm_required:
            self._axis_processor.reset()
            self._neutral_since = None
            self._neutral_samples = 0
        self._neutral_rearm_required = True
        self._neutral_rearm_reason = reason
        self._debug_emit("neutral-hold", now=now)

    def _update_neutral_hold(
        self,
        calibrated: Sequence[float],
        *,
        now: float,
    ) -> bool:
        """Return True only after a continuous centered hold has re-armed."""

        if not self._neutral_rearm_required:
            return True
        neutral = self._axis_processor.is_neutral(calibrated)
        if not neutral:
            self._neutral_since = None
            self._neutral_samples = 0
            self._axis_processor.reset()
            self._debug_emit("waiting-neutral", now=now)
            return False
        if self._neutral_since is None:
            self._neutral_since = now
            self._neutral_samples = 1
        else:
            self._neutral_samples += 1
        held_long_enough = now - self._neutral_since >= self.neutral_rearm_duration
        sampled_enough = self._neutral_samples >= self.neutral_rearm_samples
        if not (held_long_enough and sampled_enough) or self.bus.stop_pending:
            self._debug_emit("holding-neutral", now=now)
            return False
        if not self.bus.acknowledge_neutral("gamepad"):
            return False
        self._neutral_rearm_required = False
        self._neutral_rearm_reason = None
        self._neutral_since = None
        self._neutral_samples = 0
        self._axis_processor.reset()
        self._debug_emit("neutral-rearmed", force=True, now=now)
        return True

    def debug_snapshot(self, now: float | None = None) -> GamepadDebugSnapshot:
        timestamp = self._clock() if now is None else float(now)
        command = self.bus.snapshot()
        axis = self._last_axis_snapshot
        calibration = self._last_calibration_update
        center = (
            _tuple3(self._center)
            if self._calibration_ready
            else None if calibration is None else calibration.center
        )
        return GamepadDebugSnapshot(
            timestamp=timestamp,
            mode=("controller" if self._controller else "raw") if self.available else "disconnected",
            axis_bindings=self._resolved_axis_bindings,
            raw=self._last_raw,
            center=center,
            calibrated=None if axis is None else axis.calibrated,
            processed=None if axis is None else axis.processed,
            neutral=True if axis is None else axis.neutral,
            calibration_ready=self._calibration_ready,
            calibration_reason=self._calibration_reason,
            calibration_mad=None if calibration is None else calibration.mad,
            calibration_peak_to_peak=(
                None if calibration is None else calibration.peak_to_peak
            ),
            last_successful_poll=self._last_successful_poll,
            owner=command.owner,
            command=_tuple3(command.command),
            stop_pending=command.stop_pending,
        )

    @staticmethod
    def _debug_vector(value: AxisVector | None) -> str:
        if value is None:
            return "n/a"
        return "[" + ",".join(f"{component:+.3f}" for component in value) + "]"

    def _debug_emit(
        self,
        label: str,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> None:
        if not self.input_debug:
            return
        timestamp = self._clock() if now is None else float(now)
        snapshot = self.debug_snapshot(timestamp)
        signature = (
            snapshot.mode,
            snapshot.axis_bindings,
            snapshot.raw,
            snapshot.center,
            snapshot.calibrated,
            snapshot.processed,
            snapshot.neutral,
            snapshot.calibration_ready,
            snapshot.calibration_reason,
            snapshot.owner,
            snapshot.command,
            snapshot.stop_pending,
            self._neutral_rearm_required,
        )
        if signature == self._last_debug_signature and not force:
            return
        if timestamp - self._last_debug_at < self.debug_interval:
            return
        self._last_debug_at = timestamp
        self._last_debug_signature = signature
        print(
            "[INPUT] "
            f"event={label} mode={snapshot.mode} axes={snapshot.axis_bindings} "
            f"raw={self._debug_vector(snapshot.raw)} "
            f"center={self._debug_vector(snapshot.center)} "
            f"mad={self._debug_vector(snapshot.calibration_mad)} "
            f"p2p={self._debug_vector(snapshot.calibration_peak_to_peak)} "
            f"calibrated={self._debug_vector(snapshot.calibrated)} "
            f"processed={self._debug_vector(snapshot.processed)} "
            f"neutral={int(snapshot.neutral)} ready={int(snapshot.calibration_ready)} "
            f"reason={snapshot.calibration_reason or self._neutral_rearm_reason or 'none'} "
            f"owner={snapshot.owner or 'none'} "
            f"command={self._debug_vector(snapshot.command)}",
            flush=True,
        )

    def _initialize_sdl(self) -> None:
        pygame = self._pygame
        if pygame is None:
            raise RuntimeError("pygame was not imported")

        display = getattr(pygame, "display", None)
        event = getattr(pygame, "event", None)
        joystick = getattr(pygame, "joystick", None)
        if display is None or event is None or joystick is None:
            raise RuntimeError("pygame build is missing display/event/joystick support")

        display_was_init = bool(display.get_init())
        if not display_was_init:
            display.init()
            self._display_owned = True
        if display.get_surface() is None:
            flags = int(getattr(pygame, "HIDDEN", 0))
            display.set_mode((1, 1), flags=flags)
            self._display_surface_owned = True

        joystick_was_init = bool(joystick.get_init())
        if not joystick_was_init:
            joystick.init()
            self._joystick_owned = True

        try:
            import pygame._sdl2.controller as controller_mod
        except Exception:
            controller_mod = None
        self._controller_mod = controller_mod
        if controller_mod is not None:
            get_init = getattr(controller_mod, "get_init", lambda: False)
            if not bool(get_init()):
                controller_mod.init()
                self._controller_owned = True

        # A successful pump verifies that SDL's event/video subsystem is
        # actually usable; merely calling joystick.init() is insufficient.
        event.pump()

    def start(self) -> None:
        if self._running:
            self._assert_owner("start")
            return
        self._owner_thread = threading.get_ident()
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                # SDL still needs a video/event driver for joystick events in a
                # genuinely headless process.  The dummy driver creates no
                # visible window and remains fully deterministic in CI.
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
            self._pygame = pygame
            self._initialize_sdl()
        except Exception as exc:
            try:
                self._shutdown_sdl()
            finally:
                self._pygame = None
                self._owner_thread = None
            raise RuntimeError(
                "cannot initialize pygame/SDL gamepad events on the runtime "
                f"main thread: {exc}. In a headless shell, leave "
                "SDL_VIDEODRIVER unset so the runtime can select 'dummy'."
            ) from exc
        self._running = True
        self._last_reconnect = -math.inf
        self.poll()
        _LOG.info("gamepad input started (name=%r guid=%r index=%r)", self.device_name, self.device_guid, self.device_index)

    def _guid(self, device: Any) -> str:
        try:
            getter = getattr(device, "get_guid", None)
            value = getter() if callable(getter) else getattr(device, "guid", "")
            if not value:
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

    @staticmethod
    def _axis_count(device: Any) -> int | None:
        getter = getattr(device, "get_numaxes", None)
        if not callable(getter):
            return None
        count = int(getter())
        if count < 0:
            raise RuntimeError(f"gamepad reported an invalid axis count: {count}")
        return count

    def _install_device(
        self,
        device: Any,
        *,
        controller: bool,
        instance: int | None,
        info: str,
        axis_count: int | None,
    ) -> None:
        bindings = self._resolve_axis_bindings(
            controller=controller,
            axis_count=axis_count,
        )
        with self._lock:
            self._device = device
            self._controller = controller
            self._instance_id = instance
            self._resolved_axis_bindings = bindings
            self.available = True
            self.device_info = info
            self._last_successful_poll = None
            self._last_seen = 0.0
        try:
            self.begin_calibration()
        except Exception:
            with self._lock:
                self._device = None
                self._controller = False
                self._instance_id = None
                self._resolved_axis_bindings = None
                self.available = False
                self.device_info = None
            raise

    def _close_device(self, *, disconnected: bool = False) -> None:
        self._assert_owner("device close")
        with self._lock:
            device = self._device
            self._device = None
            self._controller = False
            self._instance_id = None
            self._resolved_axis_bindings = None
            had_device = self.available
            self.available = False
            self.device_info = None
            # Preserve the last proven heartbeat on a failure so diagnostics
            # can show the stale timestamp and, critically, never make a late
            # read look successful.  A normal close or a future install resets
            # it before the next device is used.
            if not disconnected:
                self._last_successful_poll = None
            self._last_seen = 0.0
        if disconnected:
            # Publish the fail-safe command/events before touching an SDL
            # handle that may already be wedged or physically gone.
            self._running = False
            self.bus.emergency_quit()
        if device is not None:
            try:
                device.quit()
            except Exception:
                pass
        if had_device or disconnected:
            if disconnected:
                _LOG.warning("gamepad disconnected; emergency stop and quit requested")
            else:
                self.bus.stop(source="gamepad")

    def _open_controller(self, index: int) -> bool:
        pygame = self._pygame
        if pygame is None:
            return False
        controller_mod = self._controller_mod
        if controller_mod is None or not self.prefer_controller:
            return False
        device: Any = None
        try:
            is_controller = getattr(controller_mod, "is_controller", None)
            if is_controller is not None and not is_controller(index):
                return False
            device = controller_mod.Controller(index)
            device.init()
            if not self._matches(device, index):
                device.quit()
                return False
            # Controller.id is not the SDL joystick instance id on every
            # pygame release.  Removal/button events are keyed by the backing
            # joystick instance, so obtain that value explicitly.
            joystick_view = device.as_joystick()
            instance_getter = getattr(joystick_view, "get_instance_id", None)
            if not callable(instance_getter):
                raise RuntimeError("SDL controller joystick view has no get_instance_id()")
            instance = int(instance_getter())
            info = self._name(device) or f"controller#{index}"
            self._install_device(
                device,
                controller=True,
                instance=instance,
                info=info,
                axis_count=self._axis_count(joystick_view),
            )
            _LOG.info("gamepad connected via SDL Controller: %s guid=%s", self.device_info, self._guid(device) or "unknown")
            return True
        except Exception as exc:
            if device is not None:
                try:
                    device.quit()
                except Exception:
                    pass
            log = _LOG.warning if isinstance(exc, ValueError) else _LOG.debug
            log("cannot open SDL controller index %s: %s", index, exc)
            return False

    def _open_joystick(self, index: int) -> bool:
        pygame = self._pygame
        if pygame is None:
            return False
        device: Any = None
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
            info = self._name(device) or f"joystick#{index}"
            self._install_device(
                device,
                controller=False,
                instance=None if instance is None else int(instance),
                info=info,
                axis_count=self._axis_count(device),
            )
            _LOG.info("gamepad connected via raw Joystick: %s guid=%s", self.device_info, self._guid(device) or "unknown")
            return True
        except Exception as exc:
            if device is not None:
                try:
                    device.quit()
                except Exception:
                    pass
            log = _LOG.warning if isinstance(exc, ValueError) else _LOG.debug
            log("cannot open joystick index %s: %s", index, exc)
            return False

    def _try_connect(self) -> bool:
        self._assert_owner("device discovery")
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
        index_by_name = {"left_x": 0, "left_y": 1, "yaw": 2}
        if name not in index_by_name:
            raise KeyError(name)
        bindings = self._resolved_axis_bindings
        if bindings is None:
            bindings = self._resolve_axis_bindings(
                controller=self._controller,
                axis_count=self._axis_count(device),
            )
            self._resolved_axis_bindings = bindings
        axis = bindings[index_by_name[name]]
        if self._controller:
            try:
                raw_value = device.get_axis(axis)
                if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
                    raw_value, (int, np.integer)
                ):
                    try:
                        numeric_value = float(raw_value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise RuntimeError(
                            f"SDL controller axis {name} is not a signed-int16 integer"
                        ) from exc
                    if not np.isfinite(numeric_value):
                        raise RuntimeError(f"SDL controller axis {name} is non-finite")
                    raise RuntimeError(
                        f"SDL controller axis {name} is not a signed-int16 integer"
                    )
                value = int(raw_value)
                if not -32768 <= value <= 32767:
                    raise RuntimeError(
                        f"SDL controller axis {name} value {value} is outside "
                        "signed-int16 range [-32768, 32767]"
                    )
                # pygame._sdl2 Controller follows SDL's signed-int16 axis
                # contract.  Always scale counts, including the +/-1 values
                # nearest center; raw Joystick values are handled separately.
                value /= 32768.0 if value < 0.0 else 32767.0
                return float(np.clip(value, -1.0, 1.0))
            except Exception as exc:
                raise RuntimeError(f"cannot read SDL controller axis {name}: {exc}") from exc
        try:
            axis_count = int(device.get_numaxes())
            if not 0 <= axis < axis_count:
                raise RuntimeError(
                    f"raw joystick axis {name} index {axis} is outside [0, {axis_count})"
                )
            value = float(device.get_axis(axis))
            if not np.isfinite(value):
                raise RuntimeError(f"raw joystick axis {name} is non-finite")
            return value
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"cannot read raw joystick axis {name}: {exc}") from exc

    def _read_raw_axes(self) -> AxisVector:
        device = self._device
        if device is None:
            raise RuntimeError("gamepad device is not open")
        attached = getattr(device, "attached", None)
        try:
            if callable(attached):
                attached = attached()
            if attached is False:
                raise RuntimeError("gamepad reports detached")
            getter = getattr(device, "get_attached", None)
            if callable(getter) and not getter():
                raise RuntimeError("gamepad reports detached")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"cannot query gamepad attachment: {exc}") from exc
        raw = (
            self._axis_value("left_x"),
            self._axis_value("left_y"),
            self._axis_value("yaw"),
        )
        try:
            return _tuple3(raw)
        except ValueError as exc:
            raise RuntimeError("gamepad axes contain NaN or infinity") from exc

    def _button_value(self, event_button: Any, action: str) -> bool:
        configured = self._button_candidates(action, controller=self._controller)
        for candidate in configured:
            try:
                if event_button == candidate:
                    return True
                if str(event_button).casefold() == str(candidate).casefold():
                    return True
            except Exception:
                continue
        return False

    def _set_axes_from_device(
        self,
        *,
        now: float | None = None,
        allow_command: bool = True,
    ) -> None:
        self._assert_owner("device read")
        read_started = self._clock() if now is None else float(now)
        if not np.isfinite(read_started):
            raise RuntimeError("gamepad poll timestamp is non-finite")
        previous_success = self._last_successful_poll
        raw = self._read_raw_axes()
        # A device getter may itself block long enough to cross the watchdog
        # deadline.  Sample the clock only after the complete three-axis read
        # and reject a late result before calibration, ownership, or command
        # state can consume it.
        read_finished = float(self._clock())
        heartbeat_origin = (
            read_started if previous_success is None else previous_success
        )
        if (
            not np.isfinite(read_finished)
            or read_finished < read_started
            or read_finished < heartbeat_origin
        ):
            _LOG.warning("gamepad watchdog clock became invalid during device read")
            self._close_device(disconnected=True)
            return
        elapsed = read_finished - heartbeat_origin
        if elapsed > self.liveness_timeout:
            _LOG.warning(
                "gamepad liveness timeout after complete read: %.3fs (limit %.3fs)",
                elapsed,
                self.liveness_timeout,
            )
            self._close_device(disconnected=True)
            return

        timestamp = read_finished
        self._last_raw = raw
        # A successful state read, even with a held stick and no SDL motion
        # event, is the liveness heartbeat.  A late read above deliberately
        # does not update this value.
        self._last_successful_poll = timestamp
        self._last_seen = timestamp

        generation = self.bus.stop_generation
        if generation != self._observed_stop_generation:
            self._observed_stop_generation = generation
            self._require_neutral_hold(
                "STOP generation changed",
                timestamp,
                reset=True,
            )
        elif self.bus.neutral_required("gamepad") and not self._neutral_rearm_required:
            self._require_neutral_hold(
                "command bus requires neutral",
                timestamp,
                reset=True,
            )

        if not self._calibration_ready:
            update = self._calibrator.add_sample(raw, timestamp)
            self._last_calibration_update = update
            self._calibration_reason = update.reason
            if not update.ready:
                self.bus.stop(source="gamepad")
                force = bool(
                    update.stable
                    and update.reason is not None
                    and (
                        "wrong axis" in update.reason
                        or "safe limit" in update.reason
                        or "held-at-connect" in update.reason
                        or "after release" in update.reason
                    )
                )
                if force and (
                    update.reason != self._last_calibration_warning
                    or timestamp - self._last_calibration_warning_at >= 2.0
                ):
                    _LOG.warning(
                        "gamepad calibration rejected: %s; velocity control remains disabled",
                        update.reason,
                    )
                    self._last_calibration_warning = update.reason
                    self._last_calibration_warning_at = timestamp
                self._debug_emit("calibrating", force=force, now=timestamp)
                return
            if update.center is None:
                raise RuntimeError("calibrator reported ready without an axis center")
            self._center = _axis_vector(update.center, "calibrated axis center")
            self._calibration_ready = True
            self._calibration_reason = None
            self._last_calibration_warning = None
            self._axis_processor.reset()
            self._require_neutral_hold(
                "post-calibration neutral hold",
                timestamp,
                reset=True,
            )
            self._debug_emit("calibrated", force=True, now=timestamp)

        axis = self._axis_processor.process(raw, self._center)
        self._last_axis_snapshot = axis
        if not self._update_neutral_hold(axis.calibrated, now=timestamp):
            return

        # A lifecycle button from this event batch must remain zero through the
        # current control tick.  Device sampling still ran above for liveness.
        if not allow_command or self.bus.stop_pending:
            self._require_neutral_hold("STOP pending", timestamp)
            return

        lx, ly, yaw = axis.processed
        limits = np.asarray(self.bus._limits, dtype=np.float64)
        # SDL/XInput gamepads expose the left-stick X axis as lateral motion
        # and Y as fore/aft motion.  SDL's positive Y points down, therefore
        # forward is +vx.  Keyboard D/E and the right-handed body convention
        # both use negative lateral/yaw values for a rightward input.
        moving = bool(lx != 0.0 or ly != 0.0 or yaw != 0.0)
        self.bus.set_command(
            float(-ly * limits[0]),
            float(-lx * limits[1]),
            float(-yaw * limits[2]),
            source="gamepad",
            active=moving,
        )
        self._debug_emit("command", now=timestamp)

    def _dispatch_button(self, button: Any) -> None:
        # Emergency-stop is intentionally checked before every lifecycle
        # action, even though overlapping reset/back mappings are rejected.
        if self._button_value(button, "back"):
            self.bus.emergency_quit()
        elif self._button_value(button, "stand"):
            self.bus.request("stand")
        elif self._button_value(button, "enable_rl"):
            self.bus.request("enable_rl")
        elif self._button_value(button, "disable_rl"):
            self.bus.request("disable_rl")
        elif self._button_value(button, "get_down"):
            self.bus.request("get_down")
        elif self._button_value(button, "reset"):
            self.bus.request("reset")

    def _handle_event(self, event: Any) -> None:
        pygame = self._pygame
        if pygame is None:
            return
        typ = getattr(event, "type", None)
        joy_added = self._event_type(pygame, "JOYDEVICEADDED")
        joy_removed = self._event_type(pygame, "JOYDEVICEREMOVED")
        ctrl_added = self._event_type(pygame, "CONTROLLERDEVICEADDED")
        ctrl_removed = self._event_type(pygame, "CONTROLLERDEVICEREMOVED")
        removed_types = {value for value in (joy_removed, ctrl_removed) if value is not None}
        added_types = {value for value in (joy_added, ctrl_added) if value is not None}
        if typ in removed_types:
            if not self.available:
                return
            instance = getattr(event, "instance_id", getattr(event, "which", None))
            if self._instance_id is None or instance is None or instance == self._instance_id:
                self._close_device(disconnected=True)
            return
        if typ in added_types:
            if not self.available:
                self._try_connect()
            return
        # SDL may emit both raw joystick and semantic controller events for a
        # mapped pad.  Consume only the API used to open the device, otherwise
        # a raw index (notably Back=6) could be misread as semantic Start=6.
        axis_name = "CONTROLLERAXISMOTION" if self._controller else "JOYAXISMOTION"
        button_name = "CONTROLLERBUTTONDOWN" if self._controller else "JOYBUTTONDOWN"
        axis_type = self._event_type(pygame, axis_name)
        button_type = self._event_type(pygame, button_name)
        axis_types = set() if axis_type is None else {axis_type}
        button_types = set() if button_type is None else {button_type}
        event_instance = getattr(
            event,
            "instance_id",
            getattr(event, "joy", getattr(event, "which", None)),
        )
        if (
            typ in axis_types | button_types
            and self._instance_id is not None
            and event_instance is not None
            and event_instance != self._instance_id
        ):
            return
        if typ in axis_types and typ is not None:
            # SDL state is sampled once after the full event batch.  Avoiding a
            # write here lets a lifecycle button in the same batch keep the
            # velocity at zero for the current control tick.
            return
        elif typ in button_types and typ is not None and self.available:
            self._dispatch_button(getattr(event, "button", None))

    def poll(self) -> None:
        """Pump SDL and read the selected gamepad on the runtime thread."""

        if not self._running:
            return
        self._assert_owner("poll")
        pygame = self._pygame
        if pygame is None:
            raise RuntimeError("gamepad poll requested before pygame initialization")

        cycle_started = float(self._clock())
        if not np.isfinite(cycle_started):
            self._close_device(disconnected=True)
            raise RuntimeError("gamepad watchdog clock returned a non-finite value")

        try:
            pygame.event.pump()
            events = tuple(pygame.event.get())
        except Exception as exc:
            self._close_device(disconnected=True)
            raise RuntimeError(f"pygame/SDL event pump failed: {exc}") from exc

        events_finished = float(self._clock())
        event_origin = (
            cycle_started
            if self._last_successful_poll is None
            else self._last_successful_poll
        )
        if (
            not np.isfinite(events_finished)
            or events_finished < cycle_started
            or events_finished < event_origin
        ):
            self._close_device(disconnected=True)
            raise RuntimeError("gamepad watchdog clock became invalid during event read")
        if self.available and events_finished - event_origin > self.liveness_timeout:
            _LOG.warning(
                "gamepad liveness timeout after event read: %.3fs (limit %.3fs)",
                events_finished - event_origin,
                self.liveness_timeout,
            )
            self._close_device(disconnected=True)
            return

        for event in events:
            try:
                self._handle_event(event)
            except Exception as exc:
                self._close_device(disconnected=True)
                raise RuntimeError(f"pygame/SDL gamepad event handling failed: {exc}") from exc
            if not self._running:
                return

        now = events_finished
        if not self.available and now - self._last_reconnect >= self.reconnect_interval:
            self._last_reconnect = now
            self._try_connect()
        if self.available:
            try:
                # Poll axes even without MOTION events.  This both catches
                # quiet/held sticks and refreshes the liveness heartbeat.
                # STOP suppresses command publication, not device polling.
                self._set_axes_from_device(
                    # For a newly opened device this includes event discovery
                    # and device-open latency in the first watchdog interval.
                    now=cycle_started,
                    allow_command=not self.bus.stop_pending,
                )
            except Exception as exc:
                _LOG.warning("gamepad read failed; disconnecting (%s)", exc)
                self._close_device(disconnected=True)

    def _shutdown_sdl(self) -> None:
        pygame = self._pygame
        controller_mod = self._controller_mod
        if controller_mod is not None and self._controller_owned:
            try:
                controller_mod.quit()
            except Exception:
                pass
        self._controller_owned = False
        self._controller_mod = None

        if pygame is not None and self._joystick_owned:
            try:
                pygame.joystick.quit()
            except Exception:
                pass
        self._joystick_owned = False

        if pygame is not None and (self._display_owned or self._display_surface_owned):
            try:
                pygame.display.quit()
            except Exception:
                pass
        self._display_owned = False
        self._display_surface_owned = False

    def close(self) -> None:
        if self._owner_thread is not None:
            self._assert_owner("close")
        self._running = False
        self._close_device()
        self._shutdown_sdl()
        self._pygame = None
        self._owner_thread = None


class InputManager:
    def __init__(self, backend: str, bus: CommandBus, joystick_axes: tuple[int, int, int], **gamepad_options: Any):
        self.backend = backend
        self.bus = bus
        self.input_debug = bool(gamepad_options.get("input_debug", False))
        self._debug_interval = float(gamepad_options.get("debug_interval", 0.25))
        self._last_debug_at = -math.inf
        self._last_debug_signature: tuple[Any, ...] | None = None
        self.keyboard = KeyboardInput(bus)
        self.joystick = JoystickInput(bus, joystick_axes, **gamepad_options)

    def _debug_keyboard(self) -> None:
        if not self.input_debug:
            return
        snapshot = self.bus.snapshot()
        signature = (
            snapshot.owner,
            _tuple3(snapshot.command),
            snapshot.stop_pending,
        )
        now = time.monotonic()
        if signature == self._last_debug_signature:
            return
        if now - self._last_debug_at < self._debug_interval:
            return
        self._last_debug_at = now
        self._last_debug_signature = signature
        command = ",".join(f"{value:+.3f}" for value in snapshot.command)
        print(
            "[INPUT] event=command mode=keyboard "
            f"owner={snapshot.owner or 'none'} stop={int(snapshot.stop_pending)} "
            f"command=[{command}]",
            flush=True,
        )

    def start(self, headless: bool) -> None:
        if self.backend in ("keyboard", "both"):
            if headless:
                self.keyboard.start_headless()
        if self.backend in ("joystick", "both"):
            self.joystick.start()

    def poll(self) -> None:
        """Poll main-thread-owned inputs once per runtime iteration."""

        if self.backend in ("joystick", "both"):
            self.joystick.poll()
        elif self.backend == "keyboard":
            self._debug_keyboard()

    def key_callback(self, key: int) -> None:
        if self.backend not in ("keyboard", "both"):
            return
        self.keyboard.handle_viewer_key(key)

    def close(self) -> None:
        self.keyboard.close()
        self.joystick.close()
