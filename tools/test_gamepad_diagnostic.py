"""Hardware-free regression tests for :mod:`tools.gamepad_diagnostic`."""

from __future__ import annotations

import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import tools.gamepad_diagnostic as diagnostic
from tools.gamepad_diagnostic import (
    AxisDiagnosticPipeline,
    DeviceInfo,
    OpenDevice,
    build_parser,
    format_diagnostic_sample,
    initialize_sdl,
    main,
    monitor,
    print_initial_snapshot,
    print_state_change,
    read_selected_axes,
    resolve_axis_bindings,
    select_device,
    select_mode,
)


class _RawDevice:
    @staticmethod
    def get_instance_id() -> int:
        return 42

    @staticmethod
    def quit() -> None:
        return None


class _SnapshotDevice(_RawDevice):
    axes = (0.0, -0.2, -1.0, 0.25)
    buttons = (0, 1, 0)
    hats = ((1, 0),)

    @classmethod
    def get_numaxes(cls) -> int:
        return len(cls.axes)

    @classmethod
    def get_axis(cls, index: int) -> float:
        return cls.axes[index]

    @classmethod
    def get_numbuttons(cls) -> int:
        return len(cls.buttons)

    @classmethod
    def get_button(cls, index: int) -> int:
        return cls.buttons[index]

    @classmethod
    def get_numhats(cls) -> int:
        return len(cls.hats)

    @classmethod
    def get_hat(cls, index: int):
        return cls.hats[index]


class GamepadDiagnosticTest(unittest.TestCase):
    def setUp(self) -> None:
        self.devices = [
            DeviceInfo(0, "Keyboard Receiver", "aaaa", "raw", 2, 4, 0),
            DeviceInfo(1, "Asura 2 Pro", "0300beef", "controller", 6, 15, 1),
        ]

    def test_device_selection_combines_index_name_and_guid_filters(self) -> None:
        selected = select_device(
            self.devices,
            index=1,
            name="asura",
            guid="BEEF",
        )
        self.assertEqual(selected, self.devices[1])
        self.assertIsNone(
            select_device(self.devices, index=0, name="asura", guid=None)
        )

    def test_summary_exposes_mode_and_physical_counts(self) -> None:
        self.assertEqual(
            self.devices[1].line(),
            "index=1 name='Asura 2 Pro' guid=0300beef mode=controller "
            "axes=6 buttons=15 hats=1",
        )

    def test_raw_yaw3_and_controller_rightx_defaults_are_distinct(self) -> None:
        raw = resolve_axis_bindings("raw", 6)
        controller = resolve_axis_bindings("controller", 6)
        self.assertEqual(raw.values(), (0, 1, 3))
        self.assertEqual(controller.values(), ("LEFTX", "LEFTY", "RIGHTX"))

        class SemanticController:
            values = (0.1, -0.2, 0.3, 0.9, -1.0, -1.0)

            @classmethod
            def get_axis(cls, index):
                return cls.values[int(index)]

        info = DeviceInfo(1, "Mapped Pad", "beef", "controller", 6, 15, 1)
        device = OpenDevice(
            info=info,
            raw=_SnapshotDevice(),
            controller=SemanticController(),
        )
        self.assertEqual(
            read_selected_axes(device, None, controller),
            (0.1, -0.2, 0.3),
        )

    def test_duplicate_and_out_of_range_axis_bindings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            resolve_axis_bindings("raw", 6, left_x=0, left_y=0, yaw=3)
        with self.assertRaisesRegex(ValueError, "outside device range"):
            resolve_axis_bindings("raw", 4, left_x=0, left_y=1, yaw=4)
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            resolve_axis_bindings(
                "controller",
                6,
                left_x="LEFTX",
                left_y="LEFTY",
                yaw=0,
            )

    def test_mode_and_axis_cli_overrides_are_parsed(self) -> None:
        args = build_parser().parse_args(
            [
                "--raw",
                "--left-x-axis",
                "0",
                "--left-y-axis",
                "1",
                "--yaw-axis",
                "3",
                "--deadzone",
                "0.18",
                "--release-deadzone",
                "0.15",
                "--center-limit",
                "0.10",
                "--calibration-seconds",
                "0.75",
            ]
        )
        self.assertTrue(args.raw)
        self.assertFalse(args.controller)
        self.assertEqual((args.left_x_axis, args.left_y_axis, args.yaw_axis), (0, 1, 3))
        self.assertEqual(args.deadzone, 0.18)
        self.assertEqual(args.release_deadzone, 0.15)
        self.assertEqual(args.center_limit, 0.10)
        self.assertEqual(args.calibration_seconds, 0.75)

        help_text = build_parser().format_help()
        self.assertIn("--enter-deadzone", help_text)
        self.assertIn("--exit-deadzone", help_text)
        self.assertIn("--center-limit", help_text)
        self.assertIn("default: 0.18", help_text)
        self.assertRegex(help_text, r"default:\s+0\.15")

    def test_controller_mode_rejects_unmapped_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "no SDL Controller mapping"):
            select_mode(self.devices[0], force_raw=False, force_controller=True)
        forced_raw = select_mode(
            self.devices[1],
            force_raw=True,
            force_controller=False,
        )
        self.assertEqual(forced_raw.mode, "raw")

    def test_unsafe_center_limit_exit_combination_fails_before_sdl(self) -> None:
        error = io.StringIO()
        with mock.patch.object(diagnostic, "initialize_sdl") as initialize, redirect_stderr(
            error
        ):
            result = main(
                [
                    "--deadzone",
                    "0.18",
                    "--exit-deadzone",
                    "0.13",
                    "--center-limit",
                    "0.15",
                ]
            )

        self.assertEqual(result, diagnostic.EXIT_UNAVAILABLE)
        initialize.assert_not_called()
        self.assertIn("unsafe center-limit/exit-deadzone", error.getvalue())

    def test_initial_snapshot_prints_all_controls_and_selected_raw_axes(self) -> None:
        info = DeviceInfo(0, "Raw Pad", "abcd", "raw", 4, 3, 1)
        device = OpenDevice(info=info, raw=_SnapshotDevice())
        bindings = resolve_axis_bindings("raw", 4)

        output = io.StringIO()
        with redirect_stdout(output):
            snapshot = print_initial_snapshot(device, bindings)

        self.assertEqual(snapshot.axes, (0.0, -0.2, -1.0, 0.25))
        self.assertEqual(snapshot.buttons, (0, 1, 0))
        self.assertEqual(snapshot.hats, ((1, 0),))
        self.assertEqual(
            read_selected_axes(device, None, bindings),
            (0.0, -0.2, 0.25),
        )
        rendered = output.getvalue()
        self.assertIn("initial mode=raw", rendered)
        self.assertIn("left_x=0 left_y=1 yaw=3", rendered)
        self.assertIn("axes=(0.0, -0.2, -1.0, 0.25)", rendered)
        self.assertIn("buttons=(0, 1, 0)", rendered)
        self.assertIn("hats=((1, 0),)", rendered)

    def test_diagnostic_pipeline_prints_calibrated_processed_command(self) -> None:
        info = DeviceInfo(0, "Raw Pad", "abcd", "raw", 4, 3, 1)
        device = OpenDevice(info=info, raw=_SnapshotDevice())
        bindings = resolve_axis_bindings("raw", 4)
        pipeline = AxisDiagnosticPipeline(
            mode="raw",
            bindings=bindings,
            deadzone=0.18,
            release_deadzone=0.15,
            calibration_seconds=0.0,
            now=0.0,
        )
        for index in range(25):
            centered = pipeline.update((0.0, -0.03, 0.0), now=index * 0.02)
        self.assertTrue(centered.calibration_ready)
        self.assertEqual(centered.command, (0.0, 0.0, 0.0))

        deflected = pipeline.update((0.0, -1.0, 0.5), now=0.6)
        output = io.StringIO()
        with redirect_stdout(output):
            print_initial_snapshot(device, bindings)
            print(format_diagnostic_sample(deflected))

        rendered = output.getvalue()
        self.assertIn("initial mode=raw", rendered)
        self.assertIn("axes=(0.0, -0.2, -1.0, 0.25)", rendered)
        self.assertIn("raw(lx,ly,yaw)=", rendered)
        self.assertIn("enter=0.18000 exit=0.15000", rendered)
        self.assertIn("center_limit=0.15000", rendered)
        self.assertIn("center=", rendered)
        self.assertIn("noise_mad=", rendered)
        self.assertIn("calibrated=", rendered)
        self.assertIn("processed vx=+1.00000", rendered)
        self.assertIn("neutral=0 ready=1", rendered)

    def test_diagnostic_reports_default_center_limit_rejection(self) -> None:
        bindings = resolve_axis_bindings("raw", 4)
        pipeline = AxisDiagnosticPipeline(
            mode="raw",
            bindings=bindings,
            deadzone=0.18,
            release_deadzone=0.15,
            calibration_seconds=0.0,
            now=0.0,
        )

        for index in range(25):
            rejected = pipeline.update((0.0, -0.2, 0.0), now=index * 0.02)

        self.assertFalse(rejected.calibration_ready)
        self.assertEqual(rejected.command, (0.0, 0.0, 0.0))
        rendered = format_diagnostic_sample(rejected)
        self.assertIn("ready=0", rendered)
        self.assertIn("reason=stable axis center exceeds the safe limit", rendered)

    def test_monitor_polls_device_state_at_fixed_frequency_without_events(self) -> None:
        selected = types.SimpleNamespace(
            device=OpenDevice(
                DeviceInfo(0, "Raw Pad", "abcd", "raw", 4, 0, 0),
                _SnapshotDevice(),
            )
        )
        pygame = types.SimpleNamespace(
            event=types.SimpleNamespace(get=lambda: []),
        )
        args = build_parser().parse_args(["--duration", "0.055"])
        now = [0.0]
        poll_times: list[float] = []
        sleeps: list[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            now[0] += seconds

        with mock.patch.object(
            diagnostic,
            "_open_selected",
            return_value=selected,
        ), mock.patch.object(
            diagnostic,
            "enumerate_devices",
            return_value=[],
        ), mock.patch.object(
            diagnostic,
            "_poll_selected",
            side_effect=lambda _selected, _controller, *, now: poll_times.append(now),
        ), mock.patch.object(
            diagnostic.time,
            "monotonic",
            side_effect=lambda: now[0],
        ), mock.patch.object(
            diagnostic.time,
            "sleep",
            side_effect=sleep,
        ):
            self.assertEqual(monitor(pygame, None, args), 0)

        npoll = len(poll_times)
        self.assertEqual(npoll, 4)
        for actual, expected in zip(poll_times, (0.0, 0.02, 0.04, 0.06)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(sleeps), npoll - 1)
        for interval in sleeps:
            self.assertAlmostEqual(interval, 1.0 / diagnostic.POLL_HZ)

    def test_repeated_identical_events_are_printed_only_once(self) -> None:
        pygame = types.SimpleNamespace(
            JOYAXISMOTION=1,
            JOYBUTTONDOWN=2,
            JOYBUTTONUP=3,
            JOYHATMOTION=4,
            CONTROLLERAXISMOTION=5,
            CONTROLLERBUTTONDOWN=6,
            CONTROLLERBUTTONUP=7,
        )
        info = DeviceInfo(0, "Raw Pad", "abcd", "raw", 3, 8, 1)
        device = OpenDevice(info=info, raw=_RawDevice())
        events = [
            types.SimpleNamespace(type=1, axis=0, value=0.5, instance_id=42),
            types.SimpleNamespace(type=1, axis=0, value=0.5, instance_id=42),
            types.SimpleNamespace(type=2, button=6, instance_id=42),
            types.SimpleNamespace(type=2, button=6, instance_id=42),
            types.SimpleNamespace(type=3, button=6, instance_id=42),
            types.SimpleNamespace(type=4, hat=0, value=(1, 0), instance_id=42),
            types.SimpleNamespace(type=4, hat=0, value=(1, 0), instance_id=42),
        ]

        output = io.StringIO()
        with redirect_stdout(output):
            handled = [print_state_change(pygame, event, device) for event in events]

        self.assertEqual(handled, [True, False, True, False, True, True, False])
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "axis index=0 value=+0.50000",
                "button index=6 pressed=1",
                "button index=6 pressed=0",
                "hat index=0 value=(1, 0)",
            ],
        )

    def test_controller_mode_ignores_duplicate_raw_axis_and_button_events(self) -> None:
        pygame = types.SimpleNamespace(
            JOYAXISMOTION=1,
            JOYBUTTONDOWN=2,
            JOYBUTTONUP=3,
            JOYHATMOTION=4,
            CONTROLLERAXISMOTION=5,
            CONTROLLERBUTTONDOWN=6,
            CONTROLLERBUTTONUP=7,
        )
        device = OpenDevice(info=self.devices[1], raw=_RawDevice())
        raw_axis = types.SimpleNamespace(type=1, axis=0, value=0.25, instance_id=42)
        controller_axis = types.SimpleNamespace(
            type=5,
            axis=0,
            value=8192,
            instance_id=42,
        )

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertFalse(print_state_change(pygame, raw_axis, device))
            self.assertTrue(print_state_change(pygame, controller_axis, device))
        self.assertEqual(output.getvalue().strip(), "axis index=0 value=+0.25001")

    def test_background_event_hint_is_set_before_sdl_initialization(self) -> None:
        observed: list[str | None] = []

        class Display:
            @staticmethod
            def init() -> None:
                observed.append(os.environ.get("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"))

            @staticmethod
            def get_surface():
                return object()

            @staticmethod
            def quit() -> None:
                return None

        class JoystickSubsystem:
            @staticmethod
            def init() -> None:
                observed.append(os.environ.get("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"))

            @staticmethod
            def quit() -> None:
                return None

        class Event:
            @staticmethod
            def pump() -> None:
                return None

        fake_pygame = types.ModuleType("pygame")
        fake_pygame.__path__ = []
        fake_pygame.display = Display()
        fake_pygame.joystick = JoystickSubsystem()
        fake_pygame.event = Event()
        fake_pygame.quit = lambda: None
        fake_controller = types.ModuleType("pygame._sdl2.controller")
        fake_controller.get_init = lambda: False
        fake_controller.init = lambda: observed.append(
            os.environ.get("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS")
        )
        fake_sdl2 = types.ModuleType("pygame._sdl2")
        fake_sdl2.__path__ = []
        fake_sdl2.controller = fake_controller
        fake_pygame._sdl2 = fake_sdl2

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
            sys.modules,
            {
                "pygame": fake_pygame,
                "pygame._sdl2": fake_sdl2,
                "pygame._sdl2.controller": fake_controller,
            },
        ):
            initialize_sdl()

        self.assertEqual(observed, ["1", "1", "1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
