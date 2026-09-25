# RI-4438 HIM MuJoCo sim2sim

This task runs the six-frame RI-4438 HIM actor in the shared combined terrain
scene. The exported policy consumes `6 x 47 = 282` observations in
frame-major, current-first order (`t, t-1, ..., t-5`) and produces 12 joint
actions. Observation normalization is already embedded in the exported ONNX;
the deploy process does not normalize the input again or clip the policy output.

## Installation and startup

Run commands from the repository root. The recommended environment is `uv`,
including the optional sim2sim dependencies:

```bash
uv sync --extra sim2sim
```

The `policy` entry in [config/sim2sim.yaml](config/sim2sim.yaml) may contain an
absolute path from the machine that exported the model. Use an explicit,
repository-relative `--policy` argument instead of relying on that path. For
example, the following commands select each supported input mode:

```bash
# Keyboard only
uv run python deploy/main.py --task ri_4438_him --input keyboard \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx

# One selected gamepad only
uv run python deploy/main.py --task ri_4438_him --input joystick \
  --gamepad-index 0 \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx

# Keyboard and gamepad together
uv run python deploy/main.py --task ri_4438_him --input both \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
```

Replace the example policy with another compatible 282-input, 12-output ONNX
file when needed. The compatibility launcher below uses the same runtime:

```bash
uv run python deploy/task/ri_4438_him/main.py --input keyboard \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
```

The default lifecycle is:

```text
PASSIVE (lying) -> GETUP -> STAND_SETTLE -> READY -> STAND -> RL -> GETDOWN -> PASSIVE (lying)
```

Startup performs the stand transition once automatically and enters RL after
the measured settle gate when `auto_enable_rl` is enabled. Returning to
`PASSIVE` does not start another automatic transition. Use
`--no-auto-stand` to remain lying until a manual stand request, or
`--auto-stand` to explicitly override a config that disables it.

## Keyboard controls

Keyboard motion is incremental, not press-and-hold or toggle control. Each
accepted motion key adds `0.1` to one command component and clamps it to the
configured limit. `Z` or keyboard `X` explicitly clears all three velocity
components.

- `W` / `S`: increase/decrease forward velocity (`vx`).
- `A` / `D`: increase/decrease lateral velocity (`vy`); `D` is right and
  therefore negative `vy`.
- `Q` / `E`: increase/decrease yaw velocity (`wz`); `E` is a right turn and
  therefore negative `wz`.
- `Z` / `X`: clear `vx`, `vy`, and `wz` once.
- `U` / Up arrow: request the controlled get-up/stand sequence.
- `L`: enable RL after the robot has reached `STAND`.
- `B`: disable RL, clear velocity, and hold the standing pose.
- `P` / `G` / Space / Down arrow: clear velocity, run the controlled get-down
  sequence, and remain in `PASSIVE`.
- `R`: clear velocity and reset the simulation, policy history, action, and
  command. The configured one-shot startup sequence is re-armed.
- `F`: toggle follow/free camera in the interactive viewer.
- `Esc`: atomically clear velocity, emergency-stop, and exit.

The keyboard `X` key means zero velocity; the gamepad X button has the
different lifecycle meaning documented below.

## Gamepad controls

pygame/SDL semantic Controller mappings are preferred. Raw Joystick mode is a
fallback for devices without an SDL mapping. The default axes are:

- Left-stick Y: forward/backward `vx` (SDL forward is negative Y and becomes
  positive `vx`).
- Left-stick X: lateral `vy`; pushing right produces negative `vy`.
- Right-stick X: yaw `wz`; pushing right produces negative `wz`.

Active stick magnitude is continuously rescaled from the exit threshold to
full travel. The default hysteresis thresholds are `enter=0.18` and
`exit=0.15`: motion must cross the enter threshold to activate, then must fall
below the exit threshold to return to exact zero. This avoids center chatter
while retaining full command range. The default calibration center limit is
`0.15`, with a `0.005` safety margin below the exit threshold.

Leave both sticks released on startup and whenever a controller is opened or
reconnected. Because an active disconnect terminates the current runtime, this
also applies when restarting after a disconnect. The runtime collects at least
25 stable samples spanning at least 0.25 seconds before enabling gamepad
velocity. A center outside the safe limit, an unstable stick, or a release
residual that cannot return below the exit threshold keeps gamepad velocity
disabled and starts a new calibration window. Do not hold a stick or trigger
during this period.

The semantic button mapping is:

- A / Cross: request get-up/stand.
- X / Square: enable RL after stand settles.
- B / Circle: disable RL, clear velocity, and hold `STAND`.
- Y / Triangle: clear velocity and request controlled get-down.
- Start: clear velocity and reset.
- Back / Select: atomically clear velocity, emergency-stop, and exit.

For the supported raw Xbox/XInput fallback, the fixed button indices are
`A=0`, `B=1`, `X=2`, `Y=3`, `Back=6`, and `Start=7`. A configuration that maps
reset and Back/Select to the same raw or semantic button is rejected at startup.
The default raw axes are `LX=0`, `LY=1`, and `RX=3`; raw axis 2 is commonly a
trigger and is deliberately not used for yaw.

If several controllers are connected, select one with `--gamepad-index`,
`--gamepad-name`, or `--gamepad-guid`. An index is only a discovery slot and
may change after reconnecting; name or GUID selection is more stable.

## Mixed-input arbitration and safety

With `--input both`, an effective non-neutral input takes ownership of the
velocity command. A keyboard motion key taking over from the gamepad starts
from a zero baseline, so it cannot inherit an old lateral or yaw component. A
neutral gamepad does not overwrite a keyboard command. When the gamepad owns
the command and returns to neutral, it emits one zero command and releases
ownership.

Zero, get-down, disable-RL, reset, emergency-stop, and quit requests clear the
velocity atomically. While such a STOP is pending, neither keyboard nor
gamepad can write a new velocity. After the STOP is consumed, the gamepad must
remain continuously neutral before a deflected stick is accepted again
(normally at least five samples over 0.10 seconds); this prevents a held stick
from immediately restarting motion. Safety events in one poll are handled
before axes, with emergency-stop, reset, get-down, and disable-RL taking
priority over stand or enable-RL requests.

Loss of the active controller, a non-finite axis value, or an SDL device/event
failure atomically clears velocity and requests emergency-stop plus exit. It is
therefore intentionally terminal for the current runtime rather than an
automatic moving reconnect. Starting with no matching controller is non-fatal;
the runtime remains safe at zero and can detect a later device addition.

## Gamepad diagnostics

Use [tools/gamepad_diagnostic.py](../../../tools/gamepad_diagnostic.py) without
starting MuJoCo or the robot FSM. First list every device visible to SDL:

```bash
uv run python tools/gamepad_diagnostic.py --list
```

Then monitor one device and move every stick and press every button. The tool
prints the device mode, name, GUID, axis indices, raw/center/calibrated values,
noise, processed command, and button transitions:

```bash
# Select by discovery index and stop after 20 seconds
uv run python tools/gamepad_diagnostic.py --index 0 --duration 20

# Select by a case-insensitive name substring; duration 0 runs until Ctrl-C
uv run python tools/gamepad_diagnostic.py --name "Xbox" --duration 0

# Select by a full GUID or GUID substring
uv run python tools/gamepad_diagnostic.py --guid 03000000 --duration 30
```

Force a backend when checking whether SDL's semantic mapping is correct. Raw
mode accepts numeric axis overrides; the supported Xbox fallback is
`LX=0`, `LY=1`, `RX=3`:

```bash
uv run python tools/gamepad_diagnostic.py --index 0 --raw \
  --left-x-axis 0 --left-y-axis 1 --yaw-axis 3 --duration 30

uv run python tools/gamepad_diagnostic.py --index 0 --controller \
  --left-x-axis LEFTX --left-y-axis LEFTY --yaw-axis RIGHTX --duration 30
```

To reproduce the runtime's deadzone and center-safety contract while allowing a
longer diagnostic calibration window, use:

```bash
uv run python tools/gamepad_diagnostic.py --index 0 \
  --enter-deadzone 0.18 --exit-deadzone 0.15 --center-limit 0.15 \
  --calibration-seconds 0.5 --duration 30
```

Keep the sticks released until `ready=1`. If calibration remains disabled,
inspect `center`, `noise_mad`, `noise_peak_to_peak`, and `reason` before changing
any threshold. A real large center offset requires increasing the enter/exit
deadzones and center limit together while still keeping the reported
physical-release residual below the exit threshold.

Use the reported index, name, or GUID with the corresponding runtime options.
The diagnostic and runtime both initialize and poll SDL on the process main
thread, so a device visible here should follow the same event-delivery path.

For live arbitration diagnostics, add `--input-debug` to sim2sim:

```bash
uv run python deploy/main.py --task ri_4438_him --input both --input-debug \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
```

The rate-limited `[INPUT]` lines show `raw`, `center`, `calibrated`, `processed`,
`owner`, and the final `command`. When a released gamepad shows a zero
`processed` value but `owner=keyboard`, the remaining command came from the
keyboard's incremental state, not gamepad drift. Press keyboard `Z` or `X` to
clear it. Alternatively, restart with `--input joystick` to isolate the
gamepad completely. If `owner=gamepad` and `processed` stays non-zero while the
sticks are released, continue with the raw/controller mapping and calibration
checks above.

When `processed` and `command` are both zero, a brief 1–2 cm displacement while
getting up or settling can come from pose interpolation and balance convergence,
not an input command. Continue drift investigation only if the robot keeps
walking after the stand has settled.

On Linux, the user must be able to read the controller's
`/dev/input/event*` and `/dev/input/js*` nodes. Inspect them with
`ls -l /dev/input`; if access is denied, add the user to the distribution's
input group or install an appropriate udev rule, then log in again. In a
container, pass through both nodes belonging to the controller, for example
`--device=/dev/input/js0 --device=/dev/input/eventN`, and preserve the required
group permissions inside the container.

When neither `DISPLAY` nor `WAYLAND_DISPLAY` is available, the runtime and
diagnostic select SDL's `dummy` video driver before importing pygame. This
provides the event subsystem without opening a visible window. If
`SDL_VIDEODRIVER` is already set to an unusable driver, initialization fails
with an explicit pygame/SDL error instead of silently ignoring the gamepad.

## Terrain and smoke tests

The robot starts on the checkerboard area. A terrain preset changes only its
spawn location while retaining the single combined scene:

```bash
uv run python deploy/main.py --task ri_4438_him --terrain stairs_5cm \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
uv run python deploy/main.py --task ri_4438_him --terrain heightfield \
  --policy logs/rsl_rl/ri_4438_him/2026-09-15_18-34-07/policy.onnx
```

For a backend/FSM smoke test without loading ONNX Runtime:

```bash
uv run python deploy/main.py --task ri_4438_him --no-policy --headless \
  --auto-stand --steps 800 --input both
```

`--no-policy` preserves the MuJoCo and FSM flow but outputs zero policy action.
The sim2sim path intentionally does not reproduce training-time observation
noise or actuator delay; those remain known dynamics-randomization gaps rather
than policy input-contract differences.

Run the focused sim2sim and gamepad regressions with:

```bash
uv run python -m unittest \
  deploy.task.ri_4438_him.test_sim2sim \
  tools.test_gamepad_diagnostic
```

The current shared validation snapshot ran 75 focused unittest cases and
recorded 75/75 passing. This count describes that validation run and is not a
promise that future versions will retain the same number of tests.

Both an 800-step no-policy run and an 800-step run using the repository-relative
example ONNX above completed successfully in headless `both` mode. No physical
controller was available during validation: SDL behavior was covered with fake
devices and the no-device path, but actual stick numbering, wireless dongle
behavior, and Linux permissions must still be confirmed with the diagnostic on
the target machine.

This launcher currently uses the MuJoCo simulator backend only. The repository
does not include the RI-4438 motor/CAN/DDS SDK, so do not connect this FSM
directly to hardware.
