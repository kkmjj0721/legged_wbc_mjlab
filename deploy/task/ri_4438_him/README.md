# RI-4438 HIM MuJoCo sim2sim

This task runs the six-frame RI-4438 HIM actor in the shared combined terrain
scene. The exported policy consumes `6 x 47 = 282` observations in
frame-major, current-first order (`t, t-1, ..., t-5`) and produces 12 joint
actions. Observation normalization is already embedded in the exported ONNX;
the deploy process does not normalize the input again or clip the policy output.

Run from the repository root:

```bash
python deploy/main.py --task ri_4438_him
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

Controls:

- `U` / Up / gamepad A: start the controlled get-up transition.
- `P` / `Space` / `G` / gamepad Y (Triangle): leave RL through the controlled
  get-down transition and remain in `PASSIVE`.
- Gamepad A (Cross): request GETUP; X (Square): enable RL after the stand
  settle sequence; B (Circle): disable RL and hold the standing pose.
- `R` / gamepad Start: reset the robot, policy history, action, and command;
  the configured one-shot auto-stand becomes available again.
- `Esc` / gamepad Back/Select: emergency-stop and exit.
- `W/S`, `A/D`, and `Q/E`: command forward/backward, lateral, and yaw motion.

The robot starts on the checkerboard area. A terrain preset changes only its
spawn location while retaining the single combined scene:

```bash
python deploy/main.py --task ri_4438_him --terrain stairs_5cm
python deploy/main.py --task ri_4438_him --terrain heightfield
```

For a backend/FSM smoke test without loading ONNX Runtime:

```bash
python deploy/main.py --task ri_4438_him --no-policy --headless --auto-stand --steps 800
```

`--no-policy` preserves the MuJoCo and FSM flow but outputs zero policy action.
The sim2sim path intentionally does not reproduce training-time observation
noise or actuator delay; those remain known dynamics-randomization gaps rather
than policy input-contract differences.

The compatibility launcher below is equivalent:

```bash
python deploy/task/ri_4438_him/main.py
```

Gamepad support uses pygame/SDL and is hot-plug safe. Install the optional
dependency with `uv sync --extra sim2sim`; the process falls back to keyboard
input when pygame is unavailable. The default SDL mapping works with the
Beitong Asura 2 Pro (星闪版). If several controllers are connected, select one
explicitly with `--gamepad-name`, `--gamepad-guid`, or `--gamepad-index`.

This launcher currently uses the MuJoCo simulator backend only. The repository
does not include the RI-4438 motor/CAN/DDS SDK, so do not connect this FSM
directly to hardware. A physical 星闪 controller was not available during
validation; confirm the detected device name/GUID and axis indices from the
startup diagnostics before relying on it.
