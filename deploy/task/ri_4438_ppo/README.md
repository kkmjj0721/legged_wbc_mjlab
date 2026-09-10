# RI-4438 MuJoCo sim2sim

The runner adds the missing floor and 12 position actuators to the RI-4438
MJCF at runtime, so the source robot XML is left unchanged. It uses the
training-compatible `0.005 s` simulation step and policy decimation `4`.

Install the optional deployment dependencies from the repository root:

```bash
uv sync --extra sim2sim
```

Run the graphical simulator with the exported actor through the unified entry point:

```bash
python deploy/main.py --task ri_4438_ppo
```

For a dependency-light model/FSM smoke test, disable ONNX and render a fixed
number of steps:

```bash
python deploy/main.py --task ri_4438_ppo --no-policy --headless --auto-stand --steps 800
```

Controls:

- Keyboard in the MuJoCo window: `W/S` for forward/backward, `A/D` for
  lateral motion, `Q/E` for yaw, `U` to stand, `P` or space for passive, `R`
  to reset, `F` to toggle robot-follow/free camera, `Esc` to quit. The camera
  follows `base_link` by default. Motion keys are toggles because the MuJoCo passive
  callback reports a key press without a release: press `W` once to move and
  press `W` again to stop. Press `P`/space before standing again to clear a
  command.
- Gamepad: left stick controls forward/lateral velocity, right-stick X controls
  yaw, `A/Cross` requests stand, `B/Circle` enters passive, Start resets, and
  Select exits.

The command-line options select the runtime, not the motion itself:

```bash
python deploy/main.py --task ri_4438_ppo --input both
```

Use `--input keyboard` or `--input joystick` to restrict input devices,
`--headless --steps N` for a fixed-duration test, `--auto-stand` to enter the
stand-up state automatically, and `--no-policy` to test the backend/FSM without
the actor.

The pose, policy path, gains, command limits, fall protection and input backend
are configurable in [config/sim2sim.yaml](config/sim2sim.yaml). `pygame` is optional; if no
gamepad is available, keyboard input remains usable.

The task configuration is also the source of truth for `policy.observation_dim`,
`policy.action_dim`, `policy.command_dim`, `policy.joint_names` and
`policy.observation_terms`. The shared code validates the ONNX shapes and the
constructed observation against those configured values instead of assuming
the RI-4438 dimensions.

Camera follow is configured under `camera` in the task YAML. `track_body`,
`distance`, `azimuth`, and `elevation` can be changed for another robot.

The reusable deployment layers live under `deploy/include/`: `fsm.py` contains
the state machine, `mujoco_bridge.py` is the simulation backend, `policy.py`
loads the actor, `inputs.py` handles keyboard/gamepad input, and `runtime.py`
is the backend-independent control loop. This layout leaves room for a real
robot backend implementing the same interface as `MujocoBackend`.
