"""Record scoring states and render genuine MuJoCo images after ranking."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .evaluation_scenarios import IMAGE_MODEL_LIMIT, TERRAIN_LABELS
from .training_logs import model_key, model_label, model_origin


class FrameRecorder:
    """Same forward-command episode and times for every model; never capture resets."""

    def __init__(self, batch, samples, commands, terrain, seed, steps, dt):
        self.batch, self.samples = batch, samples
        self.slot = next(i for i, c in enumerate(commands) if c[0] == "forward")
        self.terrain, self.seed, self.dt = terrain, seed, dt
        self.targets = {min(steps - 1, max(0, math.ceil(t / dt) - 1)) for t in (2.0, 6.0, steps * dt)}
        self.finished = set()

    def capture(self, step, sim_data, active, done, reasons, root_positions, frames):
        for index, model in enumerate(self.batch):
            env_id = index * self.samples + self.slot
            if index in self.finished or not active[env_id]:
                continue
            if step not in self.targets and not done[env_id]:
                continue
            frame = {"iteration": model["iteration"], **model_origin(model), "terrain": self.terrain, "seed": self.seed,
                     "env_id": self.slot, "command": "forward", "velocity": [0.5, 0.0, 0.0],
                     "time_s": (step + 1) * self.dt, "terminal": bool(done[env_id]),
                     "failure_reason": reasons[env_id],
                     "root_position": root_positions[env_id].detach().cpu().tolist()}
            for field in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
                value = getattr(sim_data, field, None)
                if value is not None:
                    frame[field] = value[env_id].detach().cpu().tolist()
            frames.append(frame)
            if done[env_id]:
                self.finished.add(index)


def render_simulation_images(data, output):
    """Use an isolated EGL process so rendering never changes scoring physics."""
    iterations = [model_key(r) for r in data["summary"][:IMAGE_MODEL_LIMIT]]
    directory = output / "simulation"
    request = directory / "frames.json"
    request.write_text(json.dumps({
        "model_ids": iterations,
        "frames": [f for f in data["simulation_frames"] if model_key(f) in iterations],
        "environments": data["environments"],
        "seed": data["options"]["seeds"][0],
    }, ensure_ascii=False, allow_nan=False))
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    print(f"[images] Rendering recorded scoring states for top {len(iterations)} models", flush=True)
    with (directory / "render.log").open("w") as log:
        process = subprocess.run([sys.executable, "-m", "src.utils.evaluation_images", str(request)],
                                 cwd=Path(__file__).resolve().parents[2], env=env, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        tail = (directory / "render.log").read_text()[-1800:]
        raise RuntimeError(f"simulation image rendering failed; see {directory / 'render.log'}\n{tail}")
    return json.loads((directory / "images.json").read_text())


def _render(request_path):
    import mujoco
    import numpy as np
    from .evaluation_report import _plot_setup
    from .training_logs import sha256

    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    model_ids = request.get("model_ids", request.get("iterations"))
    directory, output = request_path.parent, request_path.parent.parent
    plt = _plot_setup()
    images = []
    for terrain, description in request["environments"].items():
        model_file = output / description["render_model"]
        if sha256(model_file) != description["render_model_sha256"]:
            raise ValueError(f"recorded render model changed: {model_file}")
        model = mujoco.MjModel.from_binary_path(str(model_file))
        model.vis.global_.offwidth, model.vis.global_.offheight = 720, 450
        model.stat.extent = 6.0
        state = mujoco.MjData(model)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth, camera.elevation = 90, -20
        camera.distance = 2.0 if terrain.startswith("stairs_") else 1.5
        option = mujoco.MjvOption()
        option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        option.sitegroup[:] = 0
        fig, axes = plt.subplots(len(model_ids), 3,
                                 figsize=(15, 3.6 * len(model_ids) + 0.8), squeeze=False)
        metadata = []
        with mujoco.Renderer(model, height=450, width=720) as renderer:
            for row, identity in enumerate(model_ids):
                frames = sorted((f for f in request["frames"] if f["terrain"] == terrain and model_key(f) == identity),
                                key=lambda f: f["time_s"])
                for col, ax in enumerate(axes[row]):
                    ax.axis("off")
                    if col >= len(frames):
                        message = "首次回合已终止\n不展示重置后的画面" if frames and frames[-1]["terminal"] else "回合已结束\n无更多采样画面"
                        ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes)
                        continue
                    frame = frames[col]
                    state.qpos[:] = frame["qpos"]
                    state.qvel[:] = frame["qvel"]
                    if model.nmocap:
                        state.mocap_pos[:] = frame["mocap_pos"]
                        state.mocap_quat[:] = frame["mocap_quat"]
                    mujoco.mj_forward(model, state)
                    camera.lookat[:] = np.asarray(frame["root_position"]) + [0.35, 0, 0.1]
                    renderer.update_scene(state, camera=camera, scene_option=option)
                    rgb = renderer.render()
                    filename = f"model_{str(identity).replace(':', '_')}_{terrain}_{col}.png"
                    plt.imsave(directory / filename, rgb)
                    ax.imshow(rgb)
                    suffix = " · 首次失败" if frame["failure_reason"] else " · 回合结束" if frame["terminal"] else ""
                    ax.set_title(f"#{row+1} {model_label(frame)} · {frame['time_s']:.2f} s{suffix}", fontsize=11)
                    metadata.append({k: v for k, v in frame.items() if k not in ("qpos", "qvel", "mocap_pos", "mocap_quat")}
                                    | {"image": f"simulation/{filename}"})
        fig.suptitle(f"{TERRAIN_LABELS[terrain]} · 真实评测截图 · seed={request['seed']} · 前进 0.5 m/s\n"
                     "同一局部环境编号，采样 2 s / 6 s / 结束时；提前失败则截取首次终止状态", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.0)
        name = f"scene_{terrain}.png"
        fig.savefig(directory / name, dpi=140)
        plt.close(fig)
        images.append({"terrain": terrain, "label": TERRAIN_LABELS[terrain], "image": f"simulation/{name}",
                       "model_ids": model_ids, "seed": request["seed"], "frames": metadata,
                       "camera": {"azimuth": 90, "elevation": -20, "distance": camera.distance,
                                  "lookat_offset": [0.35, 0, 0.1], "width": 720, "height": 450},
                       "render_model_sha256": description["render_model_sha256"]})
        print(f"Rendered {terrain}: {len(metadata)} original frames", flush=True)
    (directory / "images.json").write_text(json.dumps(images, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _render(sys.argv[1])
