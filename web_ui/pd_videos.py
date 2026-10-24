"""Render measured joint-response trajectories with the imported robot geometry.

These clips replay the CPU experiment, one tested joint at a time.  They do not
run a controller on the whole robot or suggest a trained standing/walking gait.
"""

from pathlib import Path


FPS = 25
WIDTH, HEIGHT = 640, 480


def create_comparison_videos(model, nominal_data, bindings, candidates,
                             recommended_id, video_dir, progress=print):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    destination = Path(video_dir).resolve()
    artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
    if not destination.is_relative_to(artifacts):
        raise ValueError("响应视频目录必须位于本项目 .artifacts 内")
    destination.mkdir(parents=True, exist_ok=True)
    baseline = candidates[0]
    selected = next((item for item in candidates if item["id"] == recommended_id), None)
    if selected is None or selected is baseline:
        bank = [item for item in candidates if item["id"] != baseline["id"]]
        selected = min(bank, key=lambda item: item["summary"]["score"]) if bank else None
    comparisons = [baseline] + ([selected] if selected is not None else [])
    data = mujoco.MjData(model)
    nominal_qpos = nominal_data.qpos.copy()
    base_id = model.body(bindings["base_body"]).id
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = nominal_data.xpos[base_id]
    camera.distance = max(0.7, float(model.stat.extent) * 1.6)
    camera.azimuth, camera.elevation = 135, -25
    scene_options = mujoco.MjvOption()
    scene_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    clips = []
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for candidate in comparisons:
            filename = f"pd-response-{candidate['id']}.mp4"
            path = destination / filename
            progress(f"录制响应视频：{candidate['label']}（模型轨迹回放，非站立/步态测试）……")
            representatives = [next(joint for joint in candidate["joints"] if joint["role"] == role)
                               for role in ("hip", "thigh", "calf")]
            frame_count = 0
            try:
                with imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7,
                                        macro_block_size=16, ffmpeg_log_level="error") as writer:
                    for joint in representatives:
                        trace = joint["trace"]
                        duration = float(trace["time"][-1])
                        duration = max(1.0 / FPS, min(2.0, duration))
                        joint_qpos = int(model.joint(joint["name"]).qposadr[0])
                        for frame in range(max(1, round(duration * FPS))):
                            moment = frame / FPS
                            angle = float(np.interp(moment, trace["time"], trace["position"]))
                            target = float(np.interp(moment, trace["time"], trace["target"]))
                            data.qpos[:] = nominal_qpos
                            data.qpos[joint_qpos] = angle
                            data.qvel[:] = 0
                            mujoco.mj_forward(model, data)
                            renderer.update_scene(data, camera=camera, scene_option=scene_options)
                            image = Image.fromarray(renderer.render())
                            draw = ImageDraw.Draw(image, "RGBA")
                            draw.rectangle((0, 0, WIDTH, 68), fill=(252, 247, 237, 235))
                            draw.text((12, 8), f"PD response | {candidate['id']} | {joint['name']}", fill=(65, 43, 26))
                            draw.text((12, 26), f"Kp {joint['kp']:.3f}   Kd {joint['kd']:.3f}   t {moment:.2f} s", fill=(65, 43, 26))
                            draw.text((12, 44), f"q {angle:.4f} rad   target {target:.4f} rad", fill=(65, 43, 26))
                            draw.rectangle((0, HEIGHT - 30, WIDTH, HEIGHT), fill=(252, 247, 237, 235))
                            draw.text((12, HEIGHT - 21), "Fixed base / other joints locked / no gravity or contacts", fill=(65, 43, 26))
                            writer.append_data(np.asarray(image))
                            frame_count += 1
                clips.append({"filename": filename, "path": str(path), "mime": "video/mp4",
                              "label": candidate["label"], "candidate_id": candidate["id"],
                              "recommended": candidate["id"] == recommended_id,
                              "passed": candidate["summary"]["eligible"], "fps": FPS,
                              "frames": frame_count, "duration_s": frame_count / FPS,
                              "width": WIDTH, "height": HEIGHT,
                              "kind": "measured_trajectory_replay",
                              "conditions": "固定基座、其余关节锁定、无重力/接触；依次回放 hip/thigh/calf 的受控响应；不是站立或训练步态。"})
            except Exception:
                path.unlink(missing_ok=True)
                raise
    return clips
