"""Render measured PD trajectories with the imported robot geometry.

Ground clips replay every free-base and joint coordinate from the complete
contact experiment.  The legacy single-joint replay remains available for old
callers.  Neither replay represents a trained policy gait.
"""

from pathlib import Path
import math
import re


FPS = 25
WIDTH, HEIGHT = 640, 480
GROUND_FPS = 30
GROUND_WIDTH, GROUND_HEIGHT = 960, 720


def _ground_trials(candidate):
    return [trial for trial in candidate.get("ground", {}).get("trials", [])
            if trial.get("trajectory", {}).get("time")
            and trial.get("trajectory", {}).get("qpos")]


def diagnostic_candidate(candidates):
    """Choose one measured, non-reference failure for review, never application."""
    tested = [item for item in candidates if item.get("ground", {}).get("trials")
              or item.get("ground", {}).get("status") in ("passed", "failed")]
    bank = [item for item in tested if item.get("id") != "current"
            and not item.get("reference_only")]
    if not bank:
        return next((item for item in tested if item.get("id") == "current"), None)

    def score(item):
        value = item.get("summary", {}).get("score")
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else float("inf")

    qualified = [item for item in bank if item.get("summary", {}).get("joint_eligible")]
    return min(qualified, key=score) if qualified else min(
        bank, key=lambda item: (-item.get("summary", {}).get("passed_joints", 0), score(item)))


def _ground_comparisons(candidates, recommended_id):
    """Show the measured reference and recommendation, or a failed diagnostic."""
    measured = [candidate for candidate in candidates if _ground_trials(candidate)]
    if not measured:
        return []
    reference = next((item for item in measured if item.get("id") == "current"), measured[0])
    selected = next((item for item in measured if item.get("id") == recommended_id), None)
    if selected is None:
        selected = diagnostic_candidate(measured)
    return [reference] + ([selected] if selected is not None and selected is not reference else [])


def create_ground_videos(spec, bindings, parameters, analysis, candidates,
                         recommended_id, video_dir, progress=print):
    """Replay the measured, unconstrained robot contact trajectories at 720p.

    Frames select complete recorded states, including the free base.  In
    particular, no initial-pose restoration or artificial base stabilization is
    applied.  Contacts are recomputed only for displaying the recorded state;
    they are not re-simulated with a different controller during video creation.
    """
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from .pd_ground import build_model

    destination = Path(video_dir).resolve()
    artifacts = Path(__file__).resolve().parents[1] / ".artifacts"
    if not destination.is_relative_to(artifacts):
        raise ValueError("落地测试视频目录必须位于本项目 .artifacts 内")
    comparisons = _ground_comparisons(candidates, recommended_id)
    if not comparisons:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    model, initial_data, metadata = build_model(spec, bindings, parameters, analysis)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), GROUND_WIDTH)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), GROUND_HEIGHT)
    # Imported assets often have no lights.  This changes display lighting only;
    # the controller, contact geometry and measured trajectory stay identical.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    model.vis.headlight.diffuse[:] = (0.8, 0.8, 0.8)
    data = mujoco.MjData(model)
    base_id = model.body(bindings["base_body"]).id
    timing = metadata.get("timing", analysis.get("timing", {}))

    def frequency(name, fallback=None):
        value = timing.get(name, fallback)
        return f"{float(value):g}" if isinstance(value, (int, float)) and math.isfinite(value) else "?"

    physics_hz = frequency("physics_hz")
    pd_hz = frequency("pd_hz", timing.get("physics_hz"))
    policy_hz = frequency("policy_hz")
    delay = timing.get("hardware_delay_ms")
    delay_text = f"{float(delay):g} ms" if isinstance(delay, (int, float)) else "unspecified"
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = initial_data.xpos[base_id]
    # Plane size and distant world geometry must not determine the robot view.
    # Body positions give a stable, robot-sized view even with an infinite floor.
    body_spread = np.linalg.norm(initial_data.xpos[1:] - initial_data.xpos[base_id], axis=1)
    radius = float(np.max(body_spread)) if body_spread.size else 0.4
    camera.distance = max(1.0, radius * 3.5)
    camera.azimuth, camera.elevation = 135, -22
    scene_options = mujoco.MjvOption()
    floor_id = model.geom(metadata.get("floor_geom", "pd_ground")).id
    floor_z = float(initial_data.geom_xpos[floor_id, 2]) + 0.001

    def show_ground_reference(scene):
        """Draw a world-anchored 25 cm grid for judging measured slip/drift."""
        center = np.round(data.xpos[base_id, :2] / 0.25) * 0.25
        for offset in range(-8, 9):
            for axis in (0, 1):
                if scene.ngeom >= scene.maxgeom:
                    return
                start = np.array([center[0] - 2, center[1] - 2, floor_z])
                end = np.array([center[0] + 2, center[1] + 2, floor_z])
                start[axis] = end[axis] = center[axis] + offset * 0.25
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE,
                                    np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.array([0.45, 0.50, 0.56, 0.6]))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, start, end)
                geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                scene.ngeom += 1
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = small_font = ImageFont.load_default()
    clips = []
    with mujoco.Renderer(model, height=GROUND_HEIGHT, width=GROUND_WIDTH) as renderer:
        for candidate in comparisons:
            candidate_id = str(candidate["id"])
            safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", candidate_id).strip("-")
            if not safe_id:
                raise ValueError("落地测试候选名称不能生成安全的视频文件名")
            ground = candidate.get("ground", {})
            diagnostic = recommended_id is None
            ground_passed = bool(ground.get("passed"))
            passed = bool(ground_passed and candidate.get("summary", {}).get("eligible", True)
                          and not diagnostic)
            recommended = candidate_id == recommended_id and passed
            label = candidate.get("label", candidate_id)
            video_label = f"{label} · " + ("本次推荐 · 落地通过" if recommended else
                                           "未通过 · 诊断回放" if diagnostic else
                                           "基准 · 落地通过" if passed else "基准 · 未通过")
            filename = f"pd-ground-{safe_id}.mp4"
            path = destination / filename
            progress(f"录制落地测试视频：{label}（完整机器人真实接触轨迹）……")
            frame_count, trial_names = 0, []
            gain_rows = candidate.get("proposal", {}).get("control", {})
            stiffness, damping = gain_rows.get("stiffness", {}), gain_rows.get("damping", {})

            def gain_range(values):
                numbers = [float(value) for value in values.values()
                           if isinstance(value, (int, float)) and math.isfinite(value)] if isinstance(values, dict) else []
                if not numbers:
                    return "?"
                low, high = min(numbers), max(numbers)
                return f"{low:.3g}" if low == high else f"{low:.3g}..{high:.3g}"

            try:
                with imageio.get_writer(str(path), fps=GROUND_FPS, codec="libx264", quality=8,
                                        macro_block_size=16, ffmpeg_log_level="error") as writer:
                    for trial in _ground_trials(candidate):
                        trace = trial["trajectory"]
                        times = np.asarray(trace["time"], dtype=float)
                        positions = np.asarray(trace["qpos"], dtype=float)
                        if (times.ndim != 1 or positions.shape != (len(times), model.nq)
                                or not np.isfinite(times).all() or not np.isfinite(positions).all()
                                or np.any(np.diff(times) <= 0)):
                            raise ValueError(f"{candidate_id}/{trial.get('name')} 的完整落地轨迹格式无效")
                        name = str(trial.get("name", "ground"))
                        trial_names.append(name)
                        duration = max(0.0, float(times[-1] - times[0]))
                        frames = max(1, int(math.ceil(duration * GROUND_FPS)) + 1)
                        for frame in range(frames):
                            moment = min(float(times[-1]), float(times[0]) + frame / GROUND_FPS)
                            index = min(len(times) - 1, max(0, int(np.searchsorted(times, moment, side="right")) - 1))
                            data.qpos[:] = positions[index]
                            data.qvel[:] = 0
                            data.time = float(times[index])
                            mujoco.mj_forward(model, data)
                            # Follow the *measured* base, including tipping or falling.
                            camera.lookat[:] = data.xpos[base_id]
                            renderer.update_scene(data, camera=camera, scene_option=scene_options)
                            show_ground_reference(renderer.scene)
                            image = Image.fromarray(renderer.render())
                            draw = ImageDraw.Draw(image, "RGBA")
                            draw.rectangle((0, 0, GROUND_WIDTH, 112), fill=(252, 247, 237, 240))
                            status = "PASS" if passed else "FAIL"
                            role = "RECOMMENDED" if recommended else ("DIAGNOSTIC / NO RECOMMENDATION" if diagnostic else "REFERENCE")
                            draw.text((14, 9), f"Ground PD | {candidate_id} | {status} | {role}", font=font, fill=(65, 43, 26))
                            trial_status = "PASS" if trial.get("passed") else "FAIL"
                            draw.text((14, 35), f"Trial: {name} [{trial_status}]   t={float(times[index]):.2f} s   Kp {gain_range(stiffness)}   Kd {gain_range(damping)}", font=font, fill=(65, 43, 26))
                            draw.text((14, 61), f"Physics {physics_hz} Hz | PD {pd_hz} Hz | target update {policy_hz} Hz", font=font, fill=(65, 43, 26))
                            draw.text((14, 87), f"Free base | gravity + contacts | configured delay {delay_text}", font=small_font, fill=(65, 43, 26))
                            draw.rectangle((0, GROUND_HEIGHT - 58, GROUND_WIDTH, GROUND_HEIGHT), fill=(252, 247, 237, 240))
                            draw.text((14, GROUND_HEIGHT - 49), "Recorded full dynamics trajectory; camera follows the measured base", font=small_font, fill=(65, 43, 26))
                            draw.text((14, GROUND_HEIGHT - 26), "Initial pose hold and push recovery; no trained policy or gait", font=small_font, fill=(65, 43, 26))
                            writer.append_data(np.asarray(image))
                            frame_count += 1
                clips.append({
                    "filename": filename, "path": str(path), "mime": "video/mp4",
                    "label": video_label, "candidate_id": candidate_id, "recommended": recommended,
                    "passed": passed, "ground_passed": ground_passed,
                    "diagnostic": diagnostic, "trial_names": trial_names,
                    "fps": GROUND_FPS, "frames": frame_count, "duration_s": frame_count / GROUND_FPS,
                    "width": GROUND_WIDTH, "height": GROUND_HEIGHT, "kind": "ground_trajectory",
                    "timing": timing,
                    "display": {"ground_grid_m": 0.25, "camera_tracks_measured_base": True},
                    "conditions": "完整机器人自由基座，开启重力和真实地面接触；按实际测试时序保持确认初姿并测试扰动恢复。视频回放测得的全部机身与关节状态，不是训练策略步态。",
                    "reasons": ground.get("reasons", []),
                })
            except Exception:
                path.unlink(missing_ok=True)
                raise
    return clips


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
