"""First-episode accounting and descriptive stability diagnostics (CPU-only)."""

import numpy as np


COMMANDS = (
    ("stand", "站立", (0.0, 0.0, 0.0)),
    ("forward", "前进 0.5 m/s", (0.5, 0.0, 0.0)),
    ("fast", "前进 1.0 m/s", (1.0, 0.0, 0.0)),
    ("backward", "后退 0.5 m/s", (-0.5, 0.0, 0.0)),
    ("lateral", "侧移 0.5 m/s", (0.0, 0.5, 0.0)),
    ("turn", "转向 0.5 rad/s", (0.0, 0.0, 0.5)),
)


class EpisodeMetrics:
    """Terminal states count; reset episodes never contribute again."""

    def __init__(self, num_envs, dt, commands=COMMANDS, tilt_limit=20.0,
                 stair_goal=None, stair_gain=0.0):
        self.dt, self.commands, self.tilt_limit = dt, commands, tilt_limit
        self.stair_goal, self.stair_gain = stair_goal, stair_gain
        self.alive = np.ones(num_envs, dtype=bool)
        self.failed = np.zeros(num_envs, dtype=bool)
        self.count = np.zeros(num_envs, dtype=int)
        self.sums = {k: np.zeros(num_envs) for k in (
            "linear", "yaw", "tilt", "action_rate", "action_accel", "clip", "tilt_exceed",
            "roll", "roll2", "pitch", "pitch2", "body_rate", "vertical_vel", "torque", "power")}
        self.previous_action = self.previous_delta = None
        self.tilt_samples = []
        self.initial_positions = None
        self.progress = np.zeros(num_envs)
        self.height_change = np.zeros(num_envs)
        self.completed = np.zeros(num_envs, dtype=bool)
        self.completion_time = np.zeros(num_envs)

    def update(self, velocity, yaw, gravity_z, actions, command, terminated, truncated, action_clip,
               *, gravity=None, angular_velocity=None, vertical_velocity=None,
               torque=None, joint_velocity=None, position=None):
        active = self.alive.copy()
        arrays = (velocity, yaw, gravity_z, actions, command, gravity, angular_velocity,
                  vertical_velocity, torque, joint_velocity, position)
        if any(v is not None and not np.isfinite(v[active]).all() for v in arrays):
            raise ValueError("non-finite physical state/action during simulation; evaluation aborted")
        self.count += active
        zeros = np.zeros(len(active))
        tilt = np.degrees(np.arccos(np.clip(-gravity_z, -1, 1)))
        roll = zeros if gravity is None else np.degrees(np.arctan2(-gravity[:, 1], -gravity[:, 2]))
        pitch = zeros if gravity is None else np.degrees(np.arcsin(np.clip(gravity[:, 0], -1, 1)))
        delta = None if self.previous_action is None else (actions - self.previous_action) / self.dt
        values = {
            "linear": np.square(velocity - command[:, :2]).sum(axis=1),
            "yaw": np.square(yaw - command[:, 2]), "tilt": tilt**2,
            "roll": roll, "roll2": roll**2, "pitch": pitch, "pitch2": pitch**2,
            "tilt_exceed": (tilt > self.tilt_limit).astype(float),
            "action_rate": zeros if delta is None else np.square(delta).mean(axis=1),
            "action_accel": zeros if delta is None or self.previous_delta is None else
                np.square((delta - self.previous_delta) / self.dt).mean(axis=1),
            "clip": (np.abs(actions) >= action_clip - 1e-6).mean(axis=1) if action_clip is not None else zeros,
            "body_rate": zeros if angular_velocity is None else np.square(angular_velocity[:, :2]).sum(axis=1),
            "vertical_vel": zeros if vertical_velocity is None else vertical_velocity**2,
            "torque": zeros if torque is None else np.square(torque).mean(axis=1),
            "power": zeros if torque is None or joint_velocity is None else np.abs(torque * joint_velocity).sum(axis=1),
        }
        if any(not np.isfinite(v[active]).all() for v in values.values()):
            raise ValueError("non-finite derived metric during simulation")
        for key, value in values.items():
            self.sums[key] += np.where(active, value, 0)
        self.tilt_samples.append(np.where(active, tilt, np.nan))
        if position is not None and self.initial_positions is not None:
            self.progress = np.where(active, position[:, 0] - self.initial_positions[:, 0], self.progress)
            self.height_change = np.where(active, position[:, 2] - self.initial_positions[:, 2], self.height_change)
            if self.stair_goal is not None:
                reached = active & ~terminated & ~self.completed & (position[:, 0] >= self.stair_goal)
                reached &= np.abs(self.height_change - self.stair_gain) < 0.12
                self.completion_time[reached] = self.count[reached] * self.dt
                self.completed |= reached
        self.failed |= active & terminated
        self.alive &= ~(terminated | truncated)
        self.previous_action, self.previous_delta = actions.copy(), delta
        return active

    def rows(self, steps):
        rows = []
        tilts = np.asarray(self.tilt_samples)
        for i, count in enumerate(self.count):
            success = bool(not self.failed[i] and count == steps)
            row = {"env_id": i, "command": self.commands[i % len(self.commands)][0],
                   "success": success, "task_success": success and (self.stair_goal is None or bool(self.completed[i])),
                   "failed": bool(self.failed[i]), "survival_s": count * self.dt,
                   "stair_completed": bool(self.completed[i]) if self.stair_goal is not None else None,
                   "completion_time_s": float(self.completion_time[i]) if self.completed[i] else None,
                   "progress_m": float(self.progress[i]), "height_change_m": float(self.height_change[i])}
            for source, name in (("linear", "linear_rmse"), ("yaw", "yaw_rmse"), ("tilt", "tilt_rms_deg"),
                                 ("action_rate", "action_rate_rms"), ("action_accel", "action_accel_rms"),
                                 ("body_rate", "body_rate_rms"), ("vertical_vel", "vertical_vel_rms"), ("torque", "torque_rms"),
                                 ("clip", "action_clip_fraction"), ("tilt_exceed", "tilt_exceed_fraction"), ("power", "power_abs_w")):
                denominator = max(1, count - {"action_rate": 1, "action_accel": 2}.get(source, 0))
                value = self.sums[source][i] / denominator
                row[name] = float(value if source in ("clip", "tilt_exceed", "power") else np.sqrt(value))
            for angle in ("roll", "pitch"):
                variance = self.sums[angle + "2"][i] / max(1, count) - (self.sums[angle][i] / max(1, count))**2
                row[angle + "_std_deg"] = float(np.sqrt(max(0, variance)))
            samples = tilts[:count, i] if len(tilts) else np.array([0.0])
            row["tilt_p95_deg"] = float(np.percentile(samples, 95)) if count else 0.0
            row["tilt_peak_deg"] = float(np.max(samples)) if count else 0.0
            row["energy_abs_j"] = float(self.sums["power"][i] * self.dt)
            rows.append(row)
        return rows


METRIC_NAMES = ("success", "task_success", "survival_s", "linear_rmse", "yaw_rmse", "tilt_rms_deg",
                "action_rate_rms", "action_clip_fraction", "tilt_p95_deg", "tilt_peak_deg", "roll_std_deg",
                "pitch_std_deg", "body_rate_rms", "vertical_vel_rms", "tilt_exceed_fraction",
                "action_accel_rms", "torque_rms", "power_abs_w", "energy_abs_j")


def summarize(rows):
    summaries = []
    from .training_logs import model_key, model_origin
    for key in sorted({model_key(r) for r in rows}, key=str):
        sample = [r for r in rows if model_key(r) == key]
        summary = {"iteration": sample[0]["iteration"], **model_origin(sample[0]), "episodes": len(sample)}
        for key in METRIC_NAMES:
            values = [r[key] for r in sample]
            summary[key] = float(np.mean(values))
            summary[key + "_std"] = float(np.std(values))
        groups = sorted({(r.get("terrain", "flat"), r["command"]) for r in sample})
        cases = []
        for terrain, command in groups:
            group = [r for r in sample if r.get("terrain", "flat") == terrain and r["command"] == command]
            cases.append({"terrain": terrain, "command": command, "episodes": len(group),
                          **{k: float(np.mean([r[k] for r in group])) for k in METRIC_NAMES}})
        summary["cases"] = cases
        summary["worst_case_success"] = min(c["task_success"] for c in cases)
        summary["worst_case_linear_rmse"] = max(c["linear_rmse"] for c in cases)
        summary["linear_rmse_p90"] = float(np.percentile([r["linear_rmse"] for r in sample], 90))
        summary["tilt_peak_max_deg"] = max(r["tilt_peak_deg"] for r in sample)
        summary["failures"] = sum(r["failed"] for r in sample)
        seeds = sorted({r.get("seed", 0) for r in sample})
        summary["seeds"] = [{"seed": seed, **{k: float(np.mean([r[k] for r in sample if r.get("seed", 0) == seed]))
                                             for k in ("task_success", "linear_rmse", "tilt_rms_deg")}} for seed in seeds]
        summary["seed_linear_min"] = min(r["linear_rmse"] for r in summary["seeds"])
        summary["seed_linear_max"] = max(r["linear_rmse"] for r in summary["seeds"])
        summary["stair_pass_rate"] = (float(np.mean([r["task_success"] for r in sample if r["stair_completed"] is not None]))
                                      if any(r["stair_completed"] is not None for r in sample) else None)
        summaries.append(summary)
    return sorted(summaries, key=lambda r: (-r["task_success"], -r["success"], -r["survival_s"],
                                           r["linear_rmse"], r["yaw_rmse"], r["tilt_rms_deg"], r.get("source_index", 0), r["iteration"]))
