# Aligned with standalone RI-4438 HIM rewards (workspace 2026-10-12).
"""Basic reward terms for the standalone RI-4438 HIM task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_apply_yaw
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .gait import gait_phase as _gait_phase
from .motion import command_motion_mask

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    low_speed_std: float | None = None,
    low_speed_threshold: float = 0.1,
    high_speed_threshold: float = 0.5,
) -> torch.Tensor:
    """Track body-frame XY velocity with an optional command-dependent scale.

    The low-speed scale applies to every terrain and also zero translation.
    Smoothstep restores ``std`` between the two command-speed thresholds. The
    scale depends only on commanded translation, so tracking errors cannot earn
    a wider tolerance. Omitting ``low_speed_std`` preserves the original kernel.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.square(command[:, :2] - actual[:, :2]).sum(dim=1)
    if low_speed_std is None:
        return torch.exp(-xy_error / std**2)
    if not (
        math.isfinite(std)
        and math.isfinite(low_speed_std)
        and 0.0 < low_speed_std <= std
        and math.isfinite(low_speed_threshold)
        and math.isfinite(high_speed_threshold)
        and 0.0 <= low_speed_threshold < high_speed_threshold
    ):
        raise ValueError(
            "Require finite 0 < low_speed_std <= std and "
            "0 <= low_speed_threshold < high_speed_threshold."
        )
    speed = torch.linalg.norm(command[:, :2], dim=1)
    blend = (
        (speed - low_speed_threshold) / (high_speed_threshold - low_speed_threshold)
    ).clamp(0.0, 1.0)
    blend = blend.square() * (3.0 - 2.0 * blend)
    scale = low_speed_std + (std - low_speed_std) * blend
    return torch.exp(-xy_error / scale.square())


def lin_vel_z(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize squared base linear velocity along the body-frame z axis."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Track yaw velocity while penalizing roll and pitch motion."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    return torch.exp(-z_error / std**2)


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize the horizontal projection of gravity in the body frame."""
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        gravity = quat_apply_inverse(quat, asset.data.gravity_vec_w)
    else:
        gravity = asset.data.projected_gravity_b
    return torch.square(gravity[:, :2]).sum(dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    sensor: BuiltinSensor = env.scene[sensor_name]
    magnitude_sq = torch.square(sensor.data).sum(dim=-1)
    env.extras["log"]["Metrics/angular_momentum_mean"] = torch.sqrt(magnitude_sq).mean()
    return magnitude_sq


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    velocity_w = asset.data.body_link_ang_vel_w[
        :, asset_cfg.body_ids, :
    ].squeeze(1)
    quat_w = asset.data.body_link_quat_w[
        :, asset_cfg.body_ids, :
    ].squeeze(1)

    velocity_b = quat_apply_inverse(quat_w, velocity_w)
    return torch.square(velocity_b[:, :2]).sum(dim=1)


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    """Match foot contacts to a fixed-period gait, using contact time directly."""
    sensor: ContactSensor = env.scene[sensor_name]
    contact = sensor.data.current_contact_time > 0
    phase = _gait_phase(env, period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=phase.dtype).view(1, -1)
    desired = ((phase + offsets) % 1.0) < threshold
    reward = (desired == contact).float().mean(dim=1)
    command = env.command_manager.get_command(command_name)
    if command is not None:
        active = command_motion_mask(
            command, command_threshold, angular_command_threshold
        )
        reward *= active
    return reward


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    gait_name: str = "foot_gait",
    command_name: str | None = None,
    command_threshold: float = 0.1,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    """Reward each foot's ongoing support/air duration independently.

    Scores are normalized to [0, 1] before averaging over feet.
    Each foot contributes only while its contact state matches the gait.
    """
    gait = env.cfg.rewards[gait_name].params
    period = gait["period"]
    stance_fraction = gait["threshold"]

    if period <= 0.0 or not 0.0 < stance_fraction < 1.0:
        raise ValueError(
            "Gait period must be positive and stance threshold in (0, 1)."
        )

    # 支撑和腾空使用不同的参考持续时间。
    stance_duration = period * stance_fraction
    swing_duration = period * (1.0 - stance_fraction)

    sensor: ContactSensor = env.scene[sensor_name]
    air_time = sensor.data.current_air_time
    contact_time = sensor.data.current_contact_time

    assert air_time is not None and contact_time is not None, (
        "feet_air_time requires a contact sensor with track_air_time=True."
    )

    if len(gait["offset"]) != contact_time.shape[1]:
        raise ValueError(
            "Gait offsets must match the contact sensor's foot order."
        )

    # 每只脚的目标接触状态，形状为 [num_envs, num_feet]。
    phase = _gait_phase(env, period).unsqueeze(1)
    offsets = torch.as_tensor(
        gait["offset"],
        device=env.device,
        dtype=phase.dtype,
    ).view(1, -1)

    foot_phase = (phase + offsets) % 1.0
    desired_contact = foot_phase < stance_fraction

    # 每只脚实际处于哪个状态，以及该状态已经持续多久。
    in_contact = contact_time > 0.0
    mode_time = torch.where(in_contact, contact_time, air_time)

    target_duration = torch.where(
        in_contact,
        stance_duration,
        swing_duration,
    )

    # 归一化三角形得分：
    # 0 倍参考时间 -> 0 分
    # 1 倍参考时间 -> 1 分
    # 2 倍参考时间及以上 -> 0 分
    duration_ratio = mode_time / target_duration
    duration_score = (
        1.0 - torch.abs(duration_ratio - 1.0)
    ).clamp(min=0.0)

    # 各脚独立门控：某只脚不符合相位，只取消该脚的贡献。
    matches_gait = (in_contact == desired_contact).to(
        duration_score.dtype
    )
    per_foot_reward = duration_score * matches_gait

    # 不再要求恰好两脚接触，也不再取四脚最小值。
    reward = per_foot_reward.mean(dim=1)

    # 保留原有运动指令门控，站立时关闭此奖励。
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command '{command_name}' not found."

        reward *= command_motion_mask(
            command, command_threshold, angular_command_threshold
        ).to(reward.dtype)

    return reward


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_height: float = 0.25,
    gait_name: str = "foot_gait",
    scan_sensor_name: str = "terrain_scan",
    lookahead_distance: float = 0.25,
    path_half_width: float = 0.10,
    foot_radius: float = 0.01573,
    sensor_name: str = "feet_ground_contact",
    stumble_weight: float = 0.1,
    stumble_ratio: float = 5.0,
    stumble_force_threshold: float = 1.0,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    """Track a terrain-relative swing arc and penalize feet striking step risers.

    Read period/offset/stance threshold from ``gait_name`` so this term stays in
    sync with the contact gait. Site order must match that gait's offset order.
    During swing, sole clearance follows sin(pi * swing_phase), with peak
    ``min(target_height + positive_obstacle_rise, max_height)``. Heights are
    relative to the nearest scanned ground under each foot, not world zero.
    The cap limits the target, not the robot's actual motion.

    Look ahead along commanded translation plus yaw motion at each foot, even
    when it is stuck. Missing local terrain samples disable only the height
    cost. No foot-speed/contact gating is applied to swing error: a blocked
    swing must still cost something. The stumble component applies throughout
    the gait and requires world-frame contact forces (e.g. reduce="netforce").
    Set ``stumble_weight=0`` to disable that component.
    """
    if not 0.0 <= target_height <= max_height:
        raise ValueError("Require 0 <= target_height <= max_height.")
    if lookahead_distance <= 0.0 or path_half_width <= 0.0 or foot_radius < 0.0:
        raise ValueError(
            "Scan dimensions must be positive and foot_radius nonnegative."
        )
    if stumble_weight < 0.0 or stumble_ratio <= 0.0 or stumble_force_threshold < 0.0:
        raise ValueError("Invalid stumble weight, force ratio or force threshold.")

    asset: Entity = env.scene[asset_cfg.name]
    feet = asset.data.site_pos_w[:, asset_cfg.site_ids]
    gait = env.cfg.rewards[gait_name].params
    period, threshold = gait["period"], gait["threshold"]
    if period <= 0.0 or not 0.0 < threshold < 1.0:
        raise ValueError("Gait period must be positive and stance threshold in (0, 1).")
    offsets = torch.as_tensor(gait["offset"], device=feet.device, dtype=feet.dtype)
    if offsets.numel() != feet.shape[1]:
        raise ValueError("Gait offsets must match the selected foot sites in order.")
    phase = _gait_phase(env, period).unsqueeze(1)
    phase = (phase + offsets.view(1, -1)) % 1.0
    swing = phase >= threshold
    swing_phase = ((phase - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    profile = torch.sin(torch.pi * swing_phase).clamp_min(0.0)

    command = None
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command '{command_name}' not found."
    motion = (
        command
        if command is not None
        else torch.cat(
            (
                asset.data.root_link_lin_vel_b[:, :2],
                asset.data.root_link_ang_vel_b[:, 2:3],
            ),
            dim=-1,
        )
    )
    translation = torch.cat((motion[:, :2], torch.zeros_like(motion[:, :1])), dim=-1)
    translation_w = quat_apply_yaw(asset.data.root_link_quat_w, translation)[:, :2]
    relative_xy = feet[..., :2] - asset.data.root_link_pos_w[:, None, :2]
    tangent = torch.stack((-relative_xy[..., 1], relative_xy[..., 0]), dim=-1)
    velocity = translation_w[:, None] + motion[:, None, 2:3] * tangent
    direction = velocity / torch.linalg.norm(velocity, dim=-1, keepdim=True).clamp_min(
        1e-6
    )

    scan = env.scene[scan_sensor_name].data
    valid_ray = (
        (scan.distances >= 0.0)
        & torch.isfinite(scan.distances)
        & torch.isfinite(scan.hit_pos_w).all(dim=-1)
        & (scan.normals_w[..., 2] > 0.0)
    )
    hits = torch.where(valid_ray[..., None], scan.hit_pos_w, 0.0)
    delta = hits[:, None, :, :2] - feet[:, :, None, :2]
    distance_sq = (
        delta.square().sum(dim=-1).masked_fill(~valid_ray[:, None], float("inf"))
    )
    nearest_distance, nearest = distance_sq.min(dim=-1)
    valid_ground = nearest_distance <= path_half_width**2
    ground_z = hits[..., 2].gather(1, nearest)

    along = (delta * direction[:, :, None]).sum(dim=-1)
    across = delta - along[..., None] * direction[:, :, None]
    ahead = (
        valid_ray[:, None]
        & (along >= 0.0)
        & (along <= lookahead_distance)
        & (across.square().sum(dim=-1) <= path_half_width**2)
    )
    obstacle_z = (
        hits[:, None, :, 2]
        .expand_as(along)
        .masked_fill(~ahead, -float("inf"))
        .amax(dim=-1)
    )
    obstacle_z = torch.where(ahead.any(dim=-1), obstacle_z, ground_z)
    rise = (obstacle_z - ground_z).clamp_min(0.0)
    peak_height = (target_height + rise).clamp_max(max_height)
    desired_z = ground_z + foot_radius + peak_height * profile
    height_error = torch.abs(feet[..., 2] - desired_z)
    cost = torch.where(swing & valid_ground, height_error, 0.0).sum(dim=1)

    if stumble_weight > 0.0:
        force = env.scene[sensor_name].data.force
        assert force is not None, (
            "Stumble penalty requires the contact sensor's force field."
        )
        horizontal = torch.linalg.norm(force[..., :2], dim=-1)
        stumble = (
            (horizontal > stumble_force_threshold)
            & (horizontal > stumble_ratio * force[..., 2].abs())
        ).any(dim=1)
        cost = cost + stumble_weight * stumble.to(cost.dtype)

    if command is not None:
        active = command_motion_mask(
            command, command_threshold, angular_command_threshold
        )
        cost = torch.where(active, cost, 0.0)
    return cost


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos
    assert default is not None
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - default[:, asset_cfg.joint_ids]
    )
    cost = torch.square(error).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    if command is not None:
        cost *= ~command_motion_mask(
            command, command_threshold, angular_command_threshold
        )
    return cost


def hip_joint_deviation_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice) and asset_cfg.joint_names is None:
        joint_ids, _ = asset.find_joints(r".*_hip_joint")
    default = asset.data.default_joint_pos
    assert default is not None
    cost = torch.square(asset.data.joint_pos[:, joint_ids] - default[:, joint_ids]).sum(
        dim=1
    )
    return cost * (
        (torch.abs(command[:, 1]) <= command_threshold)
        & (torch.abs(command[:, 2]) <= command_threshold)
    )


class variable_posture:
    """运动时根据局部地形增加姿态容差，只计算超出容差的偏差。"""

    def __init__(self, cfg, env):
        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos
        _, names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        keys = ("std_standing", "std_walking", "std_running", "terrain_tolerance")
        self.values = torch.tensor(
            [resolve_matching_names_values(cfg.params[k], names)[2] for k in keys],
            device=env.device,
        )
        if (self.values[:3] <= 0).any() or (self.values[3] < 0).any():
            raise ValueError(
                "Pose std must be positive; terrain tolerances nonnegative."
            )

    def __call__(
        self,
        env,
        std_standing,
        std_walking,
        std_running,
        terrain_tolerance,
        asset_cfg,
        command_name,
        walking_threshold=0.1,
        running_threshold=1.0,
        scan_sensor_name="terrain_scan",
        terrain_radius=0.35,
        terrain_height_std=(0.01, 0.04),
        angular_command_threshold=None,
    ):
        lo, hi = terrain_height_std
        if not (
            terrain_radius > 0
            and 0 <= lo < hi
            and 0 <= walking_threshold < running_threshold
        ):
            raise ValueError(
                "Invalid terrain radius, height thresholds or speed thresholds."
            )

        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        speed = torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs()
        if angular_command_threshold is None:
            mode = (speed >= walking_threshold).long() + (speed >= running_threshold).long()
        else:
            moving = command_motion_mask(
                command, walking_threshold, angular_command_threshold
            )
            mode = moving.long() + (moving & (speed >= running_threshold)).long()

        # 仅统计附近有效地面；越靠近范围边缘，权重越小。
        scan = env.scene[scan_sensor_name].data
        valid = (
            (scan.distances >= 0)
            & torch.isfinite(scan.distances)
            & torch.isfinite(scan.hit_pos_w).all(-1)
            & (scan.normals_w[..., 2] > 0)
        )
        hits = torch.where(valid[..., None], scan.hit_pos_w, 0.0)
        delta = hits - asset.data.root_link_pos_w[:, None, :]
        distance_sq = delta[..., :2].square().sum(-1)
        weights = (1 - distance_sq / terrain_radius**2).clamp_min(0).square() * valid
        count = weights.sum(-1).clamp_min(1e-6)
        z = delta[..., 2]
        mean = (weights * z).sum(-1) / count
        roughness = ((weights * (z - mean[:, None]).square()).sum(-1) / count).sqrt()

        # 地形起伏映射到 0～1；站立或有效采样不足时关闭容差。
        alpha = ((roughness - lo) / (hi - lo)).clamp(0, 1)
        alpha = alpha.square() * (3 - 2 * alpha)
        alpha = torch.where((mode > 0) & ((weights > 0).sum(-1) >= 4), alpha, 0.0)

        std = self.values[mode]
        tolerance = alpha[:, None] * self.values[3]
        error = (
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - self.default_joint_pos[:, asset_cfg.joint_ids]
        )
        error = (error.abs() - tolerance).clamp_min(0.0)
        return torch.exp(-torch.mean(error.square() / std.square(), dim=1))


def stumble(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    ratio: float = 5.0,
) -> torch.Tensor:
    force = env.scene[sensor_name].data.force
    assert force is not None

    horizontal_force = torch.linalg.norm(force[..., :2], dim=-1)
    vertical_force = torch.abs(force[..., 2])

    return (horizontal_force > ratio * vertical_force).any(dim=1).float()


# Copied from mjlab 1.6.0 velocity rewards; Apache-2.0 (mjlab project).
def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor provides force history (from ``history_length > 0``),
    counts substeps where any contact force exceeds *force_threshold*.
    Falls back to the instantaneous ``found`` count otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.sum(dim=-1).float()


def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact)."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    active = command_motion_mask(
        command, command_threshold, angular_command_threshold
    ).float()
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
        num_in_contact, min=1
    )
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    angular_command_threshold: float | None = None,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            active = command_motion_mask(
                command, command_threshold, angular_command_threshold
            ).float()
            cost = cost * active
    return cost


def command_stall_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Bounded sustained low-progress cost after reset/command-change grace."""
    from .behavior_monitor import get_behavior_monitor

    return get_behavior_monitor(env).ensure_step().cost
