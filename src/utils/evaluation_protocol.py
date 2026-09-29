"""Shared, explicit benchmark geometry and paired reset states."""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul


from .evaluation_scenarios import LANE_HALF_WIDTH, STAIR_COUNT, stair_sections


@dataclass(kw_only=True)
class BenchmarkStairsCfg(SubTerrainCfg):
    step_height: float
    descending: bool = False

    def function(self, difficulty, spec, rng):
        del difficulty, rng
        body = spec.body("terrain")
        cx, cy = self.size[0] / 2, self.size[1] / 2
        geometries = []
        for i, (left, right, z) in enumerate(stair_sections(self.size[0], self.step_height, self.descending)):
            # Solid adjacent boxes with a common underside; no invisible vertical gaps.
            depth = z + 0.2
            geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 pos=(cx + (left + right) / 2, cy, z - depth / 2),
                                 size=((right - left) / 2, cy, depth / 2))
            geometries.append(TerrainGeometry(geom=geom, color=(0.25 + i * 0.025, 0.45, 0.60, 1.0)))
        origin = np.array([cx, cy, STAIR_COUNT * self.step_height if self.descending else 0.0])
        return TerrainOutput(origin=origin, geometries=geometries)


def paired_samples(seed: int, samples: int, joint_count: int, stairs: bool):
    """CPU RNG keyed by local episode index, independent of batch size/model order."""
    rng = np.random.default_rng(seed)
    limits = np.array([0.04 if stairs else 0.2, 0.04 if stairs else 0.2, 0.0,
                       0.05, 0.05, 0.025 if stairs else 0.1])
    pose = (rng.uniform(-1, 1, size=(samples, 6)) * limits).astype(np.float32)
    joints = rng.uniform(-0.02, 0.02, size=(samples, joint_count)).astype(np.float32)
    return pose, joints


def reset_paired_robot(env, env_ids, samples_per_model: int, stairs: bool):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene["robot"]
    pose, offsets = paired_samples(getattr(env, "_evaluation_seed", 0), samples_per_model,
                                   robot.data.default_joint_pos.shape[1], stairs)
    slots = env_ids % samples_per_model
    perturbation = torch.as_tensor(pose, device=env.device)[slots]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids] + perturbation[:, :3]
    root[:, 3:7] = quat_mul(root[:, 3:7], quat_from_euler_xyz(*perturbation[:, 3:6].unbind(-1)))
    robot.write_root_link_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(root[:, 7:13], env_ids=env_ids)
    positions = robot.data.default_joint_pos[env_ids] + torch.as_tensor(offsets, device=env.device)[slots]
    limits = robot.data.soft_joint_pos_limits[env_ids]
    positions = positions.clamp(limits[..., 0], limits[..., 1])
    robot.write_joint_state_to_sim(positions, robot.data.default_joint_vel[env_ids], env_ids=env_ids)


def left_stair_lane(env):
    return (env.scene["robot"].data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]).abs() > LANE_HALF_WIDTH
