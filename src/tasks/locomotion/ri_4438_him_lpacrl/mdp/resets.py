"""Nominal resets controlled by the saved, paired evaluation trial table."""
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from .curriculums import get_controller


def reset_nominal(env, env_ids):
    robot = env.scene["robot"]
    state = robot.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    zeros = torch.zeros(len(env_ids), device=env.device)
    yaw = get_controller(env).yaws[env_ids]
    state[:, 3:7] = quat_mul(state[:, 3:7], quat_from_euler_xyz(zeros, zeros, yaw))
    state[:, 7:] = 0
    robot.write_root_state_to_sim(state, env_ids=env_ids)
    positions = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)
