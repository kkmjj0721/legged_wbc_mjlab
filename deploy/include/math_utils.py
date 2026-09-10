from __future__ import annotations

import math

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    return quaternion / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quaternion(q0 + amount * (q1 - q0))
    theta = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - amount) * theta) / sin_theta * q0
        + math.sin(amount * theta) / sin_theta * q1
    )


def rotate_inverse(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector to body coordinates."""
    # Kept independent of MuJoCo so a real backend can reuse the observation
    # implementation.  MuJoCo's quaternion order is wxyz.
    w, x, y, z = quaternion_wxyz
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return rotation.T @ vector
