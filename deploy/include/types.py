from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RobotState:
    """State common to a simulated or real robot backend."""

    time: float
    position: np.ndarray
    quaternion: np.ndarray  # MuJoCo/Unitree convention: w, x, y, z.
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    angular_velocity_body: np.ndarray


class RobotBackend(Protocol):
    timestep: float

    def read_state(self) -> RobotState: ...

    def set_joint_target(self, target: np.ndarray) -> None: ...

    def set_base_pose(self, position: np.ndarray, quaternion: np.ndarray) -> None: ...

    def step(self) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
