# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms."""

from .distillation import Distillation
from .him_ppo import HIMPPO
from .ppo import PPO

__all__ = ["PPO", "HIMPPO", "Distillation"]
