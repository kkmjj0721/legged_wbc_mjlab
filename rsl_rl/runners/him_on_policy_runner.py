# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility runner entry point for HIM configurations."""

from __future__ import annotations

from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class HIMOnPolicyRunner(OnPolicyRunner):
    """Use the modern OnPolicyRunner lifecycle with HIMPPO."""
