# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .base_agent import Agent


@configclass
class AgentCfg:

    class_type: type[Agent] = Agent

    horizon: int = 200
    """Number of steps to plan ahead."""
