# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from .base_agent import Agent
from .base_agent_cfg import AgentCfg
from .time_correlated_actions import TimeCorrelatedCommandTrajectoryAgent

"""
Random Path command generators.
"""


@configclass
class TimeCorrelatedCommandTrajectoryAgentCfg(AgentCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type[Agent] = TimeCorrelatedCommandTrajectoryAgent
    """The class type of the agent."""

    @configclass
    class Ranges:
        """Ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        """Normal distribution ranges for the velocity commands."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    max_beta: float = MISSING
    """Time correleation factor for uniform time-correlation"""

    linear_ratio: float = MISSING
    normal_ratio: float = MISSING
    constant_ratio: float = MISSING
    regular_increasing_ratio: float = MISSING
    """Ratio of uniform, normal, and constant velocity commands."""

    sigma_scale: float = MISSING
    """Scale factor for the normal distribution."""
