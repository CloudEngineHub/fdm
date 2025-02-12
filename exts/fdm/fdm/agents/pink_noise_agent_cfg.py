# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from .base_agent import Agent
from .base_agent_cfg import AgentCfg
from .pink_noise_agent import PinkNoiseAgent


@configclass
class PinkNoiseAgentCfg(AgentCfg):

    class_type: type[Agent] = PinkNoiseAgent
    """The class type of the agent."""

    upper_bound = 1.2
    """Upper bound for the actions."""
    lower_bound = -1.2
    """Lower bound for the actions."""
    colored_noise_exponent = 1.0
    """Exponent for the powerlaw PSD of the noise."""
    variance = 1.0
    """Variance of the noise."""
    mean = 0.0
    """Mean of the noise."""
