# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.lab.utils import configclass

from .base_agent_cfg import AgentCfg
from .mixed_agent import MixedAgent


@configclass
class MixedAgentCfg(AgentCfg):
    """Configuration for a mix of multiple agents, each given a subset of the environment."""

    class_type: type = MixedAgent

    @configclass
    class SubsetAgentCfg:
        """Agent configuration for a subset of the environments."""

        agent_term: AgentCfg = MISSING
        """The configuration for the agent."""

        ratio: float = MISSING
        """The ratio of environments that this agent should be applied to."""

    terms: dict[str, SubsetAgentCfg] = MISSING
    """The agent terms that should be applied to the environments."""
