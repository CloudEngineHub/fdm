# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils.configclass import configclass

from .base_agent_cfg import AgentCfg
from .paper_figure_agent import PaperFigureAgent


@configclass
class PaperFigureAgentCfg(AgentCfg):

    class_type: type[PaperFigureAgent] = PaperFigureAgent

    horizon: int = 10
    """Number of steps to plan ahead."""

    platform_figure: bool = False
    """Switch commands for the platform comparison figure"""
