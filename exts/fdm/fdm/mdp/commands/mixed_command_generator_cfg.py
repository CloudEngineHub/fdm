# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .mixed_command_generator import MixedCommand


@configclass
class MixedCommandCfg(CommandTermCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type = MixedCommand

    @configclass
    class SubsetCommandTermCfg:
        """Configuration for a subset of the environments."""

        command_term: CommandTermCfg = MISSING
        """The configuration for the command term."""

        ratio: float = MISSING
        """The ratio of environments that this command term should be applied to."""

    terms: dict[str, SubsetCommandTermCfg] = MISSING
    """The command terms that should be applied to the environments."""
