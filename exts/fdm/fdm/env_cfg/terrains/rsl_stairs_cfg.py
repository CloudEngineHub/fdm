# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass

from .rsl_stairs import rsl_stairs_terrain


@configclass
class RslStairsCfg(SubTerrainBaseCfg):
    """Configuration for a stairs and ramp (both next to each other) mesh terrain."""

    function = rsl_stairs_terrain

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0.

    The border is a flat terrain with the same height as the terrain.
    """

    num_steps: int = 2
    """The number of steps on the stairs."""

    step_height: float = 0.18
    """The height of the steps (in m)."""

    step_width: float = 0.29
    """The width of the steps (in m)."""

    box_length: float = 1.5
    """The length of the box on the stairs (in m)."""

    center_platform_width: float = 2.5
    """The width of the platform in the center of the terrain"""

    platform_width: float = 2.5
    """The width of the platform at the top of the stairs"""

    wall_probability: float = 0.5
    """The probability of having a wall on the platform in between the stairs."""

    wall_height: float = 2.0
    """The height of the wall on the platform in between the stairs."""

    wall_width: float = 0.1
    """The width of the wall on the platform in between the stairs."""
