# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ray-cast sensor."""

from __future__ import annotations

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

from . import patterns


@configclass
class Lidar2DPatternCfg(PatternBaseCfg):
    """Configuration for the Lidar2D pattern for ray-casting."""

    func = patterns.lidar2d_pattern

    horizontal_fov: float = 360.0
    """Horizontal field of view (in degrees). Defaults to 360.0."""

    horizontal_res: float = 10.0
    """Horizontal resolution (in degrees). Defaults to 10.0."""
