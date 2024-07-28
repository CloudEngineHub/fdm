
"""Configuration for the ray-cast sensor."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.lab.utils import configclass
from omni.isaac.lab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg

from . import patterns


@configclass
class Lidar2DPatternCfg(PatternBaseCfg):
    """Configuration for the Lidar2D pattern for ray-casting."""

    func = patterns.lidar2d_pattern

    horizontal_fov: float = 360.0
    """Horizontal field of view (in degrees). Defaults to 360.0."""

    horizontal_res: float = 10.0
    """Horizontal resolution (in degrees). Defaults to 10.0."""


@configclass
class FootScanPatternCfg(PatternBaseCfg):

    func: Callable = patterns.foot_scan_pattern

    radii: tuple[float, ...] = (0.08, 0.16, 0.26, 0.36, 0.48)
    num_points: tuple[int, ...] = (6, 8, 10, 12, 16)
    direction: tuple = (0.0, 0.0, -1.0)
    mean: float = 0.05
    std: float = 0.1