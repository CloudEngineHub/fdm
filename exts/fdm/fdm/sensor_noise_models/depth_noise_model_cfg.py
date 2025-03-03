# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .depth_noise_model import DepthCameraNoise


@configclass
class DepthCameraNoiseCfg:
    """Configuration for the trainer."""

    # noise model class
    noise_model: DepthCameraNoise = DepthCameraNoise

    # noise model parameters
    pixel_noise_a: float = 0.25
    angular_resolution_degree: float = 85 / 80
    edge_noise_thresh: float = 0.08
    far_plane: float = 8.0
    perlin_resolution: tuple[int] = (20, 16)
