# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import patterns_cfg


def lidar2d_pattern(cfg: patterns_cfg.Lidar2DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res, device=device)

    yaw = torch.deg2rad(h.reshape(-1))
    x = torch.cos(yaw)
    y = torch.sin(yaw)
    z = torch.zeros_like(x)

    ray_directions = torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions
