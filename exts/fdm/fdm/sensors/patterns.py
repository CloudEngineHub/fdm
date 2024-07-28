from __future__ import annotations

import json
import math
import os
import torch
from typing import TYPE_CHECKING

from omni.isaac.core.utils.extensions import get_extension_path_from_name

if TYPE_CHECKING:
    from . import patterns_cfg

def lidar2d_pattern(cfg: patterns_cfg.Lidar2DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.arange(-cfg.horizontal_fov / 2, cfg.horizontal_fov / 2, cfg.horizontal_res, device=device)

    yaw = torch.deg2rad(h.reshape(-1))
    x = torch.cos(yaw)
    y = torch.sin(yaw)
    z = torch.zeros_like(x)

    ray_directions = -torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions


def foot_scan_pattern(cfg: patterns_cfg.FootScanPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The foot scan pattern for ray casting.
    Args:
        cfg (FootScanPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions
    """
    pattern = []
    for i, r in enumerate(cfg.radii):
        for j in range(cfg.num_points[i]):
            angle = 2.0 * math.pi * j / cfg.num_points[i]
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = 0.0
            pattern.append([x, y, z])
    ray_starts = torch.tensor(pattern, dtype=torch.float).to(device)

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(cfg.direction), device=device)
    return ray_starts, ray_directions