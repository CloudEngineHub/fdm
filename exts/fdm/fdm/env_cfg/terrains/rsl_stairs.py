# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import *  # noqa: F401, F403
from isaaclab.terrains.trimesh.utils import make_border, make_plane

if TYPE_CHECKING:
    from . import rsl_stairs_cfg


# -- generate the platform that account, with the random probably add a wall behind the stairs and on the sides
def add_wall(
    cfg: rsl_stairs_cfg.RslStairsCfg,
    platform_start: float,
    platform_width: float,
    terrain_center: list[float],
    meshes_list,
    stairs_center,
):
    # wall in the platform
    wall_dims = (cfg.wall_width, cfg.size[1] - 2 * cfg.border_width, cfg.wall_height)
    wall_center = (
        platform_start + platform_width / 2 - cfg.wall_width / 2,
        stairs_center[1],
        terrain_center[2] + cfg.wall_height / 2,
    )
    meshes_list.append(trimesh.creation.box(wall_dims, trimesh.transformations.translation_matrix(wall_center)))

    wall_dims = (platform_width + 2 * (cfg.num_steps * cfg.step_width), cfg.wall_width, cfg.wall_height)
    wall_center = (
        platform_start + platform_width / 2,  # + stairs_extend,
        stairs_center[1] + (cfg.size[1] - 2 * cfg.border_width) / 2 - cfg.wall_width / 2,
        terrain_center[2] + cfg.wall_height / 2,
    )
    meshes_list.append(trimesh.creation.box(wall_dims, trimesh.transformations.translation_matrix(wall_center)))
    wall_center = (
        platform_start + platform_width / 2,  # + stairs_extend,
        stairs_center[1] - (cfg.size[1] - 2 * cfg.border_width) / 2 + cfg.wall_width / 2,
        terrain_center[2] + cfg.wall_height / 2,
    )
    meshes_list.append(trimesh.creation.box(wall_dims, trimesh.transformations.translation_matrix(wall_center)))

    return meshes_list


def rsl_stairs_terrain(difficulty: float, cfg: rsl_stairs_cfg.RslStairsCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a stairs in front of the terrace and some blocks on the stairs.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """

    # modify the position of the box on the stairs
    box_position = (
        difficulty * (cfg.size[1] - 2 * cfg.border_width - cfg.box_length) + cfg.border_width + cfg.box_length / 2
    )

    # terrain locations
    # -- compute the position of the center of the terrain and the centers of the stairs as well as ramp
    terrain_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0]
    stairs_center_up = [terrain_center[0] + cfg.center_platform_width / 2, terrain_center[1]]
    stairs_center_down = [terrain_center[0] - cfg.center_platform_width / 2, terrain_center[1]]

    # restrict the space
    # initialize list of meshes
    meshes_list = list()
    # generate a ground plane for the terrain
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    # generate the border if needed
    if cfg.border_width > 0.0:
        # obtain a list of meshes for the border
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -cfg.step_height / 2]
        border_inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        make_borders = make_border(cfg.size, border_inner_size, cfg.step_height, border_center)
        # add the border meshes to the list of meshes
        meshes_list += make_borders

    # generate the terrain
    # -- generate the stair pattern
    for k in range(cfg.num_steps):
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + (k / 2.0 + 0.5) * cfg.step_height
        box_offset = (k / 2.0 + 0.5) * cfg.step_width
        # -- dimensions
        box_height = (k + 1) * cfg.step_height
        # generate the stair
        box_dims = ((cfg.num_steps - k) * cfg.step_width, cfg.size[1] - 2 * cfg.border_width, box_height)
        box_pos_up = (stairs_center_up[0] + box_offset, stairs_center_up[1], box_z)
        meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_up)))
        box_pos_down = (stairs_center_down[0] - box_offset, stairs_center_down[1], box_z)
        meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_down)))
        box_pos_up = (stairs_center_up[0] - box_offset + cfg.platform_width, stairs_center_up[1], box_z)
        meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_up)))
        box_pos_down = (stairs_center_down[0] + box_offset - cfg.platform_width, stairs_center_down[1], box_z)
        meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos_down)))

    # upper stairs platform
    platform_start = stairs_center_up[0] + (cfg.num_steps * cfg.step_width) / 2
    platform_width = cfg.platform_width - (cfg.num_steps * cfg.step_width)
    platform_center = (
        platform_start + platform_width / 2,
        stairs_center_up[1],
        terrain_center[2] + cfg.num_steps * cfg.step_height / 2,
    )
    platform_dims = (platform_width, cfg.size[1] - 2 * cfg.border_width, cfg.num_steps * cfg.step_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    # see if we add a wall
    if np.random.rand() < cfg.wall_probability:
        meshes_list = add_wall(cfg, platform_start, platform_width, terrain_center, meshes_list, stairs_center_up)
    platform_start = stairs_center_down[0] - (cfg.num_steps * cfg.step_width) / 2 - platform_width
    platform_center = (
        platform_start + platform_width / 2,
        stairs_center_down[1],
        terrain_center[2] + cfg.num_steps * cfg.step_height / 2,
    )
    platform_dims = (platform_width, cfg.size[1] - 2 * cfg.border_width, cfg.num_steps * cfg.step_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    # see if we add a wall
    if np.random.rand() < cfg.wall_probability:
        meshes_list = add_wall(cfg, platform_start, platform_width, terrain_center, meshes_list, stairs_center_down)

    # -- add box over the stairs
    platform_center = (
        stairs_center_up[0] + ((cfg.num_steps / 2 - 0.5) * cfg.step_width),
        box_position,
        terrain_center[2] + cfg.num_steps * cfg.step_height / 2,
    )
    platform_dims = (cfg.num_steps * cfg.step_width, cfg.box_length, cfg.num_steps * cfg.step_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    platform_center = (
        stairs_center_up[0] - ((cfg.num_steps / 2 - 0.5) * cfg.step_width) + cfg.platform_width,
        box_position,
        terrain_center[2] + cfg.num_steps * cfg.step_height / 2,
    )
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    platform_center = (
        stairs_center_down[0] - ((cfg.num_steps / 2 - 0.5) * cfg.step_width),
        box_position,
        terrain_center[2] + cfg.num_steps * cfg.step_height / 2,
    )
    platform_dims = (cfg.num_steps * cfg.step_width, cfg.box_length, cfg.num_steps * cfg.step_height)
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))
    platform_center = (
        stairs_center_down[0] + ((cfg.num_steps / 2 - 0.5) * cfg.step_width) - cfg.platform_width,
        box_position,
        terrain_center[2] + cfg.num_steps * cfg.step_height / 2,
    )
    meshes_list.append(trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_center)))

    # compute the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], terrain_center[2]])

    return meshes_list, origin
