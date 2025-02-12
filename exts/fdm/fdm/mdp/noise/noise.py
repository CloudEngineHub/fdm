# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torchvision.transforms import RandomRotation
from typing import TYPE_CHECKING

import kornia

if TYPE_CHECKING:
    from . import noise_cfg


def perlin_noise(data: torch.Tensor, noise_cfg: noise_cfg.PerlinNoiseCfg) -> torch.Tensor:
    """Add perlin noise to the data."""

    width = data.shape[-1]
    height = data.shape[-2]
    delta = (noise_cfg.res[0] / height, noise_cfg.res[1] / width)
    assert (
        height % noise_cfg.res[0] == 0
    ), f"Perlin height resolution {noise_cfg.res[0]} must be a multiple of height {height}"
    assert (
        width % noise_cfg.res[1] == 0
    ), f"Perlin width resolution {noise_cfg.res[1]} must be a multiple of width {width}"
    d = (height // noise_cfg.res[0], width // noise_cfg.res[1])
    batch_dim = data.shape[:-2]
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, noise_cfg.res[0], delta[0], device=data.device),
                torch.arange(0, noise_cfg.res[1], delta[1], device=data.device),
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(*batch_dim, noise_cfg.res[0] + 1, noise_cfg.res[1] + 1, device=data.device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[..., slice1[0] : slice1[1], slice2[0] : slice2[1], :]
        .repeat_interleave(d[0], -3)
        .repeat_interleave(d[1], -2)
    )

    dot = lambda grad, shift: (  # noqa: E731
        torch.stack((grid[..., :height, :width, 0] + shift[0], grid[..., :height, :width, 1] + shift[1]), dim=-1)
        * grad[..., :height, :width, :]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    if noise_cfg.fade is None:
        fade = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3  # noqa: E731
        t = fade(grid[:height, :width])
    else:
        t = noise_cfg.fade(grid[:height, :width])

    noise = math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    # noise is in range [-1, 1], scale to min and max defined in config
    noise = (noise + 1) / 2 * (noise_cfg.n_max - noise_cfg.n_min) + noise_cfg.n_min

    return data + noise


def missing_patch_noise(data: torch.Tensor, noise_cfg: noise_cfg.MissingPatchNoiseCfg) -> torch.Tensor:
    """Mask a random part of the image with a rectangle mask."""
    if data.shape[1] == 1:
        initial_queeze = True
        data = data.squeeze(1)
    else:
        initial_queeze = False

    batch, height, width = data.shape

    # Decide where the masked will be placed for each image
    # 0: top, 1: right, 2: bottom
    mask_placement = torch.randint(0, 3, size=(batch * noise_cfg.n_masks,), device=data.device)

    # Select the center of the mask based on the side
    mask_center_x = torch.zeros((batch * noise_cfg.n_masks), dtype=torch.int64, device=data.device)
    mask_center_y = torch.zeros((batch * noise_cfg.n_masks), dtype=torch.int64, device=data.device)
    mask_center_x[mask_placement != 1] = torch.randint(0, width - 1, (batch * noise_cfg.n_masks,), device=data.device)[
        mask_placement != 1
    ]
    mask_center_x[mask_placement == 1] = width - 1
    mask_center_y[mask_placement == 0] = 0
    mask_center_y[mask_placement == 1] = torch.randint(0, height - 1, (batch * noise_cfg.n_masks,), device=data.device)[
        mask_placement == 1
    ]
    mask_center_y[mask_placement == 2] = height - 1

    # generate the random center for each mask
    rect_width = torch.randint(*noise_cfg.rect_width_range, size=(batch * noise_cfg.n_masks,), device=data.device)
    rect_height = torch.randint(*noise_cfg.rect_height_range, size=(batch * noise_cfg.n_masks,), device=data.device)

    # Create rectangles' boundaries
    x_start = mask_center_x - rect_width // 2
    x_end = mask_center_x + rect_width // 2
    y_start = mask_center_y - rect_height // 2
    y_end = mask_center_y + rect_height // 2

    # Ensure boundaries stay within image dimensions
    x_start = torch.clamp(x_start, 0, width - 1)
    x_end = torch.clamp(x_end, 0, width - 1)
    y_start = torch.clamp(y_start, 0, height - 1)
    y_end = torch.clamp(y_end, 0, height - 1)

    # Create grids for x and y coordinates
    x_coords = torch.arange(width, device=data.device).unsqueeze(0).expand(batch * noise_cfg.n_masks, height, width)
    y_coords = torch.arange(height, device=data.device).unsqueeze(1).expand(batch * noise_cfg.n_masks, height, width)

    # Create boolean masks based on rectangle bounds
    mask_rect = (
        (x_coords >= x_start[:, None, None])
        & (x_coords <= x_end[:, None, None])
        & (y_coords >= y_start[:, None, None])
        & (y_coords <= y_end[:, None, None])
    )

    # generate mask tensor
    mask = torch.zeros((batch * noise_cfg.n_masks, height, width), device=data.device)
    mask[mask_rect] = 1

    # combine masks
    mask = mask.reshape(batch, noise_cfg.n_masks, height, width).max(dim=1)[0]

    # Rotate the different masks by a +- 25 degrees
    rotation = RandomRotation(degrees=25)
    mask = rotation.forward(mask)

    # apply gaussian kernel to the mask
    mask = kornia.filters.gaussian_blur2d(mask.unsqueeze(1), (9, 9), (4.5, 4.5)).squeeze(1)

    mask[mask > 0.25] = 1
    mask = mask.to(torch.int64)

    if False:
        # plot the first 25 masks in one figure
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(5, 5, figsize=(15, 15))
        for i in range(25):
            ax = axs[i // 5, i % 5]
            ax.imshow(mask[i].cpu().numpy())
            ax.axis("off")
        plt.show()

    # if additional noise is provided, apply it
    if noise_cfg.apply_noise is not None:
        data = noise_cfg.apply_noise.func(data, noise_cfg.apply_noise)

    # combine mask with the occluded points (i.e. everything above the occlusion height)
    mask = torch.logical_or(mask, data > noise_cfg.occlusion_height)

    # clip
    if noise_cfg.clip_values is not None:
        data = data.clip_(*noise_cfg.clip_values)

    # Apply mask to the image
    if noise_cfg.fill_value is None:
        indices = np.zeros((len(data.shape), *mask.shape), dtype=np.int32)

        # convert to nunmpy --> check if necessary
        mask_np = mask.cpu().numpy()

        # Compute the distance transform and nearest neighbor indices
        for i in range(mask.shape[0]):
            indices[0, i] = i
            _, indices[1:, i] = distance_transform_edt(mask_np[i], return_indices=True)

        if False:
            import matplotlib.pyplot as plt

            # show original image, mask and filled image
            plt.figure()
            plt.subplot(131)
            plt.imshow(data[0].cpu().numpy())
            plt.subplot(132)
            plt.imshow(mask[0].cpu().numpy())
            plt.subplot(133)
            plt.imshow(data[tuple(indices)][0].cpu().numpy())
            plt.show()

        # Use indices to assign nearest neighbor values
        data = data[tuple(indices)]

    else:
        data[mask == 1] = noise_cfg.fill_value

    if initial_queeze:
        data = data.unsqueeze(1)

    return data
