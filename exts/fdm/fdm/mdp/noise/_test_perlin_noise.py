# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch


def rand_perlin_2d(shape, res, device="cpu", fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3):
    width = shape[-1]
    height = shape[-2]
    delta = (res[0] / height, res[1] / width)
    assert height % res[0] == 0, f"Perlin height resolution {res[0]} must be a multiple of height {height}"
    assert width % res[1] == 0, f"Perlin width resolution {res[1]} must be a multiple of width {width}"
    d = (height // res[0], width // res[1])
    batch_dim = shape[:-2]
    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)
            ),
            dim=-1,
        )
        % 1
    )
    angles = 2 * math.pi * torch.rand(*batch_dim, res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = (
        lambda slice1, slice2: gradients[..., slice1[0] : slice1[1], slice2[0] : slice2[1], :]
        .repeat_interleave(d[0], -3)
        .repeat_interleave(d[1], -2)
    )

    dot = lambda grad, shift: (
        torch.stack((grid[..., :height, :width, 0] + shift[0], grid[..., :height, :width, 1] + shift[1]), dim=-1)
        * grad[..., :height, :width, :]
    ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:height, :width])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


import time

start = time.time()
noise = rand_perlin_2d((7168, 60, 46), (3, 2), device="cuda")
print(f"Time taken: {time.time() - start}")

import matplotlib.pyplot as plt

plt.imshow(noise.cpu().numpy()[0], cmap="gray")
plt.show()

print("DONE")
