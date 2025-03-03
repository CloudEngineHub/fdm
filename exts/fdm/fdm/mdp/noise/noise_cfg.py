# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Callable

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, UniformNoiseCfg

from . import noise


@configclass
class PerlinNoiseCfg(NoiseCfg):
    """Configuration for a additive uniform noise term."""

    func = noise.perlin_noise

    res: tuple[int, int] = (3, 2)
    """Resolution of the perlin noise. Defaults to (3, 2)."""

    fade: Callable | None = None
    """Fade function for the perlin noise. Defaults to None which equal the following function

    .. math:
        6t^5 - 15t^4 + 10t^3.

    """

    n_min: torch.Tensor | float = -0.05
    """The minimum value of the noise. Defaults to -0.05."""

    n_max: torch.Tensor | float = 0.05
    """The maximum value of the noise. Defaults to 0.05."""


@configclass
class MissingPatchNoiseCfg(NoiseCfg):
    func = noise.missing_patch_noise

    n_masks: int = 4
    """Number of masks to apply to each height scan."""

    rect_width_range: tuple[int, int] = (5, 15)
    """Range of width for the rectangle masks."""

    rect_height_range: tuple[int, int] = (5, 20)
    """Range of height for the rectangle masks."""

    fill_value: float | None = 1.5
    """The value to fill the masked region with. Defaults to 1.0.

    If None, a nearest neighbor filling is applied. Importantly, the additional noise of apply_noise is done before
    the filling to avoid that the missing patches are removed"""

    occlusion_height: float = 5.0
    """The height of the occlusion in meters. Defaults to 5.0.

    .. note::
        The occlusions are filled by the fill_value or nearest neighbor filling."""

    clip_values: tuple[float, float] | None = (0.0, 1.0)
    """The range to clip the values to. Defaults to (0.0, 1.0)."""

    apply_noise: NoiseCfg | None = UniformNoiseCfg(n_min=-0.01, n_max=0.01)
    """Noise configuration to apply in addition. Defaults to a Unoise with range [-0.01, 0.01].

    If None, no additional noise is applied."""


if __name__ == "__main__":
    # Example usage
    batch, height, width = 7156, 60, 40
    tensor = torch.rand(batch, height, width, device="cuda")
    masked_tensor = noise.missing_patch_noise(tensor.clone(), MissingPatchNoiseCfg())

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(tensor[0].cpu().numpy(), cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(masked_tensor[0].cpu().numpy(), cmap="gray")
    axs[1].set_title("Masked Image")
    plt.show()
