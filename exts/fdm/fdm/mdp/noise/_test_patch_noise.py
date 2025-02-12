# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
from torchvision.transforms import RandomRotation

import kornia


@torch.jit.script
def random_rotate_gpu(tensor: torch.Tensor, degrees: float) -> torch.Tensor:
    """
    Apply random rotation to a tensor on the GPU.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
        degrees (float): Maximum rotation angle in degrees (symmetric).

    Returns:
        torch.Tensor: Rotated tensor.
    """
    # Get the batch size and device
    n, c, h, w = tensor.size()
    device = tensor.device

    # Generate random rotation angles (in radians)
    angles = (torch.rand(n, device=device) * 2 - 1) * (degrees * torch.pi / 180)

    # Create the rotation matrix
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    rotation_matrices = torch.zeros((n, 2, 3), device=device)
    rotation_matrices[:, 0, 0] = cos
    rotation_matrices[:, 0, 1] = -sin
    rotation_matrices[:, 1, 0] = sin
    rotation_matrices[:, 1, 1] = cos

    # Create affine grid
    grid = F.affine_grid(rotation_matrices, size=tensor.size(), align_corners=False)

    # Perform the rotation using grid_sample
    rotated_tensor = F.grid_sample(tensor, grid, align_corners=False, padding_mode="zeros")

    return rotated_tensor


def apply_mask(
    height_scan: torch.Tensor,
    n_masks: int = 5,
    rect_width_range: tuple[int, int] = (5, 20),
    rect_height_range: tuple[int, int] = (5, 20),
):
    """Mask a random part of the image with a rectangle mask."""

    batch, height, width = height_scan.shape

    start = time.time()
    # Decide where the masked will be placed for each image
    # 0: top, 1: right, 2: bottom
    mask_placement = torch.randint(0, 3, size=(batch * n_masks,), device=height_scan.device)

    # Select the center of the mask based on the side
    mask_center_x = torch.zeros((batch * n_masks), dtype=torch.int64, device=height_scan.device)
    mask_center_y = torch.zeros((batch * n_masks), dtype=torch.int64, device=height_scan.device)
    mask_center_x[mask_placement != 1] = torch.randint(0, width - 1, (batch * n_masks,), device=height_scan.device)[
        mask_placement != 1
    ]
    mask_center_x[mask_placement == 1] = width - 1
    mask_center_y[mask_placement == 0] = 0
    mask_center_y[mask_placement == 1] = torch.randint(0, height - 1, (batch * n_masks,), device=height_scan.device)[
        mask_placement == 1
    ]
    mask_center_y[mask_placement == 2] = height - 1

    # generate the random center for each mask
    rect_width = torch.randint(*rect_width_range, size=(batch * n_masks,), device=height_scan.device)
    rect_height = torch.randint(*rect_height_range, size=(batch * n_masks,), device=height_scan.device)

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
    x_coords = torch.arange(width, device=height_scan.device).unsqueeze(0).expand(batch * n_masks, height, width)
    y_coords = torch.arange(height, device=height_scan.device).unsqueeze(1).expand(batch * n_masks, height, width)

    # Create boolean masks based on rectangle bounds
    mask_rect = (
        (x_coords >= x_start[:, None, None])
        & (x_coords <= x_end[:, None, None])
        & (y_coords >= y_start[:, None, None])
        & (y_coords <= y_end[:, None, None])
    )

    # generate mask tensor
    mask = torch.ones((batch * n_masks, height, width), device="cuda")
    mask[mask_rect] = 0

    print("Mask Creation Time:", time.time() - start)
    start = time.time()

    # Rotate the different masks by a +- 10 degrees
    rotation = RandomRotation(degrees=25)
    mask = (rotation.forward(mask) > 0.5).to(torch.int64)

    print("Rotation Time:", time.time() - start)

    # start = time.time()
    # rotated_mask = (random_rotate_gpu(mask.unsqueeze(1), degrees=25) > 0.5).to(torch.int64)
    # print("Rotation Time (Torch):", time.time() - start)

    # Create a Gaussian kernel
    def create_gaussian_kernel(kernel_size: int, sigma: float):
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss = gauss / gauss.sum()  # Normalize
        kernel_2d = torch.outer(gauss, gauss)
        kernel_2d = kernel_2d / kernel_2d.sum()  # Normalize again for safety
        return kernel_2d

    # Parameters
    kernel_size = 9
    sigma = 4.5
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(mask.device)  # Reshape for 2D convolution

    # Define the blur function and JIT-compile it
    @torch.jit.script
    def apply_gaussian_blur(mask: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        padding = kernel.size(-1) // 2  # Kernel size divided by 2
        mask = mask.unsqueeze(1).float()  # Add channel dimension
        blurred = F.conv2d(mask, kernel, padding=padding)
        return blurred.squeeze(1)  # Remove channel dimension

    # Example usage
    start = time.time()
    mask = apply_gaussian_blur(mask, gaussian_kernel)
    print("Gaussian Blur tuime (torch)", time.time() - start)

    start = time.time()
    # apply gaussian kernel to the mask
    mask = kornia.filters.gaussian_blur2d(mask.unsqueeze(1).float(), (5, 5), (2.5, 2.5)).squeeze(1)

    print("Gaussian Blur Time:", time.time() - start)
    start = time.time()

    # combine masks
    mask = mask.reshape(batch, n_masks, height, width).min(dim=1)[0]

    # Apply mask to the image
    height_scan[mask == 0] = torch.nan

    print("Combining Masks Time:", time.time() - start)

    return height_scan


# Example usage
batch, height, width = 7156, 60, 40
tensor = torch.rand(batch, height, width, device="cuda")
start = time.time()
masked_tensor = apply_mask(tensor.clone(), n_masks=5)
print("Time taken:", time.time() - start)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(tensor[0].cpu().numpy(), cmap="gray")
axs[0].set_title("Original Image")
axs[1].imshow(masked_tensor[0].cpu().numpy(), cmap="gray")
axs[1].set_title("Masked Image")
plt.show()
