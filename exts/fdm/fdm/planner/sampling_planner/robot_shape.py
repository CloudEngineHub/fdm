# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from math import ceil, floor

from .trajectory_optimizer_cfg import RobotCfg


def return_grid_indices(rectangles, grid_resolution):
    if len(rectangles) == 0:
        return np.array([])

    rectangles = np.array(rectangles)
    rectangle_norm = rectangles / grid_resolution
    indices = []
    for nr in range(rectangle_norm.shape[0]):

        x_forward = ceil(rectangle_norm[nr, 1, 0])
        x_backward = floor(rectangle_norm[nr, 0, 0])
        y_left = ceil(rectangle_norm[nr, 1, 1])
        y_right = floor(rectangle_norm[nr, 0, 1])

        # Vehicle Coordinate Frame
        #        x
        #        |
        #        |
        #  y------

        for x in np.arange(x_backward, x_forward):
            for y in np.arange(y_right, y_left):
                indices.append([x, y])

    unique = {tuple(pair) for pair in indices}
    unique_indices = [list(pair) for pair in unique]
    return np.array(unique_indices)


def filter(indices_to_filter, indices_for_removal):
    idx_to_remove = []
    for i in range(indices_for_removal.shape[0]):
        idx_to_remove.append(
            np.where(
                (indices_to_filter[:, 0] == indices_for_removal[i, 0])
                * (indices_to_filter[:, 1] == indices_for_removal[i, 1])
            )[0]
        )

    m = np.ones_like(indices_to_filter[:, 0], dtype=bool)
    if np.concatenate(idx_to_remove).sum() != 0:
        m[np.concatenate(idx_to_remove)] = 0

    return indices_to_filter[m]


def visu(fatal_xy, risky_xy, cautious_xy):
    import matplotlib.pyplot as plt

    # Combine all indices to find the overall bounds
    all_indices = np.vstack((fatal_xy, risky_xy, cautious_xy))

    # Find the maximum extents
    max_x, max_y = np.max(all_indices, axis=0)
    min_x, min_y = np.min(all_indices, axis=0)

    # Determine the size of the image, making it 10 pixels larger than the max indices
    img_size_w = int((max_y - min_y) + 10)
    img_size_h = int((max_x - min_x) + 10)

    # Create a white background image
    image = np.ones((img_size_h, img_size_w, 3))

    # Function to plot the indices with the given color
    # indices are in x,y ->
    def plot_indices(indices, color):
        for x, y in indices:
            # flip axis
            h = -x
            w = -y

            # center
            h = int(h - (img_size_h) / 2)
            w = int(w - (img_size_w) / 2)

            image[int(h), int(w)] = color

    # Plot each set of coordinates
    plot_indices(cautious_xy, [0, 0.5, 1])  # Blue for cautious
    plot_indices(risky_xy, [1, 0.5, 0])  # Orange for risky
    plot_indices(fatal_xy, [1, 0, 0])  # Red for fatal
    # Display the image
    plt.imshow(image)
    plt.axis("off")  # Hide axes
    plt.show()


def get_robot_shape(cfg: RobotCfg, device):
    fatal_xy = return_grid_indices(cfg.fatal, cfg.resolution)
    risky_xy = return_grid_indices(cfg.risky, cfg.resolution)
    cautious_xy = return_grid_indices(cfg.cautious, cfg.resolution)

    risky_xy = filter(risky_xy, fatal_xy)
    cautious_xy = filter(cautious_xy, fatal_xy)
    cautious_xy = filter(cautious_xy, risky_xy)

    fatal_xy = torch.from_numpy(fatal_xy).to(device).type(torch.float32)
    risky_xy = torch.from_numpy(risky_xy).to(device).type(torch.float32)
    cautious_xy = torch.from_numpy(cautious_xy).to(device).type(torch.float32)
    return fatal_xy, risky_xy, cautious_xy


if __name__ == "__main__":
    # Definition of rectangles: bottom_left, top_right
    fatal_rectangles = [
        ((-0.43, -0.235), (0.43, 0.235)),  # BODY
        ((0.43, -0.265), (0.63, 0.265)),  # Top Drive Area
        ((0.63, -0.125), (0.65, 0.125)),  # Top Face
        ((-0.63, -0.265), (-0.43, 0.265)),  # Bottom Drive Area
        ((-0.65, -0.125), (-0.63, 0.125)),  # Bottom Face
    ]
    risky_rectangels = [((-0.70, -0.58), (0.70, 0.58))]
    cautious_rectangels = [((-0.80, -0.68), (0.80, 0.68))]

    grid_resolution = 0.02

    fatal_xy = return_grid_indices(fatal_rectangles, grid_resolution)
    risky_xy = return_grid_indices(risky_rectangels, grid_resolution)
    cautious_xy = return_grid_indices(cautious_rectangels, grid_resolution)

    risky_xy = filter(risky_xy, fatal_xy)
    cautious_xy = filter(cautious_xy, fatal_xy)
    cautious_xy = filter(cautious_xy, risky_xy)

    visu(fatal_xy, risky_xy, cautious_xy)
