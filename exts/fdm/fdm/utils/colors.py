# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import colorsys


def generate_colors(num_colors, start_hue=0.0, end_hue=1.0):
    # Calculate the step size between consecutive hues
    step_size = (end_hue - start_hue) / num_colors

    # Generate the colors
    colors = []
    for i in range(num_colors):
        # Calculate the current hue
        current_hue = start_hue + i * step_size

        # Convert the hue to RGB
        rgb_color = colorsys.hsv_to_rgb(current_hue, 1.0, 1.0)
        rgb_color = (*tuple(rgb_color), 1.0)

        # Append the RGB color to the list
        colors.append(rgb_color)

    return colors
