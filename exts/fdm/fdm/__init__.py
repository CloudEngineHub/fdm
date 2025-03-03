# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for navigation tasks."""

import os
import toml

# Conveniences to other module directories via relative paths
FDM_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

FDM_DATA_DIR = os.path.join(FDM_EXT_DIR, "data")
"""Path to the extension data directory."""

FDM_METADATA = toml.load(os.path.join(FDM_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = FDM_METADATA["package"]["version"]


###
# Constants
###

TOTAL_TIME_PREDICTION_HORIZON = 6.0  # seconds
"""Currently used for Ablation: Total time of the prediction horizon in seconds (i.e. command timestep * horizon)."""

LARGE_UNIFIED_HEIGHT_SCAN = False
"""Whether to use a large unified height scan for all robot models."""

PLANNER_MODE = False
PLANNER_MODE_BASELINE = False
"""Whether to use the planner mode for the sampling-based planner."""

if LARGE_UNIFIED_HEIGHT_SCAN:
    VEL_RANGE_X = (-1.0, 1.0)  # m/s
    VEL_RANGE_Y = (-0.7, 0.7)  # m/s
    VEL_RANGE_YAW = (-1.0, 1.0)  # rad/s
elif PLANNER_MODE:
    # Restrict the movement space to make it easier for the sampling-based planner
    VEL_RANGE_X = (-0.1, 1.0)
    VEL_RANGE_Y = (-0.1, 0.1)
    # VEL_RANGE_YAW = (-0.33, 0.33)
    VEL_RANGE_YAW = (-0.66, 0.66)
elif PLANNER_MODE_BASELINE:
    VEL_RANGE_X = (-0.1, 0.5)
    VEL_RANGE_Y = (-0.2, 0.2)
    VEL_RANGE_YAW = (-1.0, 1.0)
else:
    VEL_RANGE_X = (-0.1, 1.5)  # m/s  (prev. 0.2, 1.5)
    VEL_RANGE_Y = (-0.4, 0.4)  # m/s  (prev. -0.1, 0.1)
    VEL_RANGE_YAW = (-1.0, 1.0)  # rad/s


####
# Unify Colors for plotting functions
###

PAPER_COLORS_RGB_U8 = {
    # steps for violin plot
    "step_4": (245, 204, 122),  # beige
    "step_9": (251, 151, 39),  # orange
    # paths and model predictions
    "collision": (210, 43, 38),  # red
    "future_traj": (251, 151, 39),  # orange (66, 173, 187),  # cyan
    "constant_vel": (150, 36, 145),  # mangenta
    "ours": (94, 129, 172),  # blue
    "baseline": (167, 204, 110),  # green
    # light colors
    "constant_vel_light": (223, 124, 218),  # mangenta_light
    "collision_light": (230, 121, 117),  # red_light
    "ours_light": (137, 166, 210),  # blue_light
    "step_9_light": (252, 188, 115),  # orange_light
    "future_traj_light": (252, 188, 115),  # orange_light (164, 216, 223), # cyan_light
    "baseline_light": (192, 218, 152),  # green_light
}
PAPER_COLORS_RGBA_U8 = {k: (v[0], v[1], v[2], 255) for k, v in PAPER_COLORS_RGB_U8.items()}
PAPER_COLORS_RGB_F = {
    k: (float(v[0]) / 255.0, float(v[1]) / 255.0, float(v[2]) / 255.0) for k, v in PAPER_COLORS_RGB_U8.items()
}
PAPER_COLORS_RGBA_F = {k: (v[0], v[1], v[2], 1.0) for k, v in PAPER_COLORS_RGB_F.items()}
PAPER_COLORS_HEX = {k: f"#{v[0]:02x}{v[1]:02x}{v[2]:02x}" for k, v in PAPER_COLORS_RGB_U8.items()}
