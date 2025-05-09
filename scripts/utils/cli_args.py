# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse


def add_fdm_args(parser: argparse.ArgumentParser, default_num_envs: int = 2048):
    """Add FDM arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    parser.add_argument("--num_envs", type=int, default=default_num_envs, help="Number of environments to simulate.")
    # NOTE: heuristic and rmp are only used for the planner
    parser.add_argument(
        "--env",
        type=str,
        default="height",
        choices=["baseline", "depth", "height", "heuristic"],
        help="Name of the environment to load.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="anymal_perceptive",
        choices=["anymal", "anymal_perceptive", "aow", "tytan", "tytan_quiet"],
        help="Select the robot.",
    )
    parser.add_argument("--occlusions", action="store_true", default=False, help="Add occlusion to the observations.")
    parser.add_argument("--noise", action="store_true", default=False, help="Add noise to the observations.")
    parser.add_argument("--reduced_obs", action="store_true", default=True, help="Use a reduced set of observations.")
    parser.add_argument(
        "--remove_torque", action="store_true", default=True, help="Remove the joint torque from the observations."
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=None,
        choices=["no_state_obs", "no_proprio_obs", "no_height_scan"],
        help="Ablation mode to use.",
    )
    parser.add_argument("--timestamp", type=float, default=None, help="Command timestep of the model.")
    parser.add_argument("--friction", action="store_true", default=False, help="Vary friction for each robot.")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode.")
