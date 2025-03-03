# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
parser.add_argument("--run", type=str, default=None, help="Name of the run.")
parser.add_argument("--terrain_analysis_points", type=int, default=10000, help="Number of points for terrain analysis.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=10)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import fdm.mdp as mdp
from fdm.planner import FDMPlanner, get_planner_cfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, planner_cfg_init, robot_changes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

args_cli.env = "baseline"
# args_cli.run = "Jan13_15-36-24_Baseline_NewEnv_NewCollisionShape_CorrLidar"
# args_cli.run = "Jan20_21-15-35_Baseline_NewEnv_NewCollisionShape_CorrLidar_UnifiedCollLoss"
args_cli.run = "Jan29_21-30-14_local_4mLiDAR"
# args_cli.run = "Jan28_22-45-54_Baseline_NewEnv_NewCollisionShape_CorrLidar_UnifiedCollLoss_2DEnv_NoBatchNorm"
# args_cli.num_envs = 5


def main():
    # setup runner
    cfg = planner_cfg_init(args_cli)
    # robot changes
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # set regular spawning pattern
    if args_cli.regular:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_regular
        cfg.env_cfg.events.reset_base.params.pop("pose_range")
        cfg.env_cfg.events.reset_base.params.pop("velocity_range")

    # get planner cfg
    sampling_planner_cfg_dict = get_planner_cfg(args_cli.num_envs, traj_dim=10, debug=False, device="cuda")

    if args_cli.env == "baseline":
        sampling_planner_cfg_dict["to_cfg"]["control"] = "fdm_baseline"
        # sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 1
        sampling_planner_cfg_dict["optim"]["population_size"] = 256
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_safety_factor"] = 0.0

    # build planner
    planner = FDMPlanner(cfg, sampling_planner_cfg_dict, args_cli=args_cli)
    # post modify runner and env
    planner = env_modifier_post_init(planner, args_cli=args_cli)

    # navigate
    planner.navigate()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
