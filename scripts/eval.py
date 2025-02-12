# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--terrain-cfg", type=str, default=None, help="Name of the terrain config to load.")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
parser.add_argument(
    "--runs",
    type=str,
    nargs="+",
    default="Oct25_13-04-59_MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut_NoEarlyCollFilter",
    help="Name of the run.",
)
parser.add_argument("--equal-actions", action="store_true", default=False, help="Have the same actions for all envs.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=50)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import fdm.env_cfg as env_cfg
import fdm.mdp as mdp
from fdm.runner import FDMRunner
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, robot_changes, runner_cfg_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # init runner cfg
    cfg = runner_cfg_init(args_cli)
    # select robot
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # overwrite some configs for easier debugging
    cfg.replay_buffer_cfg.trajectory_length = 50
    cfg.trainer_cfg.num_samples = 2000
    cfg.trainer_cfg.logging = False

    # switch to terrain if provided
    if isinstance(args_cli.terrain_cfg, str) and args_cli.terrain_cfg.endswith(".usd"):
        cfg.env_cfg.scene.terrain.usd_path = args_cli.terrain_cfg
    elif isinstance(args_cli.terrain_cfg, str) and hasattr(env_cfg, args_cli.terrain_cfg):
        cfg.env_cfg.scene.terrain.terrain_generator = getattr(env_cfg, args_cli.terrain_cfg)
        cfg.env_cfg.scene.terrain.terrain_generator.num_cols = 2 * len(
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains.items()
        )
        cfg.env_cfg.scene.terrain.terrain_generator.num_rows = 3
        cfg.env_cfg.scene.terrain.terrain_generator.size = (20.0, 20.0)
        cfg.env_cfg.scene.terrain.terrain_generator.curriculum = True
    elif args_cli.terrain_cfg is None:
        print(f"[INFO] Using default terrain config. {cfg.env_cfg.scene.terrain.usd_path}")
    else:
        raise ValueError(f"Unknown terrain config {args_cli.terrain_cfg}")

    # set name of the run
    if args_cli.runs is not None:
        cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

    # set regular spawning pattern
    if args_cli.regular:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_regular
        cfg.env_cfg.events.reset_base.params.pop("pose_range")
        cfg.env_cfg.events.reset_base.params.pop("velocity_range")

    # setup runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)
    # set initial state and initial predictions, plot them
    runner.evaluate()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
