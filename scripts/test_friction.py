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
parser.add_argument("--regular", action="store_true", default=True, help="Spawn robots in a regular pattern.")
parser.add_argument("--runs", type=str, nargs="+", default=None, help="Name of the run.")
parser.add_argument("--equal-actions", action="store_true", default=False, help="Have the same actions for all envs.")
parser.add_argument("--max_actions", action="store_true", default=True, help="Apply maximum xy vel command.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=50)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.friction = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import fdm.mdp as mdp
from fdm.env_cfg.env_cfg_base import CommandsCfg
from fdm.runner import FDMRunner, FDMRunnerCfg
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
    cfg.replay_buffer_cfg.trajectory_length = cfg.model_cfg.prediction_horizon + 1
    cfg.trainer_cfg.num_samples = 2000
    cfg.trainer_cfg.logging = False
    # remove terrain analysis
    cfg.env_cfg.commands = CommandsCfg()
    cfg.env_cfg.observations.planner_obs = None
    # make environment origins regular
    cfg.env_cfg.scene.terrain.regular_spawning = True
    cfg.env_cfg.scene.env_spacing = 10.0

    # swap environment
    cfg.env_cfg.scene.terrain.terrain_type = "plane"
    # make origin selection deterministic
    cfg.env_cfg.scene.terrain.random_seed = 0

    # set name of the run
    if args_cli.runs is not None:
        cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

    # set regular spawning pattern
    cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center
    pop_items = [item for item in cfg.env_cfg.events.reset_base.params.keys() if item != "asset_cfg"]
    for item in pop_items:
        cfg.env_cfg.events.reset_base.params.pop(item)

    # adjust actions taken by all robots
    cfg.agent_cfg = FDMRunnerCfg().agent_cfg
    cfg.agent_cfg.horizon = cfg.model_cfg.prediction_horizon + 1

    # remove reset when in collision
    cfg.env_cfg.terminations.base_contact = None

    # setup runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)
    # set initial state and initial predictions, plot them
    runner.test(use_planner=False)

    while simulation_app.is_running():
        runner.env.render()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
