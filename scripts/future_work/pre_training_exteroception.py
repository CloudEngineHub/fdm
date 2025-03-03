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
parser.add_argument(
    "--mode",
    type=str,
    default="dev",
    choices=["dev", "train", "eval"],
    help="Mode of the script.",
)
parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=1024)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.mode == "train":
    args_cli.headless = True
else:
    args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from fdm.env_cfg import PreTrainingFDMDepthCfg
from fdm.model import FDMExteroceptionModelCfg
from fdm.runner import FDMRunner, FDMRunnerCfg
from fdm.utils.args_cli_utils import env_modifier_post_init, robot_changes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # init runner cfg
    cfg = FDMRunnerCfg(
        model_cfg=FDMExteroceptionModelCfg(),
        env_cfg=PreTrainingFDMDepthCfg(),
    )

    # change wandb logging for exteroceptive pre-training
    cfg.trainer_cfg.experiment_name = "fdm_exteroceptive_pre_training"

    if args_cli.mode == "dev":
        # overwrite some configs for easier debugging
        cfg.replay_buffer_cfg.trajectory_length = 20
        cfg.trainer_cfg.num_samples = 4000
        cfg.trainer_cfg.logging = False
        args_cli.num_envs = 24
    elif args_cli.mode == "train":
        cfg.replay_buffer_cfg.trajectory_length = 25
        cfg.trainer_cfg.num_samples = 10000
        cfg.collection_rounds = 25
        cfg.trainer_cfg.batch_size = 256
        args_cli.num_envs = 512
    else:
        raise ValueError(f"Unknown mode {args_cli.mode}")

    runner = FDMRunner(cfg=cfg, args_cli=args_cli)

    # modify depth camera intrinsic
    args_cli.env = "depth"
    runner = robot_changes(runner, args_cli)
    runner = env_modifier_post_init(runner, args_cli)

    # set the size of the target height map
    size = cfg.env_cfg.scene.target_height_scan.pattern_cfg.size
    resolution = cfg.env_cfg.scene.target_height_scan.pattern_cfg.resolution
    runner.model.cfg.target_height_map_size = (
        torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution).shape[0],
        torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution).shape[0],
    )
    assert runner.env.scene.sensors["target_height_scan"].num_rays == int(
        runner.model.cfg.target_height_map_size[0] * runner.model.cfg.target_height_map_size[1]
    )

    # run
    runner.train()

    if args_cli.mode == "train":
        # save encoder of best model
        runner.model.load_state_dict(torch.load(runner.model.get_model_path(runner.trainer.log_dir), weights_only=True))
        torch.save(
            runner.model.obs_exteroceptive_encoder.state_dict(),
            runner.model.get_model_path(runner.trainer.log_dir, "image_encoder"),
        )
        torch.save(
            runner.model.add_obs_exteroceptive_encoder.state_dict(),
            runner.model.get_model_path(runner.trainer.log_dir, "foot_scan_encoder"),
        )


if __name__ == "__main__":
    main()
