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
parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")
parser.add_argument(
    "--domain",
    type=str,
    default="velocity",
    choices=["position", "velocity"],
    help="Predictions in which domain, either position or velocity.",
)
parser.add_argument(
    "--mode",
    type=str,
    default="dev",
    choices=["dev", "train", "eval"],
    help="Mode of the script.",
)

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=10240)
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

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg

import fdm.mdp as mdp
from fdm.env_cfg import FDMCfg
from fdm.model import FDMProprioceptionModelCfg, FDMProprioceptionVelocityModelCfg
from fdm.runner import FDMRunner, FDMRunnerCfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, robot_changes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # select correct model
    if args_cli.domain == "velocity":
        model_cfg = FDMProprioceptionVelocityModelCfg()
    elif args_cli.domain == "position":
        model_cfg = FDMProprioceptionModelCfg()
    else:
        raise ValueError(f"Unknown domain {args_cli.domain}")

    # init runner cfg
    env_cfg = FDMCfg()
    env_cfg.scene.env_spacing = 10.0
    cfg = FDMRunnerCfg(
        model_cfg=model_cfg,
        env_cfg=env_cfg,
    )
    # override terrain
    cfg.env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=True,
    )
    # override root reset
    cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_uniform
    cfg.env_cfg.events.reset_base.params["pose_range"] = {
        "x": (-10.0, 10.0),
        "y": (-10.0, 10.0),
        "yaw": cfg.env_cfg.events.reset_base.params["yaw_range"],
    }
    cfg.env_cfg.events.reset_base.params.pop("yaw_range")
    # apply robot changes
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # change wandb logging for proprioception pre-training
    cfg.trainer_cfg.experiment_name = "fdm_proprioception_pre_training"
    # remove the test datasets
    cfg.trainer_cfg.test_datasets = None

    # TRAIN
    cfg.trainer_cfg.learning_rate_warmup = 1  # acceleration and velocity loss not present at the start
    cfg.trainer_cfg.learning_rate = 3e-3

    if args_cli.mode == "train":
        cfg.trainer_cfg.num_samples = 150000
        cfg.trainer_cfg.batch_size = 1024
        cfg.collection_rounds = 10

        runner = FDMRunner(cfg=cfg, args_cli=args_cli)
        # run
        runner.train()

        # save encoder of best model
        runner.model.load_state_dict(torch.load(runner.model.get_model_path(runner.trainer.log_dir), weights_only=True))
        # proprioception_state encoder
        torch.save(
            runner.model.state_obs_proprioceptive_encoder.state_dict(),
            runner.model.get_model_path(runner.trainer.log_dir, "proprioception_state_encoder"),
        )
        # normalizer
        torch.save(
            runner.model.proprioceptive_normalizer.state_dict(),
            runner.model.get_model_path(runner.trainer.log_dir, "proprioception_normalizer"),
        )
        # action encoder
        if runner.model.action_encoder is not None:
            torch.save(
                runner.model.action_encoder.state_dict(),
                runner.model.get_model_path(runner.trainer.log_dir, "action_encoder"),
            )
        # friction predictor
        torch.save(
            runner.model.friction_predictor.state_dict(),
            runner.model.get_model_path(runner.trainer.log_dir, "friction_predictor"),
        )
        # state predictor
        torch.save(
            runner.model.state_predictor.state_dict(),
            runner.model.get_model_path(runner.trainer.log_dir, "state_predictor"),
        )
        # recurrent layer
        torch.save(
            runner.model.recurrence.state_dict(), runner.model.get_model_path(runner.trainer.log_dir, "recurrence")
        )
        # max velocity and acceleration
        torch.save(
            runner.model.velocity_limits, runner.model.get_model_path(runner.trainer.log_dir, "_velocity_limits")
        )
        torch.save(
            runner.model.acceleration_limits,
            runner.model.get_model_path(runner.trainer.log_dir, "_acceleration_limits"),
        )

    elif args_cli.mode == "dev":
        # change number of samples as have more environments
        cfg.replay_buffer_cfg.trajectory_length = 30
        cfg.trainer_cfg.logging = False
        cfg.trainer_cfg.num_samples = 1000
        args_cli.num_envs = 124

        runner = FDMRunner(cfg=cfg, args_cli=args_cli)
        # run
        runner.train()

    # EVAL
    elif args_cli.mode == "eval":
        # reduce number of environments for evaluation and add specific evaluation settings
        args_cli.num_envs = 128
        args_cli.equal_actions = False
        if args_cli.run_name is not None:
            cfg.trainer_cfg.load_run = args_cli.run_name
        cfg.trainer_cfg.logging = False
        cfg.trainer_cfg.num_samples = 1000
        cfg.env_cfg.scene.env_spacing = 10.0
        runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)

        runner.evaluate()

    else:
        raise ValueError(f"Unknown mode {args_cli.mode}")


if __name__ == "__main__":
    main()
