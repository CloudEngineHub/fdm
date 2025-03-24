# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect data in an environment for evaluation during training."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect Training Data in Testing env.")

parser.add_argument("--test_env", type=str, default="plane", help="Environment to collect data for.")
parser.add_argument("--terrain_analysis_points", type=int, default=15000, help="Number of points for terrain analysis.")
parser.add_argument("--height_threshold", type=float, default=None, help="Height threshold for samples.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=256)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pickle
import torch

import isaaclab.sim as sim_utils
import omni
from isaaclab.terrains import TerrainImporterCfg

import fdm.mdp as mdp
import fdm.runner as fdm_runner_cfg
from fdm import FDM_DATA_DIR, LARGE_UNIFIED_HEIGHT_SCAN
from fdm.env_cfg import terrain_cfg as fdm_terrain_cfg
from fdm.env_cfg.env_cfg_base import TERRAIN_ANALYSIS_CFG
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
    cfg = cfg_modifier_pre_init(cfg, args_cli, dataset_collecton=True)
    # change terrain config
    cfg.env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # visual_material=sim_utils.MdlFileCfg(
        #     mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        #     project_uvw=True,
        # ),
        debug_vis=False,
        usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    )
    # turn of logging
    cfg.trainer_cfg.logging = False
    # set test datasets to None
    cfg.trainer_cfg.test_datasets = None
    # limit number of samples
    cfg.trainer_cfg.num_samples = 50000
    # restrict goal generator to be purely goal-generated without any planner
    cfg.env_cfg.commands.command = mdp.ConsecutiveGoalCommandCfg(
        resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
        terrain_analysis=TERRAIN_ANALYSIS_CFG,
    )
    if hasattr(cfg.env_cfg.observations, "planner_obs"):
        cfg.env_cfg.observations.planner_obs.goal.func = mdp.goal_command_w_se2
        cfg.env_cfg.observations.planner_obs.goal.params = {"command_name": "command"}
    cfg.env_cfg.curriculum = None

    # height threshold
    if args_cli.height_threshold is not None:
        cfg.trainer_cfg.height_threshold = args_cli.height_threshold
        # increase number of samples as more will get filtered out
        cfg.trainer_cfg.num_samples = 100000
        cfg.replay_buffer_cfg.trajectory_length = 250

    print(f"[INFO] Collecting data for {args_cli.test_env}")
    # set next env
    if os.path.exists(args_cli.test_env):
        cfg.env_cfg.scene.terrain.terrain_type = "usd"
        cfg.env_cfg.scene.terrain.usd_path = args_cli.test_env
        dataset_path = args_cli.test_env[:-4]
    elif args_cli.test_env == "plane":
        cfg.env_cfg.scene.terrain.terrain_type = "plane"
        os.makedirs(os.path.join(FDM_DATA_DIR, "Terrains/test_datasets"), exist_ok=True)
        dataset_path = os.path.join(FDM_DATA_DIR, "Terrains/test_datasets", args_cli.test_env)
    elif hasattr(fdm_terrain_cfg, args_cli.test_env):
        cfg.env_cfg.scene.terrain.terrain_type = "generator"
        cfg.env_cfg.scene.terrain.terrain_generator = getattr(fdm_terrain_cfg, args_cli.test_env)
        os.makedirs(os.path.join(FDM_DATA_DIR, "Terrains/test_datasets"), exist_ok=True)
        dataset_path = os.path.join(FDM_DATA_DIR, "Terrains/test_datasets", args_cli.test_env)
    else:
        raise ValueError(f"Unknown terrain {args_cli.test_env}")
    # create a new stage
    omni.usd.get_context().new_stage()
    # init runner
    runner = fdm_runner_cfg.FDMRunner(cfg=cfg, args_cli=args_cli)
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)
    # collect validation dataset
    runner._collect(eval=True)
    # save dataset
    if args_cli.reduced_obs:
        dataset_path += "_reducedObs"
    if args_cli.remove_torque:
        dataset_path += "_noTorque"
    if args_cli.noise:
        dataset_path += "_noise"
    elif args_cli.occlusions:
        dataset_path += "_occlusions"
    if args_cli.env == "baseline":
        dataset_path += "_baseline"
    if LARGE_UNIFIED_HEIGHT_SCAN:
        dataset_path += "_largeUnifiedHeightScan"
    if args_cli.height_threshold is not None:
        dataset_path += f"_heightThreshold{args_cli.height_threshold}"
    with open(dataset_path + ".pkl", "wb") as fp:
        pickle.dump(runner.trainer.val_dataset, fp)
    print(f"[INFO] Data saved to {dataset_path}.pkl")


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
