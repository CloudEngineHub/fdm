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
parser = argparse.ArgumentParser(description="Collect Training Data in Testing env.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=128)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = False

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pickle
import torch
from dataclasses import MISSING

import omni

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.terrains import TerrainImporterCfg

import fdm.mdp as mdp
import fdm.runner as fdm_runner_cfg
from fdm import FDM_DATA_DIR
from fdm.env_cfg import terrain_cfg as fdm_terrain_cfg
from fdm.env_cfg.env_cfg_base import TERRAIN_ANALYSIS_CFG
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, robot_changes, runner_cfg_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

TEST_DATASET_ENV = [
    "plane",
    "PILLAR_EVAL_CFG",
    "GRID_EVAL_CFG",
    "STAIRS_WALL_EVAL_CFG",
    "STAIRS_EVAL_CFG",
    "STAIRS_RAMP_EVAL_CFG",
    os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_stairs.usd"),
    os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_perlin_stepping_stones.usd"),
    os.path.join(FDM_DATA_DIR, "Terrains", "navigation_terrain_ramp_platform.usd"),
]


def mod_terrain_cfg(cfg: fdm_runner_cfg.FDMRunnerCfg) -> fdm_runner_cfg.FDMRunnerCfg:
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
        #     mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
        #     project_uvw=True,
        # ),
        debug_vis=False,
        usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    )
    return cfg


def main():
    # iterate through envs
    for idx, test_env in enumerate(TEST_DATASET_ENV):
        # init runner cfg
        cfg = runner_cfg_init(args_cli)
        # select robot
        cfg = robot_changes(cfg, args_cli)
        # modify cfg
        cfg = cfg_modifier_pre_init(cfg, args_cli)
        # change terrain config
        cfg = mod_terrain_cfg(cfg)
        # turn of logging
        cfg.trainer_cfg.logging = False
        # set test datasets to None
        cfg.trainer_cfg.test_datasets = None
        # limit number of samples
        cfg.trainer_cfg.num_samples = 50000
        cfg.replay_buffer_cfg.trajectory_length = 20
        # restrict goal generator to be purely goal-generated without any planner
        cfg.env_cfg.commands.command = mdp.ConsecutiveGoalCommandCfg(
            resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
            terrain_analysis=TERRAIN_ANALYSIS_CFG,
        )
        if hasattr(cfg.env_cfg.observations, "planner_obs"):
            cfg.env_cfg.observations.planner_obs.goal.func = mdp.goal_command_w_se2
            cfg.env_cfg.observations.planner_obs.goal.params = {"command_name": "command"}
        cfg.env_cfg.curriculum = MISSING

        print(f"[INFO] Collecting data for {test_env}")
        # set next env
        if os.path.exists(test_env):
            cfg.env_cfg.scene.terrain.terrain_type = "usd"
            cfg.env_cfg.scene.terrain.usd_path = test_env
            dataset_path = test_env[:-4]
        elif test_env == "plane":
            cfg.env_cfg.scene.terrain.terrain_type = "plane"
            dataset_path = test_env
        elif hasattr(fdm_terrain_cfg, test_env):
            cfg.env_cfg.scene.terrain.terrain_type = "generator"
            cfg.env_cfg.scene.terrain.terrain_generator = getattr(fdm_terrain_cfg, test_env)
            dataset_path = test_env
        else:
            raise ValueError(f"Unknown terrain {test_env}")
        # set mesh path new
        cfg.env_cfg.scene.terrain.prim_path = f"/World/ground_{idx}"
        cfg.env_cfg.scene.env_sensor.mesh_prim_paths = [f"/World/ground_{idx}"]
        cfg.env_cfg.scene.foot_scanner_lf.mesh_prim_paths = [f"/World/ground_{idx}"]
        cfg.env_cfg.scene.foot_scanner_rf.mesh_prim_paths = [f"/World/ground_{idx}"]
        cfg.env_cfg.scene.foot_scanner_lh.mesh_prim_paths = [f"/World/ground_{idx}"]
        cfg.env_cfg.scene.foot_scanner_rh.mesh_prim_paths = [f"/World/ground_{idx}"]
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
        if args_cli.env == "baseline":
            dataset_path += "_baseline"
        with open(dataset_path + ".pkl", "wb") as fp:
            pickle.dump(runner.trainer.val_dataset, fp)
        print(f"[INFO] Data saved to {dataset_path}.pkl")
        runner.close()
        del runner
        del cfg


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
