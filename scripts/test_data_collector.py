"""Script to train a Forward-Dynamics-Model"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect Training Data in Testing env.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of environments to simulate.")
parser.add_argument(
    "--env",
    type=str,
    default="height",
    choices=["lidar", "depth", "height", "reduced"],
    help="Name of the environment to load.",
)
parser.add_argument("--noise", action="store_true", default=False, help="Add noise to the observations.")
parser.add_argument("--friction", action="store_true", default=False, help="Vary friction for each robot.")
parser.add_argument("-d", "--debug", action="store_true", default=False, help="Enable debug mode.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pickle
import torch

import omni

from omni.isaac.lab_assets import ISAACLAB_ASSETS_DATA_DIR

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.terrains import TerrainImporterCfg

import fdm.utils.runner_cfg as fdm_runner_cfg
from fdm.utils import FDMRunner
from fdm.utils.args_cli_utils import (
    cfg_modifier_pre_init,
    env_modifier_post_init,
    runner_cfg_init,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

TEST_DATASET_ENV = [
    os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Terrains", "navigation_terrain_stairs.usd"),
    os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Terrains", "navigation_terrain_perlin_stepping_stones.usd"),
    os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Terrains", "navigation_terrain_ramp_platform.usd"),
]


def mod_terrain_cfg(cfg: fdm_runner_cfg.FDMRunnerCfg) -> fdm_runner_cfg.FDMRunnerCfg:
    cfg.env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=True,
        usd_uniform_env_spacing=10.0,  # 10m spacing between environment origins in the usd environment
    )
    return cfg


def main():
    # init runner cfg
    cfg = runner_cfg_init(args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)
    # change terrain config
    cfg = mod_terrain_cfg(cfg)
    # vary friction of each robot
    if args_cli.friction:
        cfg.env_cfg.events.physics_material.params["static_friction_range"] = (0.2, 0.8)
    # turn of logging
    cfg.trainer_cfg.logging = False
    # set test datasets to None
    cfg.trainer_cfg.test_datasets = None

    # debug
    cfg.replay_buffer_cfg.trajectory_length = 20
    cfg.trainer_cfg.num_samples = 1000
    args_cli.num_envs = 20

    # iterate through envs
    for test_env_path in TEST_DATASET_ENV:
        print(f"[INFO] Collecting data for {test_env_path}")
        # set next env
        cfg.env_cfg.scene.terrain.usd_path = test_env_path
        # create a new stage
        omni.usd.get_context().new_stage()
        # init runner
        runner = FDMRunner(cfg=cfg, args_cli=args_cli)
        # post modify runner and env
        runner = env_modifier_post_init(runner, args_cli=args_cli)
        # collect validation dataset
        runner._collect(eval=True)
        # save dataset
        dataset_path = test_env_path[:-4]
        with open(dataset_path + ".pkl", "wb") as fp:
            pickle.dump(runner.trainer.val_dataset, fp)
        print(f"[INFO] Data saved to {dataset_path}.pkl")
        runner.env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
