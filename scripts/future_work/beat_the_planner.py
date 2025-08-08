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
parser.add_argument(
    "--run",
    type=str,
    default="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5",
    help="Name of the run.",
)
parser.add_argument(
    "--scene", default="matterport", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=2)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from omni.isaac.core.objects import VisualCuboid
from omni.isaac.matterport.config import MatterportImporterCfg
from omni.isaac.matterport.domains import MatterportRayCasterCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

import nav_tasks.sensors as nav_patterns

import fdm.mdp as mdp
from fdm import LARGE_UNIFIED_HEIGHT_SCAN
from fdm.env_cfg.env_cfg_height import HeightTerrainSceneCfg
from fdm.planner import FDMPlanner, PlannerHeightCfg, get_planner_cfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, robot_changes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@configclass
class MatterportHeightTerrainSceneCfg(HeightTerrainSceneCfg):
    # ground terrain
    terrain = MatterportImporterCfg(
        prim_path="/World/matterport",
        terrain_type="matterport",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # NOTE: this path should be absolute to load the textures correctly
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.usd",
        # obj_filepath="${USER_PATH_TO_USD}/matterport.usd",
        groundplane=True,
    )

    # lights
    sphere_1 = AssetBaseCfg(
        prim_path="/World/sphere_1",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=3000.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(8, 1, 2.0)),
    )
    sphere_2 = AssetBaseCfg(
        prim_path="/World/sphere_2",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(10.5, -5.5, 2.0)),
    )
    sphere_3 = AssetBaseCfg(
        prim_path="/World/sphere_3",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(6.0, -5.5, 2.0)),
    )
    sphere_4 = AssetBaseCfg(
        prim_path="/World/sphere_4",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -12, 2.0)),
    )


def exchange_to_matterport_sensors(cfg: MatterportHeightTerrainSceneCfg) -> MatterportHeightTerrainSceneCfg:
    # raycaster for the planner
    cfg.env_sensor = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MatterportRayCasterCfg.OffsetCfg(
            pos=(1.75, 0.0, 0.5) if not LARGE_UNIFIED_HEIGHT_SCAN else (0.0, 0.0, 0.5)
        ),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1, size=(4.5, 5.9) if not LARGE_UNIFIED_HEIGHT_SCAN else (7.9, 5.9)
        ),
        debug_vis=False,
        mesh_prim_paths=[
            "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
        ],
        max_distance=10.0,
    )

    # height scans for locomotion
    cfg.foot_scanner_lf = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT",
        offset=MatterportRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        ray_alignment="yaw",
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=[
            "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
        ],
        max_distance=10.0,
    )

    cfg.foot_scanner_rf = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RF_FOOT",
        offset=MatterportRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        ray_alignment="yaw",
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=[
            "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
        ],
        max_distance=10.0,
    )

    cfg.foot_scanner_lh = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LH_FOOT",
        offset=MatterportRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        ray_alignment="yaw",
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=[
            "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
        ],
        max_distance=10.0,
    )

    cfg.foot_scanner_rh = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RH_FOOT",
        offset=MatterportRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),  # 0.5m to allow for doors
        ray_alignment="yaw",
        pattern_cfg=nav_patterns.FootScanPatternCfg(),
        debug_vis=False,
        mesh_prim_paths=[
            "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
        ],
        max_distance=10.0,
    )

    return cfg


def main():
    # setup runner
    cfg = PlannerHeightCfg()
    # change scene (matterport, warehouse and carla)
    if args_cli.scene == "matterport":
        cfg.env_cfg.scene = MatterportHeightTerrainSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
        # change to a fix goal command with set start point
        cfg.env_cfg.commands.command = mdp.FixGoalCommandCfg(
            fix_goal_position=[8.0, -13.5, 1.0],
            resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
        )
        cfg.env_cfg.scene.robot.init_state.pos = (8.0, 0.0, 0.6)
        cfg.env_cfg.scene.robot.init_state.rot = (0.6126, 0.0327, 0.0136, -0.7896)
        # adapt viewer
        cfg.env_cfg.viewer.eye = (8.5, 3.0, 2.5)
        cfg.env_cfg.viewer.lookat = (8.5, -4.0, 0.0)
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not implemented.")
    # robot changes
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    if args_cli.scene == "matterport":
        cfg.env_cfg.scene = exchange_to_matterport_sensors(cfg.env_cfg.scene)
        cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.func = mdp.height_scan_square

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # get planner cfg
    sampling_planner_cfg_dict = get_planner_cfg(args_cli.num_envs, traj_dim=10, debug=False, device="cuda")

    # build planner
    planner = FDMPlanner(cfg, sampling_planner_cfg_dict, args_cli=args_cli)
    # post modify runner and env
    planner = env_modifier_post_init(planner, args_cli=args_cli)

    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=cfg.env_cfg.commands.command.fix_goal_position,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )

    # navigate
    planner.test()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
