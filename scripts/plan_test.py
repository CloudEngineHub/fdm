# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to test the planning of the FDM.

The script offers following modes:
   - `--mode test`: **qualitative** evaluation in a test environment
   - `--mode metric --env_type 2D`: **qualitative** evaluation in a 2D environment
   - `--mode metric --env_type 3D`: **qualitative** evaluation in a 3D environment
   - `--mode plot`: generate the Fig. **TODO** of the paper
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test Script for MPPI Planning with the FDM model.")
parser.add_argument(
    "--run",
    type=str,
    default="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque",
    help="Name of the run.",
)
parser.add_argument("--terrain_analysis_points", type=int, default=10000, help="Number of points for terrain analysis.")
parser.add_argument(
    "--mode",
    type=str,
    default="plot_video",
    choices=["metric", "test", "plot", "plot_video"],
    help="Mode of the script.",
)
parser.add_argument("--env_type", type=str, default="2D", choices=["2D", "3D"], help="Specific environment to pick.")
parser.add_argument(
    "--cost_show",
    type=str,
    default="None",
    choices=["None", "Cost", "Goal_Distance", "Collision", "Height_Scan_Cost", "Pose_Reward"],
    help="Cost visualization mode.",
)

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=24)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# script modes
if args_cli.mode == "test":
    args_cli.num_envs = 24
elif args_cli.mode == "metric":
    args_cli.num_envs = 12 * 8
    args_cli.headless = True
elif args_cli.mode == "plot" or args_cli.mode == "plot_video":
    args_cli.num_envs = 5

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import yaml

import fdm.env_cfg.terrain_cfg as fdm_terrain_cfg
import fdm.mdp as mdp
from fdm.env_cfg import TERRAIN_ANALYSIS_CFG

# activate planner mode
from fdm.planner import FDMPlanner, get_planner_cfg
from fdm.utils.args_cli_utils import (
    ablation_studies_modifications,
    cfg_modifier_pre_init,
    env_modifier_post_init,
    planner_cfg_init,
    robot_changes,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def add_env_cameras(planner: FDMPlanner, mode: str):
    from isaacsim.sensors.camera import Camera

    # add camera for each environment
    cameras = []
    for i in range(planner.env.num_envs) if mode == "plot" else range(0, planner.env.num_envs - 1, 2):
        if mode == "plot":
            camera = Camera(prim_path=f"/World/floating_camera_{i}", resolution=(3600, 2430))
            camera_pos = planner.env.scene.env_origins[i] + torch.tensor([-15, 0.0, 15], device=planner.env.device)
        else:
            camera = Camera(prim_path=f"/World/floating_camera_{i}", resolution=(7200, 2430))
            camera_pos = (planner.env.scene.env_origins[i] + planner.env.scene.env_origins[i + 1]) / 2 + torch.tensor(
                [-25, 0.0, 23], device=planner.env.device
            )
        camera.set_world_pose(position=camera_pos.tolist(), orientation=[0.9396926, 0.0, 0.3420201, 0.0])
        camera.initialize()
        cameras.append(camera)

    return cameras


def main():
    # reduce required number of samples for the terrain analysis
    if args_cli.mode in ["test", "plot", "plot_video"]:
        args_cli.terrain_analysis_points = 2000

    # setup runner
    cfg = planner_cfg_init(args_cli)
    # robot changes
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)
    # ablation studies
    cfg = ablation_studies_modifications(cfg, args_cli)

    # swap environment
    cfg.env_cfg.scene.terrain.terrain_type = "generator"
    if args_cli.mode == "test":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PLANNER_EVAL_CFG
    elif args_cli.mode == "metric" and args_cli.env_type == "2D":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PLANNER_EVAL_2D_CFG
        if args_cli.env == "baseline":
            # NOTE: for comparability, increase the obstacle height so that detactable by the 2D lidar
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["single_box"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["single_cylinder"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["single_wall"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["box_cross_pattern"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["cylinder_cross_pattern"].height_range[0] = 1.0
            cfg.env_cfg.scene.terrain.terrain_generator.sub_terrains["wall_cross_pattern"].height_range[0] = 1.0
    elif args_cli.mode == "metric" and args_cli.env_type == "3D":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PLANNER_EVAL_3D_CFG
    elif args_cli.mode == "plot" or args_cli.mode == "plot_video":
        cfg.env_cfg.scene.terrain.terrain_generator = fdm_terrain_cfg.PAPER_PLANNER_FIGURE_TERRAIN_CFG
        # change the initial spawning and resetting function
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_planner_paper_plot
        cfg.env_cfg.events.reset_base.params = {}
    else:
        raise ValueError(f"Invalid mode {args_cli.mode} and env_type {args_cli.env_type}")

    # make origin selection deterministic
    cfg.env_cfg.scene.terrain.random_seed = 0

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # modify the reset function for the robot base state
    if args_cli.mode != "plot" and args_cli.mode != "plot_video":
        # remove the randomization of yaw in the reset_base event
        cfg.env_cfg.events.reset_base.params["yaw_range"] = (0.0, 0.0)
        # enable that it is spawned relative to the env origin
        cfg.env_cfg.events.reset_base.params["spawn_in_env_frame"] = False
        # remove the velocity randomization in the reset_base event
        cfg.env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    # set a fix goal point
    cfg.env_cfg.commands.command = mdp.FixGoalCommandCfg(
        resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
        debug_vis=True,
        fix_goal_position=[4.0, 0.0, 0.5],
        relative_terrain_origin="origin",
        project_onto_terrain=True,
        terrain_analysis=TERRAIN_ANALYSIS_CFG,
        vis_line=False,
    )
    # add goal randomizations for the metric case and set number of samples
    if args_cli.mode == "metric":
        cfg.env_cfg.commands.command.goal_rand_x = (-0.4, 0.4)
        cfg.env_cfg.commands.command.goal_rand_y = (-0.1, 0.1)
        cfg.env_cfg.commands.command.trajectory_num_samples = 500

    # get planner cfg
    sampling_planner_cfg_dict = get_planner_cfg(
        args_cli.num_envs, traj_dim=10, debug=False, device="cuda", population_size=512
    )

    if args_cli.env == "heuristic":
        sampling_planner_cfg_dict["to_cfg"]["control"] = "velocity_control"
        sampling_planner_cfg_dict["to_cfg"]["states_cost_w_cost_map"] = True
        sampling_planner_cfg_dict["to_cfg"]["state_cost_w_fatal_trav"] = sampling_planner_cfg_dict["to_cfg"][
            "collision_cost_high_risk_factor"
        ]
        # Elevate height scan to make sure all obstacles are captured
        pos_offset = list(cfg.env_cfg.scene.env_sensor.offset.pos)
        pos_offset[2] = 2.0
        cfg.env_cfg.scene.env_sensor.offset.pos = tuple(pos_offset)
    elif args_cli.env == "baseline":
        sampling_planner_cfg_dict["to_cfg"]["control"] = "fdm_baseline"
        sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 3
        sampling_planner_cfg_dict["optim"]["population_size"] = 256
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_safety_factor"] = -0.1
        sampling_planner_cfg_dict["to_cfg"]["collision_cost_high_risk_factor"] = 10 if args_cli.env_type == "2D" else 20
        cfg.env_cfg.episode_length_s = 120.0
        cfg.max_path_time = 120.0
    elif args_cli.env == "height":
        # Elevate height scan to make sure all obstacles are captured
        pos_offset = list(cfg.env_cfg.scene.env_sensor.offset.pos)
        pos_offset[2] = 2.0
        cfg.env_cfg.scene.env_sensor.offset.pos = tuple(pos_offset)
        sampling_planner_cfg_dict["to_cfg"]["num_neighbors"] = 4

    # build planner
    planner = FDMPlanner(cfg, sampling_planner_cfg_dict, args_cli=args_cli)
    # post modify runner and env
    planner = env_modifier_post_init(planner, args_cli=args_cli)

    # set the defined cost visualization
    if args_cli.cost_show != "None":
        planner.env._window.current_cost_viz_mode = args_cli.cost_show.replace("_", " ")

    if args_cli.mode == "plot" or args_cli.mode == "plot_video":
        cameras = add_env_cameras(planner, args_cli.mode)
        # navigate
        planner.test(cameras)
        # planner.test()
    elif args_cli.mode == "test":
        # navigate
        planner.test()
    else:
        # navigate
        metrics = planner.navigate()

        # save the predictions
        save_dir = os.path.abspath(os.path.join("logs", "fdm", "planner_eval"))
        os.makedirs(save_dir, exist_ok=True)
        with open(save_dir + f"/planner_eval_metric_method_{args_cli.env}_env_{args_cli.env_type}.yaml", "w") as f:
            yaml.dump(metrics, f)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
