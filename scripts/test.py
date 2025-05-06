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
parser.add_argument("--terrain-cfg", type=str, default=None, help="Name of the terrain config to load.")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
parser.add_argument(
    "--runs",
    type=str,
    nargs="+",
    default="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5",
    help="Name of the run.",
)
# parser.add_argument("--runs", type=str, nargs="+", default="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque", help="Name of the run.")
# parser.add_argument("--runs", type=str, nargs="+", default="Dec03_20-25-59_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoTorque", help="Name of the run.")
parser.add_argument("--equal-actions", action="store_true", default=False, help="Have the same actions for all envs.")
parser.add_argument("--paper-figure", action="store_true", default=True, help="Run paper figure test.")
parser.add_argument(
    "--paper-platform-figure", action="store_true", default=False, help="Run paper platform figure test."
)
parser.add_argument("--terrain_analysis_points", type=int, default=2000, help="Number of points for terrain analysis.")
parser.add_argument("--record", action="store_true", default=False, help="Record the simulation.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=360)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# adapt number of environments for the paper figure
if args_cli.paper_figure:
    args_cli.num_envs = 16
    args_cli.enable_cameras = True
    args_cli.terrain_analysis_points = 500
elif args_cli.paper_platform_figure:
    args_cli.num_envs = 4
    args_cli.enable_cameras = True
    args_cli.terrain_analysis_points = 500

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import torch.utils.benchmark as benchmark

from isaacsim.sensors.camera import Camera

import isaaclab.sim.spawners as sim_spawners
from isaaclab.assets import AssetBaseCfg
from isaaclab_tasks.utils import get_checkpoint_path

from nav_suite.collectors import TrajectorySamplingCfg

import fdm.env_cfg.terrain_cfg as terrain_cfg
import fdm.mdp as mdp
from fdm.agents import PaperFigureAgentCfg
from fdm.env_cfg.env_cfg_base import TERRAIN_ANALYSIS_CFG
from fdm.env_cfg.env_cfg_base_mixed import PlannerObservationsCfg
from fdm.model.utils import TorchPolicyExporter
from fdm.runner import FDMRunner, FDMRunnerCfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, robot_changes, runner_cfg_init

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def export_to_jit(runner: FDMRunner):
    with torch.inference_mode():
        # export cuda model
        torch_policy_exporter = TorchPolicyExporter(runner, device="cuda")
        resume_path = get_checkpoint_path(
            runner.trainer.log_root_path, runner.trainer.cfg.load_run, runner.trainer.cfg.load_checkpoint
        )
        dir_path, _ = os.path.split(resume_path)
        torch_policy_exporter.export(os.path.join(dir_path, "export"), "model_cuda_jit.pth")
        # export cpu model
        torch_policy_exporter = TorchPolicyExporter(runner, device="cpu")
        torch_policy_exporter.export(os.path.join(dir_path, "export"), "model_cpu_jit.pth")
        # check that exported model generates the same output as original model
        jit_model_cuda = torch.jit.load(os.path.join(dir_path, "export", "model_cuda_jit.pth")).to(runner.model.device)
        jit_model_cuda.eval()
        jit_model_cpu = torch.jit.load(os.path.join(dir_path, "export", "model_cpu_jit.pth")).to("cpu")
        jit_model_cpu.eval()
        example_input = [torch.rand(curr_size) for curr_size in torch_policy_exporter.example_input]
        output_jit_gpu = jit_model_cuda(example_input)
        output_orig = runner.model(example_input)
        assert torch.allclose(output_jit_gpu[0], output_orig[0])
        assert torch.allclose(output_jit_gpu[1], output_orig[1])
        assert torch.allclose(output_jit_gpu[2], output_orig[2])
        output_jit_cpu = jit_model_cpu(example_input)
        assert torch.allclose(output_jit_cpu[0], output_orig[0].to("cpu"), atol=1e-3)
        assert torch.allclose(output_jit_cpu[1], output_orig[1].to("cpu"), atol=1e-3)
        assert torch.allclose(output_jit_cpu[2], output_orig[2].to("cpu"), atol=1e-3)
        # benchmark jit compiled vs original model
        timer_model = benchmark.Timer(
            stmt="jit_model_cuda(example_input)",
            globals={"jit_model_cuda": jit_model_cuda, "example_input": example_input},
            num_threads=1,
            label="jit_model_cuda",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (jit, cuda)\t :", time_value / 1e-6, "us")

        timer_model = benchmark.Timer(
            stmt="model(example_input)",
            globals={"model": runner.model, "example_input": example_input},
            num_threads=1,
            label="runner.model",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (torch, cuda)\t :", time_value / 1e-6, "us")

        timer_model = benchmark.Timer(
            stmt="jit_model_cpu(example_input)",
            globals={"jit_model_cpu": jit_model_cpu, "example_input": example_input},
            num_threads=1,
            label="jit_model_cpu",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (jit, cpu)\t :", time_value / 1e-6, "us")

        runner.model.device = "cpu"
        runner.model.proprioceptive_normalizer.to("cpu")
        timer_model = benchmark.Timer(
            stmt="model(example_input)",
            globals={"model": runner.model.to("cpu"), "example_input": example_input},
            num_threads=1,
            label="runner.model",
        )
        time_value = timer_model.blocked_autorange().median
        print("\tTime for model (torch, cpu)\t :", time_value / 1e-6, "us")

    # make sure model is again on the gpu
    runner.model.device = "cuda"
    runner.model.proprioceptive_normalizer.to("cuda")
    runner.model.to(runner.model.device)


def main():
    # init runner cfg
    cfg = runner_cfg_init(args_cli)
    # select robot
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # overwrite some configs for easier debugging
    cfg.replay_buffer_cfg.trajectory_length = (
        50 if not args_cli.paper_figure and not args_cli.paper_platform_figure else 15
    )
    cfg.trainer_cfg.num_samples = 2000
    cfg.trainer_cfg.logging = False

    # swap environment
    cfg.env_cfg.scene.terrain.terrain_type = "generator"
    if args_cli.paper_figure:
        cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.PAPER_FIGURE_TERRAIN_CFG
    elif args_cli.paper_platform_figure:
        cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.PAPER_PLATFORM_FIGURE_TERRAIN_CFG
    else:
        cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.PILLAR_TERRAIN_EVAL_CFG

    # FIXME: debug for baseline
    if args_cli.env == "baseline":
        cfg.env_cfg.scene.terrain.terrain_generator = terrain_cfg.FDM_TERRAINS_CFG
        cfg.env_cfg.scene.terrain.terrain_generator.num_cols = 6
        cfg.env_cfg.scene.terrain.terrain_generator.num_rows = 12
    # make origin selection deterministic
    cfg.env_cfg.scene.terrain.random_seed = 0
    # make environment origins regular
    cfg.env_cfg.scene.terrain.regular_spawning = True

    # set name of the run
    if args_cli.runs is not None:
        cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

    # set regular spawning pattern
    if args_cli.paper_figure or args_cli.paper_platform_figure:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_paper_plot
    else:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_center
    pop_items = [item for item in cfg.env_cfg.events.reset_base.params.keys() if item != "asset_cfg"]
    for item in pop_items:
        cfg.env_cfg.events.reset_base.params.pop(item)
    # restrict agent to be purely random, temporal-correlated actions with adjusted horizon
    if args_cli.paper_figure or args_cli.paper_platform_figure:
        cfg.agent_cfg = PaperFigureAgentCfg(
            horizon=cfg.model_cfg.prediction_horizon + 1, platform_figure=args_cli.paper_platform_figure
        )
    else:
        cfg.agent_cfg = FDMRunnerCfg().agent_cfg
        cfg.agent_cfg.horizon = cfg.model_cfg.prediction_horizon + 1
    # add planner observations
    cfg.env_cfg.observations.planner_obs = PlannerObservationsCfg.PlannerObsCfg()
    # restrict goal generator to be purely goal-generated without any planner
    cfg.env_cfg.commands.command = mdp.GoalCommandCfg(
        resampling_time_range=(1000000.0, 1000000.0),  # only resample once at the beginning
        infite_sampling=False,
        debug_vis=False,
        traj_sampling=TrajectorySamplingCfg(
            terrain_analysis=TERRAIN_ANALYSIS_CFG,
        ),
    )
    cfg.env_cfg.observations.planner_obs.goal.func = mdp.goal_command_w_se2
    cfg.env_cfg.observations.planner_obs.goal.params = {"command_name": "command"}

    # remove reset when in collision
    cfg.env_cfg.terminations.base_contact = None

    # add distance light to the scene
    if args_cli.paper_figure:
        cfg.env_cfg.scene.light_1 = AssetBaseCfg(
            prim_path="/World/light_1",
            spawn=sim_spawners.SphereLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=200,  # 500.0,
                exposure=8.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, -6.0, 10.0),
            ),
        )
        cfg.env_cfg.scene.light_2 = AssetBaseCfg(
            prim_path="/World/light_2",
            spawn=sim_spawners.SphereLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=200,  # 500.0,
                exposure=8.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, -18.0, 10.0),
            ),
        )
        cfg.env_cfg.scene.light_3 = AssetBaseCfg(
            prim_path="/World/light_3",
            spawn=sim_spawners.SphereLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=200,  # 500.0,
                exposure=8.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 6.0, 10.0),
            ),
        )
        cfg.env_cfg.scene.light_4 = AssetBaseCfg(
            prim_path="/World/light_4",
            spawn=sim_spawners.SphereLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=200,  # 500.0,
                exposure=8.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 18.0, 10.0),
            ),
        )
        cfg.env_cfg.scene.light.spawn.intensity = 2000.0
    elif args_cli.paper_platform_figure:
        cfg.env_cfg.scene.light_1 = AssetBaseCfg(
            prim_path="/World/light_1",
            spawn=sim_spawners.SphereLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=30,  # 500.0,
                exposure=8.0,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 10.0),
            ),
        )
        cfg.env_cfg.scene.light.spawn.intensity = 2000.0

    # setup runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)

    # add a camera to the scene for the paper figure
    if args_cli.paper_figure or args_cli.paper_platform_figure:
        import omni
        from PIL import Image

        if args_cli.paper_figure:
            # camera = Camera(
            #     prim_path="/World/floating_camera",
            #     resolution=(8640, 2160) if args_cli.paper_figure else (3600, 2430)  # (4320, 2430),
            # )
            # camera.set_world_pose(position=[0.0, 0.0, 120], orientation=[0.707, 0.0, 0.707, 0.0])
            # camera.initialize()

            # camera per terrain origin
            cameras = []
            for idx in range(runner.env.scene.terrain.terrain_origins.shape[1]):
                cameras.append(
                    Camera(
                        prim_path=f"/World/floating_camera_{idx}",
                        resolution=(3600, 2430),
                    )
                )
                cameras[-1].set_world_pose(
                    position=(
                        runner.env.scene.terrain.terrain_origins[0, idx]
                        + torch.tensor([25, 0.0, 28], device=runner.env.device)
                    ).tolist(),
                    orientation=[0.4146932, 0.0, 0.9099613, 0.0],
                )
                cameras[-1].initialize()
        else:
            camera = Camera(
                prim_path="/World/floating_camera",
                resolution=(8640, 2160) if args_cli.paper_figure else (3600, 2430),  # (4320, 2430),
            )
            camera.set_world_pose(position=[-32, 0.0, 32], orientation=[0.9250441, 0.0, 0.3798598, 0.0])
            camera.initialize()
            camera_rob_1 = Camera(prim_path="/World/floating_camera_robot_1", resolution=(3600, 2430))
            camera_rob_1.set_world_pose(position=[-5, 0.0, 4], orientation=[0.9250441, 0.0, 0.3798598, 0.0])
            camera_rob_1.initialize()
            camera_rob_2 = Camera(
                prim_path="/World/floating_camera_robot_2",
                resolution=(3600, 2430),
            )
            camera_rob_2.set_world_pose(position=[-5, 0.0, 4], orientation=[0.9250441, 0.0, 0.3798598, 0.0])
            camera_rob_2.initialize()

    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)

    # exportmodel to jit
    if args_cli.env != "baseline" and not args_cli.paper_figure and not args_cli.paper_platform_figure:
        export_to_jit(runner)

    # run test script
    if args_cli.paper_figure and args_cli.record:
        runner.test(cameras=cameras)
    else:
        runner.test()

    if args_cli.paper_figure or args_cli.paper_platform_figure:

        if args_cli.paper_figure:
            for idx, camera in enumerate(cameras):
                camera.get_current_frame()
                img = Image.fromarray(camera.get_rgba()[:, :, :3])
                img.save(f"fdm_perceptive_demo_sim_with_traj_white_env{idx}.png")
        elif args_cli.paper_platform_figure:
            camera.get_current_frame()
            img = Image.fromarray(camera.get_rgba()[:, :, :3])
            img.save(f"fdm_perceptive_demo_sim_with_traj_white_{args_cli.robot}.png")

            # get the robot position
            robot_pos = runner.env.scene["robot"].data.root_pos_w.clone()
            if args_cli.robot == "anymal_perceptive" or args_cli.robot == "tytan":
                robot_pos_1 = (robot_pos[1] + torch.tensor([-11.0, -1.0, 10], device=runner.env.device)).tolist()
                robot_pos_2 = (robot_pos[2] + torch.tensor([-10.0, 0.0, 9], device=runner.env.device)).tolist()
            elif args_cli.robot == "aow":
                robot_pos_1 = (robot_pos[1] + torch.tensor([-13.5, -0.5, 11], device=runner.env.device)).tolist()
                robot_pos_2 = (robot_pos[2] + torch.tensor([-13.0, -0.5, 11], device=runner.env.device)).tolist()
            elif args_cli.robot == "tytan_quiet":
                robot_pos_1 = (robot_pos[1] + torch.tensor([-11.0, -1.0, 10], device=runner.env.device)).tolist()
                robot_pos_2 = (robot_pos[2] + torch.tensor([-9.0, 1.0, 9], device=runner.env.device)).tolist()
            camera_rob_1.set_world_pose(position=robot_pos_1, orientation=[0.9250441, 0.0, 0.3798598, 0.0])
            camera_rob_2.set_world_pose(position=robot_pos_2, orientation=[0.9250441, 0.0, 0.3798598, 0.0])

            # update camera image
            _app = omni.kit.app.get_app_interface()
            _app.update()

            # save the images
            camera_rob_1.get_current_frame()
            img = Image.fromarray(camera_rob_1.get_rgba()[:, :, :3])
            img.save(f"fdm_perceptive_demo_sim_with_traj_white_{args_cli.robot}_rob1.png")
            camera_rob_2.get_current_frame()
            img = Image.fromarray(camera_rob_2.get_rgba()[:, :, :3])
            img.save(f"fdm_perceptive_demo_sim_with_traj_white_{args_cli.robot}_rob2.png")

        return

    print("Simulation will keep running until the user closes it.")
    while simulation_app.is_running():
        runner.env.render()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
