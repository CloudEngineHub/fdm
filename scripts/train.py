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
    default="debug",
    choices=["train", "eval", "debug", "train-real-world"],
    help="Mode of the script.",
)
parser.add_argument("--run_name", type=str, default="local_4mLiDAR-2DEnv", help="Name of the run.")
parser.add_argument("--resume", type=str, default=None, help="Resume from an experiment (provide the experiment name).")
parser.add_argument("--regular", action="store_true", default=False, help="Spawn robots in a regular pattern.")
# parser.add_argument("--runs", type=str, nargs="+", default="Oct25_13-04-59_MergeSingleObjTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_ModPreTrained_Bs2048_DropOut_NoEarlyCollFilter", help="Name of the run.")
parser.add_argument(
    "--runs",
    type=str,
    nargs="+",
    default="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5",
    help="Name of the run.",
)
parser.add_argument(
    "--real-world-dilution",
    type=int,
    default=10,
    help="Dilution factor for the real-world datasets, i.e. how many simulation batches per real-world batches.",
)

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=4096)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

if args_cli.mode == "debug":
    args_cli.headless = False
    args_cli.num_envs = 2
    args_cli.terrain_analysis_points = 5000
elif args_cli.mode == "eval":
    args_cli.headless = True
    args_cli.num_envs = 2048
    args_cli.terrain_analysis_points = 5000

# args_cli.resume = "Jan29_17-25-19_local_test"
# args_cli.resume = "Jan29_14-40-33"
# args_cli.env = "baseline"
# args_cli.num_envs = 4096

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch

import omni

from fdm.runner import FDMRunner
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, robot_changes, runner_cfg_init
from fdm.utils.model_comp_plot import ViolinPlotter, meta_summarize, plot_metrics_with_grid

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def load_cfg():
    # init runner cfg
    cfg = runner_cfg_init(args_cli)
    # select robot
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)
    return cfg


def run_eval(runner: FDMRunner):
    # eval environments
    meta_eval, test_meta = runner.eval_metric()

    # create violin plot
    vilion_plotter = ViolinPlotter(
        datasets=(
            ["train"] if runner.trainer.dataloader is None else ["train"] + list(runner.trainer.test_datasets.keys())
        ),
        env_sensor_cfg=runner.cfg.env_cfg.scene.env_sensor,
    )
    vilion_plotter.update_data(runner.model, runner.trainer.dataloader, "train")
    if runner.trainer.test_datasets is not None:
        for test_dataset_name, test_dataset in runner.trainer.test_datasets.items():
            vilion_plotter.update_data(runner.model, test_dataset, test_dataset_name)

    if args_cli.mode == "eval":
        fdm_model_log_dir = runner.trainer.resume_path
    else:
        fdm_model_log_dir = runner.trainer.log_dir

    # close prev environment
    runner.close()
    # create a new stage
    omni.usd.get_context().new_stage()
    # del runner
    del runner

    # include baseline in the tests --> load baseline method and rerun the evaluation
    args_cli.env = "baseline"
    args_cli.runs = None
    cfg = load_cfg()
    # limit number of samples
    cfg.trainer_cfg.num_samples = 50000
    # increase the resolution of the
    cfg.model_cfg.eval_distance_interval = 0.1
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)
    baseline_meta_eval, baseline_test_meta = runner.eval_metric()
    baseline_meta_eval_new = {}
    for key, value in baseline_meta_eval.items():
        new_key = key.replace("plot", "plot baseline", 1)
        baseline_meta_eval_new[new_key] = value
    for key, value in baseline_test_meta.items():
        new_key = key.replace("_baseline", " baseline", 1)
        baseline_meta_eval_new[new_key] = value

    # get path to save the plots
    dir_path, _ = os.path.split(fdm_model_log_dir)
    os.makedirs(os.path.join(dir_path, "plots"), exist_ok=True)

    # create violin plot
    vilion_plotter.update_data(runner.model, runner.trainer.dataloader, "train", baseline=True)
    if runner.trainer.test_datasets is not None:
        for test_dataset_name, test_dataset in runner.trainer.test_datasets.items():
            vilion_plotter.update_data(
                runner.model, test_dataset, test_dataset_name.removesuffix("_baseline"), baseline=True
            )
    # create the vilion plot
    vilion_plotter.plot_data(os.path.join(dir_path, "plots"))

    # Save the metrics in a grid plot
    # -- plot without baseline
    distance_meta_summary, step_meta_summary = meta_summarize(meta_eval | test_meta)
    plot_metrics_with_grid(distance_meta_summary, os.path.join(dir_path, "plots"))
    plot_metrics_with_grid(step_meta_summary, os.path.join(dir_path, "plots"), step=True)
    plot_metrics_with_grid(
        distance_meta_summary, os.path.join(dir_path, "plots"), only_first_row=True, suffix="_first_row"
    )
    plot_metrics_with_grid(
        step_meta_summary, os.path.join(dir_path, "plots"), step=True, only_first_row=True, suffix="_first_row"
    )

    # -- plot with baseline
    distance_meta_summary, step_meta_summary = meta_summarize(meta_eval | test_meta | baseline_meta_eval_new)
    plot_metrics_with_grid(distance_meta_summary, os.path.join(dir_path, "plots"), suffix="_baseline")
    plot_metrics_with_grid(step_meta_summary, os.path.join(dir_path, "plots"), step=True, suffix="_baseline")
    plot_metrics_with_grid(
        distance_meta_summary, os.path.join(dir_path, "plots"), suffix="_baseline_first_row", only_first_row=True
    )
    plot_metrics_with_grid(
        step_meta_summary, os.path.join(dir_path, "plots"), step=True, suffix="_baseline_first_row", only_first_row=True
    )

    print("Done evaluated the model")


def main():
    # load cfg
    cfg = load_cfg()
    if args_cli.mode == "eval":
        # increase the resolution of the
        cfg.model_cfg.eval_distance_interval = 0.1
        # limit number of samples
        cfg.trainer_cfg.num_samples = 50000
        # set name of the run
        if args_cli.runs is not None:
            cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs
    elif args_cli.mode == "debug":
        # overwrite some configs for easier debugging
        cfg.replay_buffer_cfg.trajectory_length = 200
        cfg.trainer_cfg.num_samples = 1000
        cfg.trainer_cfg.logging = False
        cfg.trainer_cfg.test_datasets = None
        if cfg.env_cfg.curriculum is not None:
            cfg.env_cfg.curriculum.command_ratios.func.cfg.update_interval = 200
    elif args_cli.mode == "train-real-world":
        # check that reduced obs are selected when real-world datasets are given for training
        assert args_cli.reduced_obs, "Real world datasets require reduced observations."
        # reduce number of epochs for the training with the datasets
        cfg.trainer_cfg.epochs = 3
        # reduce learning rate for the fine tuning
        cfg.trainer_cfg.learning_rate = 1e-4
        # reduce number of collection steps
        cfg.collection_rounds = 10
        # set name of the run
        if args_cli.runs is not None:
            cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs
        # add train datasets
        name_suffix = "_noTorque" if args_cli.remove_torque else ""
        name_suffix += "_nearest_neighbor_filling" if args_cli.noise else ""

        cfg.trainer_cfg.real_world_train_datasets = [
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-09-23-10-52-57_urban/real_world_dataset_reducedObs{name_suffix}_train.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-03-07-52-45_moenchsjoch_fenced_1/real_world_dataset_reducedObs{name_suffix}_train.pkl",
            # "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-07-57-34_moenchsjoch_fenced_2/real_world_dataset_reducedObs{name_suffix}_train.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-03-08-17-23_moenchsjoch_outside_1/real_world_dataset_reducedObs{name_suffix}_train.pkl",
            # "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-08-42-30_moenchsjoch_outside_2/real_world_dataset_reducedObs{name_suffix}_train.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-14-14-36-02_forest_kaeferberg_entanglement/real_world_dataset_reducedObs{name_suffix}_train.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-15-12-06-03_forest_albisguetli_slippery_slope/real_world_dataset_reducedObs{name_suffix}_train.pkl",
        ]
        cfg.trainer_cfg.real_world_test_datasets = [
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-09-23-10-52-57_urban/real_world_dataset_reducedObs{name_suffix}_val.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-03-07-52-45_moenchsjoch_fenced_1/real_world_dataset_reducedObs{name_suffix}_val.pkl",
            # "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-07-57-34_moenchsjoch_fenced_2/real_world_dataset_reducedObs{name_suffix}_val.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-03-08-17-23_moenchsjoch_outside_1/real_world_dataset_reducedObs{name_suffix}_val.pkl",
            # "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-08-42-30_moenchsjoch_outside_2/real_world_dataset_reducedObs{name_suffix}_val.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-14-14-36-02_forest_kaeferberg_entanglement/real_world_dataset_reducedObs{name_suffix}_val.pkl",
            "{LOG_DIR}"
            + f"/real_world_datasets/2024-11-15-12-06-03_forest_albisguetli_slippery_slope/real_world_dataset_reducedObs{name_suffix}_val.pkl",
        ]

        # reduce the size of the replay buffer to replect the smaller size of samples needed
        cfg.replay_buffer_cfg.trajectory_length = int(
            3000 * args_cli.real_world_dilution * cfg.model_cfg.prediction_horizon / args_cli.num_envs
        )
    elif args_cli.mode == "train" and args_cli.env == "baseline":
        # FIXME: remove later
        cfg.replay_buffer_cfg.trajectory_length = 200

    # init runner
    runner = FDMRunner(cfg=cfg, args_cli=args_cli, eval=args_cli.mode == "eval")
    # post modify runner and env
    runner = env_modifier_post_init(runner, args_cli=args_cli)

    if args_cli.mode == "train":
        # run
        runner.train()
        # run eval
        runner.model.cfg.eval_distance_interval = 0.1
        runner.trainer.cfg.num_samples = 50000
        run_eval(runner)
    elif args_cli.mode == "eval":
        # run
        run_eval(runner)
    elif args_cli.mode == "debug":
        # run
        runner.train()
    elif args_cli.mode == "train-real-world":
        # change number of samples to match the desired dilution factor
        assert runner.trainer.real_world_train_datasets is not None, "No real world datasets given."
        runner.trainer.cfg.num_samples = (
            sum(runner.trainer.real_world_train_datasets.dataset.cumulative_sizes) * args_cli.real_world_dilution
        )

        # run
        runner.train()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
