# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate a Forward-Dynamics-Model on Real-World datasets."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--runs",
    type=str,
    nargs="+",
    default=[
        # "Jan09_11-06-38_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5_RealWorld_Dilution5",
        # "Jan09_11-07-09_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5_RealWorld_Dilution10",
        "Jan09_23-16-19_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5_RealWorld_Dilution5_WithForest",
        # "Jan09_23-17-12_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5_RealWorld_Dilution10_WithForest",
        "Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5",
    ],
    help="Name of the run.",
)
parser.add_argument(
    "--names", type=str, nargs="+", default=["Ours (Fine Tuned)", "Ours (Pure Sim)"], help="Name of the run."
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/home/pascal/orbit/IsaacLab/logs/fdm/fdm_fine_tuning_results",
    help="Path where to save the results",
)
parser.add_argument(
    "--collision_split",
    action="store_true",
    default=False,
    help="Include split in collision and non-collision samples in the polt.",
)

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=16)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# require reduced obs
args_cli.reduced_obs = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import pickle
import re
import torch
from torch.utils.data import DataLoader

from isaaclab_tasks.utils import get_checkpoint_path

from fdm import LARGE_UNIFIED_HEIGHT_SCAN
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, robot_changes, runner_cfg_init
from fdm.utils.model_comp_plot import ViolinPlotter

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
    # add train datasets
    # NOTE: {LOG_DIR} is replaced with the log directory
    name_suffix = "_noTorque" if args_cli.remove_torque else ""
    name_suffix += "_nearest_neighbor_filling" if args_cli.noise else ""
    # cfg.trainer_cfg.real_world_test_datasets = [
    #     "{LOG_DIR}" + f"/real_world_datasets/2024-09-23-10-52-57_urban/real_world_dataset_reducedObs{name_suffix}_val.pkl",
    #     # "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-07-52-45_moenchsjoch_fenced_1/real_world_dataset_reducedObs{name_suffix}_val.pkl",
    #     "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-07-57-34_moenchsjoch_fenced_2/real_world_dataset_reducedObs{name_suffix}_train.pkl",  # not used during training
    #     # "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-08-17-23_moenchsjoch_outside_1/real_world_dataset_reducedObs{name_suffix}_val.pkl",
    #     "{LOG_DIR}" + f"/real_world_datasets/2024-11-03-08-42-30_moenchsjoch_outside_2/real_world_dataset_reducedObs{name_suffix}_train.pkl",  # not used during training
    #     "{LOG_DIR}" + f"/real_world_datasets/2024-11-14-14-36-02_forest_kaeferberg_entanglement/real_world_dataset_reducedObs{name_suffix}_val.pkl",
    #     "{LOG_DIR}" + f"/real_world_datasets/2024-11-15-12-06-03_forest_albisguetli_slippery_slope/real_world_dataset_reducedObs{name_suffix}_val.pkl",
    # ]
    cfg.trainer_cfg.real_world_test_datasets = [
        "{LOG_DIR}"
        + f"/real_world_datasets/2024-09-23-10-52-57_urban/real_world_dataset_reducedObs{name_suffix}_val.pkl",
        "{LOG_DIR}" + f"/real_world_datasets/forest/test_forest_dataset_reducedObs{name_suffix}.pkl",
        "{LOG_DIR}" + f"/real_world_datasets/snow/test_snow_dataset_reducedObs{name_suffix}.pkl",
    ]
    # check that reduced obs are selected when real-world datasets are given for training
    if not args_cli.reduced_obs:
        raise ValueError("Real world datasets require reduced observations.")

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # open all datasets
    datasets = {}
    log_dir = os.path.abspath(os.path.join("logs", "fdm"))
    # -- load the current train dataset (if a sampled dataset exists)
    if False:  # cfg.env_cfg.scene.terrain.terrain_type == "usd":
        # get dataset path
        terrain_name = os.path.splitext(os.path.split(cfg.env_cfg.scene.terrain.usd_path)[1])[0]
        # follow naming convention of test datasets (used in args_cli_utils.py)
        suffix = ""
        if args_cli.env != "baseline":
            if hasattr(args_cli, "reduced_obs") and args_cli.reduced_obs:
                suffix += "_reducedObs"
            if hasattr(args_cli, "remove_torque") and args_cli.remove_torque:
                suffix += "_noTorque"
            if hasattr(args_cli, "noise") and args_cli.noise:
                suffix += "_noise"
            elif hasattr(args_cli, "occlusions") and args_cli.occlusions:
                suffix += "_occlusions"
        else:
            if hasattr(args_cli, "noise") and args_cli.noise:
                suffix = "_noise_baseline"
            else:
                suffix = "_baseline"
        if LARGE_UNIFIED_HEIGHT_SCAN:
            suffix += "_largeHeightScan"
        eval_dataset_path = os.path.join(log_dir, "test_datasets", f"{terrain_name}{suffix}_dataset.pkl")
        if not os.path.isfile(eval_dataset_path):
            print("[WARNING]: Eval dataset not found. Proceeding without datasets of the training environment.")
        else:
            print("[INFO]: Using existing dataset of the training environment.")
            # load dataset
            with open(eval_dataset_path, "rb") as test_dataset:
                datasets[terrain_name] = DataLoader(
                    pickle.load(test_dataset),
                    batch_size=cfg.trainer_cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.trainer_cfg.num_workers,
                    pin_memory=True,
                )

    if cfg.trainer_cfg.real_world_test_datasets is not None:
        if isinstance(cfg.trainer_cfg.real_world_test_datasets, str):
            cfg.trainer_cfg.real_world_test_datasets = [cfg.trainer_cfg.real_world_test_datasets]

        for real_world_test_dataset_path in cfg.trainer_cfg.real_world_test_datasets:
            # if {LOG_DIR} is in the path, replace it with the current log directory
            real_world_test_dataset_path = real_world_test_dataset_path.replace("{LOG_DIR}", log_dir)

            if not os.path.isfile(real_world_test_dataset_path):
                print(
                    f"[WARNING] Real World Test Dataset {real_world_test_dataset_path} not found! Will proceed without!"
                )
                continue

            # get the name of the dataset (given as the directory name)
            directory = os.path.basename(os.path.dirname(real_world_test_dataset_path))
            # Remove numbers using regex
            cleaned_directory = re.sub(r"\d+", "", directory)
            # Remove any leftover underscores or dashes at the start or end
            basename = cleaned_directory.strip("_-")
            # make the first letter uppercase
            basename = basename[0].upper() + basename[1:]
            with open(real_world_test_dataset_path, "rb") as real_world_test_dataset:
                datasets[basename] = DataLoader(
                    pickle.load(real_world_test_dataset),
                    batch_size=cfg.trainer_cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.trainer_cfg.num_workers,
                    pin_memory=True,
                )

    if False:  # cfg.trainer_cfg.test_datasets is not None:
        if isinstance(cfg.trainer_cfg.test_datasets, str):
            cfg.trainer_cfg.test_datasets = [cfg.trainer_cfg.test_datasets]

        for test_dataset_path in cfg.trainer_cfg.test_datasets:
            # if {LOG_DIR} is in the path, replace it with the current log directory
            test_dataset_path = test_dataset_path.replace("{LOG_DIR}", log_dir)

            if not os.path.isfile(test_dataset_path):
                print(f"[WARNING] Test Dataset {test_dataset_path} not found! Will proceed without!")
                continue

            basename = os.path.splitext(os.path.split(test_dataset_path)[1])[0]
            with open(test_dataset_path, "rb") as test_dataset:
                datasets[basename] = DataLoader(
                    pickle.load(test_dataset),
                    batch_size=cfg.trainer_cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.trainer_cfg.num_workers,
                    pin_memory=True,
                )

    print(f"[INFO] Found the following datasets: {datasets.keys()}")

    # get path to save the plots
    if len(args_cli.runs) > 1:
        # get path to save the plots
        plot_dir = os.path.join(args_cli.save_path, "multi_modal")
    else:
        plot_dir = os.path.join(args_cli.save_path, args_cli.runs[0])
    os.makedirs(plot_dir, exist_ok=True)

    # create violin plot
    vilion_plotter = ViolinPlotter(
        datasets=list(datasets.keys()),
        models=args_cli.names + ["Constant Vel."],
        filter_percept_range=True,
        env_sensor_cfg=cfg.env_cfg.scene.env_sensor,
        collision_split=args_cli.collision_split,
        percept_range_relax_factor=0.15,
    )

    # iterate over all runs
    for idx, run in enumerate(args_cli.runs):
        print(f"[INFO] Evaluating model {run}")
        # set the new weights of the model
        resume_path = get_checkpoint_path(log_dir + "/fdm_se2_prediction_depth", run, cfg.trainer_cfg.load_checkpoint)
        # load the model
        model = cfg.model_cfg.class_type(cfg.model_cfg, device=device)
        model.to(device)
        model.load(resume_path)
        model.eval()
        # update the vilion_plotter
        for dataset_name, dataset in datasets.items():
            print(f"[INFO] Updating data for {dataset_name}")
            vilion_plotter.update_data(model, dataset, dataset_name, model_name=args_cli.names[idx], save_path=plot_dir)

    # create the vilion plot
    vilion_plotter.plot_data(plot_dir)
    # also save the statistics as latex
    vilion_plotter.save_statistics_as_latex(os.path.join(plot_dir, "statistics.tex"))

    print("Done evaluated the model")


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
