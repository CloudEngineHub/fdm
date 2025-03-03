# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to combine multiple PyTorch datasets."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Combine multiple PyTorch datasets.")
parser.add_argument(
    "--save_path",
    help="Path to save the combined dataset.",
    # default="/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/snow/test_snow_dataset_reducedObs_noTorque_nearest_neighbor_filling.pkl",
    default="/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/forest/test_forest_dataset_reducedObs_noTorque_nearest_neighbor_filling.pkl",
)
parser.add_argument(
    "--dataset_paths",
    nargs="+",
    help="Paths to the pickle files containing the datasets.",
    # snow
    # default=[
    #     "/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/2024-11-03-07-57-34_moenchsjoch_fenced_2/real_world_dataset_reducedObs_noTorque_nearest_neighbor_filling_train.pkl",
    #     "/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/2024-11-03-07-57-34_moenchsjoch_fenced_2/real_world_dataset_reducedObs_noTorque_nearest_neighbor_filling_val.pkl",
    #     "/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/2024-11-03-08-42-30_moenchsjoch_outside_2/real_world_dataset_reducedObs_noTorque_nearest_neighbor_filling_train.pkl",
    #     "/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/2024-11-03-08-42-30_moenchsjoch_outside_2/real_world_dataset_reducedObs_noTorque_nearest_neighbor_filling_val.pkl",
    # ]
    # forest
    default=[
        "/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/2024-11-14-14-36-02_forest_kaeferberg_entanglement/real_world_dataset_reducedObs_noTorque_nearest_neighbor_filling_val.pkl",
        "/home/pascal/orbit/IsaacLab/logs/fdm/real_world_datasets/2024-11-15-12-06-03_forest_albisguetli_slippery_slope/real_world_dataset_reducedObs_noTorque_nearest_neighbor_filling_val.pkl",
    ],
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import pickle
from pathlib import Path
from torch.utils.data import ConcatDataset, Dataset


def combine_torch_datasets(dataset_paths, save_path):
    """
    Combines multiple PyTorch datasets (saved as pickle files) into a single dataset
    and saves it to the specified path.

    Parameters:
        dataset_paths (list of str): Paths to the pickle files containing the datasets.
        save_path (str): Path to save the combined dataset.

    Returns:
        None
    """
    datasets = []

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"File not found: {dataset_path}")
            continue

        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, Dataset):
            datasets.append(data)
            print(f"Loaded dataset from: {dataset_path} with {len(data)} samples")
        else:
            raise ValueError(f"Unsupported dataset format in {dataset_path}. Expected a PyTorch Dataset.")

    if not datasets:
        raise ValueError("No valid datasets were loaded. Please check the paths and formats.")

    # Combine the datasets using ConcatDataset
    combined_dataset = ConcatDataset(datasets)

    # Save the combined dataset
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(combined_dataset, f)
    print(f"Combined dataset saved to: {save_path} with {len(combined_dataset)} samples")


if __name__ == "__main__":
    combine_torch_datasets(args_cli.dataset_paths, args_cli.save_path)
    # close sim app
    simulation_app.close()
