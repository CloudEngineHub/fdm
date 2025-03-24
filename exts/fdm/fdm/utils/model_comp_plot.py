# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# avoid to use type 3 fonts
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import torch
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from matplotlib.patches import Rectangle
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sns
from isaaclab.sensors import RayCasterCfg
from tabulate import tabulate
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, F1Score

from fdm import PAPER_COLORS_HEX, PAPER_COLORS_RGB_F, PAPER_COLORS_RGBA_F, VEL_RANGE_X, VEL_RANGE_Y, VEL_RANGE_YAW
from fdm.model import FDMModel

matplotlib.rcParams["text.usetex"] = True


def fill_nearest_neighbor(height_map, unknown_value=1.5):
    # apply the median filter
    import torch.nn.functional as F

    import kornia

    padding = (2, 2, 2, 2)
    height_map = F.pad(height_map, padding, mode="replicate")
    height_map = kornia.filters.median_blur(height_map.unsqueeze(1), (5, 5)).squeeze(1)
    height_map = height_map[:, 2:-2, 2:-2]

    # Convert to NumPy array
    height_map_np = height_map.numpy()

    # Binary mask of unknown values
    mask = height_map_np == unknown_value
    indices = np.zeros((len(height_map_np.shape), *mask.shape), dtype=np.int32)

    # Compute the distance transform and nearest neighbor indices
    for i in range(mask.shape[0]):
        indices[0, i] = i
        _, indices[1:, i] = distance_transform_edt(mask[i], return_indices=True)

    # Use indices to assign nearest neighbor values
    filled_map = height_map_np[tuple(indices)]
    return torch.tensor(filled_map)


def meta_summarize(  # noqa: C901
    meta_data: dict,
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], dict[str, dict[str, dict[str, list[float]]]]]:
    """
    Summarize the meta data
    """
    # Regular expression to capture metric, distance ranges, and optionally std
    # -- distance based metrics
    metric_regex = r"(\w+)\s([\w\s]+) (\d+\.\d{2}) - (\d+\.\d{2})m \[Batch\]"
    std_regex = r"(\w+)\s([\w\s]+) Std (\d+\.\d{2}) - (\d+\.\d{2})m \[Batch\]"
    collision_regex = r"(\w+)\s([\w\s]+) (\d+\.\d{2}) - (\d+\.\d{2})m \[Collision\]"
    collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+\.\d{2}) - (\d+\.\d{2})m \[Collision\]"
    non_collision_regex = r"(\w+)\s([\w\s]+) (\d+\.\d{2}) - (\d+\.\d{2})m \[Non-Collision\]"
    non_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+\.\d{2}) - (\d+\.\d{2})m \[Non-Collision\]"
    pred_collision_regex = r"(\w+)\s([\w\s]+) (\d+\.\d{2}) - (\d+\.\d{2})m \[Pred Collision\]"
    pred_non_collision_regex = r"(\w+)\s([\w\s]+) (\d+\.\d{2}) - (\d+\.\d{2})m \[Pred Non-Collision\]"
    pred_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+\.\d{2}) - (\d+\.\d{2})m \[Pred Collision\]"
    pred_non_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+\.\d{2}) - (\d+\.\d{2})m \[Pred Non-Collision\]"
    # -- step based metrics
    step_metric_regex = r"(\w+)\s([\w\s]+) (\d+) \[Batch\]"
    step_std_regex = r"(\w+)\s([\w\s]+) Std (\d+) \[Batch\]"
    step_collision_regex = r"(\w+)\s([\w\s]+) (\d+) \[Collision\]"
    step_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+) \[Collision\]"
    step_non_collision_regex = r"(\w+)\s([\w\s]+) (\d+) \[Non-Collision\]"
    step_non_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+) \[Non-Collision\]"
    step_pred_collision_regex = r"(\w+)\s([\w\s]+) (\d+) \[Pred Collision\]"
    step_pred_non_collision_regex = r"(\w+)\s([\w\s]+) (\d+) \[Pred Non-Collision\]"
    step_pred_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+) \[Pred Collision\]"
    step_pred_non_collision_std_regex = r"(\w+)\s([\w\s]+) Std (\d+) \[Pred Non-Collision\]"

    def get_entry(
        match,
        curr_meta_data: dict,
        std: bool = False,
        collision: bool = False,
        non_collision: bool = False,
        step: bool = False,
        pred: bool = False,
    ) -> dict:
        dataset = match.group(1)  # Dataset name
        metric = match.group(2).strip()  # Metric name
        if step:
            x_value = int(match.group(3))  # Step
        else:
            min_distance = float(match.group(3))  # Min distance
            max_distance = float(match.group(4))  # Max distance
            x_value = (min_distance + max_distance) / 2.0  # Compute the mean distance

        if pred:
            metric = metric + "_pred"

        if collision:
            metric = metric + "_collision"
        elif non_collision:
            metric = metric + "_non_collision"

        if dataset not in curr_meta_data:
            curr_meta_data[dataset] = {}

        if metric not in curr_meta_data[dataset]:
            curr_meta_data[dataset][metric] = {"x": [], "y": [], "std": []}

        # Find the matching distance in the existing data and assign the X and Y values
        if std:
            try:
                idx = curr_meta_data[dataset][metric]["x"].index(x_value)
                curr_meta_data[dataset][metric]["std"][idx] = value
            except ValueError:
                curr_meta_data[dataset][metric]["x"].append(x_value)
                curr_meta_data[dataset][metric]["y"].append(None)  # Placeholder for y
                curr_meta_data[dataset][metric]["std"].append(value)
        else:
            try:
                idx = curr_meta_data[dataset][metric]["x"].index(x_value)
                curr_meta_data[dataset][metric]["y"][idx] = value
            except ValueError:
                curr_meta_data[dataset][metric]["x"].append(x_value)
                curr_meta_data[dataset][metric]["y"].append(value)
                curr_meta_data[dataset][metric]["std"].append(None)  # Placeholder for std

        return curr_meta_data

    # Dictionary to store data for each metric
    metric_data = {}
    step_metric_data = {}

    for key, value in meta_data.items():

        # distance based metrics
        metric_match = re.match(metric_regex, key)
        std_match = re.match(std_regex, key)
        collision_match = re.match(collision_regex, key)
        non_collision_match = re.match(non_collision_regex, key)
        collision_std_match = re.match(collision_std_regex, key)
        non_collision_std_match = re.match(non_collision_std_regex, key)
        pred_collision_match = re.match(pred_collision_regex, key)
        pred_non_collision_match = re.match(pred_non_collision_regex, key)
        pred_collision_std_match = re.match(pred_collision_std_regex, key)
        pred_non_collision_std_match = re.match(pred_non_collision_std_regex, key)

        # step based metrics
        step_metric_match = re.match(step_metric_regex, key)
        step_std_match = re.match(step_std_regex, key)
        step_collision_match = re.match(step_collision_regex, key)
        step_non_collision_match = re.match(step_non_collision_regex, key)
        step_collision_std_match = re.match(step_collision_std_regex, key)
        step_non_collision_std_match = re.match(step_non_collision_std_regex, key)
        step_pred_collision_match = re.match(step_pred_collision_regex, key)
        step_pred_non_collision_match = re.match(step_pred_non_collision_regex, key)
        step_pred_collision_std_match = re.match(step_pred_collision_std_regex, key)
        step_pred_non_collision_std_match = re.match(step_pred_non_collision_std_regex, key)

        # metric based metrics
        if std_match:
            metric_data = get_entry(std_match, metric_data, std=True)
        elif metric_match:
            metric_data = get_entry(metric_match, metric_data)
        elif collision_std_match:
            metric_data = get_entry(collision_std_match, metric_data, std=True, collision=True)
        elif collision_match:
            metric_data = get_entry(collision_match, metric_data, collision=True)
        elif non_collision_std_match:
            metric_data = get_entry(non_collision_std_match, metric_data, std=True, non_collision=True)
        elif non_collision_match:
            metric_data = get_entry(non_collision_match, metric_data, non_collision=True)
        elif pred_collision_std_match:
            metric_data = get_entry(pred_collision_std_match, metric_data, std=True, collision=True, pred=True)
        elif pred_collision_match:
            metric_data = get_entry(pred_collision_match, metric_data, collision=True, pred=True)
        elif pred_non_collision_std_match:
            metric_data = get_entry(pred_non_collision_std_match, metric_data, std=True, non_collision=True, pred=True)
        elif pred_non_collision_match:
            metric_data = get_entry(pred_non_collision_match, metric_data, non_collision=True, pred=True)

        # step based metrics
        elif step_std_match:
            step_metric_data = get_entry(step_std_match, step_metric_data, std=True, step=True)
        elif step_metric_match:
            step_metric_data = get_entry(step_metric_match, step_metric_data, step=True)
        elif step_collision_std_match:
            step_metric_data = get_entry(
                step_collision_std_match, step_metric_data, std=True, collision=True, step=True
            )
        elif step_collision_match:
            step_metric_data = get_entry(step_collision_match, step_metric_data, collision=True, step=True)
        elif step_non_collision_std_match:
            step_metric_data = get_entry(
                step_non_collision_std_match, step_metric_data, std=True, non_collision=True, step=True
            )
        elif step_non_collision_match:
            step_metric_data = get_entry(step_non_collision_match, step_metric_data, non_collision=True, step=True)
        elif step_pred_collision_std_match:
            step_metric_data = get_entry(
                step_pred_collision_std_match, step_metric_data, std=True, collision=True, step=True, pred=True
            )
        elif step_pred_collision_match:
            step_metric_data = get_entry(
                step_pred_collision_match, step_metric_data, collision=True, step=True, pred=True
            )

        elif step_pred_non_collision_std_match:
            step_metric_data = get_entry(
                step_pred_non_collision_std_match, step_metric_data, std=True, non_collision=True, step=True, pred=True
            )
        elif step_pred_non_collision_match:
            step_metric_data = get_entry(
                step_pred_non_collision_match, step_metric_data, non_collision=True, step=True, pred=True
            )

    return metric_data, step_metric_data


def plot_metrics_with_grid(
    meta_summary: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
    step: bool = False,
    suffix: str = "",
    only_first_row: bool = False,
    only_mean_all_datasets: bool = True,
):
    # Determine whether to use larger text based on the number of datasets
    if len(meta_summary) > 3:
        plt.rcParams.update({"font.size": 14})  # Larger font size
    else:
        plt.rcParams.update({"font.size": 10})  # Default font size

    # Define metrics for each row
    if step:
        name_prefix = ""
    else:
        name_prefix = "Final "

    # NOTE: ours last as it has the smallest variance
    if only_first_row:
        metric_row_mapping = [
            [
                "Perfect Velocity Position Offset",
                f"baseline {name_prefix}Position Offset",
                f"{name_prefix}Position Offset",
            ],
        ]
        suffix = "_first_row" + suffix
    else:
        metric_row_mapping = [
            [
                "Perfect Velocity Position Offset",
                f"baseline {name_prefix}Position Offset",
                f"{name_prefix}Position Offset",
            ],
            [
                "Perfect Velocity Position Offset_collision",
                f"baseline {name_prefix}Position Offset_collision",
                f"{name_prefix}Position Offset_collision",
            ],
            [
                "Perfect Velocity Position Offset_non_collision",
                f"baseline {name_prefix}Position Offset_non_collision",
                f"{name_prefix}Position Offset_non_collision",
            ],
            [
                "Perfect Velocity Position Offset_pred_collision",
                f"baseline {name_prefix}Position Offset_pred_collision",
                f"{name_prefix}Position Offset_pred_collision",
            ],
            [
                "Perfect Velocity Position Offset_pred_non_collision",
                f"baseline {name_prefix}Position Offset_pred_non_collision",
                f"{name_prefix}Position Offset_pred_non_collision",
            ],
            [f"{name_prefix}Relative Position Offset", f"baseline {name_prefix}Relative Position Offset"],
            ["Relative Perfect Velocity Position Offset"],
        ]

    # Get the list of datasets
    datasets = list(meta_summary.keys())

    # Determine the number of rows and columns
    n_cols = len(datasets)
    n_rows = len(metric_row_mapping)

    # Create a figure with subplots (n_rows x n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    axs = axs if n_rows > 1 else axs.reshape(1, -1)

    # Loop over datasets and metrics
    for col_idx, (dataset, metrics) in enumerate(meta_summary.items()):
        for row_idx, metric_names in enumerate(metric_row_mapping):
            for metric_name in metric_names:
                if metric_name in metrics:
                    values = metrics[metric_name]
                    # Handle None values in std
                    yerr = [std if std is not None and not math.isnan(std) else 0 for std in values["std"]]

                    # convert to numpy array
                    x = np.array(values["x"])
                    mean = np.array(values["y"])
                    yerr = np.array(yerr)

                    # replace the metrics name for the plotting label
                    if "baseline" in metric_name:
                        plt_metric_name = metric_name.replace(
                            f"baseline {name_prefix}Position Offset", "Kim et al."
                        ).replace("_", " ")
                        plt_color = PAPER_COLORS_RGB_F["baseline"]
                        plt_color_light = PAPER_COLORS_RGB_F["baseline_light"]
                        alpha = 0.5
                    elif "Perfect Velocity" in metric_name:
                        plt_metric_name = metric_name.replace(
                            "Perfect Velocity Position Offset", "Constant Vel."
                        ).replace("_", " ")
                        plt_color = PAPER_COLORS_RGB_F["constant_vel"]
                        plt_color_light = PAPER_COLORS_RGB_F["constant_vel_light"]
                        alpha = 0.25
                    else:
                        plt_metric_name = metric_name.replace(f"{name_prefix}Position Offset", "Ours").replace("_", " ")
                        plt_color = PAPER_COLORS_RGB_F["ours"]
                        plt_color_light = PAPER_COLORS_RGB_F["ours_light"]
                        alpha = 0.5

                    # axs[row_idx, col_idx].errorbar(
                    #     values["x"], values["y"], yerr=yerr, label=plt_metric_name, capsize=5, marker="o", linestyle="-"
                    # )

                    axs[row_idx, col_idx].plot(
                        x, mean, label=plt_metric_name, marker="o", linestyle="-", color=plt_color
                    )
                    axs[row_idx, col_idx].fill_between(x, mean - yerr, mean + yerr, color=plt_color_light, alpha=alpha)

            # Set titles and labels
            axs[row_idx, col_idx].set_title(f"{dataset}")
            axs[row_idx, col_idx].set_xlabel("Prediction Step" if step else "GT Path Distance (m)")
            axs[row_idx, col_idx].set_ylabel("Position Offset (m)")
            axs[row_idx, col_idx].grid(True, alpha=0.5)
            axs[row_idx, col_idx].legend(loc="upper left")

    # Save the plot
    os.makedirs(save_path, exist_ok=True)
    if step:
        fig.savefig(os.path.join(save_path, f"metrics_step_grid_plot{suffix}.pdf"), dpi=300)
    else:
        fig.savefig(os.path.join(save_path, f"metrics_distance_grid_plot{suffix}.pdf"), dpi=300)
    plt.close(fig)

    if not only_mean_all_datasets:
        return

    # Initialize a dictionary to store averaged results
    avg_metrics = {metric: {"x": [], "y": [], "std": []} for row in metric_row_mapping for metric in row}

    # Loop over datasets and metrics to accumulate results
    for dataset, metrics in meta_summary.items():
        for row_idx, metric_names in enumerate(metric_row_mapping):
            for metric_name in metric_names:
                if metric_name in metrics:
                    values = metrics[metric_name]
                    for x, y, std in zip(values["x"], values["y"], values["std"]):
                        if x not in avg_metrics[metric_name]["x"]:
                            avg_metrics[metric_name]["x"].append(x)
                            avg_metrics[metric_name]["y"].append(y)
                            avg_metrics[metric_name]["std"].append(std)
                        else:
                            idx = avg_metrics[metric_name]["x"].index(x)
                            avg_metrics[metric_name]["y"][idx] += y
                            if std is not None and avg_metrics[metric_name]["std"][idx] is not None:
                                avg_metrics[metric_name]["std"][idx] += std
                            elif std is not None:
                                avg_metrics[metric_name]["std"][idx] = std

    # Average the results
    for metric_name, values in avg_metrics.items():
        count = len(meta_summary)
        avg_metrics[metric_name]["y"] = [y / count for y in values["y"]]
        avg_metrics[metric_name]["std"] = [std / count if std is not None else None for std in values["std"]]

    # Create a figure for the averaged results
    fig_avg, axs_avg = plt.subplots(n_rows, 1, figsize=(5, 4 * n_rows), constrained_layout=True)
    axs_avg = axs_avg if n_rows > 1 else [axs_avg]

    # Plot the averaged results
    for row_idx, metric_names in enumerate(metric_row_mapping):
        for metric_name in metric_names:
            if metric_name in avg_metrics:
                values = avg_metrics[metric_name]

                # convert to numpy array
                x = np.array(values["x"])
                mean = np.array(values["y"])
                yerr = np.array([std if std is not None and not math.isnan(std) else 0 for std in values["std"]])

                # replace the metrics name for the plotting label
                if "baseline" in metric_name:
                    plt_metric_name = "Kim et al."
                    plt_color = PAPER_COLORS_RGB_F["baseline"]
                    plt_color_light = PAPER_COLORS_RGB_F["baseline_light"]
                    alpha = 0.5
                elif "Perfect Velocity" in metric_name:
                    plt_metric_name = "Constant Vel."
                    plt_color = PAPER_COLORS_RGB_F["constant_vel"]
                    plt_color_light = PAPER_COLORS_RGB_F["constant_vel_light"]
                    alpha = 0.25
                else:
                    plt_metric_name = "Ours"
                    plt_color = PAPER_COLORS_RGB_F["ours"]
                    plt_color_light = PAPER_COLORS_RGB_F["ours_light"]
                    alpha = 0.5

                # axs_avg[row_idx].errorbar(
                #     values["x"], values["y"], yerr=yerr, label=plt_metric_name, capsize=5, marker="o", linestyle="-"
                # )

                axs_avg[row_idx].plot(x, mean, label=plt_metric_name, marker="o", linestyle="-", color=plt_color)
                axs_avg[row_idx].fill_between(x, mean - yerr, mean + yerr, color=plt_color_light, alpha=alpha)

        axs_avg[row_idx].set_xlabel("Prediction Step" if step else "GT Path Distance (m)")
        axs_avg[row_idx].set_ylabel("Position Offset (m)")
        axs_avg[row_idx].grid(True, alpha=0.5)
        axs_avg[row_idx].legend(loc="upper left")

    # Save the averaged plot
    if step:
        fig_avg.savefig(os.path.join(save_path, f"metrics_step_avg_plot{suffix}.pdf"), dpi=300)
    else:
        fig_avg.savefig(os.path.join(save_path, f"metrics_distance_avg_plot{suffix}.pdf"), dpi=300)
    plt.close(fig_avg)


class ViolinPlotter:
    """Class to plot violin plots for the position delta data."""

    def __init__(
        self,
        steps: list[int] = [4, 9],
        models: list[str] = ["fdm", "Constant Vel.", "baseline"],
        datasets: list[str] = ["train"],
        filter_actions: bool = True,
        filter_percept_range: bool = False,
        env_sensor_cfg: RayCasterCfg | None = None,
        nearest_neighbor_interpolation: bool = False,
        collision_split: bool = True,
        percept_range_relax_factor: float = 0.0,
        correct_collision_estimation_split: bool = False,
        height_scan_plots: bool = False,
        ablation_mode: str | None = None,
    ):
        """
        Vilion Plot constructor.

        Args:
            steps: List of steps to do the plotting for
            models: List of models to include in the plot
            datasets: List of datasets to include in the plot
            filter_actions: Filter the actions based on the limits
            filter_percept_range: Filter the paths based on the perceptive range of the environment sensor
            env_sensor_cfg: Environment sensor configuration
            nearest_neighbor_interpolation: Execute the nearest neighbor interpolation of the environment sensor data. This should not be done anymore as the filter is directly available in the noise augmentations.
            collision_split: Split the data into collision and non-collision data
            percept_range_relax_factor: Factor to relax the percept range filter
            correct_collision_estimation_split: Create an extra plot for the correct collision estimation samples
            height_scan_plots: Create the height scan plots with the prediction, ground truth, and perfect velocity positions
            ablation_mode: Ablation mode to use for the plot
        """

        self.steps = steps
        self.models = models
        self.datasets = datasets
        self.filter_actions = filter_actions
        self.filter_percept_range = filter_percept_range
        self.nearest_neighbor_interpolation = nearest_neighbor_interpolation
        self.collision_split = collision_split
        self.percept_range_relax_factor = percept_range_relax_factor
        self.correct_collision_estimation_split = correct_collision_estimation_split
        self.height_scan_plots = height_scan_plots
        self.ablation_mode = ablation_mode

        # expand models by nn-interpolated data models
        if self.nearest_neighbor_interpolation:
            print(
                "[WARNING] Should not be used anymore, now directly included in the noise augmentations for both real"
                " world and synthetic data!"
            )
            self.models = [
                model_name + " nn"
                for model_name in self.models
                if model_name != "Constant Vel." and model_name != "baseline"
            ] + self.models

        # buffer lists for datasets
        # orgnaization: {model: {dataset: {step: [data]}}}
        #   model: fdm, pv, baseline
        #   dataset: mixed, test, test_1, test_2, ...
        self.position_delta_step: dict[str, dict[str, dict[int, list[torch.tensor]]]] = {
            model: {dataset: {step: [] for step in steps} for dataset in datasets} for model in self.models
        }
        if self.collision_split:
            self.collision_position_delta_step: dict[str, dict[str, dict[int, list[torch.tensor]]]] = {
                model: {dataset: {step: [] for step in steps} for dataset in datasets} for model in self.models
            }
            self.non_collision_position_delta_step: dict[str, dict[str, dict[int, list[torch.tensor]]]] = {
                model: {dataset: {step: [] for step in steps} for dataset in datasets} for model in self.models
            }
        if self.correct_collision_estimation_split:
            self.correct_coll_position_delta_step: dict[str, dict[str, dict[int, list[torch.tensor]]]] = {
                model: {dataset: {step: [] for step in steps} for dataset in datasets} for model in self.models
            }
        # buffer for metric results
        self.collision_metrics: dict[str, dict[str, dict[str, list[float]]]] = {
            model: {dataset: {"precision": [], "recall": [], "accuracy": [], "f1score": []} for dataset in datasets}
            for model in self.models
            if model != "Constant Vel."
        }

        self.low_limits = torch.tensor([VEL_RANGE_X[0], VEL_RANGE_Y[0], VEL_RANGE_YAW[0]])
        self.high_limits = torch.tensor([VEL_RANGE_X[1], VEL_RANGE_Y[1], VEL_RANGE_YAW[1]])
        if env_sensor_cfg is not None:
            self.high_scan_low_limit = (
                -torch.tensor(env_sensor_cfg.pattern_cfg.size) / 2 + torch.tensor(env_sensor_cfg.offset.pos)[:2]
            )
            self.high_scan_high_limit = (
                torch.tensor(env_sensor_cfg.pattern_cfg.size) / 2 + torch.tensor(env_sensor_cfg.offset.pos)[:2]
            )
        else:
            self.high_scan_low_limit = None
            self.high_scan_high_limit = None

        self.out_of_limits_count = 0
        self.percept_out_of_limits_count = 0

    @torch.inference_mode()
    def update_data(
        self,
        model: FDMModel,
        dataloader: DataLoader,
        dataset_name: str,
        baseline: bool = False,
        model_name: str | None = None,
        save_path: str | None = None,
        filter_height_diff: bool = False,
    ):
        # iterate over the eval dataset
        if model_name is None:
            model_name = "fdm" if not baseline else "baseline"

        # if metrics have not been initialized yet, initialize them with the first time the data is updated
        if not hasattr(self, "metric_presision"):
            self.metric_presision = BinaryPrecision(threshold=model.cfg.collision_threshold).to(model.device)
            self.metric_recall = BinaryRecall(threshold=model.cfg.collision_threshold).to(model.device)
            self.metric_accuracy = BinaryAccuracy(threshold=model.cfg.collision_threshold).to(model.device)
            self.metric_f1 = F1Score(task="binary", threshold=model.cfg.collision_threshold).to(model.device)

        out_of_limits_count = 0
        percept_out_of_limits_count = 0
        non_correct_collision_estimation_count = 0
        below_min_height_count = 0
        num_collision_samples = 0

        for batch_idx, inputs in enumerate(dataloader):
            # filter actions according to the current limits
            if self.filter_actions:
                out_of_bounds_idx = torch.logical_or(
                    torch.any(inputs[3] < self.low_limits, dim=-1), torch.any(inputs[3] > self.high_limits, dim=-1)
                )
                out_of_bounds_idx = torch.any(out_of_bounds_idx, dim=-1)
                out_of_limits_count += torch.sum(out_of_bounds_idx).item()
                if torch.any(out_of_bounds_idx):
                    inputs = [inp[~out_of_bounds_idx] for inp in inputs]

            # filter the percept range
            if self.filter_percept_range:
                out_of_percept_range_idx = torch.logical_or(
                    torch.any(
                        inputs[5][..., :2] < (self.high_scan_low_limit) * (1 - self.percept_range_relax_factor), dim=-1
                    ),
                    torch.any(
                        inputs[5][..., :2] > self.high_scan_high_limit * (1 + self.percept_range_relax_factor), dim=-1
                    ),
                )
                out_of_percept_range_idx = torch.any(out_of_percept_range_idx, dim=-1)
                percept_out_of_limits_count += torch.sum(out_of_percept_range_idx).item()
                if torch.any(out_of_percept_range_idx):
                    inputs = [inp[~out_of_percept_range_idx] for inp in inputs]

            # filter for height difference
            # if filter_height_diff:
            #     pv_positions = inputs[6][..., [1, 0]]
            #     pv_positions_idx = (pv_positions - self.high_scan_low_limit[[1, 0]].unsqueeze(0).unsqueeze(0)) / 0.1
            #     env_idx = torch.arange(pv_positions_idx.shape[0], dtype=torch.int64)[:, None].repeat(1, pv_positions_idx.shape[1]).reshape(-1)
            #     pv_positions_idx = pv_positions_idx.reshape(-1, 2).to(torch.int64)
            #     pv_positions_idx[:, 0] = pv_positions_idx[:, 0].clamp(0, inputs[2].shape[2] - 1)
            #     pv_positions_idx[:, 1] = pv_positions_idx[:, 1].clamp(0, inputs[2].shape[3] - 1)
            #     heights = inputs[2][env_idx, 0, pv_positions_idx[:, 0], pv_positions_idx[:, 1]]
            #     heights = heights.reshape(pv_positions.shape[0], pv_positions.shape[1])
            #     # get the maximum height difference between two points along the trajectory
            #     max_height_diff = torch.max(torch.abs(heights[:, 1:] - heights[:, :-1]), dim=1).values
            #     # filter the data based on the height difference
            #     below_min_height_diff_idx = max_height_diff < 0.1
            #     below_min_height_count += torch.sum(below_min_height_diff_idx).item()
            #     if torch.any(below_min_height_diff_idx):
            #         inputs = [inp[~below_min_height_diff_idx] for inp in inputs]

            # for ablation studies that zero out the input
            if self.ablation_mode is not None and not baseline:
                if self.ablation_mode == "no_height_scan":
                    inputs[2] = torch.zeros_like(inputs[2])
                elif self.ablation_mode == "no_proprio_obs":
                    inputs[1] = torch.zeros_like(inputs[1])
                elif self.ablation_mode == "no_state_obs":
                    inputs[0] = torch.zeros_like(inputs[0])

            outputs = model(inputs[:5])
            collision_idx = torch.any(inputs[5][..., 4].to(model.device) == 1, dim=1)
            curr_position_delta_step = torch.norm(outputs[0][:, :, :2] - inputs[5][..., :2].to(model.device), dim=-1)
            curr_pv_position_delta_step = torch.norm(inputs[6][..., :2] - inputs[5][..., :2], dim=-1).to(model.device)
            correct_collision_estimation_idx = collision_idx == torch.any(
                outputs[1] >= model.cfg.collision_threshold, dim=1
            )
            non_correct_collision_estimation_count += torch.sum(~correct_collision_estimation_idx).item()

            # calculate the precision, recall, and f1 score for the collision estimation
            # -- precision: TP / (TP + FP)
            # -- recall: TP / (TP + FN)
            # -- f1: 2 * (precision * recall) / (precision + recall)
            # -- TP: True Positives, FP: False Positives, FN: False Negatives
            self.collision_metrics[model_name][dataset_name]["precision"].append(
                self.metric_presision(outputs[1].max(dim=-1)[0], inputs[5][..., 4].to(model.device).max(dim=-1)[0])
            )
            self.collision_metrics[model_name][dataset_name]["recall"].append(
                self.metric_recall(outputs[1].max(dim=-1)[0], inputs[5][..., 4].to(model.device).max(dim=-1)[0])
            )
            self.collision_metrics[model_name][dataset_name]["accuracy"].append(
                self.metric_accuracy(outputs[1].max(dim=-1)[0], inputs[5][..., 4].to(model.device).max(dim=-1)[0])
            )
            self.collision_metrics[model_name][dataset_name]["f1score"].append(
                self.metric_f1(outputs[1].max(dim=-1)[0], inputs[5][..., 4].to(model.device).max(dim=-1)[0])
            )
            num_collision_samples += torch.sum(collision_idx).item()

            # fill the non-filled part of the height map with nearest neighbor
            if self.nearest_neighbor_interpolation:
                inputs_filled = [curr_inp.clone() for curr_inp in inputs]
                inputs_filled[2] = fill_nearest_neighbor(
                    inputs_filled[2].squeeze(1), inputs_filled[2].max().item()
                ).unsqueeze(1)
                outputs_nn = model(inputs_filled[:5])
                # -- inputs[5] is the model targets
                # -- inputs[6] is the pv target
                filled_curr_position_delta_step = torch.norm(
                    outputs_nn[0][:, :, :2] - inputs[5][..., :2].to(model.device), dim=-1
                )

            for step_idx in self.steps:
                self.position_delta_step[model_name][dataset_name][step_idx].append(
                    curr_position_delta_step[:, step_idx]
                )
                if self.collision_split:
                    self.collision_position_delta_step[model_name][dataset_name][step_idx].append(
                        curr_position_delta_step[collision_idx, step_idx]
                    )
                    self.non_collision_position_delta_step[model_name][dataset_name][step_idx].append(
                        curr_position_delta_step[~collision_idx, step_idx]
                    )

                if self.correct_collision_estimation_split:
                    self.correct_coll_position_delta_step[model_name][dataset_name][step_idx].append(
                        curr_position_delta_step[correct_collision_estimation_idx, step_idx]
                    )

                if not baseline and self.nearest_neighbor_interpolation:
                    self.position_delta_step[model_name + " nn"][dataset_name][step_idx].append(
                        filled_curr_position_delta_step[:, step_idx]
                    )
                    if self.collision_split:
                        self.collision_position_delta_step[model_name + " nn"][dataset_name][step_idx].append(
                            filled_curr_position_delta_step[collision_idx, step_idx]
                        )
                        self.non_collision_position_delta_step[model_name + " nn"][dataset_name][step_idx].append(
                            filled_curr_position_delta_step[~collision_idx, step_idx]
                        )

                if not baseline and "Constant Vel." in self.models:
                    self.position_delta_step["Constant Vel."][dataset_name][step_idx].append(
                        curr_pv_position_delta_step[:, step_idx]
                    )
                    if self.collision_split:
                        self.collision_position_delta_step["Constant Vel."][dataset_name][step_idx].append(
                            curr_pv_position_delta_step[collision_idx, step_idx]
                        )
                        self.non_collision_position_delta_step["Constant Vel."][dataset_name][step_idx].append(
                            curr_pv_position_delta_step[~collision_idx, step_idx]
                        )
                    if self.correct_collision_estimation_split:
                        self.correct_coll_position_delta_step["Constant Vel."][dataset_name][step_idx].append(
                            curr_pv_position_delta_step[correct_collision_estimation_idx, step_idx]
                        )

            if self.high_scan_low_limit is not None and self.height_scan_plots:
                self.plot_traj_on_height_map(
                    curr_position_delta_step,
                    inputs,
                    outputs,
                    dataset_name,
                    model_name,
                    save_path,
                    batch_idx,
                    collision_threshold=model.cfg.collision_threshold,
                )
                self.plot_traj_on_height_map(
                    curr_position_delta_step,
                    inputs,
                    outputs,
                    dataset_name,
                    model_name,
                    save_path,
                    batch_idx,
                    collision_threshold=model.cfg.collision_threshold,
                    largest=False,
                )
                if self.nearest_neighbor_interpolation:
                    self.plot_traj_on_height_map(
                        filled_curr_position_delta_step,
                        inputs_filled,
                        outputs_nn,
                        dataset_name,
                        model_name + " nn",
                        save_path,
                        batch_idx,
                        collision_threshold=model.cfg.collision_threshold,
                    )
                    self.plot_traj_on_height_map(
                        filled_curr_position_delta_step,
                        inputs_filled,
                        outputs_nn,
                        dataset_name,
                        model_name + " nn",
                        save_path,
                        batch_idx,
                        collision_threshold=model.cfg.collision_threshold,
                        largest=False,
                    )
                    # plot highest pv errors
                    self.plot_traj_on_height_map(
                        curr_pv_position_delta_step,
                        inputs_filled,
                        outputs_nn,
                        dataset_name,
                        "Constant Vel.",
                        save_path,
                        batch_idx,
                        collision_threshold=model.cfg.collision_threshold,
                    )
                    self.plot_traj_on_height_map(
                        curr_pv_position_delta_step,
                        inputs_filled,
                        outputs_nn,
                        dataset_name,
                        "Constant Vel.",
                        save_path,
                        batch_idx,
                        collision_threshold=model.cfg.collision_threshold,
                        largest=False,
                    )
                else:
                    self.plot_traj_on_height_map(
                        curr_pv_position_delta_step,
                        inputs,
                        outputs,
                        dataset_name,
                        "Constant Vel.",
                        save_path,
                        batch_idx,
                        collision_threshold=model.cfg.collision_threshold,
                    )
                    self.plot_traj_on_height_map(
                        curr_pv_position_delta_step,
                        inputs,
                        outputs,
                        dataset_name,
                        "Constant Vel.",
                        save_path,
                        batch_idx,
                        collision_threshold=model.cfg.collision_threshold,
                        largest=False,
                    )

        print(
            f"Model {model_name} - Dataset {dataset_name} - Collision Percentage"
            f" {num_collision_samples / len(dataloader.dataset) * 100:.2f}%"
        )
        print(f"Out of limits count: \t\t{out_of_limits_count}")
        print(f"Percept out of limits count: \t\t{percept_out_of_limits_count}")
        print(
            "Percentage of filtered samples:"
            f" \t{(out_of_limits_count + percept_out_of_limits_count) / len(dataloader.dataset) * 100:.2f}%"
        )
        if self.correct_collision_estimation_split:
            print(f"Non correct collision estimation count: \t{non_correct_collision_estimation_count}")
            print(
                "Percentage of non correct collision estimation samples:"
                f" \t{non_correct_collision_estimation_count / len(dataloader.dataset) * 100:.2f}%"
            )
        if filter_height_diff:
            print(f"Below min height difference count: \t{below_min_height_count}")
            print(
                "Percentage of below min height difference samples:"
                f" \t{below_min_height_count / (len(dataloader.dataset) - out_of_limits_count - percept_out_of_limits_count) * 100:.2f}%"
            )

    def plot_traj_on_height_map(
        self,
        position_delta_step: torch.Tensor,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        dataset_name: str,
        model_name: str,
        save_path: str | None = None,
        batch_idx: int = 0,
        largest: bool = True,
        collision_threshold: float = 0.5,
    ):
        # get the highest losses
        loss_values, loss_idx = torch.topk(position_delta_step[:, -1], k=10, largest=largest)
        height_scan = inputs[2][loss_idx.to("cpu")]
        pred_positions = outputs[0][loss_idx.to("cpu")][..., [1, 0]]
        pred_positions_idx = (
            pred_positions - self.high_scan_low_limit[[1, 0]].unsqueeze(0).unsqueeze(0).to(pred_positions.device)
        ) / 0.1
        pred_positions_idx = pred_positions_idx.to(torch.int64).clamp(0, 59)
        true_positions = inputs[5][loss_idx.to("cpu")][..., [1, 0]]
        true_positions_idx = (true_positions - self.high_scan_low_limit[[1, 0]].unsqueeze(0).unsqueeze(0)) / 0.1
        true_positions_idx = true_positions_idx.to(torch.int64).clamp(0, 59)
        pv_positions = inputs[6][loss_idx.to("cpu")][..., [1, 0]]
        pv_positions_idx = (pv_positions - self.high_scan_low_limit[[1, 0]].unsqueeze(0).unsqueeze(0)) / 0.1
        pv_positions_idx = pv_positions_idx.to(torch.int64).clamp(0, 59)
        initial_position = (-self.high_scan_low_limit[[1, 0]] / 0.1).to(torch.int64).clamp(0, 59)

        # plot the height maps with the positions in one figure for all 10
        fig, axs = plt.subplots(2, 5, figsize=(24, 12))
        for idx in range(10):
            print(f"Model {model_name} - Dataset {dataset_name} - Loss: {loss_values[idx].item():.2f} on image {idx}")
            ax = axs[idx // 5, idx % 5]
            img = ax.imshow(height_scan[idx].squeeze(0).to("cpu"), cmap="gray")

            if torch.any(outputs[1][loss_idx[idx]] >= collision_threshold):
                ax.scatter(
                    pred_positions_idx[idx, :, 1].to("cpu"),
                    pred_positions_idx[idx, :, 0].to("cpu"),
                    color="red",
                    s=5,
                    label="Predicted (Collision)",
                )
                ax.plot(
                    pred_positions_idx[idx, :, 1].to("cpu"),
                    pred_positions_idx[idx, :, 0].to("cpu"),
                    color="red",
                    linewidth=1,
                    linestyle="-",
                    alpha=0.7,
                )
            else:
                ax.scatter(
                    pred_positions_idx[idx, :, 1].to("cpu"),
                    pred_positions_idx[idx, :, 0].to("cpu"),
                    color="green",
                    s=5,
                    label="Predicted (Non-Collision)",
                )
                ax.plot(
                    pred_positions_idx[idx, :, 1].to("cpu"),
                    pred_positions_idx[idx, :, 0].to("cpu"),
                    color="green",
                    linewidth=1,
                    linestyle="-",
                    alpha=0.7,
                )

            if torch.any(inputs[5][loss_idx[idx], :, 4] == 1):
                ax.scatter(
                    true_positions_idx[idx, :, 1].to("cpu"),
                    true_positions_idx[idx, :, 0].to("cpu"),
                    color="yellow",
                    s=5,
                    label="Ground Truth (Collision)",
                )
                ax.plot(
                    true_positions_idx[idx, :, 1].to("cpu"),
                    true_positions_idx[idx, :, 0].to("cpu"),
                    color="yellow",
                    linewidth=1,
                    linestyle="-",
                    alpha=0.7,
                )
            else:
                ax.scatter(
                    true_positions_idx[idx, :, 1].to("cpu"),
                    true_positions_idx[idx, :, 0].to("cpu"),
                    color="blue",
                    s=5,
                    label="Ground Truth (Non-Collision)",
                )
                ax.plot(
                    true_positions_idx[idx, :, 1].to("cpu"),
                    true_positions_idx[idx, :, 0].to("cpu"),
                    color="blue",
                    linewidth=1,
                    linestyle="-",
                    alpha=0.7,
                )

            ax.scatter(
                pv_positions_idx[idx, :, 1].to("cpu"),
                pv_positions_idx[idx, :, 0].to("cpu"),
                color="orange",
                s=5,
                label="Perfect Velocity",
            )
            ax.plot(
                pv_positions_idx[idx, :, 1].to("cpu"),
                pv_positions_idx[idx, :, 0].to("cpu"),
                color="orange",
                linewidth=1,
                linestyle="-",
                alpha=0.7,
            )
            ax.scatter(
                initial_position[1].item(), initial_position[0].item(), color="green", s=5, label="Initial Position"
            )

            ax.set_title(f"Loss: {loss_values[idx].item():.2f}")
            ax.legend()

            # Add colorbar
            cbar = fig.colorbar(img, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Height")

        plt.tight_layout()
        name_suffix = "" if largest else "_min"

        if save_path is not None:
            os.makedirs(os.path.join(save_path, "height_map_plots"), exist_ok=True)
            plt.savefig(
                os.path.join(
                    save_path,
                    "height_map_plots",
                    f"height_map_loss{name_suffix}_{dataset_name}_{model_name.replace(' ', '_')}_{f'{batch_idx}'.zfill(2)}.png",
                ),
                dpi=300,
            )
        else:
            plt.savefig(f"height_map_loss{name_suffix}_{dataset_name}_{model_name.replace(' ', '_')}.png", dpi=300)
        plt.close(fig)

        # save the corresponding actions in a csv file
        actions = inputs[3][loss_idx.to("cpu")].to("cpu").numpy().reshape(-1, 2)
        if save_path is not None:
            np.savetxt(
                os.path.join(save_path, f"actions{name_suffix}_{dataset_name}_{model_name.replace(' ', '_')}.csv"),
                actions,
                delimiter=",",
                fmt="%.2f",
            )
        else:
            np.savetxt(
                f"actions{name_suffix}_{dataset_name}_{model_name.replace(' ', '_')}.csv",
                actions,
                delimiter=",",
                fmt="%.2f",
            )

    def plot_data(  # noqa: C901
        self,
        save_path: str,
        apply_clip: bool = True,
        names: list[str] | None = None,
        clip_upper_percentile: int = 95,
        prefix: str = "fine_tune_",
        dataset_name_map: dict | None = None,
        log_scale: bool = False,
    ):
        """Plot the data in a violin plot."""
        # Determine whether to use larger text based on the number of datasets
        if len(self.datasets) > 3:
            plt.rcParams.update({"font.size": 14})  # Larger font size
        else:
            plt.rcParams.update({"font.size": 10})  # Default font size

        for step in self.steps:
            # Number of rows = 3 (Position delta, Collision, Non-collision)
            fig, axes = plt.subplots(
                3 if self.collision_split else 1,
                len(self.datasets),
                figsize=(5 * len(self.datasets), 12 if self.collision_split else 4),
            )  # , sharey='row')
            fig.suptitle("Violin Plots for Position Delta, Collision, and Non-Collision Data", fontsize=16)

            if len(axes.shape) == 1 and self.collision_split:
                axes = axes[:, None]
            elif len(axes.shape) == 1:
                axes = axes[None, :]

            # Iterate over datasets (columns)
            for col_idx, dataset in enumerate(self.datasets):

                # Iterate over rows (Position delta, Collision, Non-collision)
                for row_idx in range(3) if self.collision_split else [0]:

                    ax = axes[row_idx, col_idx]

                    # Prepare data for each model dynamically
                    data_by_model = []
                    for model in self.models:
                        if row_idx == 0:  # Position delta
                            data_by_model.append(torch.concatenate(self.position_delta_step[model][dataset][step]))
                        elif row_idx == 1:  # Collision data
                            data_by_model.append(
                                torch.concatenate(self.collision_position_delta_step[model][dataset][step])
                            )
                        elif row_idx == 2:  # Non-collision data
                            data_by_model.append(
                                torch.concatenate(self.non_collision_position_delta_step[model][dataset][step])
                            )

                    assert all(
                        [data.min() >= 0.0 if len(data) > 0 else True for data in data_by_model]
                    ), "Data contains negative values!"

                    # Combine all data into one dataframe-like structure for seaborn plotting
                    if apply_clip:
                        # Clip data to reduce outliers impact
                        data_clipped = [
                            self.clip_data(data.cpu().numpy(), upper_percentile=clip_upper_percentile)
                            for data in data_by_model
                        ]

                        # Use clipped data for violin plot
                        all_data = np.concatenate(data_clipped)
                        model_labels = sum(
                            (([model] * len(data)) for model, data in zip(self.models, data_clipped)), []
                        )
                    else:
                        all_data = np.concatenate([data.cpu().numpy() for data in data_by_model])
                        model_labels = sum(
                            (([model] * len(data)) for model, data in zip(self.models, data_by_model)), []
                        )

                    assert all_data.min() >= 0.0 if len(all_data) > 0 else True, "All data contains negative values!"

                    # Create violin plot in the current axis
                    sns.violinplot(x=model_labels, y=all_data, ax=ax, inner="quart", cut=0)
                    if dataset_name_map is not None and dataset.lower() in dataset_name_map:
                        formatted_title = dataset_name_map[dataset.lower()]
                    else:
                        formatted_title = " ".join(word.capitalize() for word in dataset.replace("_", " ").split())
                    ax.set_title(formatted_title + f" - Step {step}")

                    # Set row-specific y-labels
                    if col_idx == 0:
                        if row_idx == 0:
                            ax.set_ylabel("Position Delta")
                        elif row_idx == 1:
                            ax.set_ylabel("Position Delta [Collision]")
                        elif row_idx == 2:
                            ax.set_ylabel("Position Delta [No Collision]")

                    # add grid lines
                    ax.grid(True, alpha=0.5)

                    # Set log scale if specified
                    if log_scale:
                        linthresh = 1
                        ax.set_yscale("symlog", linthresh=linthresh)

                    if col_idx == 0:
                        # Add custom labels for "linear" and "logarithmic"
                        axes[row_idx, col_idx].annotate(
                            "linear",
                            xy=(0, linthresh / 2),  # Position halfway between 0 and linthresh
                            xytext=(-65, 0),  # Adjust label alignment to match the axis
                            textcoords="offset points",
                            rotation=90,
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="black",
                        )

                        axes[row_idx, col_idx].annotate(
                            "logarithmic",
                            xy=(0, linthresh * 2),  # Position above linthresh in the logarithmic region
                            xytext=(-65, 0),  # Adjust label alignment
                            textcoords="offset points",
                            rotation=90,
                            ha="center",
                            va="center",
                            fontsize=10,
                            color="black",
                        )

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Adjust title spacing
            os.makedirs(save_path, exist_ok=True)
            suffix = "_with_collision" if self.collision_split else ""
            fig.savefig(os.path.join(save_path, f"{prefix}violin_plot_step{step}{suffix}.pdf"), dpi=300)
            plt.close(fig)

        # If there are two steps, create a combined violin plot with hue differentiation
        if len(self.steps) == 2:
            # Number of rows = 3 (Position delta, Collision, Non-collision) if collision_split is True else 1 (position delta)
            fig, axes = plt.subplots(
                3 if self.collision_split else 1,
                len(self.datasets),
                figsize=(5 * len(self.datasets), 12 if self.collision_split else 4),
                sharey="row",
                dpi=300,
            )
            # fig.suptitle('Combined Violin Plots for Step Comparisons', fontsize=16)

            if len(axes.shape) == 1 and self.collision_split:
                axes = axes[:, None]
            elif len(axes.shape) == 1:
                axes = axes[None, :]

            for col_idx, dataset in enumerate(self.datasets):

                for row_idx in range(3) if self.collision_split else [0]:

                    ax = axes[row_idx, col_idx]

                    # Prepare data for each model dynamically
                    data_step1_by_model = []
                    data_step2_by_model = []
                    for model in self.models:
                        if row_idx == 0:  # Position delta
                            data_step1_by_model.append(
                                torch.concatenate(self.position_delta_step[model][dataset][self.steps[0]])
                            )
                            data_step2_by_model.append(
                                torch.concatenate(self.position_delta_step[model][dataset][self.steps[1]])
                            )
                        elif row_idx == 1:  # Collision data
                            data_step1_by_model.append(
                                torch.concatenate(self.collision_position_delta_step[model][dataset][self.steps[0]])
                            )
                            data_step2_by_model.append(
                                torch.concatenate(self.collision_position_delta_step[model][dataset][self.steps[1]])
                            )
                        elif row_idx == 2:  # Non-collision data
                            data_step1_by_model.append(
                                torch.concatenate(self.non_collision_position_delta_step[model][dataset][self.steps[0]])
                            )
                            data_step2_by_model.append(
                                torch.concatenate(self.non_collision_position_delta_step[model][dataset][self.steps[1]])
                            )

                    # Combine all data into one dataframe-like structure for seaborn plotting
                    if apply_clip:
                        # Clip data to reduce outliers impact
                        data_step1_clipped = [
                            self.clip_data(data.cpu().numpy(), upper_percentile=clip_upper_percentile)
                            for data in data_step1_by_model
                        ]
                        data_step2_clipped = [
                            self.clip_data(data.cpu().numpy(), upper_percentile=clip_upper_percentile)
                            for data in data_step2_by_model
                        ]

                        # Prepare data and labels for combined steps
                        all_data = np.concatenate(data_step1_clipped + data_step2_clipped)
                        model_labels = sum(
                            (([model] * len(data)) for model, data in zip(self.models, data_step1_clipped)), []
                        ) + sum((([model] * len(data)) for model, data in zip(self.models, data_step2_clipped)), [])
                        step_labels = sum(
                            ([f"Step {self.steps[0]}"] * len(data) for data in data_step1_clipped), []
                        ) + sum(([f"Step {self.steps[1]}"] * len(data) for data in data_step2_clipped), [])
                    else:
                        all_data = np.concatenate(
                            [data.cpu().numpy() for data in data_step1_by_model + data_step2_by_model]
                        )
                        model_labels = sum(
                            (([model] * len(data)) for model, data in zip(self.models, data_step1_by_model)), []
                        ) + sum((([model] * len(data)) for model, data in zip(self.models, data_step2_by_model)), [])
                        step_labels = sum(
                            ([f"Step {self.steps[0]}"] * len(data) for data in data_step1_by_model), []
                        ) + sum(([f"Step {self.steps[1]}"] * len(data) for data in data_step2_by_model), [])

                    # Create violin plot with hue for the steps
                    sns.violinplot(
                        x=model_labels,
                        y=all_data,
                        hue=step_labels,
                        split=True,
                        ax=ax,
                        inner="quart",
                        cut=0,
                        gap=0.1,
                        palette={
                            f"Step {self.steps[0]}": PAPER_COLORS_RGBA_F["step_4"],
                            f"Step {self.steps[1]}": PAPER_COLORS_RGBA_F["step_9"],
                        },
                    )
                    if dataset_name_map is not None and dataset.lower() in dataset_name_map:
                        formatted_title = dataset_name_map[dataset.lower()]
                    else:
                        formatted_title = " ".join(word.capitalize() for word in dataset.replace("_", " ").split())
                    ax.set_title(formatted_title)

                    # Get the renderer to measure text sizes
                    renderer = ax.figure.canvas.get_renderer()

                    # Customize x-axis labels with a square
                    for idx, (tick, label) in enumerate(zip(ax.get_xticks(), ax.get_xticklabels())):
                        if "fine tuned" in label.get_text().lower():
                            color = PAPER_COLORS_HEX["ours"]
                            x_offset_factor = 2.12
                        elif "pure sim" in label.get_text().lower():
                            color = PAPER_COLORS_HEX["ours"]
                            x_offset_factor = 2.15
                        elif "ours" in label.get_text().lower():
                            color = PAPER_COLORS_HEX["ours"]
                            x_offset_factor = 4.25
                        elif "baseline" in label.get_text().lower() or "kim et al." in label.get_text().lower():
                            color = PAPER_COLORS_HEX["baseline"]
                            x_offset_factor = 3.3
                        elif "constant vel." in label.get_text().lower() and prefix == "fine_tune_":
                            color = PAPER_COLORS_HEX["constant_vel"]
                            x_offset_factor = 2.2
                        elif "constant vel." in label.get_text().lower():
                            color = PAPER_COLORS_HEX["constant_vel"]
                            x_offset_factor = 3.1
                        else:
                            color = "#000000"  # Default color if no match is found

                        # Measure the width of the label
                        bbox = label.get_window_extent(renderer=renderer)
                        text_width = bbox.width / ax.figure.dpi  # Convert to inches

                        # Convert text width to data units
                        x_offset = text_width * (ax.get_xlim()[1] - ax.get_xlim()[0]) / ax.figure.get_size_inches()[0]
                        # Create a small square patch with a fixed size
                        square = Rectangle((0, 0), width=0.03, height=0.05, color=color)  # Adjust size here

                        # Wrap the square in an AuxTransformBox
                        box = AuxTransformBox(ax.transAxes)
                        box.add_artist(square)

                        # Position the square after the text
                        ab = AnnotationBbox(
                            box,
                            (
                                tick - x_offset * x_offset_factor,
                                -0.05,
                            ),  # Add x_offset to move the square after the text
                            xycoords=("data", "axes fraction"),  # Align x with tick, y relative to the axis
                            box_alignment=(0, 0.5),  # Align the square's center with the label
                            frameon=False,  # Remove the bounding box around the square
                        )

                        # Add the AnnotationBbox to the axis
                        ax.add_artist(ab)

                    # Set row-specific y-labels
                    if col_idx == 0:
                        if row_idx == 0:
                            ax.set_ylabel("Position Delta (m)")
                        elif row_idx == 1:
                            ax.set_ylabel("Position Delta [Collision] (m)")
                        elif row_idx == 2:
                            ax.set_ylabel("Position Delta [No Collision] (m)")

                    # Remove legends for all axes except the first one
                    if not (col_idx == 0 and row_idx == 0):  # Keep legend only for the first plot
                        ax.legend_.remove()
                    else:
                        # Set the legend position for the first plot
                        ax.legend(loc="upper left")  # Adjust the location as needed (e.g., 'upper left', 'best')

                    # add grid lines
                    ax.grid(True, alpha=0.5)

                    # Set log scale if specified
                    if log_scale:
                        linthresh = 1
                        ax.set_yscale("symlog", linthresh=linthresh)

                if col_idx == 0:
                    # Add custom labels for "linear" and "logarithmic"
                    axes[row_idx, col_idx].annotate(
                        "linear",
                        xy=(0, linthresh / 2),  # Position halfway between 0 and linthresh
                        xytext=(-65, 0),  # Adjust label alignment to match the axis
                        textcoords="offset points",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )

                    axes[row_idx, col_idx].annotate(
                        "logarithmic",
                        xy=(0, linthresh * 2),  # Position above linthresh in the logarithmic region
                        xytext=(-65, 0),  # Adjust label alignment
                        textcoords="offset points",
                        rotation=90,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="black",
                    )

            # Adjust layout
            plt.tight_layout()
            # plt.subplots_adjust(bottom=0.2, left=0.1, top=0.9)  # Adjust title spacing
            suffix = "_with_collision" if self.collision_split else ""
            fig.savefig(os.path.join(save_path, f"{prefix}combined_violin_plot{suffix}.pdf"))
            plt.close(fig)

        if self.correct_collision_estimation_split and len(self.steps) == 2:
            fig, axes = plt.subplots(1, len(self.datasets), figsize=(5 * len(self.datasets), 4), sharey="row")

            for idx, dataset in enumerate(self.datasets):
                ax = axes[idx]

                data_step1_by_model = []
                data_step2_by_model = []
                for model in self.models:
                    data_step1_by_model.append(
                        torch.concatenate(self.correct_coll_position_delta_step[model][dataset][self.steps[0]])
                    )
                    data_step2_by_model.append(
                        torch.concatenate(self.correct_coll_position_delta_step[model][dataset][self.steps[1]])
                    )

                # Combine all data into one dataframe-like structure for seaborn plotting
                if apply_clip:
                    # Clip data to reduce outliers impact
                    data_step1_clipped = [
                        self.clip_data(data.cpu().numpy(), upper_percentile=clip_upper_percentile)
                        for data in data_step1_by_model
                    ]
                    data_step2_clipped = [
                        self.clip_data(data.cpu().numpy(), upper_percentile=clip_upper_percentile)
                        for data in data_step2_by_model
                    ]

                    # Prepare data and labels for combined steps
                    all_data = np.concatenate(data_step1_clipped + data_step2_clipped)
                    model_labels = sum(
                        (([model] * len(data)) for model, data in zip(self.models, data_step1_clipped)), []
                    ) + sum((([model] * len(data)) for model, data in zip(self.models, data_step2_clipped)), [])
                    step_labels = sum(([f"Step {self.steps[0]}"] * len(data) for data in data_step1_clipped), []) + sum(
                        ([f"Step {self.steps[1]}"] * len(data) for data in data_step2_clipped), []
                    )
                else:
                    all_data = np.concatenate(
                        [data.cpu().numpy() for data in data_step1_by_model + data_step2_by_model]
                    )
                    model_labels = sum(
                        (([model] * len(data)) for model, data in zip(self.models, data_step1_by_model)), []
                    ) + sum((([model] * len(data)) for model, data in zip(self.models, data_step2_by_model)), [])
                    step_labels = sum(
                        ([f"Step {self.steps[0]}"] * len(data) for data in data_step1_by_model), []
                    ) + sum(([f"Step {self.steps[1]}"] * len(data) for data in data_step2_by_model), [])

                # Create violin plot with hue for the steps
                sns.violinplot(
                    x=model_labels, y=all_data, hue=step_labels, split=True, ax=ax, inner="quart", cut=0, gap=0.1
                )
                if dataset_name_map is not None and dataset.lower() in dataset_name_map:
                    formatted_title = dataset_name_map[dataset.lower()]
                else:
                    formatted_title = " ".join(word.capitalize() for word in dataset.replace("_", " ").split())
                ax.set_title(formatted_title)

                # Remove legends for all axes except the first one
                if idx != 0:  # Keep legend only for the first plot
                    ax.legend_.remove()
                else:
                    # Set the legend position for the first plot
                    ax.legend(loc="upper left")  # Adjust the location as needed (e.g., 'upper left', 'best')

                # add grid lines
                ax.grid(True, alpha=0.5)

                # Set log scale if specified
                if log_scale:
                    linthresh = 1
                    ax.set_yscale("symlog", linthresh=linthresh)

            axes[0].set_ylabel("Position Delta (m)")

            if log_scale:
                # Add custom labels for "linear" and "logarithmic"
                axes[0].annotate(
                    "linear",
                    xy=(0, linthresh / 2),  # Position halfway between 0 and linthresh
                    xytext=(-65, 0),  # Adjust label alignment to match the axis
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

                axes[0].annotate(
                    "logarithmic",
                    xy=(0, linthresh * 2),  # Position above linthresh in the logarithmic region
                    xytext=(-65, 0),  # Adjust label alignment
                    textcoords="offset points",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Adjust title spacing
            suffix = "_with_collision" if self.collision_split else ""
            fig.savefig(os.path.join(save_path, f"{prefix}corr_coll_violin_plot{suffix}.pdf"), dpi=300)
            plt.close(fig)

    def save_statistics_as_latex(self, save_path: str):
        """Compute mean and standard deviation of the data and save the computed statistics as a LaTeX table."""
        rows = []
        coll_rows = []

        for model in self.models:
            for dataset in self.datasets:
                for step in self.steps:
                    # Position Delta
                    pos_delta = torch.concatenate(self.position_delta_step[model][dataset][step])
                    mean_pos = pos_delta.mean().item()
                    std_pos = pos_delta.std().item()

                    row = [model, dataset, step, f"{mean_pos:.2f} ± {std_pos:.2f}"]

                    if self.collision_split:
                        # Collision Position Delta
                        collision_delta = torch.concatenate(self.collision_position_delta_step[model][dataset][step])
                        mean_collision = collision_delta.mean().item()
                        std_collision = collision_delta.std().item()

                        # Non-Collision Position Delta
                        non_collision_delta = torch.concatenate(
                            self.non_collision_position_delta_step[model][dataset][step]
                        )
                        mean_non_collision = non_collision_delta.mean().item()
                        std_non_collision = non_collision_delta.std().item()

                        row += [
                            f"{mean_collision:.2f} ± {std_collision:.2f}",
                            f"{mean_non_collision:.2f} ± {std_non_collision:.2f}",
                        ]

                    rows.append(row)

                if model == "Constant Vel.":
                    continue

                # Collision Estimation Metrics
                precision = torch.vstack(self.collision_metrics[model][dataset]["precision"]).mean().item() * 100
                recall = torch.vstack(self.collision_metrics[model][dataset]["recall"]).mean().item() * 100
                accuracy = torch.vstack(self.collision_metrics[model][dataset]["accuracy"]).mean().item() * 100
                f1score = torch.vstack(self.collision_metrics[model][dataset]["f1score"]).mean().item()

                coll_rows.append(
                    [model, dataset, f"{precision:.2f}", f"{recall:.2f}", f"{accuracy:.2f}", f"{f1score:.2f}"]
                )

        # Create a DataFrame for formatting as LaTeX
        columns = ["Model", "Dataset", "Step", "Mean Pos ± Std"]
        if self.collision_split:
            columns += ["Mean Collision ± Std", "Mean Non-Collision ± Std"]
        df = pd.DataFrame(rows, columns=columns)

        # Create a DataFrame for collision estimation metrics
        coll_columns = ["Model", "Dataset", "Precision", "Recall", "Accuracy", "F1 Score"]
        coll_df = pd.DataFrame(coll_rows, columns=coll_columns)

        # Generate LaTeX table
        latex_table = tabulate(df, headers="keys", tablefmt="latex", showindex=False)
        coll_latex_table = tabulate(coll_df, headers="keys", tablefmt="latex", showindex=False)

        # Save to file
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        with open(save_path.replace(".tex", "_coll_metrics.tex"), "w", encoding="utf-8") as f:
            f.write(coll_latex_table)

        # Print the LaTeX table to the command line
        print("\nGenerated LaTeX Table:\n")
        print(latex_table)
        print("\nGenerated Collision Estimation Metrics LaTeX Table:\n")
        print(coll_latex_table)

    @staticmethod
    def clip_data(data, lower_percentile=0, upper_percentile=95):
        if len(data) == 0:
            return data

        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        return np.clip(data, lower_bound, upper_bound)
