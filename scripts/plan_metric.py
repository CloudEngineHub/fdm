# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate planning performance against baseline methods given random paths in a defined environment.

Does a comparison of 10000 paths in simulation with the following methods:
- MPPI using the presented FDM
- MPPI using the baseline FDM
- MPPI using a cost-map
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# local imports
import utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Do planning comparison with multiple methods.")
parser.add_argument(
    "--run",
    type=str,
    default="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque",
    help="Name of the run.",
)
# parser.add_argument("--run", type=str, default="Nov20_15-34-02_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_NoEarlyCollFilter", help="Name of the run.")
parser.add_argument("--mode", type=str, default="debug", choices=["full", "debug"], help="Mode of the script.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=20)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = args_cli.mode != "debug"
args_cli.num_envs = 2 if args_cli.mode == "debug" else 200

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time
import torch
import yaml

import omni
import wandb
from isaaclab_tasks.utils import get_checkpoint_path
from tabulate import tabulate

from fdm.env_cfg import TERRAIN_ANALYSIS_CFG
from fdm.planner import FDMPlanner, get_planner_cfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, env_modifier_post_init, planner_cfg_init, robot_changes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def load_planner() -> FDMPlanner:
    # setup runner
    cfg = planner_cfg_init(args_cli)
    # robot changes
    cfg = robot_changes(cfg, args_cli)
    # modify cfg
    cfg = cfg_modifier_pre_init(cfg, args_cli)

    # define the number of runs
    cfg.env_cfg.commands.command.trajectory_config = (
        {
            "num_paths": 40,
            "max_path_length": 8,
            "min_path_length": 3,
        }
        if args_cli.mode == "debug"
        else {
            "num_paths": 1000,
            "max_path_length": 8,
            "min_path_length": 3,
        }
    )
    cfg.env_cfg.commands.command.traj_sampling.terrain_analysis.max_path_length = max(
        cfg.env_cfg.commands.command.trajectory_config["max_path_length"]
    )

    # set name of the run
    if args_cli.run is not None:
        cfg.load_run = args_cli.run

    # get planner cfg
    sampling_planner_cfg_dict = get_planner_cfg(args_cli.num_envs, traj_dim=10, debug=False, device="cuda")
    if args_cli.env == "heuristic":
        sampling_planner_cfg_dict["to_cfg"]["control"] = "velocity_control"
        sampling_planner_cfg_dict["to_cfg"]["states_cost_w_cost_map"] = True

    # build planner
    planner = FDMPlanner(cfg, sampling_planner_cfg_dict, args_cli=args_cli)
    # post modify runner and env
    planner = env_modifier_post_init(planner, args_cli=args_cli)

    return planner


def main():
    TERRAIN_ANALYSIS_CFG.sample_points = 10000

    # load config
    planner = load_planner()
    print(f"[INFO] Planner loaded with run: {planner.cfg.load_run}")
    # init metric dict
    metrics: dict[str, dict[str, dict[str, float]]] = {}

    # init wandb logging
    if args_cli.mode == "full":
        wb_entity = os.getenv("WANDB_ENTITY")
        wb_mode = os.getenv("WANDB_MODE", "online")
        wb_api_key = os.getenv("WANDB_API_KEY")

        if not wb_api_key:
            print("[WARNING] WANDB_API_KEY environment variable not set. Wandb logging will be disabled.")
            return

        try:
            wandb.init(
                project="planner_eval",
                entity=wb_entity,
                name=args_cli.run,
                config=planner.cfg.to_dict() | planner.planner_cfg,
                dir=os.path.join("logs", "fdm", "fdm_se2_prediction_depth", args_cli.run),
                mode=wb_mode,
            )
        except:  # noqa: E722
            print("[WARNING: Wandb not available")

    num_paths = sum(planner.cfg.env_cfg.commands.command.trajectory_config["num_paths"])
    min_length = min(planner.cfg.env_cfg.commands.command.trajectory_config["min_path_length"])
    max_length = max(planner.cfg.env_cfg.commands.command.trajectory_config["max_path_length"])

    ###
    # Ours --> MPPI with FDM
    ###

    # check if results already available
    if os.path.exists(
        os.path.join(
            planner.log_root_path,
            planner.cfg.load_run,
            f"planner_eval_metric_mppi_fdm_num{num_paths}_min{min_length}_max{max_length}.yaml",
        )
    ):
        print("[INFO] Planner evaluation metrics for mppi_fdm already available.")
        with open(
            os.path.join(
                planner.log_root_path,
                planner.cfg.load_run,
                f"planner_eval_metric_mppi_fdm_num{num_paths}_min{min_length}_max{max_length}.yaml",
            ),
        ) as f:
            metrics["mppi_fdm"] = yaml.safe_load(f)
        planner.print_metrics(metrics["mppi_fdm"])
    else:
        print("[INFO] Evaluating planner: mppi_fdm")
        start = time.time()
        metrics["mppi_fdm"] = planner.navigate()
        print(f"[INFO] Time taken: {time.time() - start}")

        # save the predictions
        with open(
            os.path.join(
                planner.log_root_path,
                planner.cfg.load_run,
                f"planner_eval_metric_mppi_fdm_num{num_paths}_min{min_length}_max{max_length}.yaml",
            ),
            "w",
        ) as f:
            yaml.dump(metrics["mppi_fdm"], f)

    ###
    # Baseline --> MPPI with heuristic traversability estimation
    ###

    # switch planner to MPPI with heuristic cost map
    applied_planner = "mppi_heuristic"
    metrics_file = os.path.join(
        planner.log_root_path,
        f"planner_eval_metric_{applied_planner}_num{num_paths}_min{min_length}_max{max_length}.yaml",
    )
    # -- check if results already available
    if os.path.exists(metrics_file):
        print(f"[INFO] Planner evaluation metrics for {applied_planner} already available.")
        with open(metrics_file) as f:
            metrics["mppi_heuristic"] = yaml.safe_load(f)

        planner.print_metrics(metrics["mppi_heuristic"])
    else:
        # --swap the model to the baseline model
        planner.close()
        del planner
        # create a new stage
        omni.usd.get_context().new_stage()
        # change the environment to heuristic
        args_cli.env = "heuristic"
        # init the planner
        planner = load_planner()

        print(f"[INFO] Evaluating planner: {applied_planner}")
        start = time.time()
        metrics["mppi_heuristic"] = planner.navigate()
        print(f"[INFO] Time taken: {time.time() - start}")

        # save the predictions
        with open(metrics_file, "w") as f:
            yaml.dump(metrics["mppi_heuristic"], f)

    ###
    # Baseline --> MPPI with Kim et al. FDM
    ###
    # --swap the model to the baseline model
    planner.close()
    del planner
    # create a new stage
    omni.usd.get_context().new_stage()
    # include baseline in the tests --> load baseline method and rerun the evaluation
    args_cli.env = "baseline"
    args_cli.run = None
    planner = load_planner()
    # -- get the used model path
    model_path = get_checkpoint_path(planner.log_root_path, planner.cfg.load_run, planner.cfg.load_checkpoint)
    log_dir, _ = os.path.split(model_path)
    metrics_file = os.path.join(
        log_dir, f"planner_eval_metric_mppi_baseline_num{num_paths}_min{min_length}_max{max_length}.yaml"
    )
    # check if the results are already available
    if os.path.exists(metrics_file):
        print("[INFO] Planner evaluation metrics for mppi_baseline already available.")
        with open(metrics_file) as f:
            metrics["mppi_baseline"] = yaml.safe_load(f)
        planner.print_metrics(metrics["mppi_baseline"])
    else:
        print("[INFO] Evaluating planner: mppi_baseline")

        planner.planner.to_cfg.control = "fdm_baseline"
        start = time.time()
        metrics["mppi_baseline"] = planner.navigate()
        print(f"[INFO] Time taken: {time.time() - start}")

        # save the predictions
        with open(metrics_file) as f:
            yaml.dump(metrics["mppi_baseline"], f)

    ###
    # Show results
    ###

    # show results as a table
    table_data = []
    columns = {metric_name: list(metric_data.keys()) for metric_name, metric_data in metrics["mppi_fdm"].items()}
    for planner_name, planner_metrics in metrics.items():
        # buffer for each subpart of the metric
        curr_metric = []
        for metric, metric_subparts in columns.items():
            for subpart in metric_subparts:
                curr_metric.append(planner_metrics[metric][subpart])
        table_data.append([planner_name] + curr_metric)

    # Define the header with multi-level columns
    headers = ["Planner"] + [f"{metric}_{subpart}" for subpart in columns.values() for metric in columns.keys()]

    # Generate the basic LaTeX table using tabulate
    latex_table = tabulate(table_data, headers, tablefmt="latex", floatfmt=".3f")

    # Modify the LaTeX table to add multi-line header
    multiline_header = r"""
        \begin{table}[ht]
        \centering
        \begin{tabular}{lccccccccccccc}
        \hline
        \multirow{2}{*}{Planner} & \multicolumn{3}{c}{Finished Paths} & \multicolumn{3}{c}{SPL} & \multicolumn{3}{c}{Mean Time} & \multicolumn{3}{c}{Mean Length} \\
        & Success & Fail & All & Success & Fail & All & Success & Fail & All & Success & Fail & All \\
        \hline
    """

    # Add closing parts of the table
    latex_table = latex_table.replace(r"\begin{tabular}", multiline_header)
    latex_table = latex_table.replace(r"\end{tabular}", r"\hline \end{tabular} \end{table}")

    # Print the LaTeX table
    print(latex_table)

    # save the table as a latex
    log_dir = os.path.abspath(os.path.join("logs", "fdm"))
    with open(os.path.join(log_dir, "comp_plots", "planner_statistics.tex"), "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
