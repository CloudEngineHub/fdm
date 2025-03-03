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
    "--runs",
    type=str,
    nargs="+",
    default="Nov19_20-56-45_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_reducedObs_Occlusion_NoEarlyCollFilter_NoTorque",
    help="Name of the run.",
)
# parser.add_argument("--runs", type=str, nargs="+", default="Dec03_20-27-43_MergeSingleObjMazeTerrain_HeightScan_lr3e3_Ep8_CR20_AllOnceStructure_NonUniColl_NOPreTrained_Bs2048_Noise_reducedObs_Occlusion_NoTorque_NewHeightScanNoise_NewNNTrainNoise_SchedEp10_Wait4_Decay5e5", help="Name of the run.")
parser.add_argument("--equal-actions", action="store_true", default=False, help="Have the same actions for all envs.")
parser.add_argument("--only_test_envs", action="store_true", default=True, help="Only test on the test environments.")
parser.add_argument("--terrain_analysis_points", type=int, default=2000, help="Number of points for terrain analysis.")
parser.add_argument("--height_threshold", type=float, default=None, help="Height threshold for samples.")

# append common FDM cli arguments
cli_args.add_fdm_args(parser, default_num_envs=250)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# DEBUG
args_cli.noise = False
args_cli.reduced_obs = True
args_cli.occlusion = False
args_cli.remove_torque = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import os
import torch

import omni

import fdm.mdp as mdp
import fdm.runner as fdm_runner_cfg
from fdm.utils.args_cli_utils import cfg_modifier_pre_init, robot_changes, runner_cfg_init
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
    cfg = cfg_modifier_pre_init(cfg, args_cli, dataset_collecton=True)

    # limit number of samples
    cfg.trainer_cfg.num_samples = 50000

    # increase the resolution of the
    cfg.model_cfg.eval_distance_interval = 0.1

    # restrict number of samples
    # set name of the run
    if args_cli.runs is not None:
        cfg.trainer_cfg.load_run = args_cli.runs[0] if isinstance(args_cli.runs, list) else args_cli.runs

    # set regular spawning pattern
    if args_cli.regular:
        cfg.env_cfg.events.reset_base.func = mdp.reset_root_state_regular
        cfg.env_cfg.events.reset_base.params.pop("pose_range")
        cfg.env_cfg.events.reset_base.params.pop("velocity_range")

    return cfg


def combine_dicts(dicts):
    # Combine data
    combined_dict = {}
    combined_dict_count = {}

    for curr_dict in dicts:
        for env, meta_dict in curr_dict.items():
            if env not in combined_dict:
                combined_dict[env] = {}
                combined_dict_count[env] = {}
            for metric, values in meta_dict.items():
                if "Perfect Velocity" in metric:
                    metric = metric.removeprefix("baseline ")

                if metric not in combined_dict[env]:
                    combined_dict[env][metric] = {"x": values["x"], "y": values["y"], "std": values["std"]}
                    combined_dict_count[env][metric] = {
                        "y_sum": 1,
                        "std_count": 0 if np.any(np.array(values["std"]) is None) else 1,
                    }
                else:
                    combined_dict[env][metric]["y"] = list(np.add(combined_dict[env][metric]["y"], values["y"]))
                    combined_dict_count[env][metric]["y_sum"] += 1
                    if not np.any(np.array(combined_dict[env][metric]["std"]) is None) and not np.any(
                        np.array(values["std"]) is None
                    ):
                        combined_dict[env][metric]["std"] = list(
                            np.add(combined_dict[env][metric]["std"], values["std"])
                        )
                        combined_dict_count[env][metric]["std_count"] += 1
                    elif not np.any(np.array(values["std"]) is None):
                        combined_dict[env][metric]["std"] = values["std"]
                        combined_dict_count[env][metric]["std_count"] += 1

    # Calculate averages
    for env, meta_dict in combined_dict.items():
        for metric, values in meta_dict.items():
            combined_dict[env][metric]["y"] = list(
                np.array(combined_dict[env][metric]["y"]) / combined_dict_count[env][metric]["y_sum"]
            )
            if combined_dict_count[env][metric]["std_count"] > 0:
                combined_dict[env][metric]["std"] = list(
                    np.array(combined_dict[env][metric]["std"]) / combined_dict_count[env][metric]["std_count"]
                )

    return combined_dict


def run_eval(runner: fdm_runner_cfg.FDMRunner, args_cli: argparse.Namespace):
    # eval environments
    meta_eval, test_meta = runner.eval_metric()

    # get path to save the plots
    fdm_model_log_dir = runner.trainer.resume_path
    dir_path, _ = os.path.split(fdm_model_log_dir)
    os.makedirs(os.path.join(dir_path, "plots"), exist_ok=True)
    if args_cli.only_test_envs:
        assert runner.trainer.test_datasets is not None, "No test datasets available"
        datasets = list(runner.trainer.test_datasets.keys())
    else:
        datasets = (
            ["train"] if runner.trainer.dataloader is None else ["train"] + list(runner.trainer.test_datasets.keys())
        )

    # clean test datasets names
    combined_suffix = ""
    if args_cli.reduced_obs:
        combined_suffix += "_reducedObs"
    if args_cli.remove_torque:
        combined_suffix += "_noTorque"
    if args_cli.noise:
        combined_suffix += "_noise"
    elif args_cli.occlusion:
        combined_suffix += "_occlusions"
    for i, dataset in enumerate(datasets):
        if args_cli.height_threshold is not None and "stairs" in dataset.lower():
            dataset = dataset.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
        datasets[i] = dataset.removesuffix(combined_suffix)
        datasets[i] = datasets[i].removesuffix("_EVAL_CFG")  # remove the eval cfg suffix

    # # create violin plot
    vilion_plotter = ViolinPlotter(
        datasets=datasets,
        models=["Ours", "Kim et al.", "Constant Vel."],
        env_sensor_cfg=runner.env.cfg.scene.env_sensor,
        correct_collision_estimation_split=True,
        collision_split=True,
    )
    if not args_cli.only_test_envs:
        vilion_plotter.update_data(
            runner.model,
            runner.trainer.dataloader,
            "train",
            model_name="Ours",
            save_path=os.path.join(dir_path, "plots"),
        )
    if runner.trainer.test_datasets is not None:
        for test_dataset_name, test_dataset in runner.trainer.test_datasets.items():
            vilion_plotter.update_data(
                runner.model,
                test_dataset,
                test_dataset_name.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
                .removesuffix(combined_suffix)
                .removesuffix("_EVAL_CFG"),
                model_name="Ours",
                save_path=os.path.join(dir_path, "plots"),
            )

    # save fdm batch size
    fdm_batch_size = runner.trainer.cfg.batch_size

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
    # avoid resampling the environment if generator is selected
    if args_cli.only_test_envs:
        cfg.env_cfg.scene.terrain.terrain_type = "usd"
    if args_cli.noise:
        cfg.trainer_cfg.load_run = "Jan09_23-18-02_Baseline_NewEnv_NewCollisionShape_noise"
    else:
        cfg.trainer_cfg.load_run = "Jan13_15-36-24_Baseline_NewEnv_NewCollisionShape_CorrLidar"
        # cfg.trainer_cfg.load_run = "Jan13_15-39-48_Baseline_NewEnv_NewCollisionShape_CorrLidar_Plane"
    cfg.trainer_cfg.batch_size = fdm_batch_size
    runner = fdm_runner_cfg.FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)
    baseline_meta_eval, baseline_test_meta = runner.eval_metric()
    baseline_meta_eval_new = {}
    for key, value in baseline_meta_eval.items():
        new_key = key.replace("plot", "plot baseline", 1)
        baseline_meta_eval_new[new_key] = value
    for key, value in baseline_test_meta.items():
        new_key = key.replace("_baseline", " baseline", 1)
        baseline_meta_eval_new[new_key] = value

    # create violin plot
    if not args_cli.only_test_envs:
        vilion_plotter.update_data(
            runner.model,
            runner.trainer.dataloader,
            "train",
            model_name="Kim et al.",
            baseline=True,
            save_path=os.path.join(dir_path, "plots"),
        )
    if runner.trainer.test_datasets is not None:
        for test_dataset_name, test_dataset in runner.trainer.test_datasets.items():
            vilion_plotter.update_data(
                runner.model,
                test_dataset,
                test_dataset_name.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
                .removesuffix("_baseline")
                .removesuffix("_EVAL_CFG"),
                model_name="Kim et al.",
                baseline=True,
                save_path=os.path.join(dir_path, "plots"),
            )
    # create the vilion plot
    vilion_plotter.plot_data(
        os.path.join(dir_path, "plots"),
        clip_upper_percentile=99,
        prefix="sim_comp_",
        dataset_name_map={
            "train": "Train",
            "plane": "Plane",
            "pillar": "2D",
            "stairs_wall": "2D-3D",
            "stairs_ramp": "3D",
        },
        log_scale=True,
    )
    # also save the statistics as latex
    vilion_plotter.save_statistics_as_latex(os.path.join(dir_path, "plots", "statistics.tex"))

    # Save the metrics in a grid plot
    # -- plot without baseline
    distance_meta_summary, step_meta_summary = meta_summarize(meta_eval | test_meta)

    # remove the train dataset
    if args_cli.only_test_envs:
        distance_meta_summary.pop("plot")
        step_meta_summary.pop("plot")

    # keys are the dataset names, apply the the same remove of suffix and EVAL_CFG for each of the keys
    distance_meta_summary = {
        key.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
        .removesuffix(combined_suffix)
        .removesuffix("_EVAL_CFG")
        .removeprefix("plot_"): value
        for key, value in distance_meta_summary.items()
    }
    step_meta_summary = {
        key.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
        .removesuffix(combined_suffix)
        .removesuffix("_EVAL_CFG")
        .removeprefix("plot_"): value
        for key, value in step_meta_summary.items()
    }

    plot_metrics_with_grid(step_meta_summary, os.path.join(dir_path, "plots"), step=True)
    plot_metrics_with_grid(distance_meta_summary, os.path.join(dir_path, "plots"))
    plot_metrics_with_grid(step_meta_summary, os.path.join(dir_path, "plots"), step=True, only_first_row=True)
    plot_metrics_with_grid(distance_meta_summary, os.path.join(dir_path, "plots"), only_first_row=True)

    # -- plot with baseline
    distance_meta_summary, step_meta_summary = meta_summarize(meta_eval | test_meta | baseline_meta_eval_new)

    # remove the train dataset
    if args_cli.only_test_envs:
        distance_meta_summary.pop("plot")
        step_meta_summary.pop("plot")

    # keys are the dataset names, apply the the same remove of suffix and EVAL_CFG for each of the keys
    step_meta_summary_clean = combine_dicts([
        {
            key.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
            .removesuffix(combined_suffix)
            .removesuffix("_EVAL_CFG")
            .removeprefix("plot_"): value
        }
        for key, value in step_meta_summary.items()
    ])
    # NOTE: update is technically not the correct version, instead use combine function but currently now working for distance metrics
    distance_meta_summary_clean = {}
    for key, value in distance_meta_summary.items():
        new_key = (
            key.removesuffix(f"_heightThreshold{args_cli.height_threshold}")
            .removesuffix(combined_suffix)
            .removesuffix("_EVAL_CFG")
            .removeprefix("plot_")
        )
        if new_key in distance_meta_summary_clean:
            distance_meta_summary_clean[new_key].update(value)
        else:
            distance_meta_summary_clean[new_key] = value
    # distance_meta_summary_clean = combine_dicts([{key.removesuffix(combined_suffix).removesuffix("_EVAL_CFG").removeprefix("plot_"): value} for key, value in distance_meta_summary.items()])

    plot_metrics_with_grid(step_meta_summary_clean, os.path.join(dir_path, "plots"), step=True, suffix="_baseline")
    plot_metrics_with_grid(distance_meta_summary_clean, os.path.join(dir_path, "plots"), suffix="_baseline")
    plot_metrics_with_grid(
        step_meta_summary_clean, os.path.join(dir_path, "plots"), step=True, suffix="_baseline", only_first_row=True
    )
    plot_metrics_with_grid(
        distance_meta_summary_clean, os.path.join(dir_path, "plots"), suffix="_baseline", only_first_row=True
    )

    print("Done evaluated the model")


def main():
    # load cfg
    cfg = load_cfg()

    # setup runner
    runner = fdm_runner_cfg.FDMRunner(cfg=cfg, args_cli=args_cli, eval=True)

    # run
    run_eval(runner, args_cli=args_cli)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
