# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import fdm.env_cfg.robot_cfg as env_robot_cfg
import fdm.mdp as mdp
import fdm.model.robot_cfg as model_robot_cfg
import fdm.planner as fdm_planner
import fdm.runner as fdm_runner
from fdm import LARGE_UNIFIED_HEIGHT_SCAN, TOTAL_TIME_PREDICTION_HORIZON
from fdm.env_cfg.env_cfg_base import TERRAIN_ANALYSIS_CFG
from fdm.env_cfg.env_cfg_depth import CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
from fdm.env_cfg.env_cfg_height import OccludedObsExteroceptiveCfg
from fdm.env_cfg.env_cfg_heuristic_planner import HeuristicsOccludedObsExteroceptiveCfg
from fdm.mdp.noise import MissingPatchNoiseCfg, PerlinNoiseCfg


def runner_cfg_init(args_cli) -> fdm_runner.FDMRunnerCfg:
    # setup runner
    if args_cli.env == "baseline":
        cfg = fdm_runner.RunnerBaselineCfg()
    elif args_cli.env == "depth":
        cfg = fdm_runner.RunnerDepthCfg()
        # cfg = fdm_runner.RunnerDepthFlatCfg()
        # cfg = fdm_runner.RunnerPreTrainedDepthCfg()
    elif args_cli.env == "height":
        # cfg = fdm_runner.RunnerHeightCfg()
        cfg = fdm_runner.RunnerMixedHeightCfg()
        # cfg = fdm_runner.RunnerAllPreTrainedHeightCfg()
        # cfg = fdm_runner.RunnerAllPreTrainedMixedHeightCfg()
        # cfg = fdm_runner.RunnerAllPreTrainedHeightSingleStepCfg()
        # cfg = fdm_runner.RunnerAllPreTrainedHeightSingleStepHeightAdjustCfg()
    else:
        raise ValueError(f"Unknown environment {args_cli.env}")

    return cfg


def robot_changes(
    cfg: fdm_runner.FDMRunnerCfg | fdm_planner.FDMPlannerCfg, args_cli
) -> fdm_runner.FDMRunnerCfg | fdm_planner.FDMPlannerCfg:

    # Tytan
    if args_cli.robot.lower() == "tytan":
        print("[INFO] Tytan")
        cfg.env_cfg = env_robot_cfg.tytan_env(cfg.env_cfg)
        cfg.model_cfg = model_robot_cfg.tytan_model(cfg.model_cfg)
        # NOTE: SHANK is selected as the collision shape of the foot is combined with the one of the shank for increased
        #       stability as the mass of the foot is very small. This was necessary in IsaacGym, possible that in
        #       IsaacLab the collision shape can be directly added to the foot.
        cfg.body_regex_contact_checking = ".*SHANK"

        # currently cancel out all the pre-trained models
        if isinstance(cfg, fdm_runner.FDMRunnerCfg):
            cfg.trainer_cfg.encoder_resume = None

        # FIXME: remove when test datasets are available
        cfg.trainer_cfg.test_datasets = None

    # Tytan quiet
    elif args_cli.robot.lower() == "tytan_quiet":
        print("[INFO] Tytan quiet")
        cfg.env_cfg = env_robot_cfg.tytan_env(cfg.env_cfg, quiet=True)
        cfg.model_cfg = model_robot_cfg.tytan_model(cfg.model_cfg)
        # NOTE: SHANK is selected as the collision shape of the foot is combined with the one of the shank for increased
        #       stability as the mass of the foot is very small. This was necessary in IsaacGym, possible that in
        #       IsaacLab the collision shape can be directly added to the foot.
        cfg.body_regex_contact_checking = ".*SHANK"

        # currently cancel out all the pre-trained models
        if isinstance(cfg, fdm_runner.FDMRunnerCfg):
            cfg.trainer_cfg.encoder_resume = None

        # FIXME: remove when test datasets are available
        cfg.trainer_cfg.test_datasets = None

    # Anymal on wheels (AOW)
    elif args_cli.robot.lower() == "aow":
        print("[INFO] Anymal on wheels (AOW)")
        cfg.env_cfg = env_robot_cfg.aow_env(cfg.env_cfg, args_cli.env)
        cfg.model_cfg = model_robot_cfg.aow_model(cfg.model_cfg, args_cli.env)
        cfg.body_regex_contact_checking = ".*WHEEL_L"

        # remove the thighs from the collision checking
        cfg.env_cfg.observations.fdm_state.base_collision.params["sensor_cfg"].body_names = "base"
        cfg.env_cfg.terminations.base_contact.params["sensor_cfg"].body_names = "base"

        # currently cancel out all the pre-trained models
        if isinstance(cfg, fdm_runner.FDMRunnerCfg):
            cfg.trainer_cfg.encoder_resume = None

        # FIXME: remove when test datasets are available
        cfg.trainer_cfg.test_datasets = None

    # ANYmal with perceptive walking policy
    elif args_cli.robot.lower() == "anymal_perceptive":
        print("[INFO] ANYmal with perceptive walking policy")
        cfg.env_cfg = env_robot_cfg.anymal_perceptive(cfg.env_cfg)

        # remove the added cpg state
        if args_cli.env == "baseline":
            cfg.env_cfg.observations.fdm_obs_proprioception.cpg_state = None
        else:
            cfg.model_cfg = model_robot_cfg.anymal_perceptive_model(cfg.model_cfg)

    # Standard ANYmal
    elif args_cli.robot.lower() == "anymal":
        print("[INFO] Standard ANYmal")
        return cfg

    else:
        raise ValueError(f"Unknown robot {args_cli.robot}")

    return cfg


def cfg_modifier_pre_init(  # noqa: C901
    cfg: fdm_runner.FDMRunnerCfg | fdm_planner.FDMPlannerCfg, args_cli, dataset_collecton: bool = False
) -> fdm_runner.FDMRunnerCfg | fdm_planner.FDMPlannerCfg:
    """
    Modify the configuration before the initialization of the runner or planner.

    Args:
        cfg: The configuration object.
        args_cli: The command line arguments.
        dataset_collecton: If the configuration is used for dataset collection or training/ eval.
    """
    # add occlusions to the exteroceptive observation term
    if hasattr(args_cli, "occlusions") and args_cli.occlusions:
        print("[INFO] Adding occlusions to the exteroceptive observation term")
        if args_cli.env == "height":
            cfg.env_cfg.observations.fdm_obs_exteroceptive = OccludedObsExteroceptiveCfg()
        elif args_cli.env == "heuristic":
            cfg.env_cfg.observations.fdm_obs_exteroceptive = HeuristicsOccludedObsExteroceptiveCfg()
        else:
            print("[WARNING] Occlusions are not supported for the current environment")

    # add noise to observations
    if hasattr(args_cli, "noise") and args_cli.noise:

        cfg.env_cfg.observations.fdm_obs_proprioception.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        cfg.env_cfg.observations.fdm_obs_proprioception.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)

        if args_cli.env != "baseline":
            cfg.env_cfg.observations.fdm_obs_proprioception.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_torque.noise = Unoise(n_min=-0.1, n_max=0.1)
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)

            if args_cli.robot.lower() == "anymal" or args_cli.robot.lower() == "anymal_perceptive":
                cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx0.noise = Unoise(n_min=-1.5, n_max=1.5)
                cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx0.noise = Unoise(
                    n_min=-0.01, n_max=0.01
                )
                cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx2.noise = Unoise(
                    n_min=-0.01, n_max=0.01
                )
                cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx4.noise = Unoise(
                    n_min=-0.01, n_max=0.01
                )
                cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx2.noise = Unoise(n_min=-1.5, n_max=1.5)
                cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx4.noise = Unoise(n_min=-1.5, n_max=1.5)

            if args_cli.env == "height" and hasattr(cfg.env_cfg.observations.fdm_obs_exteroceptive, "env_sensor"):
                # cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.noise = MissingPatchNoiseCfg(fill_value=cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.clip[1], apply_noise=PerlinNoiseCfg(), clip_values=cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.clip)
                cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.noise = MissingPatchNoiseCfg(
                    fill_value=None,
                    apply_noise=PerlinNoiseCfg(),
                    clip_values=cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.clip,
                )
                cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.clip = None

        else:
            cfg.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.noise = Unoise(n_min=-0.1, n_max=0.1)

        # enable noise augmentation in the trainer
        if isinstance(cfg, fdm_runner.FDMRunnerCfg) and not dataset_collecton:
            cfg.trainer_cfg.apply_noise = True
            cfg.env_cfg.observations.fdm_obs_proprioception.enable_corruption = False
            cfg.env_cfg.observations.fdm_obs_exteroceptive.enable_corruption = False
        else:
            cfg.env_cfg.observations.fdm_obs_proprioception.enable_corruption = True
            cfg.env_cfg.observations.fdm_obs_exteroceptive.enable_corruption = True

    # reduced observation space
    if hasattr(args_cli, "reduced_obs") and args_cli.reduced_obs and args_cli.env != "baseline":
        print("[INFO] Reduced observation space")
        # remove the friction from the state space
        cfg.model_cfg.exclude_state_idx_from_input = [-4, -3, -2, -1]  # size 4
        cfg.model_cfg.state_obs_proprioception_encoder.input_size -= 4

        if args_cli.robot.lower() == "anymal" or args_cli.robot.lower() == "anymal_perceptive":
            # reduce the proprioceptive observation space
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx0 = None  # size 12
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx2 = None  # size 12
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx4 = None  # size 12
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx2 = None  # size 12
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx4 = None  # size 12
            # adjust the size of the layers
            cfg.model_cfg.state_obs_proprioception_encoder.input_size -= 60
            cfg.model_cfg.empirical_normalization_dim -= 60

            if hasattr(cfg.env_cfg.observations.fdm_obs_proprioception, "cpg_state"):
                cfg.env_cfg.observations.fdm_obs_proprioception.cpg_state = None
                cfg.model_cfg.state_obs_proprioception_encoder.input_size -= 8
                cfg.model_cfg.empirical_normalization_dim -= 8

        if hasattr(args_cli, "remove_torque") and args_cli.remove_torque:
            cfg.env_cfg.observations.fdm_obs_proprioception.joint_torque = None
            if args_cli.robot.lower() == "aow":
                cfg.model_cfg.state_obs_proprioception_encoder.input_size -= 16
                cfg.model_cfg.empirical_normalization_dim -= 16
            else:
                cfg.model_cfg.state_obs_proprioception_encoder.input_size -= 12
                cfg.model_cfg.empirical_normalization_dim -= 12

    # change the tests datasets
    if isinstance(cfg, fdm_runner.FDMRunnerCfg) and cfg.trainer_cfg.test_datasets is not None:
        if args_cli.env != "baseline":
            if hasattr(args_cli, "reduced_obs") and args_cli.reduced_obs:
                cfg.trainer_cfg.test_datasets = [
                    f"{dataset[:-4]}_reducedObs.pkl" for dataset in cfg.trainer_cfg.test_datasets
                ]
            if hasattr(args_cli, "remove_torque") and args_cli.remove_torque:
                cfg.trainer_cfg.test_datasets = [
                    f"{dataset[:-4]}_noTorque.pkl" for dataset in cfg.trainer_cfg.test_datasets
                ]
            # NOTE: when noise then it should also have occlusions
            if hasattr(args_cli, "noise") and args_cli.noise:
                cfg.trainer_cfg.test_datasets = [
                    f"{dataset[:-4]}_noise.pkl" for dataset in cfg.trainer_cfg.test_datasets
                ]
            elif hasattr(args_cli, "occlusion") and args_cli.occlusion:
                cfg.trainer_cfg.test_datasets = [
                    f"{dataset[:-4]}_occlusions.pkl" for dataset in cfg.trainer_cfg.test_datasets
                ]
        else:
            if hasattr(args_cli, "noise") and args_cli.noise:
                cfg.trainer_cfg.test_datasets = [
                    f"{dataset[:-4]}_noise.pkl" for dataset in cfg.trainer_cfg.test_datasets
                ]
            cfg.trainer_cfg.test_datasets = [
                f"{dataset[:-4]}_baseline.pkl" for dataset in cfg.trainer_cfg.test_datasets
            ]

        # FIXME: remove that once the test datasets are available
        if LARGE_UNIFIED_HEIGHT_SCAN:
            raise ValueError("Test datasets are not available for the large unified height scan")
            cfg.trainer_cfg.test_datasets = [
                f"{dataset[:-4]}_largeUnifiedHeightScan.pkl" for dataset in cfg.trainer_cfg.test_datasets
            ]

        if hasattr(args_cli, "height_threshold") and args_cli.height_threshold is not None:
            cfg.trainer_cfg.test_datasets = [
                (
                    f"{dataset[:-4]}_heightThreshold{args_cli.height_threshold}.pkl"
                    if "stairs" in dataset.lower()
                    else dataset
                )
                for dataset in cfg.trainer_cfg.test_datasets
            ]

    # adjust given a timestamp
    if hasattr(args_cli, "timestamp") and args_cli.timestamp:
        print(f"[INFO] Adjusting the model for a timestamp of {args_cli.timestamp}")
        cfg.model_cfg.command_timestep = args_cli.timestamp
        cfg.model_cfg.prediction_horizon = int(TOTAL_TIME_PREDICTION_HORIZON / args_cli.timestamp)
        # adjust the length of the replay buffer
        cfg.replay_buffer_cfg.trajectory_length = int(15 * cfg.model_cfg.prediction_horizon)
        if cfg.model_cfg.state_predictor.output != 3:
            # model is not a single step model, adjust the output size
            cfg.model_cfg.state_predictor.output = cfg.model_cfg.prediction_horizon * 3
            cfg.model_cfg.collision_predictor.output = cfg.model_cfg.prediction_horizon
            cfg.model_cfg.energy_predictor.output = cfg.model_cfg.prediction_horizon
            # adjust input sizes of the predictor networks
            cfg.model_cfg.state_predictor.input = (
                cfg.model_cfg.recurrence.hidden_size * cfg.model_cfg.prediction_horizon
            )
            cfg.model_cfg.collision_predictor.input = (
                cfg.model_cfg.recurrence.hidden_size * cfg.model_cfg.prediction_horizon
            )
            cfg.model_cfg.energy_predictor.input = (
                cfg.model_cfg.recurrence.hidden_size * cfg.model_cfg.prediction_horizon
            )

    # adjust the resume of the model
    if isinstance(cfg, fdm_runner.FDMRunnerCfg) and hasattr(args_cli, "resume") and args_cli.resume:
        print(f"[INFO] Adjusting the model for a resume of {args_cli.resume}")

        cfg.trainer_cfg.resume = True
        cfg.trainer_cfg.load_run = args_cli.resume
        cfg.trainer_cfg.load_checkpoint = "model.pth"

        # set the encoder resume to None
        cfg.trainer_cfg.encoder_resume = None
        cfg.trainer_cfg.encoder_resume_add_to_optimizer = False

    # vary friction linearly for each robot
    if hasattr(args_cli, "friction") and args_cli.friction:
        cfg.env_cfg.events.physics_material.params["static_friction_range"] = (1e-3, 0.4)
        if hasattr(args_cli, "regular") and args_cli.regular:
            cfg.env_cfg.events.physics_material.params["regular"] = True

    # adjust the number of samples in the terrain analysis
    if hasattr(args_cli, "terrain_analysis_points") and args_cli.terrain_analysis_points is not None:
        if isinstance(cfg.env_cfg.commands.command, mdp.MixedCommandCfg):
            cfg.env_cfg.commands.command.terms["planner"].command_term.terrain_analysis.sample_points = (
                args_cli.terrain_analysis_points
            )
        if isinstance(cfg.env_cfg.events.reset_base.func, mdp.TerrainAnalysisRootReset):
            cfg.env_cfg.events.reset_base.func.cfg.sample_points = args_cli.terrain_analysis_points
        TERRAIN_ANALYSIS_CFG.sample_points = args_cli.terrain_analysis_points
    return cfg


def env_modifier_post_init(entity: fdm_runner.FDMRunner | fdm_planner.FDMPlanner, args_cli):
    if hasattr(args_cli, "env") and args_cli.env == "depth":
        # left and right zed x mini add camera intrinsic
        camera_intrinsic = torch.tensor([[380.0831, 0.0, 467.7916], [0.0, 380.0831, 262.0532], [0.0, 0.0, 1.0]])
        camera_intrinsic[0, 2] = camera_intrinsic[0, 2] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        camera_intrinsic[1, 2] = camera_intrinsic[1, 2] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        camera_intrinsic[0, 0] = camera_intrinsic[0, 0] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        camera_intrinsic[1, 1] = camera_intrinsic[1, 1] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        # set the intrinsic matrix
        entity.env.scene.sensors["env_sensor_right"].set_intrinsic_matrices(
            matrices=camera_intrinsic.repeat(entity.env.num_envs, 1, 1)
        )
        entity.env.scene.sensors["env_sensor_left"].set_intrinsic_matrices(
            matrices=camera_intrinsic.repeat(entity.env.num_envs, 1, 1)
        )

        # front and rear zed x add camera intrinsic
        camera_intrinsic = torch.tensor([[369.7771, 0.0, 489.9926], [0.0, 369.7771, 275.9385], [0.0, 0.0, 1.0]])
        camera_intrinsic[0, 2] = camera_intrinsic[0, 2] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        camera_intrinsic[1, 2] = camera_intrinsic[1, 2] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        camera_intrinsic[0, 0] = camera_intrinsic[0, 0] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        camera_intrinsic[1, 1] = camera_intrinsic[1, 1] / CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
        entity.env.scene.sensors["env_sensor"].set_intrinsic_matrices(
            matrices=camera_intrinsic.repeat(entity.env.num_envs, 1, 1)
        )

    return entity


def planner_cfg_init(args_cli) -> fdm_planner.FDMPlannerCfg:
    # setup runner
    if args_cli.env == "depth":
        cfg = fdm_planner.PlannerDepthCfg()
    elif args_cli.env == "height":
        cfg = fdm_planner.PlannerHeightCfg()
        # cfg = fdm_planner.PlannerHeightSingleStepCfg()
        # cfg = fdm_planner.PlannerHeightSingleStepHeightAdjustCfg()
    elif args_cli.env == "baseline":
        cfg = fdm_planner.PlannerBaselineCfg()
    elif args_cli.env == "heuristic":
        cfg = fdm_planner.PlannerHeuristicCfg()
    elif args_cli.env == "rmp":  # noqa: R506
        raise NotImplementedError("RMP planner is not yet supported")
    else:
        raise ValueError(f"Unknown/ Not yet supported environment {args_cli.env}")

    return cfg
