

import torch

from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import fdm.utils.planner_cfg as fdm_planner_cfg
import fdm.utils.runner_cfg as fdm_runner_cfg
from fdm.env_cfg.env_cfg_depth import CAMERA_SIM_RESOLUTION_DECREASE_FACTOR
from fdm.utils import FDMPlanner, FDMRunner


def runner_cfg_init(args_cli) -> fdm_runner_cfg.FDMRunnerCfg:
    # setup runner
    if args_cli.env == "lidar":
        cfg = fdm_runner_cfg.RunnerLidarCfg()
    elif args_cli.env == "depth":
        # cfg = fdm_runner_cfg.RunnerDepthCfg()
        # cfg = fdm_runner_cfg.RunnerPerceptiveDepthCfg()
        cfg = fdm_runner_cfg.RunnerPerceptiveDepthFlatCfg()
        # cfg = fdm_runner_cfg.RunnerPreTrainedPerceptiveDepthCfg()
    elif args_cli.env == "height":
        # cfg = fdm_runner_cfg.RunnerPerceptiveHeightCfg()
        # cfg = fdm_runner_cfg.RunnerPerceptiveLargeHeightCfg()
        cfg = fdm_runner_cfg.RunnerAllPreTrainedPerceptiveHeightCfg()
        # cfg = fdm_runner_cfg.RunnerAllPreTrainedPerceptiveHeightSingleStepCfg()
        # cfg = fdm_runner_cfg.RunnerAllPreTrainedPerceptiveHeightSingleStepHeightAdjustCfg()
    else:
        raise ValueError(f"Unknown environment {args_cli.env}")

    return cfg


def cfg_modifier_pre_init(
    cfg: fdm_runner_cfg.FDMRunnerCfg | fdm_planner_cfg.FDMPlannerCfg, args_cli
) -> fdm_runner_cfg.FDMRunnerCfg | fdm_planner_cfg.FDMPlannerCfg:
    # add noise to observations
    if hasattr(args_cli, "noise") and args_cli.noise:
        cfg.env_cfg.observations.fdm_obs_proprioception.projected_gravity.noise = Unoise(n_min=-0.05, n_max=0.05)
        cfg.env_cfg.observations.fdm_obs_proprioception.base_lin_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        cfg.env_cfg.observations.fdm_obs_proprioception.base_ang_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_torque.noise = Unoise(n_min=-0.1, n_max=0.1)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx0.noise = Unoise(n_min=-1.5, n_max=1.5)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx0.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx2.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_pos_error_idx4.noise = Unoise(n_min=-0.01, n_max=0.01)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx2.noise = Unoise(n_min=-1.5, n_max=1.5)
        cfg.env_cfg.observations.fdm_obs_proprioception.joint_vel_idx4.noise = Unoise(n_min=-1.5, n_max=1.5)

    # change recursive units to S4RNN
    if hasattr(args_cli, "S4RNN") and args_cli.S4RNN:
        obs_proprioceptive_encoder_params = cfg.model_cfg.state_obs_proprioception_encoder.to_dict()
        obs_proprioceptive_encoder_params.pop("type")
        obs_proprioceptive_encoder_params.pop("bias")
        cfg.model_cfg.state_obs_proprioception_encoder = cfg.model_cfg.S4RNNConfig(**obs_proprioceptive_encoder_params)
        recurrence_params = cfg.model_cfg.recurrence.to_dict()
        recurrence_params.pop("type")
        recurrence_params.pop("bias")
        cfg.model_cfg.recurrence = cfg.model_cfg.S4RNNConfig(**recurrence_params)

    return cfg


def env_modifier_post_init(entity: FDMRunner | FDMPlanner, args_cli):
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
