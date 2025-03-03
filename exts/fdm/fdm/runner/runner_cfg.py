# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import fdm.env_cfg as fdm_env_cfg
import fdm.mdp as mdp
import fdm.model as fdm_model_cfg
from fdm import VEL_RANGE_X, VEL_RANGE_Y, VEL_RANGE_YAW
from fdm.agents import AgentCfg, MixedAgentCfg, SamplingPlannerAgentCfg, TimeCorrelatedCommandTrajectoryAgentCfg
from fdm.data_buffers import ReplayBufferCfg

from .trainer import TrainerBaseCfg


@configclass
class FDMRunnerCfg:
    model_cfg: fdm_model_cfg.FDMBaseModelCfg = MISSING
    """Model config class"""
    env_cfg: fdm_env_cfg.FDMCfg = MISSING
    """Environment config class"""
    trainer_cfg: TrainerBaseCfg = TrainerBaseCfg()
    """Trainer config class"""
    agent_cfg: AgentCfg = TimeCorrelatedCommandTrajectoryAgentCfg(
        ranges=TimeCorrelatedCommandTrajectoryAgentCfg.Ranges(
            lin_vel_x=VEL_RANGE_X,
            lin_vel_y=VEL_RANGE_Y,
            ang_vel_z=VEL_RANGE_YAW,
        ),
        linear_ratio=0.6,
        normal_ratio=0.4,
        constant_ratio=0.0,
        regular_increasing_ratio=0.0,
        max_beta=0.3,
        sigma_scale=0.3,
    )
    """Agent config class"""
    replay_buffer_cfg: ReplayBufferCfg = ReplayBufferCfg()
    """Replay buffer config class"""

    # general configurations
    collection_rounds: int = 20
    """Number of collection rounds. For each round, ``epochs`` number of epochs are trained."""

    body_regex_contact_checking: str = ".*FOOT"
    """Regex to select the bodies for contact checking.

    During data collection, reset environments that haven't been in contact with the ground for a certain number of steps.
    This regex describes which bodies are used to determine whether the robot is in contact with the ground.
    """

    def __post_init__(self):
        # set the correct length of the max torque buffer if the term exists
        if not isinstance(self.env_cfg, type(MISSING)) and self.model_cfg.hard_contact_metric == "torque":
            self.env_cfg.observations.fdm_state.hard_contact = mdp.MaxJointTorqueCfg(history_length=10)
            self.env_cfg.observations.fdm_state.hard_contact.history_length = math.ceil(
                self.model_cfg.command_timestep
                / (self.env_cfg.sim.dt * self.env_cfg.decimation * self.model_cfg.history_length)
            )
        if not isinstance(self.env_cfg, type(MISSING)) and self.model_cfg.hard_contact_metric == "contact":
            self.env_cfg.observations.fdm_state.hard_contact = mdp.MaxContactForceObsCfg(
                history_length=10,
                params={
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces", body_names=["LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"]
                    )
                },
            )
            self.env_cfg.observations.fdm_state.hard_contact.history_length = math.ceil(
                self.model_cfg.command_timestep
                / (self.env_cfg.sim.dt * self.env_cfg.decimation * self.model_cfg.history_length)
            )


###
# Baseline Config
###


@configclass
class RunnerBaselineCfg(FDMRunnerCfg):
    """Configuration for the Runner to replicate the Baseline Paper:
    Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation
    """

    trainer_cfg = TrainerBaseCfg(
        learning_rate=3e-4,
        batch_size=512,
        num_samples=45000,
        epochs=8,
        weight_decay=0,
        lr_scheduler=False,
        small_motion_ratio=None,
        experiment_name="fdm_baseline",
    )
    env_cfg: fdm_env_cfg.FDMBaselineEnvCfg = fdm_env_cfg.FDMBaselineEnvCfg()
    model_cfg: fdm_model_cfg.FDMBaselineCfg = fdm_model_cfg.FDMBaselineCfg()


###
# Config with Standard Locomotion Policy
###


@configclass
class RunnerDepthCfg(FDMRunnerCfg):
    """Configuration for the Runner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMDepthCfg = fdm_env_cfg.FDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthModelCfg = fdm_model_cfg.FDMDepthModelCfg()

    def __post_init__(self):
        """Post initialization."""
        # change exteroceptive noise model
        # self.trainer_cfg.extereoceptive_noise_model = DepthCameraNoiseCfg()
        # adjust batch size
        self.trainer_cfg.batch_size = 128


@configclass
class RunnerHeightCfg(FDMRunnerCfg):
    """Configuration for the Runner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMHeightCfg = fdm_env_cfg.FDMHeightCfg()
    model_cfg: fdm_model_cfg.FDMHeightModelMultiStepCfg = fdm_model_cfg.FDMHeightModelMultiStepCfg()

    def __post_init__(self):
        return super().__post_init__()


@configclass
class RunnerDepthHeightCfg(RunnerDepthCfg):
    """Configuration for the Runner with depth images and additional height scan observations using the percpetive locomotion policy."""

    env_cfg: fdm_env_cfg.FDMDepthCfg = fdm_env_cfg.FDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthHeightScanModelCfg = fdm_model_cfg.FDMDepthHeightScanModelCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()


###
# Data Generation with Mix of Random and Planner Generated Commands
###


@configclass
class RunnerMixedHeightCfg(RunnerHeightCfg):

    env_cfg: fdm_env_cfg.MixedFDMHeightCfg = fdm_env_cfg.MixedFDMHeightCfg()
    agent_cfg: AgentCfg = MixedAgentCfg(
        terms={
            "time_correlated": MixedAgentCfg.SubsetAgentCfg(
                agent_term=TimeCorrelatedCommandTrajectoryAgentCfg(
                    ranges=TimeCorrelatedCommandTrajectoryAgentCfg.Ranges(
                        lin_vel_x=VEL_RANGE_X,
                        lin_vel_y=VEL_RANGE_Y,
                        ang_vel_z=VEL_RANGE_YAW,
                    ),
                    linear_ratio=0.6,
                    normal_ratio=0.4,
                    constant_ratio=0.0,
                    regular_increasing_ratio=0.0,
                    max_beta=0.3,
                    sigma_scale=0.3,
                ),
                ratio=fdm_env_cfg.RANDOM_RATIO,
            ),
            "planner": MixedAgentCfg.SubsetAgentCfg(
                agent_term=SamplingPlannerAgentCfg(
                    ranges=SamplingPlannerAgentCfg.Ranges(
                        lin_vel_x=VEL_RANGE_X,
                        lin_vel_y=VEL_RANGE_Y,
                        ang_vel_z=VEL_RANGE_YAW,
                    ),
                ),
                ratio=fdm_env_cfg.PLANNER_RATIO,
            ),
        },
        horizon=10,
    )

    def __post_init__(self):
        super().__post_init__()


###
# Pre-Trained Perception and Exteroceptive Encoder
###
EXTEROCEPTIVE_PRE_TRAINING_RUN = "Jun04_19-03-33_depth_footscans_bs1024"
# PROPRIOCEPTION_PRE_TRAINING_RUN = "Jun08_19-55-23_atOncePred_noDropout_velPredCorr_bs1024"
PROPRIOCEPTION_PRE_TRAINING_RUN = "Aug06_14-42-52_atOncePred_noDropout_velPredCorr_bs1024_noBatchNorm"


# get pre-trained model dict
def get_pre_trained_model_paths(pre_training_run: str) -> dict:
    return {
        # proprioception pre-training
        "state_obs_proprioceptive_encoder": os.path.join(
            "logs/fdm/fdm_proprioception_pre_training",
            pre_training_run,
            "modelproprioception_state_encoder.pth",
        ),
        "proprioceptive_normalizer": os.path.join(
            "logs/fdm/fdm_proprioception_pre_training",
            pre_training_run,
            "modelproprioception_normalizer.pth",
        ),
        "action_encoder": os.path.join(
            "logs/fdm/fdm_proprioception_pre_training", pre_training_run, "modelaction_encoder.pth"
        ),
        "friction_predictor": os.path.join(
            "logs/fdm/fdm_proprioception_pre_training",
            pre_training_run,
            "modelfriction_predictor.pth",
        ),
    }


@configclass
class RunnerAllPreTrainedHeightCfg(RunnerHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # resume the encoder for depth images
        self.trainer_cfg.encoder_resume = get_pre_trained_model_paths(PROPRIOCEPTION_PRE_TRAINING_RUN)


@configclass
class RunnerAllPreTrainedMixedHeightCfg(RunnerMixedHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # resume the encoder for depth images
        self.trainer_cfg.encoder_resume = get_pre_trained_model_paths(PROPRIOCEPTION_PRE_TRAINING_RUN)


@configclass
class RunnerAllPreTrainedHeightSingleStepCfg(RunnerAllPreTrainedHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMHeightModelSingleStepCfg = fdm_model_cfg.FDMHeightModelSingleStepCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class RunnerAllPreTrainedMixedHeightSingleStepCfg(RunnerAllPreTrainedMixedHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMHeightModelSingleStepCfg = fdm_model_cfg.FDMHeightModelSingleStepCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class RunnerAllPreTrainedHeightSingleStepHeightAdjustCfg(RunnerAllPreTrainedHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMModelVelocitySingleStepHeightAdjustCfg = (
        fdm_model_cfg.FDMModelVelocitySingleStepHeightAdjustCfg()
    )

    def __post_init__(self):
        super().__post_init__()

        # adjust the height scan pattern
        self.env_cfg.scene.env_sensor.pattern_cfg.resolution = 0.05
        self.env_cfg.scene.env_sensor.pattern_cfg.size = (4.55, 5.95)
        self.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"] = (120, 92)

        # adjust the model parameters
        # TODO: should be able to set them
        # self.model_cfg.height_scan_res = self.env_cfg.scene.env_sensor.pattern_cfg.resolution
        # self.model_cfg.height_scan_shape = list(self.env_cfg.observations.fdm_obs_exteroceptive.env_sensor.params["shape"])


@configclass
class RunnerAllPreTrainedDepthCfg(RunnerDepthCfg):
    """Configuration for the Runner with depth images and additional height scan observations using the percpetive locomotion policy."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # resume the encoder for depth images
        self.trainer_cfg.encoder_resume = {
            # exteroceptive pre-training
            "obs_exteroceptive_encoder": os.path.join(
                "logs/fdm/fdm_exteroceptive_pre_training", EXTEROCEPTIVE_PRE_TRAINING_RUN, "modelimage_encoder.pth"
            ),
            "add_obs_exteroceptive_encoder": os.path.join(
                "logs/fdm/fdm_exteroceptive_pre_training", EXTEROCEPTIVE_PRE_TRAINING_RUN, "modelfoot_scan_encoder.pth"
            ),
            # proprioception pre-training
            "state_obs_proprioceptive_encoder": os.path.join(
                "logs/fdm/fdm_proprioception_pre_training",
                PROPRIOCEPTION_PRE_TRAINING_RUN,
                "modelproprioception_state_encoder.pth",
            ),
            "proprioceptive_normalizer": os.path.join(
                "logs/fdm/fdm_proprioception_pre_training",
                PROPRIOCEPTION_PRE_TRAINING_RUN,
                "modelproprioception_normalizer.pth",
            ),
            "action_encoder": os.path.join(
                "logs/fdm/fdm_proprioception_pre_training", PROPRIOCEPTION_PRE_TRAINING_RUN, "modelaction_encoder.pth"
            ),
            "friction_predictor": os.path.join(
                "logs/fdm/fdm_proprioception_pre_training",
                PROPRIOCEPTION_PRE_TRAINING_RUN,
                "modelfriction_predictor.pth",
            ),
        }
