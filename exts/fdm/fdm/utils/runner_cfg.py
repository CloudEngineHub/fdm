

from __future__ import annotations

import os
from dataclasses import MISSING

from omni.isaac.lab_assets import ISAACLAB_ASSETS_EXT_DIR

from omni.isaac.lab.utils import configclass

import fdm.env_cfg as fdm_env_cfg
import fdm.model as fdm_model_cfg
from fdm.agents import AgentCfg, TimeCorrelatedCommandTrajectoryAgentCfg

from ..sensor_noise_models import DepthCameraNoiseCfg
from .replay_buffer_cfg import ReplayBufferCfg
from .trainer_cfg import TrainerBaseCfg


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
            # FIXME: get complete y velocity range again
            lin_vel_x=(0.2, 1.5),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-1.0, 1.0),  # rad/s
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
    collection_rounds: int = 30
    """Number of collection rounds. For each round, ``epochs`` number of epochs are trained."""


@configclass
class RunnerLidarCfg(FDMRunnerCfg):
    """Configuration for the Runner with LiDAR data."""

    env_cfg: fdm_env_cfg.FDMLidarCfg = fdm_env_cfg.FDMLidarCfg()
    model_cfg: fdm_model_cfg.FDMLidarModelCfg = fdm_model_cfg.FDMLidarModelCfg()


@configclass
class RunnerDepthCfg(FDMRunnerCfg):
    """Configuration for the Runner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMDepthCfg = fdm_env_cfg.FDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthModelCfg = fdm_model_cfg.FDMDepthModelCfg()

    def __post_init__(self):
        """Post initialization."""
        # change extereoceptive observation precision to float16
        self.trainer_cfg.exteroceptive_obs_precision = "float16"
        # change exteroceptive noise model
        # TODO: add noise once training is stable
        # self.trainer_cfg.extereoceptive_noise_model = DepthCameraNoiseCfg()
        # adjust batch size
        self.trainer_cfg.batch_size = 128


@configclass
class RunnerHeightCfg(FDMRunnerCfg):
    """Configuration for the Runner with height scanner data."""

    env_cfg: fdm_env_cfg.FDMHeightCfg = fdm_env_cfg.FDMHeightCfg()
    model_cfg: fdm_model_cfg.FDMHeightModelMultiStepCfg = fdm_model_cfg.FDMHeightModelMultiStepCfg()

    def __post_init__(self):
        """Post initialization."""
        # adjust proprioceptive observation size
        self.trainer_cfg.num_samples = 70000
        self.trainer_cfg.batch_size = 512


##
# Configs with the perceptive locomotion policy
##


@configclass
class RunnerPerceptiveHeightCfg(FDMRunnerCfg):
    """Configuration for the Runner with height scanner data."""

    env_cfg: fdm_env_cfg.PerceptiveFDMHeightCfg = fdm_env_cfg.PerceptiveFDMHeightCfg()
    model_cfg: fdm_model_cfg.FDMHeightModelMultiStepCfg = fdm_model_cfg.FDMHeightModelMultiStepCfg()


@configclass
class RunnerPerceptiveLargeHeightCfg(FDMRunnerCfg):
    """Configuration for the Runner with height scanner data."""

    env_cfg: fdm_env_cfg.PerceptiveFDMHeightCfg = fdm_env_cfg.PerceptiveFDMHeightCfg()
    model_cfg: fdm_model_cfg.FDMLargeHeightModelCfg = fdm_model_cfg.FDMLargeHeightModelCfg()

    def __post_init__(self):
        """Post initialization."""
        # increase height scanner pattern
        self.env_cfg.scene.env_sensor.pattern_cfg.resolution = 0.05
        self.env_cfg.scene.env_sensor.pattern_cfg.size = [10.0, 10.0]


@configclass
class RunnerPerceptiveDepthCfg(RunnerDepthCfg):
    """Configuration for the Runner with depth images and additional height scan observations using the percpetive locomotion policy."""

    env_cfg: fdm_env_cfg.PerceptiveFDMDepthCfg = fdm_env_cfg.PerceptiveFDMDepthCfg()
    model_cfg: fdm_model_cfg.FDMDepthHeightScanModelCfg = fdm_model_cfg.FDMDepthHeightScanModelCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()


@configclass
class RunnerPerceptiveDepthFlatCfg(RunnerPerceptiveDepthCfg):
    """Configuration for the Runner with depth images and additional height scan observations using the percpetive locomotion policy."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        self.env_cfg.observations.fdm_obs_exteroceptive.concatenate_dim = 1
        # concatenate depth images along their vertical axis, adjust encoder input size
        self.model_cfg.obs_exteroceptive_encoder.input_channels = 1
        self.model_cfg.obs_exteroceptive_encoder.downsample_MLP.input = 1792


###
# Pre-Trained Perception and Exteroceptive Encoder
###
EXTEROCEPTIVE_PRE_TRAINING_RUN = "Jun04_19-03-33_depth_footscans_bs1024"
PROPRIOCEPTION_PRE_TRAINING_RUN = "Jun08_19-55-23_atOncePred_noDropout_velPredCorr_bs1024"


@configclass
class RunnerAllPreTrainedPerceptiveHeightCfg(RunnerPerceptiveHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # resume the encoder for depth images
        self.trainer_cfg.encoder_resume = {
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


@configclass
class RunnerAllPreTrainedPerceptiveHeightSingleStepCfg(RunnerAllPreTrainedPerceptiveHeightCfg):
    """Configuration for the Runner with height scan observations using the percpetive locomotion policy.

    Select the model that predicts one step and feedbacks the collision, energy and correction velocity back to the
    input of the recurrent layer."""

    model_cfg: fdm_model_cfg.FDMHeightModelSingleStepCfg = fdm_model_cfg.FDMHeightModelSingleStepCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class RunnerAllPreTrainedPerceptiveHeightSingleStepHeightAdjustCfg(RunnerAllPreTrainedPerceptiveHeightCfg):
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
class RunnerAllPreTrainedPerceptiveDepthCfg(RunnerPerceptiveDepthCfg):
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


###
# Pre-Trained PerceptNet
###


@configclass
class RunnerPreTrainedPerceptNetPerceptiveDepthCfg(RunnerPerceptiveDepthCfg):
    """Configuration for the Runner with depth images and additional height scan observations using the percpetive locomotion policy."""

    model_cfg: fdm_model_cfg.FDMDepthPreTrainedModelCfg = fdm_model_cfg.FDMDepthPreTrainedModelCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # resume the encoder for depth images
        self.trainer_cfg.encoder_resume = {
            "obs_exteroceptive_encoder": ISAACLAB_ASSETS_EXT_DIR + "/Encoders/perceptnet_emb256_low_resolution_SD.pt"
        }
