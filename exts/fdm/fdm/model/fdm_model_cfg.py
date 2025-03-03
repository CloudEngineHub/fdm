# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from isaaclab.utils import configclass

from fdm import LARGE_UNIFIED_HEIGHT_SCAN

from .fdm_model import (
    FDMModel,
    FDMModelVelocityMultiStep,
    FDMModelVelocitySingleStep,
    FDMModelVelocitySingleStepHeightAdjust,
)
from .model_base_cfg import BaseModelCfg


@configclass
class FDMBaseModelCfg(BaseModelCfg):
    """
    Model part configs
    """

    class_type: type[FDMModel] = FDMModel

    ###
    # Encoder
    ###

    # state and proprioception encoder
    state_obs_proprioception_encoder: BaseModelCfg.GRUConfig | BaseModelCfg.MLPConfig = BaseModelCfg.GRUConfig(
        input_size=142, hidden_size=64, num_layers=2, dropout=0.0
    )
    # env encoder
    obs_exteroceptive_encoder: BaseModelCfg.MLPConfig | BaseModelCfg.CNNConfig | None = None
    # command encoder
    action_encoder: BaseModelCfg.MLPConfig | None = BaseModelCfg.MLPConfig(
        input=3, output=16, shape=None, dropout=0.2, batchnorm=False, activation="LeakyReLU"  # , batchnorm=True
    )
    # additional env encoder
    add_obs_exteroceptive_encoder: BaseModelCfg.MLPConfig | None = None

    ###
    # Prediction
    ###

    # recurrent layers
    recurrence: BaseModelCfg.GRUConfig | BaseModelCfg.MLPConfig = BaseModelCfg.GRUConfig(
        input_size=16, hidden_size=160, num_layers=2, dropout=0.2
    )
    # state predictor
    state_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=160, output=3, shape=[64, 32, 16], dropout=0.2, batchnorm=False, activation="LeakyReLU"  # batchnorm=True
    )
    # collision predictior
    collision_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=160, output=1, shape=[64, 32, 16], dropout=0.2, batchnorm=False, activation="LeakyReLU"  # batchnorm=True
    )
    # energy predictor
    energy_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=160, output=1, shape=[64], dropout=0.2, batchnorm=False, activation="LeakyReLU"  # batchnorm=True
    )
    # friction predictor
    friction_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=64, output=4, shape=[32], activation="LeakyReLU"
    )

    # empirical normalization
    empirical_normalization_dim: int = 132
    """The dimension of the empirical normalization.

    Should be applied on the proprioception, i.e. the state_obs_proprioception_encoder input size - state_dim.
    """

    # exclude part of the state from the input but keep as label
    exclude_state_idx_from_input: list[int] | None = None
    """List of indices of the state that should be excluded from the input but kept as label."""

    # hard contact prediction observatiion metric
    hard_contact_metric: Literal["contact", "torque", "energy"] = "energy"
    """Metric used for the hard contact prediction."""

    """
    Loss-Parameters
    """

    loss_weights: dict[str, float] = {
        "collision": 2.0,
        "position": 1.7,
        "heading": 1.7,
        "velocity": 0.0,
        "acceleration": 0.0,
        "stop": 1.0,
        "energy": 1.0,
    }
    """Loss weights for the different terms."""
    progress_scaling: dict[str, bool] = {
        "collision": False,
        "position": False,
        "heading": False,
        "velocity": False,
        "acceleration": False,
        "stop": False,
        "energy": False,
    }
    """Whether to scale the loss with the progress of the learning."""
    unified_failure_prediction: bool = False
    """Summarize the failure predicition of each command into a unified value that is used for the cost."""
    weight_inverse_distance: bool = False
    """Weight the position loss with the inverse distance to the goal.

    Done to focus the model on smaller distances. Is multiplied by the mean of the norm to keep the general loss scale
    similar to the unscaled one. The loss is then calculated as:

    .. math:: MSE(pos, pos_gt) * 1 / ||pos - pos_gt|| * mean(||pos - pos_gt||)

    Default is False."""
    pos_loss_norm: Literal["mse", "l1", "l2"] = "mse"
    """Norm used for the loss calculation.

    L2 smoothes the path more whereas l1 captures sudden changes better.
    Default is l2 which is the mean squared error."""
    prediction_horizon: int = 10
    """Prediction horizon for the model."""
    command_timestep: float = 0.5
    """Timestep between new commands are sampled in sec."""
    history_length: int = 10
    """Number of robot states history included in the state as part of the .

    The states are recorded at a frequency of ``command_timestep / history_length``.
    """
    collision_threshold: float = 0.5
    """Collision threshold for the collision prediction. Default is 0.5."""
    eval_distance_interval: float = 1.0
    """Distance interval for the evaluation metrics."""
    zero_collision_actions: bool = False
    """set actions to zero if predicted that they are in collision"""


###
# Height Scan Models
###


@configclass
class FDMHeightModelMultiStepCfg(FDMBaseModelCfg):

    class_type: type[FDMModelVelocityMultiStep] = FDMModelVelocityMultiStep

    obs_exteroceptive_encoder = FDMBaseModelCfg.CNNConfig(
        in_channels=1,
        out_channels=[32, 64, 128, 256] if not LARGE_UNIFIED_HEIGHT_SCAN else [32, 64, 128, 128, 512],
        stride=[1, 2, 2, 2] if not LARGE_UNIFIED_HEIGHT_SCAN else [1, 1, 2, 2, 2],
        kernel_size=(
            [(7, 7), (3, 3), (3, 3), (3, 3)]
            if not LARGE_UNIFIED_HEIGHT_SCAN
            else [(7, 7), (3, 3), (3, 3), (3, 3), (3, 3)]
        ),
        max_pool=[True, False, False, False] if not LARGE_UNIFIED_HEIGHT_SCAN else [True, False, False, False, False],
        activation="LeakyReLU",
        batchnorm=False,
        avg_pool=LARGE_UNIFIED_HEIGHT_SCAN,
        flatten=not LARGE_UNIFIED_HEIGHT_SCAN,
    )

    def __post_init__(self):
        # adjust recurrent layer
        self.recurrence.input_size = (
            self.action_encoder.output  # 64
            + self.state_obs_proprioception_encoder.hidden_size  # 64
            + self.friction_predictor.output  # 4
            + 512  # height scan
        )
        # TINY-Recurrent
        self.recurrence.hidden_size = 128
        # NORMAL-Recurrent
        # self.recurrence.hidden_size = 256
        self.recurrence.dropout = 0.2
        self.state_obs_proprioception_encoder.dropout = 0.2

        # adjust output sizes to predict all timesteps at once
        self.state_predictor.output = self.state_predictor.output * self.prediction_horizon
        self.collision_predictor.output = self.prediction_horizon
        self.energy_predictor.output = self.prediction_horizon

        # adjust input sizes of the predictor networks
        self.state_predictor.input = self.recurrence.hidden_size * self.prediction_horizon
        self.collision_predictor.input = self.recurrence.hidden_size * self.prediction_horizon
        self.energy_predictor.input = self.recurrence.hidden_size * self.prediction_horizon

        # adjust shape of the predictor networks (TINY-Decoder)
        self.state_predictor.shape = [128, 64]
        self.collision_predictor.shape = [64]
        self.energy_predictor.shape = [64]
        # adjust shape of the predictor networks (NORMAL-Decoder)
        # self.state_predictor.shape = [256, 128, 64]
        # self.collision_predictor.shape = [128, 64]
        # self.energy_predictor.shape = [128, 64]


@configclass
class FDMHeightModelSingleStepCfg(FDMHeightModelMultiStepCfg):

    class_type: type[FDMModelVelocitySingleStep] = FDMModelVelocitySingleStep

    def __post_init__(self):
        super().__post_init__()

        # adjust output sizes to predict all timesteps at once
        self.state_predictor.output = 3
        self.collision_predictor.output = 1
        self.energy_predictor.output = 1

        # adjust recurrent layer
        self.recurrence.input_size += (
            self.state_predictor.output + self.collision_predictor.output + self.energy_predictor.output
        )

        # adjust input sizes of the predictor networks
        self.state_predictor.input = self.recurrence.hidden_size
        self.collision_predictor.input = self.recurrence.hidden_size
        self.energy_predictor.input = self.recurrence.hidden_size


@configclass
class FDMModelVelocitySingleStepHeightAdjustCfg(FDMHeightModelSingleStepCfg):

    class_type: type[FDMModelVelocitySingleStepHeightAdjust] = FDMModelVelocitySingleStepHeightAdjust

    height_scan_res: float = 0.05
    """Resolution of the height scan."""  # TODO: set later by sensor cfg

    scan_cut_dim_x: list[float] = [-0.5, 1.0]
    """Cut dimensions of the height scan in x direction."""
    scan_cut_dim_y: list[float] = [-1.0, 1.0]
    """Cut dimensions of the height scan in y direction."""

    height_scan_shape: list[int] = [120, 92]
    """Shape of the height scan."""  # TODO: set later by sensor cfg

    def __post_init__(self):
        super().__post_init__()

        # adjust exteroceptive encoder
        # self.obs_exteroceptive_encoder.kernel_size[0] = (3, 3)
        self.obs_exteroceptive_encoder.max_pool[0] = False
        self.obs_exteroceptive_encoder.flatten = False
        self.obs_exteroceptive_encoder.avg_pool = True

        # adjust recurrent layer
        self.recurrence.input_size = (
            self.action_encoder.output
            + self.state_obs_proprioception_encoder.hidden_size
            + self.friction_predictor.output
            + 256  # height scan
            + self.state_predictor.output
            + self.collision_predictor.output
            + self.energy_predictor.output
        )

        # add height_scan_robot_center
        self.height_scan_robot_center = self.height_scan_shape[0] / 2, 0.5 / self.height_scan_res


@configclass
class FDMLargeHeightModelCfg(FDMBaseModelCfg):
    obs_exteroceptive_encoder = FDMBaseModelCfg.CNNConfig(
        in_channels=1,
        out_channels=[8, 16, 16, 32, 32],
        kernel_size=[(7, 7), (5, 5), (5, 5), (3, 3), (3, 3)],
        max_pool=[True, True, True, True, False],
        batchnorm=False,
        activation="LeakyReLU",
        compress_MLP_layers=FDMBaseModelCfg.MLPConfig(
            input=2048, output=32, shape=[256], activation="LeakyReLU", dropout=0.2, batchnorm=False  # batchnorm=True
        ),
    )


###
# Depth Image Models
###


@configclass
class FDMDepthModelCfg(FDMBaseModelCfg):
    """Exteroceptive observation are depth images of size (240, 320)"""

    state_obs_proprioception_encoder: BaseModelCfg.GRUConfig = BaseModelCfg.GRUConfig(
        input_size=142, hidden_size=32, num_layers=2, dropout=0.2
    )
    obs_exteroceptive_encoder: FDMBaseModelCfg.ResNetConfig = FDMBaseModelCfg.ResNetConfig(
        layers=[2, 2, 2, 2],
        individual_channel_encoding=True,
        avg_pool=True,
        downsample_MLP=None,
        # downsample_MLP=FDMBaseModelCfg.MLPConfig(
        #     input=768, output=128, shape=None, activation="LeakyReLU", dropout=0.2, batchnorm=False
        # ),
    )
    """Encoder for the exteroceptive observation."""
    recurrence: BaseModelCfg.GRUConfig = BaseModelCfg.GRUConfig(
        input_size=16, hidden_size=160, num_layers=2, dropout=0.2
    )
    """Recurrent layers."""
    state_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=160, output=4, shape=[64, 16], dropout=0.2, batchnorm=False, activation="LeakyReLU"  # batchnorm=True
    )
    """State predictor."""
    collision_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=160, output=1, shape=[64], dropout=0.2, batchnorm=False, activation="LeakyReLU"  # batchnorm=True
    )
    """Collision predictor."""


@configclass
class FDMDepthHeightScanModelCfg(FDMDepthModelCfg):
    """Exteroceptive observation are depth images of size (240, 320) and height scan around the feed. Proprioception
    observations are extended by the cpg state."""

    add_obs_exteroceptive_encoder: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(
        input=208, output=20, shape=[64], dropout=0.2, batchnorm=False, activation="LeakyReLU"
    )
    """Height Scan Encoder."""

    def __post_init__(self):
        # adjust hidden and input sizes due to additional encoder
        self.recurrence.hidden_size = 180
        self.state_predictor.input = 180
        self.collision_predictor.input = 180
        self.energy_predictor.input = 180
        # adjust proprioception encoder to account for cpg state
        self.state_obs_proprioception_encoder.input_size = 150
        # force to encode each channel individually
        self.obs_exteroceptive_encoder.individual_channel_encoding = True
        # self.obs_exteroceptive_encoder.downsample_MLP.batchnorm = False


###
# Pre-trained ResNet-18 Depth Image Encoder
###


@configclass
class FDMDepthPreTrainedModelCfg(FDMDepthModelCfg):
    # obs_exteroceptive_encoder: BaseModelCfg.ResNetConfig = BaseModelCfg.ResNetConfig(
    #     layers=[2, 2, 2, 2],
    #     layer_planes=[64, 128, 256, 512],
    #     avg_pool=True,
    #     inplanes=64,
    #     dilation=1,
    #     layer_stride=[2, 2, 2, 2],
    # )
    obs_exteroceptive_encoder: BaseModelCfg.PerceptNetCfg = BaseModelCfg.PerceptNetCfg(
        layers=[2, 2, 2, 2],
        avg_pool=True,
    )

    def __post_init__(self):
        # adjust hidden size of GRU
        # adjust hidden and input sizes due to additional encoder
        self.recurrence.hidden_size = 288
        self.state_predictor.input = 288
        self.collision_predictor.input = 288
        self.energy_predictor.input = 288
