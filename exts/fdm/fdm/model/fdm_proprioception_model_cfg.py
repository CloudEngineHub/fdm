# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass

from .fdm_model_cfg import FDMBaseModelCfg
from .fdm_proprioception_model import FDMProprioceptionModel, FDMProprioceptionVelocityModel


@configclass
class FDMProprioceptionModelCfg(FDMBaseModelCfg):
    """Configuration class for the FDM proprioception model."""

    class_type: type[FDMProprioceptionModel] = FDMProprioceptionModel

    def __post_init__(self):
        # introduce the additional loss weights
        self.loss_weights["acceleration"] = 0.5
        self.loss_weights["friction"] = 1.0

        # adjust recurrent layer
        self.recurrence.hidden_size = 128
        self.recurrence.input_size = self.action_encoder.output + self.state_obs_proprioception_encoder.hidden_size
        self.recurrence.dropout = 0.0

        # adjust state predictor
        self.state_predictor.input = self.recurrence.hidden_size * self.prediction_horizon
        self.state_predictor.output = 40
        self.state_predictor.shape = [128, 64]

        # change position loss to rmse
        # self.pos_loss_norm = "l2"


# @configclass
# class FDMProprioceptionModelCfg(FDMBaseModelCfg):
#     """Configuration class for the FDM proprioception model."""

#     class_type: type[FDMProprioceptionModel] = FDMProprioceptionModel

#     state_obs_proprioception_encoder: BaseModelCfg.GRUConfig = BaseModelCfg.GRUConfig(
#         input_size=150, hidden_size=64, num_layers=2, dropout=0.2
#     )

#     friction_predictor: BaseModelCfg.MLPConfig = BaseModelCfg.MLPConfig(input=64, output=4, shape=[32], activation="LeakyReLU")

#     empirical_normalization_dim: int = 140
#     """The dimension of the empirical normalization.

#     Should be applied on the proprioception, i.e. the state_obs_proprioception_encoder input size - state_dim.
#     """

#     def __post_init__(self):
#         # introduce the additional loss weights
#         self.loss_weights["acceleration"] = 1.0
#         self.loss_weights["friction"] = 1.0

#         # adjust recurrent layer
#         self.state_predictor.output = 4
#         self.recurrence.hidden_size = 128
#         self.recurrence.input_size = self.action_encoder.output + self.state_obs_proprioception_encoder.hidden_size + self.state_predictor.output * 2

#         # adjust state predictor
#         self.state_predictor.input = self.recurrence.hidden_size
#         self.state_predictor.shape = [128, 64]

#         # change position loss to rmse
#         self.pos_loss_norm = "l2"


@configclass
class FDMProprioceptionVelocityModelCfg(FDMProprioceptionModelCfg):
    """Configuration class for the FDM proprioception model."""

    class_type: type[FDMProprioceptionVelocityModel] = FDMProprioceptionVelocityModel

    def __post_init__(self):
        super().__post_init__()

        # adjust state predictor to only predict corrected velocity
        self.state_predictor.output = 30
