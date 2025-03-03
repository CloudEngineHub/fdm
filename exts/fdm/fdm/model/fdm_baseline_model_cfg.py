# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .fdm_baseline_model import FDMBaseline
from .model_base_cfg import BaseModelCfg


@configclass
class FDMBaselineCfg(BaseModelCfg):
    """
    Learning Forward Dynamics Model and Informed Trajectory Sampler for Safe Quadruped Navigation
    Yunho Kim, Chanyoung Kim, Jemin Hwangbo
    https://arxiv.org/abs/2204.08647
    """

    class_type: type[FDMBaseline] = FDMBaseline

    """
    Planner structure
    """
    state_encoder: dict = {
        "input": 450,  # COM_encoder output is included in the input  (360 + 32)
        "output": 100,
        "shape": [256, 256, 128, 128],
        "activation": "leakyrelu",
        "dropout": 0.2,
        "batchnorm": False,  # True,  #
    }
    """State encoder configuration."""
    command_encoder: dict = {
        "input": 3,
        "output": 64,
        "shape": [32],
        "activation": "leakyrelu",
        "dropout": 0.2,
        "batchnorm": False,  # True,  #
    }
    """Command encoder configuration."""
    recurrence: dict = {
        "input": 64,
        "hidden": 100,
        "layer": 2,
        "dropout": 0.2,
    }
    """Recurrence configuration."""
    traj_predictor: dict = {
        "input": 100,
        "shape": [64, 32, 16],
        "activation": "leakyrelu",
        "dropout": 0.2,
        "batchnorm": False,  # True,  #
        "collision": {"output": 1},
        "coordinate": {"output": 2},
    }
    """Trajectory predictor configuration."""

    max_grad_norm: float = 2.0
    """Maximum gradient norm for the optimizer."""

    """
    Loss-Parameters
    """

    interpolate_probability: bool = False
    """Whether to interpolate the predicted trajectory."""
    loss_weights: dict[str, float] = {
        "collision": 2.0,
        "coordinate": 1.7,
    }
    """Loss weights for the different terms."""
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
    cvae_retrain: bool = False
    """Whether to retrain the cvae model."""

    unified_failure_prediction = True
    """Calculate loss over max of the failure prediction"""

    ###
    # Additional parameters to comply with the rest of the code
    ###

    exclude_state_idx_from_input = None
    hard_contact_metric = "energy"
