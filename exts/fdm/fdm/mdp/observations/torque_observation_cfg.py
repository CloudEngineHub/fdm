# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .torque_observation import MaxJointTorque


@configclass
class MaxJointTorqueCfg(ObservationTermCfg):

    func: Callable[..., torch.Tensor] = MaxJointTorque

    history_length: int = MISSING
    """Length of the history to consider for the max joint torque."""

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    """Configuration of the asset to observe."""
