# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.lab.managers import ObservationTermCfg, SceneEntityCfg
from omni.isaac.lab.utils import configclass

from .max_contact_observation import MaxContactForceObs


@configclass
class MaxContactForceObsCfg(ObservationTermCfg):

    func: Callable[..., torch.Tensor] = MaxContactForceObs

    history_length: int = MISSING
    """Length of the history to consider for the max joint torque."""

    sensor_cfg: SceneEntityCfg = MISSING
    """Configuration of the contact sensor to observe."""
