# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import carb
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_delayed(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, delay: int = 1
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold after a certain delay.

    This allows to record the observations when the robot is in contact and do not directly terminate.
    Useful in navigation tasks/ forward dynamics model learning.
    The delay is multiplied by the decimation of the low-level policy to make sure that the obstacles
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # make sure that delay is not larger than the history length
    delay_physics_timestep = delay * env.cfg.decimation
    if delay_physics_timestep >= net_contact_forces.shape[1]:
        carb.log_warn(
            f"Delay {delay} requires a history length of {delay_physics_timestep} but current length is only"
            f" {net_contact_forces.shape[1]}.Setting delay to {net_contact_forces.shape[1] - 1}"
        )
        delay_physics_timestep = net_contact_forces.shape[1] - 1
    # check if any contact force exceeds the threshold
    # newest contact force is at history idx 0, so delay is removing the newest contact force
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, delay_physics_timestep:, sensor_cfg.body_ids], dim=-1), dim=1)[0]
        > threshold,
        dim=1,
    )
