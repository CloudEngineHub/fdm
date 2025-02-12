# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import ManagerTermBase

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

    from .torque_observation_cfg import MaxJointTorqueCfg


class MaxJointTorque(ManagerTermBase):
    """Observation of the maximum joint torque over a history of steps."""

    cfg: MaxJointTorqueCfg
    """Configuration of the observation."""

    def __init__(self, cfg: MaxJointTorqueCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset: Articulation = self._env.scene[self.cfg.asset_cfg.name]
        # history buffer for the max torque
        self.torque_buffer = torch.zeros((env.num_envs, self.cfg.history_length, 1), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self.torque_buffer[env_ids] *= 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        # roll buffer
        self.torque_buffer = torch.roll(self.torque_buffer, 1, 1)

        # save max torque
        self.torque_buffer[:, 0] = torch.max(self.asset.data.applied_torque, dim=-1)[0].unsqueeze(-1)

        # return max torque over the history
        return torch.max(self.torque_buffer, dim=1)[0]
