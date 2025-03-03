# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .max_contact_observation_cfg import MaxContactForceObsCfg


class MaxContactForceObs(ManagerTermBase):
    """Observation of the maximum contact force over a history of steps."""

    cfg: MaxContactForceObsCfg
    """Configuration of the observation."""

    def __init__(self, cfg: MaxContactForceObsCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # history buffer
        self._buffer = torch.zeros((env.num_envs, self.cfg.history_length), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._buffer[env_ids] *= 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        # roll buffer
        self._buffer = torch.roll(self._buffer, 1, 1)

        # check if any contact force exceeds the threshold
        normed_force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1).flatten(start_dim=1)
        self._buffer[:, 0] = torch.max(normed_force, dim=1)[0]
        return torch.max(self._buffer, dim=1)[0].unsqueeze(1)
