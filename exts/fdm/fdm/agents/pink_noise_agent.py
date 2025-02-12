# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_agent import Agent
from .utils import powerlaw_psd_gaussian

if TYPE_CHECKING:
    from fdm.runner import FDMRunner

    from .pink_noise_agent_cfg import PinkNoiseAgentCfg


class PinkNoiseAgent(Agent):
    cfg: PinkNoiseAgentCfg
    """Pink noise agent configuration."""

    def __init__(self, cfg: PinkNoiseAgentCfg, runner: FDMRunner):
        super().__init__(cfg, runner=runner)
        # reset
        self.reset(obs=None)

    def plan(self, obs: torch.Tensor | None = None, env_ids: torch.Tensor | None = None):
        # allow to reset individual envs
        if env_ids is None:
            env_ids = self._ALL_INDICES

        plan = powerlaw_psd_gaussian(
            self.cfg.colored_noise_exponent,
            size=(self._runner.env.num_envs, self.action_dim, self.cfg.horizon),
            device=self.device,
        )

        plan = torch.minimum(plan * torch.sqrt(self.cfg.variance) + self.cfg.mean, self.cfg.upper_bound)
        plan = torch.maximum(plan, self.cfg.lower_bound)

        self._plan_step[env_ids] = 0
        self._plan[env_ids] = plan
        return plan
