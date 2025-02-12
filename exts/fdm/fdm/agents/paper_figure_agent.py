# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_agent import Agent

if TYPE_CHECKING:
    from fdm.runner import FDMRunner

    from .paper_figure_agent_cfg import PaperFigureAgentCfg


class PaperFigureAgent(Agent):
    """Command generator that generates a command for the submission figure"""

    cfg: PaperFigureAgentCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: PaperFigureAgentCfg, runner: FDMRunner):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, runner=runner)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "PaperFigureAgent:\n"
        msg += f"\tCommand dimension: {tuple(self._plan.shape)}\n"  # TODO: check that this is correct
        msg += f"\tPlanning horizon: {self.cfg.horizon}\n"
        return msg

    """
    Operations
    """

    def plan(self, env_ids: torch.Tensor, obs: torch.Tensor | None = None, random_init: bool = True):
        """Update the velocity commands which are correlated to the prev. command."""
        if torch.all(self._plan == 0.0) and self.cfg.platform_figure:
            # check if there are any envs to plan
            self._plan = (
                torch.tensor(
                    [
                        # facing down
                        [
                            [1.21, -0.03, -0.07],
                            [0.93, -0.01, -0.09],
                            [0.81, -0.05, -0.06],
                            [1.15, -0.08, -0.01],
                            [0.72, 0.43, 0.12],
                            [0.85, 0.38, 0.23],
                            [1.04, 0.23, 0.53],
                            [1.07, 0.07, 0.43],
                            [0.96, 0.12, 0.47],
                            [1.27, 0.15, 0.21],
                            [1.41, 0.15, 0.04],
                        ],
                        # facing right
                        [
                            [1.01, 0.23, 0.07],
                            [1.41, 0.34, 0.35],
                            [1.22, 0.23, 0.41],
                            [1.32, 0.19, 0.53],
                            [1.12, 0.05, 0.62],
                            [1.23, 0.02, 0.81],
                            [1.34, 0.10, 0.91],
                            [1.12, 0.07, 0.79],
                            [1.14, 0.12, 0.58],
                            [1.12, 0.15, 0.73],
                            [1.16, 0.15, 0.82],
                        ],
                        # facing up
                        [
                            [1.0, 0.10, 0.03],
                            [1.23, 0.12, 0.64],
                            [1.11, 0.14, 0.44],
                            [0.97, 0.19, 0.52],
                            [1.05, 0.05, 0.18],
                            [0.91, 0.02, -0.2],
                            [0.89, 0.10, -0.3],
                            [0.11, 0.07, 0.01],
                            [1.04, 0.12, 0.21],
                            [1.04, 0.15, 0.21],
                            [1.07, 0.15, 0.21],
                        ],
                        # facing left
                        [
                            [1.47, 0.10, 0.02],
                            [1.35, 0.12, 0.32],
                            [1.22, 0.14, 0.51],
                            [1.32, 0.19, 0.73],
                            [1.12, 0.05, 0.22],
                            [1.23, 0.02, 0.03],
                            [1.34, 0.10, 0.34],
                            [1.12, 0.07, 0.47],
                            [1.14, 0.12, 0.06],
                            [1.12, 0.15, -0.24],
                            [1.16, 0.15, -0.42],
                        ],
                    ],
                    device=self.device,
                )[:, None, :, :]
                .repeat(1, int(self._runner.env.num_envs / 4), 1, 1)
                .reshape(-1, self.cfg.horizon, 3)
            )

        elif torch.all(self._plan == 0.0):
            # check if there are any envs to plan
            self._plan = (
                torch.tensor(
                    [
                        # facing down
                        [
                            [0.15, -0.12, -0.23],
                            [0.23, -0.15, -0.31],
                            [0.43, -0.10, -0.36],
                            [0.50, -0.20, -0.42],
                            [0.62, -0.23, -0.23],
                            [0.85, -0.15, -0.04],
                            [1.04, -0.10, 0.10],
                            [1.21, 0.07, 0.15],
                            [0.96, 0.12, -0.02],
                            [1.14, 0.15, -0.12],
                            [1.21, 0.15, -0.22],
                        ],
                        # facing right
                        [
                            [1.47, 0.10, -0.40],
                            [1.50, 0.34, -0.35],
                            [1.22, 0.14, -0.41],
                            [1.32, 0.19, -0.53],
                            [1.12, 0.05, -0.62],
                            [1.23, 0.02, -0.81],
                            [1.34, 0.10, -0.91],
                            [1.12, 0.07, -0.97],
                            [1.14, 0.12, -0.92],
                            [1.12, 0.15, -0.87],
                            [1.16, 0.15, -0.82],
                        ],
                        # facing up
                        [
                            [1.0, 0.10, 0.2],
                            [1.0, 0.12, 0.2],
                            [1.0, 0.14, 0.2],
                            [1.0, 0.19, 0.0],
                            [1.0, 0.05, 0.0],
                            [1.0, 0.02, -0.2],
                            [1.0, 0.10, -0.3],
                            [1.0, 0.07, 0.0],
                            [1.0, 0.12, 0.2],
                            [1.0, 0.15, 0.2],
                            [1.0, 0.15, 0.2],
                        ],
                        # facing left
                        [
                            [1.47, 0.10, 0.12],
                            [1.35, 0.12, 0.02],
                            [1.22, 0.14, 0.41],
                            [1.32, 0.19, 0.73],
                            [1.12, 0.05, 0.22],
                            [1.23, 0.02, 0.03],
                            [1.34, 0.10, 0.34],
                            [1.12, 0.07, 0.47],
                            [1.14, 0.12, 0.06],
                            [1.12, 0.15, -0.24],
                            [1.16, 0.15, -0.42],
                        ],
                    ],
                    device=self.device,
                )[:, None, :, :]
                .repeat(1, int(self._runner.env.num_envs / 4), 1, 1)
                .reshape(-1, self.cfg.horizon, 3)
            )

    def plan_reset(self, obs: dict, env_ids: torch.Tensor):
        self._plan[env_ids] = torch.roll(self._plan[env_ids], shifts=-1, dims=1)

    def reset(
        self, obs: dict | None = None, env_ids: torch.Tensor | None = None, return_actions: bool = True
    ) -> torch.Tensor | None:
        if return_actions:
            return self._plan[self._ALL_INDICES, self._plan_step]
        else:
            return None
