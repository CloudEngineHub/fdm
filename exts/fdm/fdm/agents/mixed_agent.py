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

"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from .base_agent import Agent

if TYPE_CHECKING:
    from fdm.runner import FDMRunner

    from .mixed_agent_cfg import MixedAgentCfg


class MixedAgent(Agent):
    """"""

    cfg: MixedAgentCfg
    """Configuration for the command."""

    def __init__(self, cfg: MixedAgentCfg, runner: FDMRunner):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            runner: The runner object.
        """
        super().__init__(cfg, runner=runner)

        # Override the horizon config if provided
        if cfg.horizon is not None:
            for term in cfg.terms.values():
                term.agent_term.horizon = cfg.horizon

        # Initialize the individual command generators
        self._agents: list[Agent] = [term.agent_term.class_type(term.agent_term, runner) for term in cfg.terms.values()]
        self._names: list[str] = list(cfg.terms.keys())
        self.update_ratios([term.ratio for term in cfg.terms.values()])

    def update_ratios(self, ratios: list[float]):
        # normalize the ratios
        self._ratio = [ratio / sum(ratios) for ratio in ratios]
        # get the environment indices for each command term
        self._switch_idx = (
            [0]
            + [
                int(sum(self._ratio[: ratio_idx + 1]) * self._runner.env.num_envs)
                for ratio_idx in range(len(self._ratio))
            ]
            + [self._runner.env.num_envs]
        )
        self._envs_idx: list[list[int]] = [
            list(range(self._switch_idx[term_idx], self._switch_idx[term_idx + 1]))
            for term_idx in range(len(self._ratio))
        ]

    def plan(self, env_ids: torch.Tensor, obs: torch.Tensor | None = None, random_init: bool = True):
        """Generate a action plan for each robot.

        Args:
            env_ids: The environment indices.
            obs: The observation tensor.
            random_init: Whether to initialize the plan randomly.

        Returns:
            The plan for the agent.
        """

        # check if there are any envs to plan
        if env_ids.shape[0] == 0:
            return

        for term_idx, agent_term in enumerate(self._agents):
            # get the environment indices for the current command term
            env_ids_subset = env_ids[
                torch.logical_and(env_ids >= self._switch_idx[term_idx], env_ids < self._switch_idx[term_idx + 1])
            ]

            if len(env_ids_subset) == 0:
                continue

            agent_term.plan(env_ids_subset, obs, random_init)
            self._plan[env_ids_subset] = agent_term._plan[env_ids_subset]

    def plan_reset(self, obs: torch.Tensor, env_ids: torch.Tensor):
        for term_idx, agent_term in enumerate(self._agents):
            # get the environment indices for the current command term
            env_ids_subset = env_ids[
                torch.logical_and(env_ids >= self._switch_idx[term_idx], env_ids < self._switch_idx[term_idx + 1])
            ]

            if len(env_ids_subset) == 0:
                continue

            agent_term.plan_reset(obs, env_ids_subset)
            assert torch.all(
                self._plan_step[env_ids_subset] == 1
            ), "The plan_reset should only be called for environment with plan_step == 1"
            self._plan[env_ids_subset, 1:] = agent_term._plan[env_ids_subset, :-1]

    def debug_viz(self, env_ids: torch.Tensor | None = None):
        """Visualize the planning results."""
        # check if there are any envs to visualize
        if env_ids is None:
            env_ids = self._ALL_INDICES
        elif len(env_ids) == 0:
            return

        for term_idx, agent_term in enumerate(self._agents):
            # get the environment indices for the current command term
            env_ids_subset = env_ids[
                torch.logical_and(env_ids >= self._switch_idx[term_idx], env_ids < self._switch_idx[term_idx + 1])
            ]

            if len(env_ids_subset) == 0:
                continue

            agent_term.debug_viz(env_ids_subset)
