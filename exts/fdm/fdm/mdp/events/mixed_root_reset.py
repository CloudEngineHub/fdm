# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward Dynamics Model specific randomization utilities."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .mixed_root_reset_cfg import MixedRootResetEventCfg


class MixedRootResetEvent:
    def __init__(self, cfg: MixedRootResetEventCfg):
        self.cfg = cfg

        # initialize the sub reset events
        self._terms: list[Callable] = [term.event_term_cfg.func for term in cfg.terms.values()]
        self._ratio: list[float] = [term.ratio for term in cfg.terms.values()]
        self._names: list[str] = list(cfg.terms.keys())

        # normalize the ratios
        self._ratio = [0] + [ratio / sum(self._ratio) for ratio in self._ratio]

        # ratio cumsum
        self._ratio_cumsum = torch.cumsum(torch.tensor(self._ratio), dim=0)

    """
    Operations
    """

    def get_subterm(self, name: str) -> Callable:
        """Get the event term function with the specified name.

        Args:
            name (str): The name of the command term.

        Returns:
            Callable: The event term function with the specified name.

        Raises:
            ValueError: If the event term with the specified name is not found.
        """
        if name not in self._names:
            raise ValueError(f"Event term with name '{name}' not found.")

        return self._terms[self._names.index(name)]

    """
    Implementation specific functions.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        yaw_range: tuple[float, float],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        for term_idx, reset_event in enumerate(self._terms):
            # get the environment indices for the current command term
            env_ids_subset = env_ids[
                torch.logical_and(
                    env_ids >= self._ratio_cumsum[term_idx] * env.num_envs,
                    env_ids < self._ratio_cumsum[term_idx + 1] * env.num_envs,
                )
            ]
            if len(env_ids_subset) == 0:
                continue
            # apply reset function
            reset_event(env, env_ids_subset, yaw_range, velocity_range, asset_cfg)

    def __name__(self):
        return "MixedRootResetEvent"
