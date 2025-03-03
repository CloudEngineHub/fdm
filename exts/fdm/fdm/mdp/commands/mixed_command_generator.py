# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Sub-module containing command generators for the position-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import NullCommand
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .mixed_command_generator_cfg import MixedCommandCfg


class MixedCommand(CommandTerm):
    """Command generator that combines multiple generators each for a subset of the environments.

    Requirement is that the individual command generators produce a command of the same dimensions."""

    cfg: MixedCommandCfg
    """Configuration for the command."""

    def __init__(self, cfg: MixedCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command class.

        Args:
            cfg: The configuration parameters for the command.
            env: The environment object.
        """
        # Initialize the individual command generators
        self._terms: list[CommandTerm] = [
            term.command_term.class_type(term.command_term, env) for term in cfg.terms.values()
        ]
        # initialize buffer for combined command
        self.combined_command = torch.zeros_like(self._terms[-1].command)
        super().__init__(cfg, env)

        # update ratios and term names
        ratios: list[float] = [term.ratio for term in cfg.terms.values()]
        self.update_ratios(ratios)
        self._names: list[str] = list(cfg.terms.keys())

    def __str__(self) -> str:
        msg = "MixedCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += "\tIndividal Terms:\n"
        for term_idx, term in enumerate(self._terms):
            msg += f"\t\t{term.__str__}\n"
            msg += f"\t\tRatio: {self._ratio[term_idx]}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base position in base frame. Shape is (num_envs, 3)."""
        return self.combined_command

    @property
    def get_term_names(self) -> list[str]:
        """Get the names of the individual command terms."""
        return self._names

    """
    Operations
    """

    def get_subterm(self, name: str) -> CommandTerm:
        """Get the command term with the specified name.

        Args:
            name (str): The name of the command term.

        Returns:
            CommandTerm: The command term with the specified name.

        Raises:
            ValueError: If the command term with the specified name is not found.
        """
        if name not in self._names:
            raise ValueError(f"Command term with name '{name}' not found.")

        return self._terms[self._names.index(name)]

    def update_ratios(self, ratios: list[float]):
        """Update the ratios for the individual command terms.

        Args:
            ratios (Sequence[float]): The new ratios for the individual command terms.
        """
        # normalize the ratios
        self._ratio = [ratio / sum(ratios) for ratio in ratios]
        # get the environment indices for each command term
        self._switch_idx = (
            [0]
            + [int(sum(self._ratio[: ratio_idx + 1]) * self._env.num_envs) for ratio_idx in range(len(self._ratio))]
            + [self._env.num_envs]
        )
        self._envs_idx: list[list[int]] = [
            list(range(self._switch_idx[term_idx], self._switch_idx[term_idx + 1]))
            for term_idx in range(len(self._ratio))
        ]

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        """Sample new goal commands for the specified environments.

        Args:
            env_ids (Sequence[int]): The list of environment IDs to resample.
        """
        for term in self._terms:
            if isinstance(term, NullCommand):
                continue
            term._resample_command(env_ids)

    def _update_command(self):
        """Re-target the position command to the current root position and heading."""
        for term_idx, term in enumerate(self._terms):
            if isinstance(term, NullCommand):
                continue
            term._update_command()
            self.combined_command[self._envs_idx[term_idx]] = term.command[self._envs_idx[term_idx]]

    def _update_metrics(self):
        """Update metrics."""
        for term in self._terms:
            if isinstance(term, NullCommand):
                continue
            term._update_metrics()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization for the command.

        Args:
            debug_vis (bool): Whether to enable debug visualization.
        """
        for term in self._terms:
            if isinstance(term, NullCommand):
                continue
            term._set_debug_vis_impl(debug_vis)

    def _debug_vis_callback(self, event):
        """Callback function for the debug visualization."""
        for term_ids, term in enumerate(self._terms):
            if isinstance(term, NullCommand):
                continue

            try:
                term._debug_vis_callback(event, env_ids=self._envs_idx[term_ids])
            except KeyError:
                term._debug_vis_callback(event)
