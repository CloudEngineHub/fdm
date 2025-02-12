# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import torch
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fdm.runner import FDMRunner


def L2Loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.norm(input - target, p=2, dim=-1))


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        # flattent the proprioception over the history length
        x = x.view(-1, x.shape[-1])

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class TorchPolicyExporter(torch.nn.Module):
    """Exporter of fdm models into JIT file."""

    def __init__(self, runner: FDMRunner, device: str | None = None):
        super().__init__()

        # copy normalizer
        self.proprioceptive_normalizer = copy.deepcopy(runner.model.proprioceptive_normalizer)

        # copy encoder layers
        self.state_obs_proprioceptive_encoder = copy.deepcopy(runner.model.state_obs_proprioceptive_encoder)
        self.obs_exteroceptive_encoder = copy.deepcopy(runner.model.obs_exteroceptive_encoder)
        if runner.model.action_encoder is not None:
            self.action_encoder = copy.deepcopy(runner.model.action_encoder)
        else:
            self.action_encoder = None
        if runner.model.add_obs_exteroceptive_encoder is not None:
            self.add_obs_exteroceptive_encoder = copy.deepcopy(runner.model.add_obs_exteroceptive_encoder)
        else:
            self.add_obs_exteroceptive_encoder = None

        # copy prediction layers
        self.recurrence = copy.deepcopy(runner.model.recurrence)
        self.state_predictor = copy.deepcopy(runner.model.state_predictor)
        self.collision_predictor = copy.deepcopy(runner.model.collision_predictor)
        self.energy_predictor = copy.deepcopy(runner.model.energy_predictor)
        self.friction_predictor = copy.deepcopy(runner.model.friction_predictor)
        self.sigmoid = copy.deepcopy(runner.model.sigmoid)

        # copy forward function
        self.forward = copy.deepcopy(runner.model.forward)
        self.state_encoder_forward = copy.deepcopy(runner.model.state_encoder_forward)
        self.recurrence_forward = copy.deepcopy(runner.model.recurrence_forward)

        # copy config parameters necessary for forward function
        self.param_command_timestep = copy.deepcopy(runner.model.param_command_timestep)
        self.param_unified_failure_prediction = copy.deepcopy(runner.model.param_unified_failure_prediction)
        self.param_collision_threshold = copy.deepcopy(runner.model.param_collision_threshold)
        self.param_zero_collision_actions = copy.deepcopy(runner.model.param_zero_collision_actions)

        # determine device
        if device is not None:
            self.device = device
        else:
            self.device = runner.model.device

        # construct the example input
        self.example_input = [
            # states
            (
                1,
                runner.model.cfg.history_length,
                runner.model.cfg.state_obs_proprioception_encoder.input_size
                - runner.model.cfg.empirical_normalization_dim,
            ),
            # obs proprioceptive
            (1, runner.model.cfg.history_length, runner.model.cfg.empirical_normalization_dim),
            # obs exteroceptive
            (1, *runner.replay_buffer.exteroceptive_observation_dim),
            # action
            (1, runner.model.cfg.prediction_horizon, runner.model.cfg.action_encoder.input),
            # add obs exteroceptive
            (
                (1, runner.model.cfg.add_obs_exteroceptive_encoder.input)
                if runner.model.add_obs_exteroceptive_encoder is not None
                else (1, 1)
            ),
        ]

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to(self.device)
        traced_script_module = torch.jit.script(self, example_inputs=self.example_input)
        traced_script_module.save(path)
