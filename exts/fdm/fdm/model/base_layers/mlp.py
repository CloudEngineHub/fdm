# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from torch import nn as nn

from ..model_base_cfg import BaseModelCfg


class MLP(nn.Module):
    def __init__(self, cfg: BaseModelCfg.MLPConfig):
        """
        Multi-Layer Perceptron model.

        .. note::
            Do not save config to allow for the model to be jit compiled.
        """
        super().__init__()

        # get activation function
        assert isinstance(cfg.activation, str), "Activation function must be a string"
        assert hasattr(nn, cfg.activation), f"Activation function {cfg.activation} not found in torch.nn"
        activation_function = getattr(nn, cfg.activation)

        if cfg.shape:
            # build MLP model
            modules = [nn.Linear(cfg.input, cfg.shape[0]), activation_function()]
            scale = [math.sqrt(2)]

            for idx in range(len(cfg.shape) - 1):
                modules.append(nn.Linear(cfg.shape[idx], cfg.shape[idx + 1]))
                if cfg.batchnorm:
                    modules.append(nn.BatchNorm1d(cfg.shape[idx + 1]))
                modules.append(activation_function())
                if cfg.dropout != 0.0:
                    modules.append(nn.Dropout(cfg.dropout))
                scale.append(math.sqrt(2))

            modules.append(nn.Linear(cfg.shape[-1], cfg.output))
            self.architecture = nn.Sequential(*modules)
            scale.append(math.sqrt(2))
        else:
            # build single layer perceptron
            modules = [nn.Linear(cfg.input, cfg.output)]
            if cfg.batchnorm:
                modules.append(nn.BatchNorm1d(cfg.output))
            modules.append(activation_function())
            if cfg.dropout != 0.0:
                modules.append(nn.Dropout(cfg.dropout))
            self.architecture = nn.Sequential(*modules)
            scale = [math.sqrt(2)]

        # initialize weights
        self.init_weights(self.architecture, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.architecture(x)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
