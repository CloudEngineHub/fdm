# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch import nn as nn

from ..model_base_cfg import BaseModelCfg
from .mlp import MLP


class CNN(nn.Module):
    def __init__(self, cfg: BaseModelCfg.CNNConfig):
        """
        Convolutional Neural Network model.

        .. note::
            Do not save config to allow for the model to be jit compiled.
        """
        super().__init__()

        if isinstance(cfg.batchnorm, bool):
            cfg.batchnorm = [cfg.batchnorm] * len(cfg.out_channels)
        if isinstance(cfg.max_pool, bool):
            cfg.max_pool = [cfg.max_pool] * len(cfg.out_channels)
        if isinstance(cfg.kernel_size, tuple):
            cfg.kernel_size = [cfg.kernel_size] * len(cfg.out_channels)
        if isinstance(cfg.stride, int):
            cfg.stride = [cfg.stride] * len(cfg.out_channels)

        # get activation function
        activation_function = getattr(nn, cfg.activation)

        # build model layers
        modules = []

        for idx in range(len(cfg.out_channels)):
            in_channels = cfg.in_channels if idx == 0 else cfg.out_channels[idx - 1]
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=cfg.out_channels[idx],
                    kernel_size=cfg.kernel_size[idx],
                    stride=cfg.stride[idx],
                )
            )
            if cfg.batchnorm[idx]:
                modules.append(nn.BatchNorm2d(num_features=cfg.out_channels[idx]))
            modules.append(activation_function())
            if cfg.max_pool[idx]:
                modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        if cfg.compress_MLP_layers:
            modules.append(nn.Flatten())
            modules.append(MLP(cfg.compress_MLP_layers))

        self.architecture = nn.Sequential(*modules)

        if cfg.avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None

        # initialize weights
        self.init_weights(self.architecture)

        # save flatten config for forward function
        self.flatten = cfg.flatten

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.architecture(x)
        if self.flatten:
            x = x.flatten(start_dim=1)
        elif self.avgpool is not None:
            x = self.avgpool(x)
            x = x.flatten(start_dim=1)
        return x

    @staticmethod
    def init_weights(sequential):
        [
            torch.nn.init.xavier_uniform_(module.weight)
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Conv2d))
        ]
