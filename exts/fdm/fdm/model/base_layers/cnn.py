

from __future__ import annotations

import torch
from torch import nn as nn

from ..model_base_cfg import BaseModelCfg
from .mlp import MLP


class CNN(nn.Module):
    def __init__(self, cfg: BaseModelCfg.CNNConfig):
        super().__init__()
        self.cfg = cfg
        if isinstance(self.cfg.batchnorm, bool):
            self.cfg.batchnorm = [self.cfg.batchnorm] * len(self.cfg.out_channels)
        if isinstance(self.cfg.max_pool, bool):
            self.cfg.max_pool = [self.cfg.max_pool] * len(self.cfg.out_channels)
        if isinstance(self.cfg.kernel_size, tuple):
            self.cfg.kernel_size = [self.cfg.kernel_size] * len(self.cfg.out_channels)
        if isinstance(self.cfg.stride, int):
            self.cfg.stride = [self.cfg.stride] * len(self.cfg.out_channels)

        # get activation function
        activation_function = getattr(nn, self.cfg.activation)

        # build model layers
        modules = []

        for idx in range(len(self.cfg.out_channels)):
            in_channels = self.cfg.in_channels if idx == 0 else self.cfg.out_channels[idx - 1]
            modules.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.cfg.out_channels[idx],
                    kernel_size=self.cfg.kernel_size[idx],
                    stride=self.cfg.stride[idx],
                )
            )
            if self.cfg.batchnorm[idx]:
                modules.append(nn.BatchNorm2d(num_features=self.cfg.out_channels[idx]))
            modules.append(activation_function())
            if self.cfg.max_pool[idx]:
                modules.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        if self.cfg.compress_MLP_layers:
            modules.append(nn.Flatten())
            modules.append(MLP(self.cfg.compress_MLP_layers))

        self.architecture = nn.Sequential(*modules)

        if self.cfg.avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None

        # initialize weights
        self.init_weights(self.architecture)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.architecture(x)
        if self.cfg.flatten:
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
