

from __future__ import annotations

import math
import torch
from torch import nn as nn

from ..model_base_cfg import BaseModelCfg


class MLP(nn.Module):
    def __init__(self, cfg: BaseModelCfg.MLPConfig):
        super().__init__()
        self.cfg = cfg

        # get activation function
        assert isinstance(self.cfg.activation, str), "Activation function must be a string"
        assert hasattr(nn, self.cfg.activation), f"Activation function {self.cfg.activation} not found in torch.nn"
        activation_function = getattr(nn, self.cfg.activation)

        if self.cfg.shape:
            # build MLP model
            modules = [nn.Linear(self.cfg.input, self.cfg.shape[0]), activation_function()]
            scale = [math.sqrt(2)]

            for idx in range(len(self.cfg.shape) - 1):
                modules.append(nn.Linear(self.cfg.shape[idx], self.cfg.shape[idx + 1]))
                if self.cfg.batchnorm:
                    modules.append(nn.BatchNorm1d(self.cfg.shape[idx + 1]))
                modules.append(activation_function())
                if self.cfg.dropout != 0.0:
                    modules.append(nn.Dropout(self.cfg.dropout))
                scale.append(math.sqrt(2))

            modules.append(nn.Linear(self.cfg.shape[-1], self.cfg.output))
            self.architecture = nn.Sequential(*modules)
            scale.append(math.sqrt(2))
        else:
            # build single layer perceptron
            modules = [nn.Linear(self.cfg.input, self.cfg.output)]
            if self.cfg.batchnorm:
                modules.append(nn.BatchNorm1d(self.cfg.output))
            modules.append(activation_function())
            if self.cfg.dropout != 0.0:
                modules.append(nn.Dropout(self.cfg.dropout))
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
