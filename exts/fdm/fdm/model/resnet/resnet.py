# Copyright (c) 2025, ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from ..base_layers import MLP
from ..model_base_cfg import BaseModelCfg


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, cfg: BaseModelCfg.ResNetConfig):
        super().__init__()

        # save config
        self.cfg = cfg

        # check configuration
        assert (
            len(self.cfg.layers)
            == len(self.cfg.layer_planes)
            == len(self.cfg.layer_stride)
            == len(self.cfg.replace_stride_with_dilation)
        ), "layers, layer_planes, layer_stride, and replace_stride_with_dilation should have the same length"

        # change configs if each channel is handled individually and add layer to reduce number of channels
        if self.cfg.individual_channel_encoding:
            self.conv2 = nn.Conv2d(
                self.cfg.layer_planes[-1] * self.cfg.input_channels,
                self.cfg.layer_planes[-1],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.cfg.input_channels = 1

        # build network components
        self.conv1 = nn.Conv2d(
            self.cfg.input_channels, self.cfg.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = [
            self._make_layer(
                BasicBlock,
                self.cfg.layer_planes[i],
                self.cfg.layers[i],
                stride=self.cfg.layer_stride[i],
                dilate=self.cfg.replace_stride_with_dilation[i],
            )
            for i in range(len(self.cfg.layers))
        ]
        self.layers = nn.Sequential(*self.layers)

        # downsample layers
        if self.cfg.downsample_MLP is not None:
            self.downsample = MLP(self.cfg.downsample_MLP)
        else:
            self.downsample = None

        # average max pool over high and width dimensions
        if self.cfg.avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.cfg.dilation
        if dilate:
            self.cfg.dilation *= stride
            stride = 1
        if stride != 1 or self.cfg.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.cfg.inplanes, planes * block.expansion, stride),
            )

        layers = []
        layers.append(
            block(
                self.cfg.inplanes,
                planes,
                stride,
                downsample,
                self.cfg.groups,
                self.cfg.base_width,
                previous_dilation,
            )
        )
        self.cfg.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.cfg.inplanes,
                    planes,
                    groups=self.cfg.groups,
                    base_width=self.cfg.base_width,
                    dilation=self.cfg.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x.shape = BS, 3, 90, 160
        if x.shape[1] != self.cfg.input_channels:
            x = torch.movedim(x, -1, 1)

        # let every channel be handled individually
        if self.cfg.individual_channel_encoding:
            # BS: Batch Size, W: Width, H: Height
            BS, C, H, W = x.shape
            # shape that each channel is a batch (works by first taking all channels of the first image, then all channels of the second image, etc.)
            x = x.reshape(BS * C, H, W).unsqueeze(1)  # x.shape = BS * C, 1, 90, 160

        x = self.conv1(x)  # x.shape = BS, C (32), 45, 80
        x = self.relu(x)
        x = self.maxpool(x)  # x.shape = BS, C (32), 23, 40

        # forward through layers
        # internal shapes --> (BS, C (64), 12, 20), (BS, C (128), 6, 10), (BS, C (256), 3, 5)
        x = self.layers(x)  # x.shape = BS, C (256), 3, 5

        # average max pool over high and width dimensions
        if self.avgpool is not None:
            x = self.avgpool(x)  # x.shape = BS, C (256), 1, 1

        # connect channels again for every image, then reduce number of channels
        if self.cfg.individual_channel_encoding:
            x = x.reshape(BS, x.shape[1] * C, x.shape[2], x.shape[3])  # x.shape = BS, C (768), 1, 1
            x = x.squeeze(-1).squeeze(-1)  # x.shape = BS, C (768)

        # downsample MLP
        if self.downsample is not None:
            x = torch.flatten(x, start_dim=1)
            x = self.downsample(x)

        return x
