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
from .resnet import BasicBlock, conv1x1


class ResNetFPN(nn.Module):
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
            self.cfg.input_channels = 1

        # build network components
        self.conv1 = nn.Conv2d(
            self.cfg.input_channels, self.cfg.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            BasicBlock,
            self.cfg.layer_planes[0],
            self.cfg.layers[0],
            stride=self.cfg.layer_stride[0],
            dilate=self.cfg.replace_stride_with_dilation[0],
        )
        self.layer2 = self._make_layer(
            BasicBlock,
            self.cfg.layer_planes[1],
            self.cfg.layers[1],
            stride=self.cfg.layer_stride[1],
            dilate=self.cfg.replace_stride_with_dilation[1],
        )
        self.layer3 = self._make_layer(
            BasicBlock,
            self.cfg.layer_planes[2],
            self.cfg.layers[2],
            stride=self.cfg.layer_stride[2],
            dilate=self.cfg.replace_stride_with_dilation[2],
        )
        # self.layer4 = self._make_layer(BasicBlock, 512, self.cfg.layers[3], stride=self.cfg.layer_stride[3],
        #                                dilate=self.cfg.replace_stride_with_dilation[3])

        # downsample layers
        if self.cfg.downsample_MLP is not None:
            self.downsample = MLP(self.cfg.downsample_MLP)
        else:
            self.downsample = None

        # average max pool over high and width dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # extra layers for feature extraction
        self.maxpool_features_0 = nn.MaxPool2d(kernel_size=(10, 10), stride=(9, 9), padding=1)
        self.maxpool_features_1 = nn.MaxPool2d(kernel_size=(6, 10), stride=(5, 9), padding=1)
        self.adaptive_pool_feature_2 = nn.AdaptiveAvgPool2d((2, 1))

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

        # BS: Batch Size, W: Width, H: Height
        BS, C, H, W = x.shape

        # let every channel be handled individually
        if self.cfg.individual_channel_encoding:
            # shape that each channel is a batch (works by first taking all channels of the first image, then all channels of the second image, etc.)
            x = x.reshape(BS * C, H, W).unsqueeze(1)  # x.shape = BS * C, 1, 90, 160

        x = self.conv1(x)  # x.shape = BS, C (32), 45, 80
        x = self.relu(x)
        x = self.maxpool(x)  # x.shape = BS, C (32), 23, 40
        features_0 = self.maxpool_features_0(x)  # features_0.shape = BS, C (32), 2, 4
        features_0 = torch.flatten(features_0, start_dim=1)  # features_0.shape = BS, C (256)

        # layer 1
        x = self.layer1(x)  # x.shape = (BS, C (64), 12, 20)
        features_1 = self.maxpool_features_1(x)  # features_1.shape = (BS, C (64), 2, 2)
        features_1 = torch.flatten(features_1, start_dim=1)  # features_1.shape = (BS, C (256))

        # layer 2
        x = self.layer2(x)  # x.shape = (BS, C (128), 6, 10)
        features_2 = self.adaptive_pool_feature_2(x)  # features_2.shape = (BS, C (128), 2, 1)
        features_2 = torch.flatten(features_2, start_dim=1)  # features_2.shape = (BS, C (256))

        # layer 3
        x = self.layer3(x)  # x.shape = (BS, C (256), 3, 5)
        x = self.avgpool(x)  # x.shape = BS, C (256), 1, 1
        x = x.squeeze(-1).squeeze(-1)  # x.shape = BS, C (256)

        # concatenate features
        x = torch.cat([features_0, features_1, features_2, x], dim=1)  # x.shape = BS, C (1024)

        # connect channels again for every image, then reduce number of channels
        if self.cfg.individual_channel_encoding:
            x = x.reshape(BS, x.shape[1] * C)  # x.shape = BS, C (1024)

        # downsample MLP
        if self.downsample is not None:
            x = self.downsample(x)

        return x
