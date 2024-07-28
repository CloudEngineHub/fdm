

# Adatped from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
import torch.nn as nn

import torchvision.transforms as transforms


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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
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


class PerceptNet(nn.Module):

    def __init__(
        self,
        layers=[2, 2, 2, 2],
        block=BasicBlock,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        avg_pool=False,
        individual_channel_encoding=True,
        img_size=(180, 320),
    ):

        super().__init__()

        # save config
        self.individual_channel_encoding = individual_channel_encoding

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # average max pool over high and width dimensions
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = None

        # Transform layer
        self.depth_transform = transforms.Compose([transforms.Resize(tuple(img_size), antialias=True)])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation)
            )

        return nn.Sequential(*layers)

    def forward(self, x):  # x: [N, 3, 360, 640]   --> here 180, 320 is the resolution of the image
        if x.shape[1] != 3:
            x = torch.movedim(x, -1, 1)

        # transform depth image to desired size
        x = self.depth_transform(x)

        # let every channel be handled individually
        if self.individual_channel_encoding:
            # BS: Batch Size, W: Width, H: Height
            BS, C, H, W = x.shape
            # shape that each channel is a batch (works by first taking all channels of the first image, then all channels of the second image, etc.)
            x = x.reshape(BS * C, H, W).unsqueeze(1)
            # increase channel to 3 again
            x = x.repeat(1, 3, 1, 1)

        x = self.conv1(x)  # x_new: [N, 64, 180, 320]
        x = self.relu(x)  # x_new: [N, 64, 180, 320]
        x = self.maxpool(x)  # x_new: [N, 64, 90, 160]

        x = self.layer1(x)  # x_new: [N, 64, 90, 160]
        x = self.layer2(x)  # x_new: [N, 128, 45, 80]
        x = self.layer3(x)  # x_new: [N, 256, 23, 40]
        # x = self.layer4(x) # x_new: [N, 512, 12, 20]

        # connect channels again for every image, then reduce number of channels
        if self.individual_channel_encoding:
            x = x.reshape(BS, x.shape[1] * C, x.shape[2], x.shape[3])

        # average max pool over high and width dimensions
        if self.avgpool is not None:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)  # x_new: [N, 512] / [N, 512 * C]

        return x
