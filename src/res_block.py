#!/usr/bin/env python
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
from torch import nn

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):

        super(ResBlock, self).__init__()

        expansion = 4
        middle_dim = int(out_channels / expansion)
        self.stride = stride
        self.is_downsample = (in_channels != out_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=middle_dim, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middle_dim)
        self.conv2 = nn.Conv2d(in_channels=middle_dim, out_channels=middle_dim, kernel_size=kernel_size, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(middle_dim)
        self.conv3 = nn.Conv2d(in_channels=middle_dim, out_channels=middle_dim*expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(middle_dim*expansion)
        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=middle_dim*expansion, kernel_size=1, stride=self.stride, bias=False),
                nn.BatchNorm2d(middle_dim*expansion),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        skip = x
        if self.is_downsample:
            skip = self.downsample(x)
        out += skip
        return self.relu(out)
