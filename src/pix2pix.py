#!/usr/bin/env python

import torch
from torch import nn

from res_block import ResBlock, SNResBlock

class DownSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):

        super(DownSampleBlock, self).__init__()
        block = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            ]
        if normalize:
            block.append(nn.InstanceNorm2d(num_features=out_channels))
        block.append(nn.LeakyReLU(negative_slope=0.2))
        if dropout:
            block.append(nn.Dropout(p=dropout))

        self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)

class UpSampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.0):

        super(UpSampleBlock, self).__init__()
        block = [
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            ]
        if dropout:
            block.append(nn.Dropout(p=dropout))

        self.model = nn.Sequential(*block)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class Generator(nn.Module):

    def __init__(self, middle_dim, in_channels, out_channels, dropout):

        super(Generator, self).__init__()
        self.down1 = DownSampleBlock(in_channels, middle_dim//8, normalize=False)
        self.down2 = DownSampleBlock(middle_dim//8, middle_dim//4)
        self.down3 = DownSampleBlock(middle_dim//4, middle_dim//2)
        self.down4 = DownSampleBlock(middle_dim//2, middle_dim, dropout=dropout)
        self.down5 = DownSampleBlock(middle_dim, middle_dim, dropout=dropout)
        self.down6 = DownSampleBlock(middle_dim, middle_dim, normalize=False, dropout=dropout)

        self.up1 = UpSampleBlock(middle_dim, middle_dim, dropout=dropout)
        self.up2 = UpSampleBlock(middle_dim*2, middle_dim, dropout=dropout)
        self.up3 = UpSampleBlock(middle_dim*2, middle_dim//2)
        self.up4 = UpSampleBlock(middle_dim, middle_dim//4)
        self.up5 = UpSampleBlock(middle_dim//2, middle_dim//8)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d(padding=(1, 0, 1, 0)),
            nn.Conv2d(in_channels=middle_dim//4, out_channels=out_channels, kernel_size=4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)

class Discriminator(nn.Module):

    def __init__(self, middle_dim, in_channels, out_channels):

        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            SNResBlock(in_channels=in_channels+out_channels, out_channels=middle_dim//8, kernel_size=4, stride=2),
            SNResBlock(in_channels=middle_dim//8, out_channels=middle_dim//4, kernel_size=4, stride=2),
            SNResBlock(in_channels=middle_dim//4, out_channels=middle_dim//2, kernel_size=4, stride=2),
            SNResBlock(in_channels=middle_dim//2, out_channels=middle_dim, kernel_size=4, stride=2),
            nn.ZeroPad2d(padding=(1, 0, 1, 0)),
            nn.Conv2d(in_channels=middle_dim, out_channels=1, kernel_size=4, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_A, img_B):

        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

