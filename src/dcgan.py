#!/usr/bin/env python

import torch
from torch import nn

from res_block import ResBlock

class Generator(nn.Module):

    def __init__(self, latent_dim, middle_dim, img_size, channels):

        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.middle_dim = middle_dim
        self.init_size = img_size // 16

        self.l = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.middle_dim*4*self.init_size*self.init_size)
        )

        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=self.middle_dim*4),
            nn.Upsample(scale_factor=2),
            ResBlock(in_channels=self.middle_dim*4, out_channels=self.middle_dim*4, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2),
            ResBlock(in_channels=self.middle_dim*4, out_channels=self.middle_dim*2, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2),
            ResBlock(in_channels=self.middle_dim*2, out_channels=self.middle_dim*2, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2),
            ResBlock(in_channels=self.middle_dim*2, out_channels=self.middle_dim, kernel_size=3, stride=1),
            nn.Conv2d(in_channels=self.middle_dim, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):

        m = self.l(z)
        m = m.view(m.shape[0], self.middle_dim*4, self.init_size, self.init_size)
        return self.layers(m)

class Discriminator(nn.Module):

    def __init__(self, middle_dim, img_size, channels):

        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):

            block = [
                nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.layers = nn.Sequential(
            *discriminator_block(channels, middle_dim, bn=False),
            *discriminator_block(middle_dim, middle_dim*2),
            *discriminator_block(middle_dim*2, middle_dim*4),
            *discriminator_block(middle_dim*4, middle_dim*8),
        )

        size = img_size // 2**4
        self.l = nn.Sequential(
            nn.Linear(in_features=middle_dim*8*size*size, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input):

        m = self.layers(input)
        m = m.view(m.shape[0], -1)
        return self.l(m)
