#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from PIL import Image

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class GraySamplerDataset(Dataset):

    def __init__(self, root, datamode, transform, latent_dim):

        image_dir = os.path.join(root, datamode)
        self.image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
        self.transform = transform
        self.latent_dim = latent_dim

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        latent = torch.randn(self.latent_dim)
        img_path = self.image_paths[index]
        img = Image.open(img_path)

        if not self.transform is None:
            img = self.transform(img)

        return latent, img

class ColoringDataset(Dataset):

    def __init__(self, root, datamode, transform1, transform2):

        image_dir = os.path.join(root, datamode)
        self.image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
        self.gray_transform = transform1
        self.rgb_transform = transform2

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        img_path = self.image_paths[index]
        gray_img = self.gray_transform(Image.open(img_path))
        rgb_img = self.rgb_transform(Image.open(img_path))

        return gray_img, rgb_img

