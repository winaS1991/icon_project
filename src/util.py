#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
from PIL import Image

class MyDataset(Dataset):

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

