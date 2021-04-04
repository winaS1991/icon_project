#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
print(torch.__version__)

import numpy as np

from dcgan import Generator as DCGAN_Generator
from pix2pix import Generator as pix2pix_Generator
from util import GraySamplerDataset

# ----- Device Setting -----
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----- Set Param -----
DATA_DIR = "../data/anime"
SAVE_DIR = "../result"
MODEL_DIR = "../weight"
GRAY_MODEL_NAME = "/resnet_G.pth"
COLOR_MODEL_NAME = "/pix2pix_G.pth"

latent_dim = 100
gray_middle_dim = 128
color_middle_dim = 512
test_batch_size = 64

# ----- Dataset Setting -----
transforms = transforms.Compose( \
    [transforms.RandomHorizontalFlip(), \
     transforms.Grayscale(), \
     transforms.ToTensor()])
test_dataset = GraySamplerDataset(DATA_DIR, datamode="test", transform=transforms, latent_dim=latent_dim)
img_channels = test_dataset[0][1].shape[0]
out_channels = 3
img_size = test_dataset[0][1].shape[1]

# ----- DataLoader Setting -----
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

# ----- Network Setting -----
gray_img_sampler = DCGAN_Generator(latent_dim, gray_middle_dim, img_size, img_channels)
gray_img_sampler.cuda()
print(gray_img_sampler)
gray_img_sampler.load_state_dict(torch.load(MODEL_DIR+GRAY_MODEL_NAME))
gray_img_sampler.eval()

coloring = pix2pix_Generator(color_middle_dim, img_channels, out_channels)
coloring.cuda()
print(coloring)
coloring.load_state_dict(torch.load(MODEL_DIR+COLOR_MODEL_NAME))
coloring.eval()

# ----- Generate Image -----
test_latent, test_img = iter(test_loader).next()
test_latent = Variable(test_latent.type(Tensor))
test_fake_img = gray_img_sampler(test_latent)
test_fake_img = coloring(test_fake_img)

grid_img = make_grid(test_fake_img, nrow=8, padding=0)
save_image(grid_img, SAVE_DIR+"/samples.png", nrow=1)

for i in range(len(test_fake_img)):
    save_image(test_fake_img[i], SAVE_DIR+"/sample{}.png".format(i))
