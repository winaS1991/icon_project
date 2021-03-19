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

from dcgan import Generator
from util import MyDataset

# ----- Device Setting -----
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----- Set Param -----
DATA_DIR = "../data/anime"
SAVE_DIR = "../result"
MODEL_DIR = "../weight"
MODEL_NAME = "/dcgan_G.pth"

latent_dim = 100
middle_dim = 128
test_batch_size = 64

# ----- Dataset Setting -----
transforms = transforms.Compose( \
    [transforms.RandomHorizontalFlip(), \
     transforms.Grayscale(), \
     transforms.ToTensor()])
test_dataset = MyDataset(DATA_DIR, datamode="test", transform=transforms, latent_dim=latent_dim)
img_channels = test_dataset[0][1].shape[0]
img_size = test_dataset[0][1].shape[1]

# ----- DataLoader Setting -----
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

# ----- Network Setting -----
model_G = Generator(latent_dim, middle_dim, img_size, img_channels)
model_G.cuda()
print(model_G)
model_G.load_state_dict(torch.load(MODEL_DIR+MODEL_NAME))
model_G.eval()

# ----- Generate Image -----
test_latent, test_img = iter(test_loader).next()
test_latent = Variable(test_latent.type(Tensor))
test_fake_img = model_G(test_latent)

grid_img = make_grid(test_fake_img, nrow=8, padding=0)
grid_img = grid_img.mul(0.5).add_(0.5)
save_image(grid_img, SAVE_DIR+"/samples.png", nrow=1)

for i in range(len(test_fake_img)):
    save_image(test_fake_img[i], SAVE_DIR+"/sample{}.png".format(i))
