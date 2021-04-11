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

from dcgan import Discriminator as DCGAN_Discriminator
from util import GraySamplerDataset

# ----- Device Setting -----
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----- Set Param -----
DATA_DIR = "../data/anime"
SAVE_DIR = "../result"
MODEL_DIR = "../weight"
MODEL_NAME = "/model_D1.pth"

latent_dim = 100
gray_middle_dim = 128
test_batch_size = 1

# ----- Dataset Setting -----
transforms = transforms.Compose( \
    [transforms.Grayscale(), \
     transforms.ToTensor()])
test_dataset = GraySamplerDataset(DATA_DIR, datamode="train2", transform=transforms, latent_dim=latent_dim)
img_channels = test_dataset[0][1].shape[0]
img_size = test_dataset[0][1].shape[1]
print("image shape: [%d, %d, %d]"%(img_channels, img_size, img_size))

# ----- DataLoader Setting -----
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

# ----- Network Setting -----
model = DCGAN_Discriminator(gray_middle_dim, img_size, img_channels)
model.cuda()
print(model)
model.load_state_dict(torch.load(MODEL_DIR+MODEL_NAME))
model.eval()

miss_img = []
count = 0
# ----- Generate Image -----
for latent, img in test_loader:
    img = Variable(img.type(Tensor))
    if model(img).item() > 0.5:
        save_image(img, SAVE_DIR+"/sample%0.5d.png"%(count))
        count += 1

