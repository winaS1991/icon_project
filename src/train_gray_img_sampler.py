#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
print(torch.__version__)

import numpy as np

from dcgan import Generator
from dcgan import Discriminator
from util import MyDataset

# ----- Device Setting -----
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----- Set Param -----
DATA_DIR = "../data/anime"
SAVE_DIR = "../result"

latent_dim = 100
middle_dim = 128
train_batch_size = 64
test_batch_size = 64
epoch = 500

# ----- Dataset Setting -----
transforms = transforms.Compose( \
    [transforms.RandomHorizontalFlip(), \
     transforms.Grayscale(), \
     transforms.ToTensor()])
train_dataset = MyDataset(DATA_DIR, datamode="train", transform=transforms, latent_dim=latent_dim)
test_dataset = MyDataset(DATA_DIR, datamode="test", transform=transforms, latent_dim=latent_dim)
print("train dataset size: %d"%len(train_dataset))
print("test dataset size: %d"%len(test_dataset))
img_channels = train_dataset[0][1].shape[0]
img_size = train_dataset[0][1].shape[1]
print("image shape: [%d, %d, %d]"%(img_channels, img_size, img_size))

# ----- DataLoader Setting -----
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

# ----- Network Setting -----
model_G = Generator(latent_dim, middle_dim, img_size, img_channels)
model_D = Discriminator(middle_dim, img_size, img_channels)
print(model_G)
print(model_D)
model_D.train()
model_G.train()

# ----- Loss Setting -----
adversarial_loss = torch.nn.BCELoss()

if cuda:
    adversarial_loss.cuda()
    model_D.cuda()
    model_G.cuda()

# ----- Optimizer Setting -----
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ----- Training -----
for i in range(epoch):
    print("epoch: %d"%i)

    step = 0

    for latent, imgs in train_loader:

        ones = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        zeros = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        latent = Variable(latent.type(Tensor))
        real_img = Variable(imgs.type(Tensor))
        fake_img = model_G(latent)

        # ----- Train Generator -----
        optimizer_G.zero_grad()
        pred_fake = model_D(fake_img)
        g_loss = adversarial_loss(pred_fake, ones)
        g_loss.backward()
        optimizer_G.step()

        # ----- Train Discriminator -----
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(model_D(real_img), ones)
        fake_loss = adversarial_loss(model_D(fake_img.detach()), zeros)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"%( \
            i+1, epoch, step, len(train_loader), d_loss.item(), g_loss.item()))
        step += 1

    torch.save(model_D.state_dict(), SAVE_DIR+"/model_D.pth")
    torch.save(model_G.state_dict(), SAVE_DIR+"/model_G.pth")

    test_latent, test_img = iter(test_loader).next()
    ones = Variable(Tensor(test_img.shape[0], 1).fill_(1.0), requires_grad=False)
    zeros = Variable(Tensor(test_img.shape[0], 1).fill_(0.0), requires_grad=False)
    test_img = Variable(test_img.type(Tensor))
    test_latent = Variable(test_latent.type(Tensor))
    test_fake_img = model_G(test_latent)
    test_real_loss = adversarial_loss(model_D(test_img), ones)
    test_fake_loss = adversarial_loss(model_D(test_fake_img.detach()), zeros)
    test_d_loss = (test_real_loss + test_fake_loss) / 2
    print("[Epoch %d/%d] [D loss %f] [D loss (real): %f] [G loss (fake): %f]"%( \
        i+1, epoch, test_d_loss.item(), test_real_loss.item(), test_fake_loss.item()))

    grid_img = make_grid(test_fake_img, nrow=8, padding=0)
    grid_img = grid_img.mul(0.5).add_(0.5)
    save_image(grid_img, SAVE_DIR+"/{}.png".format(i), nrow=1)
