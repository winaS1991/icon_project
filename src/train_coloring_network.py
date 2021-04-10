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

import csv
import numpy as np

from pix2pix import Generator
from pix2pix import Discriminator
from util import weights_init_normal
from util import ColoringDataset

# ----- Device Setting -----
cuda = True if torch.cuda.is_available() else False
if not cuda: print("[WARNING] cuda is not used.")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----- Set Param -----
DATA_DIR = "../data/anime"
SAVE_DIR = "../result"

middle_dim = 512
train_batch_size = 64
test_batch_size = 64
epoch = 500
num_iter = 5

# ----- Dataset Setting -----
gray_transforms = transforms.Compose( \
    [transforms.Grayscale(), \
     transforms.ToTensor(), \
     transforms.Normalize((0.5,), (0.5,))])
rgb_transforms = transforms.Compose( \
    [transforms.ToTensor(), \
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = ColoringDataset(DATA_DIR, datamode="train", transform1=gray_transforms, transform2=rgb_transforms)
test_dataset = ColoringDataset(DATA_DIR, datamode="test", transform1=gray_transforms, transform2=rgb_transforms)
print("train dataset size: %d"%len(train_dataset))
print("test dataset size: %d"%len(test_dataset))
in_channels = train_dataset[0][0].shape[0]
out_channels = train_dataset[0][1].shape[0]
img_size = train_dataset[0][1].shape[1]
print("image shape: [%d, %d, %d] -> [%d, %d, %d]"%(in_channels, img_size, img_size, out_channels, img_size, img_size))

# ----- DataLoader Setting -----
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)

# ----- Network Setting -----
model_G = Generator(middle_dim, in_channels, out_channels, dropout=0.5)
model_D = Discriminator(middle_dim, in_channels, out_channels)
print(model_G)
print(model_D)
model_D.train()
model_G.train()

# ----- Loss Setting -----
adversarial_loss = torch.nn.BCELoss()
pixel_loss = torch.nn.L1Loss()
patch = (1, img_size // 2 ** 4, img_size // 2 ** 4)

if cuda:
    adversarial_loss.cuda()
    pixel_loss.cuda()
    model_D.cuda()
    model_G.cuda()

model_G.apply(weights_init_normal)
model_D.apply(weights_init_normal)

# ----- Optimizer Setting -----
optimizer_G = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

lambda_pixel = 100

# ----- Training -----
for i in range(epoch):
    print("epoch: %d"%i)

    step = 0

    for gray_imgs, rgb_imgs in train_loader:

        ones = Variable(Tensor(np.ones((rgb_imgs.size(0), *patch))), requires_grad=False)
        zeros = Variable(Tensor(np.zeros((rgb_imgs.size(0), *patch))), requires_grad=False)
        real_rgb_img = Variable(rgb_imgs.type(Tensor))
        real_gray_img = Variable(gray_imgs.type(Tensor))
        fake_rgb_img = model_G(real_gray_img)

        # ----- Train Generator -----
        if step % num_iter == 0:
            optimizer_G.zero_grad()
            pred_fake = model_D(fake_rgb_img, real_gray_img)
            a_loss = adversarial_loss(pred_fake, ones)
            p_loss = pixel_loss(fake_rgb_img, real_rgb_img)
            g_loss = a_loss + lambda_pixel * p_loss
            g_loss.backward()
            optimizer_G.step()

        # ----- Train Discriminator -----
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(model_D(real_rgb_img, real_gray_img), ones)
        fake_loss = adversarial_loss(model_D(fake_rgb_img.detach(), real_gray_img), zeros)
        d_loss = (real_loss + fake_loss) / 2.0
        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"%( \
            i+1, epoch, step, len(train_loader), d_loss.item(), g_loss.item()))
        step += 1

    torch.save(model_D.state_dict(), SAVE_DIR+"/model_D.pth")
    torch.save(model_G.state_dict(), SAVE_DIR+"/model_G.pth")

    test_gray_imgs, test_rgb_imgs = iter(test_loader).next()
    ones = Variable(Tensor(np.ones((test_rgb_imgs.size(0), *patch))), requires_grad=False)
    zeros = Variable(Tensor(np.zeros((test_rgb_imgs.size(0), *patch))), requires_grad=False)
    test_real_rgb_img = Variable(test_rgb_imgs.type(Tensor))
    test_real_gray_img = Variable(test_gray_imgs.type(Tensor))
    test_fake_rgb_img = model_G(test_real_gray_img)
    test_real_loss = adversarial_loss(model_D(test_real_rgb_img, test_real_gray_img), ones)
    test_fake_loss = adversarial_loss(model_D(test_fake_rgb_img.detach(), test_real_gray_img), zeros)
    test_d_loss = (test_real_loss + test_fake_loss) / 2
    print("[Epoch %d/%d] [D loss %f] [D loss (real): %f] [G loss (fake): %f]"%( \
        i+1, epoch, test_d_loss.item(), test_real_loss.item(), test_fake_loss.item()))
    with open(SAVE_DIR+'/log_loss.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([i+1, test_d_loss.item(), test_real_loss.item(), test_fake_loss.item()])

    grid_img = make_grid(test_real_rgb_img, nrow=8, padding=0)
    save_image(grid_img, SAVE_DIR+"/real{}.png".format(i), nrow=1)
    grid_img = make_grid(test_fake_rgb_img, nrow=8, padding=0)
    save_image(grid_img, SAVE_DIR+"/fake{}.png".format(i), nrow=1)

