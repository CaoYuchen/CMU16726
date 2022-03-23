# CMU 16-726 Learning-Based Image Synthesis / Spring 2022, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - CycleGenerator     --> Used in the CycleGAN in Part 2
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
#   - PatchDiscriminator --> Used in the CycleGAN in Part 2
# For the assignment, you are asked to create the architectures of these three networks by
# filling in the __init__ and forward methods in the
# DCGenerator, CycleGenerator, DCDiscriminator, and PatchDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def up_conv(in_channels, out_channels, kernel_size=4, stride=1, padding=1, scale_factor=2, norm='instance'):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm='batch', init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):
    def __init__(self, noise_size, conv_dim):
        super(DCGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################
        self.up_conv1 = nn.ConvTranspose2d(in_channels=noise_size, out_channels=conv_dim * 8, kernel_size=4, stride=1,
                                           padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(conv_dim * 8)

        # self.up_conv2 = nn.ConvTranspose2d(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=4, stride=2,
        #                                    padding=1, bias=False)
        # self.bn2 = nn.InstanceNorm2d(conv_dim * 4)
        # self.up_conv3 = nn.ConvTranspose2d(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4, stride=2,
        #                                    padding=1, bias=False)
        # self.bn3 = nn.InstanceNorm2d(conv_dim * 2)
        # self.up_conv4 = nn.ConvTranspose2d(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4, stride=2,
        #                                    padding=1, bias=False)
        # self.bn4 = nn.InstanceNorm2d(conv_dim)
        # self.up_conv5 = nn.ConvTranspose2d(in_channels=conv_dim, out_channels=3, kernel_size=4, stride=2,
        #                                    padding=1, bias=False)
        self.up_conv2 = up_conv(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=4, stride=2, padding=1,
                                scale_factor=4,
                                norm='instance')
        self.up_conv3 = up_conv(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=4, stride=2, padding=1,
                                scale_factor=4,
                                norm='instance')
        self.up_conv4 = up_conv(in_channels=conv_dim * 2, out_channels=conv_dim, kernel_size=4, stride=2, padding=1,
                                scale_factor=4,
                                norm='instance')
        self.up_conv5 = up_conv(in_channels=conv_dim, out_channels=3, kernel_size=4, stride=2, padding=1,
                                scale_factor=4, norm='none')

    def forward(self, z):
        """Generates an image given a sample of random noise.

            Input
            -----
                z: BS x noise_size x 1 x 1   -->  16x100x1x1

            Output
            ------
                out: BS x channels x image_width x image_height  -->  16x3x64x64
        """

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        z = F.relu(self.bn1(self.up_conv1(z)))
        z = F.relu(self.up_conv2(z))
        z = F.relu(self.up_conv3(z))
        z = F.relu(self.up_conv4(z))

        z = F.tanh(self.up_conv5(z))

        # z = F.relu(self.bn1(self.up_conv2(z)))
        # z = F.relu(self.bn1(self.up_conv3(z)))
        # z = F.relu(self.bn1(self.up_conv4(z)))
        # z = F.tanh(self.up_conv5(z))

        return z


class ResnetBlock(nn.Module):
    def __init__(self, conv_dim, norm):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1,
                               norm=norm)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64, init_zero_weights=False, norm='batch'):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        self.conv1 = 0
        self.conv2 = 0

        # 2. Define the transformation part of the generator
        self.resnet_block = 0

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.up_conv1 = 0
        self.up_conv2 = 0

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """

        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        pass


class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=32, norm='batch'):
        super(DCDiscriminator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        self.conv1 = conv(in_channels=3, out_channels=conv_dim, kernel_size=4, norm='instance',
                          init_zero_weights=True)
        self.conv2 = conv(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=4, norm='instance',
                          init_zero_weights=True)
        self.conv3 = conv(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=4, norm='instance',
                          init_zero_weights=True)
        self.conv4 = conv(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=4, norm='instance',
                          init_zero_weights=True)
        self.conv5 = conv(in_channels=conv_dim * 8, out_channels=1, kernel_size=4, stride=1, padding=0, norm='none',
                          init_zero_weights=True)

    def forward(self, x):
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.conv2(x), 0.2, True)
        x = F.leaky_relu(self.conv3(x), 0.2, True)
        x = F.leaky_relu(self.conv4(x), 0.2, True)

        x = F.sigmoid(self.conv5(x))
        return x


class PatchDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64, norm='batch'):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # Hint: it should look really similar to DCDiscriminator.

    def forward(self, x):
        ###########################################
        ##   FILL THIS IN: FORWARD PASS   ##
        ###########################################

        pass
