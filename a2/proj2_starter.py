# --------------------------------------------------------
# Written by Yufei Ye and modified by Sheng-Yu Wang (https://github.com/JudyYe)
# Convert from MATLAB code https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj3/gradient_starter.zip
# --------------------------------------------------------
from __future__ import print_function

import argparse
import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
import time


def toy_recon(image, loop=False):
    imh, imw = image.shape
    n_pixel = imh * imw
    n_equation = n_pixel * 2 + 1  # x, y direction, (0,0) coord
    im2var = np.arange(n_pixel).reshape((imh, imw)).astype(int)
    A = sp.lil_matrix((n_equation, n_pixel))
    b = np.zeros((n_equation, 1))
    # loop method
    if loop:
        e = 0
        # objective 1
        for y in range(imh):
            for x in range(imw - 1):
                A[e, im2var[y, x + 1]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = image[y, x + 1] - image[y, x]
                e += 1
        # objective 2
        for y in range(imh - 1):
            for x in range(imw):
                A[e, im2var[y + 1, x]] = 1
                A[e, im2var[y, x]] = -1
                b[e] = image[y + 1, x] - image[y, x]
                e += 1
        # (0,0)
        A[e, im2var[0, 0]] = 1
        b[e] = image[0, 0]
    # non-loop method
    if not loop:
        s = image
        a1 = np.eye(n_pixel, n_pixel, dtype=int)
        a2 = np.roll(a1, -1, axis=1)
        a2[:, -1] = a1[:, -1]
        a3 = (a2 - a1).reshape(n_pixel, imh, imw)
        a3 = np.transpose(a3, (0, 2, 1)).reshape(n_pixel, n_pixel)
        A[0:n_pixel, :] = a2 - a1
        A[n_pixel:-1, :] = a3
        A[-1, im2var[0, 0]] = 1

        b1 = s
        b2 = np.roll(b1, -1, axis=1)
        b2[:, -1] = b1[:, -1]
        b3 = np.roll(b1, -1, axis=0)
        b3[-1, :] = b1[-1, :]
        b[0:n_pixel] = (b2 - b1).reshape((n_pixel, 1))
        b[n_pixel:-1] = (b3 - b1).reshape((n_pixel, 1))
        b[-1] = s[0, 0]

    v = lsqr(A.tocsr(), b)[0] * 255
    output = v.reshape((imh, imw)).astype(int)
    return output


def poisson_blend(fg, mask, bg):
    """
    Poisson Blending.
    :param fg: (H, W, C) source texture / foreground object
    :param mask: (H, W, 1)
    :param bg: (H, W, C) target image / background
    :return: (H, W, C)
    """
    imh, imw, channels = min(fg.shape, bg.shape)
    n_pixels = imh * imw
    im2var = np.arange(n_pixels).reshape((imh, imw)).astype(int)
    v_rgb = np.empty((imh, imw, channels), dtype=int)
    A = sp.lil_matrix((n_pixels, n_pixels))
    mask_index = np.where(mask == True)
    n_mask = len(mask_index[0])
    for c in range(channels):
        b = np.zeros((n_pixels, 1))
        for index in range(n_mask):
            y = mask_index[0][index]
            x = mask_index[1][index]
            e = (y - 1) * imw + x
            # construct A, only construct once because A is same for 3 channels
            if c == 0:
                if mask[y - 1, x]:
                    A[e, im2var[y - 1, x]] = -1
                if mask[y + 1, x]:
                    A[e, im2var[y + 1, x]] = -1
                if mask[y, x - 1]:
                    A[e, im2var[y, x - 1]] = -1
                if mask[y, x + 1]:
                    A[e, im2var[y, x + 1]] = -1
                # center point is 4
                A[e, im2var[y, x]] = 4

            # construct b, construct 3 times because b is different for r, g, b
            b[e] = 4 * fg[y, x, c] - fg[y - 1, x, c] - fg[y + 1, x, c] - fg[y, x + 1, c] - fg[y, x - 1, c]
            if not mask[y - 1, x]:
                b[e] += bg[y - 1, x, c]
            if not mask[y + 1, x]:
                b[e] += bg[y + 1, x, c]
            if not mask[y, x - 1]:
                b[e] += bg[y, x - 1, c]
            if not mask[y, x + 1]:
                b[e] += bg[y, x + 1, c]

        # calculate lsq, only the mask area is what we want
        v = lsqr(A.tocsr(), b)[0] * 255
        v_rgb[:, :, c] = v.reshape((imh, imw)).astype(int)

    return v_rgb / 255. * mask + bg * (1 - mask)


def mixed_blend(fg, mask, bg):
    """EC: Mix gradient of source and target"""
    return fg * mask + bg * (1 - mask)


def color2gray(rgb_image):
    """Naive conversion from an RGB image to a gray image."""
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def mixed_grad_color2gray(rgb_image):
    """EC: Convert an RGB image to gray image using mixed gradients."""
    return np.zeros_like(rgb_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Poisson blending.")
    parser.add_argument("-q", "--question", required=True, choices=["toy", "blend", "mixed", "color2gray"])
    args, _ = parser.parse_known_args()

    # Example script: python proj2_starter.py -q toy
    if args.question == "toy":
        image = imageio.imread('./data/toy_problem.png') / 255.

        timer = time.time()
        image_hat = toy_recon(image, loop=True)
        print("Time used:" + str(time.time() - timer))

        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(image_hat, cmap='gray')
        plt.title('Output')
        plt.show()

    # Example script: python proj2_starter.py -q blend -s data/source_01_newsource.png -t data/target_01.jpg -m data/target_01_mask.png
    if args.question == "blend":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        timer = time.time()
        blend_img = poisson_blend(fg, mask, bg)
        print("Time used:" + str(time.time() - timer))
        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Poisson Blend')
        plt.show()

    if args.question == "mixed":
        parser.add_argument("-s", "--source", required=True)
        parser.add_argument("-t", "--target", required=True)
        parser.add_argument("-m", "--mask", required=True)
        args = parser.parse_args()

        # after alignment (masking_code.py)
        ratio = 1
        fg = cv2.resize(imageio.imread(args.source), (0, 0), fx=ratio, fy=ratio)
        bg = cv2.resize(imageio.imread(args.target), (0, 0), fx=ratio, fy=ratio)
        mask = cv2.resize(imageio.imread(args.mask), (0, 0), fx=ratio, fy=ratio)

        fg = fg / 255.
        bg = bg / 255.
        mask = (mask.sum(axis=2, keepdims=True) > 0)

        blend_img = mixed_blend(fg, mask, bg)

        plt.subplot(121)
        plt.imshow(fg * mask + bg * (1 - mask))
        plt.title('Naive Blend')
        plt.subplot(122)
        plt.imshow(blend_img)
        plt.title('Mixed Blend')
        plt.show()

    if args.question == "color2gray":
        parser.add_argument("-s", "--source", required=True)
        args = parser.parse_args()

        rgb_image = imageio.imread(args.source)
        gray_image = color2gray(rgb_image)
        mixed_grad_img = mixed_grad_color2gray(rgb_image)

        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('rgb2gray')
        plt.subplot(122)
        plt.imshow(mixed_grad_img, cmap='gray')
        plt.title('mixed gradient')
        plt.show()

    plt.close()
