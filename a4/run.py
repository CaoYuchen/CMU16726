import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss
import argparse

import os
import imageio
import torchvision.transforms as transforms

"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
tag = 'conv_4'


def get_model_and_losses(cnn, style_img, content_img,
                         content_layers=content_layers_default,
                         style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses

    # get the optimizer

    # run model training, with one weird caveat
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    # def closure():
    # here
    # which does the following:
    # clear the gradients
    # compute the loss and it's gradient
    # return the loss

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step

    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_image_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            if use_style:
                for sl in style_losses:
                    style_score += sl.loss
                style_score *= style_weight

            if use_content:
                for cl in content_losses:
                    content_score += cl.loss
                content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                if use_style:
                    print('Style Loss : {:4f}'.format(style_score.item()))
                if use_content:
                    print('Content Loss: {:4f}'.format(content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    # make sure to clamp once you are done

    return input_img


def main(style_img_path, content_img_path, output_path):
    torch.cuda.empty_cache()
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)

    # interative MPL
    plt.ion()

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    # plt.figure()
    # imshow(style_img, title='Style Image')
    #
    # plt.figure()
    # imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # input_img = random noise of the size of content_img on the correct device
    # output = reconstruct the image from the noise
    input_img = torch.rand_like(content_img, device=device)
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False)

    fig = plt.figure()
    title = 'Reconstructed Image'
    imshow(output, title=title)
    save_images(fig, output, output_path, title)

    # # texture synthesis
    # print("Performing Texture Synthesis from white noise initialization")
    # # input_img = random noise of the size of content_img on the correct device
    # # output = synthesize a texture like style_image
    # input_img = torch.rand_like(content_img, device=device)
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=False, use_style=True)
    #
    # fig = plt.figure()
    # title = 'Synthesized Texture'
    # imshow(output, title=title)
    # save_images(fig, output_path, title, tag)
    #
    # # style transfer
    # # input_img = random noise of the size of content_img on the correct device
    # # output = transfer the style from the style_img to the content image
    # input_img = torch.rand_like(content_img, device=device)
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True)
    #
    # fig = plt.figure()
    # title = 'Output Image from Noise'
    # imshow(output, title=title)
    # save_images(fig, output_path, title, tag)
    #
    #
    # print("Performing Style Transfer from content image initialization")
    # # input_img = content_img.clone()
    # # output = transfer the style from the style_img to the content image
    # input_img = content_img.clone()
    # output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True)
    #
    # fig = plt.figure()
    # title = 'Output Image from Content Image'
    # imshow(output, title=title)
    # save_images(fig, output_path, title, tag)
    #
    #
    plt.ioff()
    # plt.show()


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Input Image Path
    parser.add_argument('--style_img_path', type=str, default="./images/style/picasso.jpg")
    parser.add_argument('--content_img_path', type=str, default="./images/content/dancing.jpg")
    parser.add_argument('--output_path', type=str, default="./output/")

    return parser


def save_images(fig, dir, name, tag):
    path = os.path.join(dir, '{:s}_{:s}.png'.format(name, tag))
    fig.savefig(path)
    print('Saved {}'.format(path))


# def save_images(images, output_dir, name):
#     image = images.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)  # remove the fake batch dimension
#     # img  = transforms.ToPILImage(image)
#     unload = transforms.ToPILImage()
#     img = unload(image)
#     path = os.path.join(output_dir, '{:s}_{:s}.png'.format(name, tag))
#     imageio.imwrite(path, img)
#     # plt.savefig(path)
#     print('Saved {}'.format(path))


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    # args = sys.argv[1:3]
    main(args.style_img_path, args.content_img_path, args.output_path)
