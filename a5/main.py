# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe), modified by Zhiqiu Lin (zl279@cornell.edu)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np

from LBFGS import FullBatchLBFGS

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import torchvision.utils as vutils
from torchvision.models import vgg19

from dataloader import get_data_loader


def build_model(name):
    if name.startswith('vanilla'):
        z_dim = 100
        model_path = 'pretrained/%s.ckpt' % name
        pretrain = torch.load(model_path)
        from vanilla.models import DCGenerator
        model = DCGenerator(z_dim, 32, 'instance')
        model.load_state_dict(pretrain)

    elif name == 'stylegan':
        model_path = 'pretrained/%s.ckpt' % name
        import sys
        sys.path.insert(0, 'stylegan')
        from stylegan import dnnlib, legacy
        with dnnlib.util.open_url(model_path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
            z_dim = model.z_dim
    else:
        return NotImplementedError('model [%s] is not implemented', name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, z_dim


class Wrapper(nn.Module):
    """The wrapper helps to abstract stylegan / vanilla GAN, z / w latent"""

    def __init__(self, args, model, z_dim):
        super().__init__()
        self.model, self.z_dim = model, z_dim
        self.latent = args.latent
        self.is_style = args.model == 'stylegan'

    def forward(self, param):
        if self.latent == 'z':
            if self.is_style:
                image = self.model(param, None)
            else:
                image = self.model(param)
        # w / wp
        else:
            assert self.is_style
            if self.latent == 'w':
                param = param.repeat(1, self.model.mapping.num_ws, 1)
            image = self.model.synthesis(param)
        return image


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PerceptualLoss(nn.Module):
    def __init__(self, add_layer=['conv_5']):
        super().__init__()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        norm = Normalization(cnn_normalization_mean, cnn_normalization_std)
        cnn = vgg19(pretrained=True).features.to(device).eval()

        # TODO (Part 1): implement the Perceptual/Content loss
        #                hint: hw4
        # You may split the model into different parts and store each part in 'self.model'
        self.model = nn.ModuleList()
        self.model.append(norm)
        self.add_layer = add_layer
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            self.model.append(layer)

            if name == add_layer[-1]:
                break

        # self.model = nn.Sequential(norm, *self.model)

    def forward(self, pred, target):

        if isinstance(target, tuple):
            target, mask = target

        loss = 0.
        i = 0
        for net in self.model:
            pred = net(pred)
            target = net(target)

            # TODO (Part 1): implement the forward call for perceptual loss
            #                free feel to rewrite the entire forward call based on your
            #                implementation in hw4
            # TODO (Part 3): if mask is not None, then you should mask out the gradient
            #                based on 'mask==0'. You may use F.adaptive_avg_pool2d() to 
            #                resize the mask such that it has the same shape as the feature map.
            if isinstance(net, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
                if name in self.add_layer:
                    # add content loss:
                    loss += F.mse_loss(pred, target.detach())

        return loss


class Criterion(nn.Module):
    def __init__(self, args, mask=False, layer=['conv_5']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        self.l1_wgt = args.l1_wgt  # weight for l1 loss/mask loss
        self.l2_wgt = args.l2_wgt
        self.bce_wgt = args.bce_wgt
        self.loss_type = args.loss_type
        self.mask = mask

        self.perc = PerceptualLoss(layer)

    def forward(self, pred, target):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        if self.mask:
            target, mask = target
            # TODO (Part 3): loss with mask
            pass
        else:
            # TODO (Part 1): loss w/o mask
            if self.loss_type == "l1":
                lp_loss = self.l1_wgt * torch.mean(torch.linalg.matrix_norm(target - pred, ord=1))
            elif self.loss_type == "l2":
                lp_loss = self.l2_wgt * torch.mean(torch.linalg.matrix_norm(target - pred))
            elif self.loss_type == "bce":
                bce = nn.BCELoss(reduction='none')
                lp_loss = self.bce_wgt * bce(pred, target)
            else:
                raise NotImplementedError('%s is not supported' % self.loss_type)
            loss = self.perc_wgt * self.perc(pred, target) + lp_loss
        return loss


def save_images(image, fname, col=8):
    image = image.cpu().detach()
    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, None, col) for each in image_list]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    imageio.mimsave(fname + '.gif', image_list)


def sample_noise(dim, device, latent, model, N=1, from_mean=False):
    """
    To generate a noise vector, just sample from a normal distribution.
    To generate a style latent, you need to map the noise (z) to the style (W) space given the `model`.
    You will be using model.mapping for this function.
    Specifically,
    if from_mean=False,
        sample N noise vector (z) or N style latent(w/w+) depending on latent value.
    if from_mean=True
        if latent == 'z': Return zero vectors since zero is the mean for standard gaussian
        if latent == 'w'/'w+': You should sample N=10000 z to generate w/w+ and then take the mean.
    Some hint on the z-mapping can be found at stylegan/generate_gif.py L70:81.
    Additionally, you can look at stylegan/training/networks.py class Generator L477:500
    :return: Tensor on device in shape of (N, dim) if latent == z
             Tensor on device in shape of (N, 1, dim) if latent == w
             Tensor on device in shape of (N, nw, dim) if latent == w+
    """
    # TODO (Part 1): Finish the function below according to the comment above
    Nw = 10000
    if latent == 'z':
        vector = torch.randn(N, dim, device=device) if not from_mean else torch.zeros(N, dim, device=device)
    elif latent == 'w':
        if from_mean:
            vector = np.random.randn(Nw, dim, device=device)
            vector = model.mapping(vector, None)
            vector = torch.mean(vector, dim=0, keepdim=True)
            vector = vector[:, 0, :]
        else:
            vector = model.mapping(torch.randn(Nw, dim, device=device), None)
            vector = vector[:, 0, :]
    elif latent == 'w+':
        if from_mean:
            vectors = np.random.randn(Nw, dim, device=device)
            vectors = model.mapping(vectors, None)
            vector = torch.mean(vectors, dim=0, keepdim=True)
        else:
            vector = model.mapping(torch.randn(Nw, dim, device=device), None)
    else:
        raise NotImplementedError('%s is not supported' % latent)
    return vector


def optimize_para(wrapper, param, target, criterion, num_step, save_prefix=None, res=False):
    """
    wrapper: image = wrapper(z / w/ w+): an interface for a generator forward pass.
    param: z / w / w+
    target: (1, C, H, W)
    criterion: loss(pred, target)
    """
    delta = torch.zeros_like(param)
    delta = delta.requires_grad_().to(device)
    optimizer = FullBatchLBFGS([delta], lr=.1, line_search='Wolfe')
    iter_count = [0]

    def closure():
        iter_count[0] += 1
        # TODO (Part 1): Your optimiztion code. Free free to try out SGD/Adam.
        optimizer.zero_grad()
        image = wrapper(param + delta)
        loss = criterion(image, target)

        if iter_count[0] % 250 == 0:
            # visualization code
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            if save_prefix is not None:
                iter_result = image.data.clamp_(-1, 1)
                save_images(iter_result, save_prefix + '_%d' % iter_count[0])
        return loss

    loss = closure()
    loss.backward()
    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
    image = wrapper(param)
    return param + delta, image


def sample(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    batch_size = 16
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    noise = sample_noise(z_dim, device, args.latent, model, batch_size)
    image = wrapper(noise)
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)


def project(args):
    # load images
    loader = get_data_loader(args.input, args.resolution, is_train=False)

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    print('model {} loaded'.format(args.model))
    criterion = Criterion(args)
    # project each image
    for idx, (data, _) in enumerate(loader):
        target = data.to(device)
        save_images(data, 'output/project/%d_data' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, model)
        optimize_para(wrapper, param, target, criterion, args.n_iters,
                      'output/project/%d_%s_%s_%g' % (idx, args.model, args.latent, args.perc_wgt))
        if idx >= 0:
            break


def draw(args):
    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution, alpha=True)
    criterion = Criterion(args, True)
    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb, 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        # TODO (Part 3): optimize sketch 2 image
        #                hint: Set from_mean=True when sampling noise vector


def interpolate(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution)
    criterion = Criterion(args)
    for idx, (image, _) in enumerate(loader):
        save_images(image, 'output/interpolate/%d' % (idx))
        target = image.to(device)
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)
        param, recon = optimize_para(wrapper, param, target, criterion, args.n_iters)
        save_images(recon, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx % 2 == 0:
            src = param
            continue
        dst = param
        alpha_list = np.linspace(0, 1, 50)
        image_list = []
        with torch.no_grad():
            # TODO (Part 2): interpolation code
            #                hint: Write a for loop to append the convex combinations to image_list
            image_list.append(dst)  # change dst
        save_gifs(image_list, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx >= 3:
            break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan'])
    parser.add_argument('--mode', type=str, default='project', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='z', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=1000,
                        help="number of optimization steps in the image projection")
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'l2', 'bce'])
    parser.add_argument('--perc_wgt', type=float, default=0.01, help="perc loss weight")
    parser.add_argument('--l1_wgt', type=float, default=10., help="L1 pixel loss weight")
    parser.add_argument('--l2_wgt', type=float, default=10., help="L2 pixel loss weight")
    parser.add_argument('--bce_wgt', type=float, default=10., help="BCE pixel loss weight")
    parser.add_argument('--resolution', type=int, default=64, help='Resolution of images')
    parser.add_argument('--input', type=str, default='data/cat/*.png', help="path to the input image")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    if torch.cuda.is_available():
        device = 'cuda'
        print('Models moved to GPU.')
    else:
        device = 'cpu'
    if args.mode == 'sample':
        sample(args)
    elif args.mode == 'project':
        project(args)
    elif args.mode == 'draw':
        draw(args)
    elif args.mode == 'interpolate':
        interpolate(args)
