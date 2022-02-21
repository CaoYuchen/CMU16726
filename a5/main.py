# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
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


class Criterion(nn.Module):
    def __init__(self, args, mask=False, layer=['conv_1']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        self.mask = mask

        self.perc = PerceptualLoss(layer)

    def forward(self, pred, target):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        if self.mask:
            target, mask = target
            # todo: loss with mask
        else:
            # todo: loss w/o mask
            pass
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
    sample (take the mean if from_mean=True) N noise vector (z) or N style latent(w/w+) depending on latent value.
    To generate a noise vector, just sample from a normal distribution.
    To generate a style latent, you need to map the noise (z) to the style (W) space given the `model`.
    Some hint on the z-mapping can be found at stylegan/generate_gif.py L70:81.
    :return: Tensor on device in shape of (N, dim) if latent == z
             Tensor on device in shape of (N, 1, dim) if latent == w
             Tensor on device in shape of (N, nw, dim) if latent == w+
    """
    if latent == 'z':
        vector = torch.randn(N, dim, device) if not from_mean else torch.zeros(N, dim, device)
    elif latent == 'w':
        if from_mean:
            vector =
        else:
            vector =
    elif latent == 'w+':
        if from_mean:
            vector =
        else:
            vector = 
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
    param = param.requires_grad_().to(device)
    optimizer = FullBatchLBFGS([param], lr=.1, line_search='Wolfe')
    iter_count = [0]
    def closure():
        # todo: your optimiztion
        if iter_count[0] % 250 == 0 and save_prefix is not None:
            # visualization code
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            iter_result = image.data.clamp_(-1, 1)
            save_images(iter_result, save_prefix + '_%d' % iter_count[0])
        return loss

    loss = closure()
    loss.backward()
    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
    image = wrapper(param)
    return param, image


def sample(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    batch_size = 16

    # todo: complete sample_noise and wrapper
    noise = sample_noise(z_dim, device, args.latent, model, batch_size)
    image = wrapper(noise)
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)


def project(args):
    # load images
    loader = get_data_loader(args.input, is_train=False)

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    print('model {} loaded'.format(args.model))
    # todo: implement your criterion here.
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
    loader = get_data_loader(args.input, alpha=True)
    criterion = Criterion(args, True)
    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb, 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        # todo: optimize sketch 2 image


def interpolate(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input)
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
        image_list = []
        with torch.no_grad():
            # todo: interpolation code
        save_gifs(image_list, 'output/interpolate/%d_%s_%s' % (idx, args.model, args.latent))
        if idx >= 3:
            break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan'])
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='z', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=1000, help="number of optimization steps in the image projection")
    parser.add_argument('--perc_wgt', type=float, default=0., help="perc loss lambda")
    parser.add_argument('--input', type=str, default='data/cat/*.png', help="path to the input image")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    if torch.cuda.is_available():
        device = 'cuda'
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
