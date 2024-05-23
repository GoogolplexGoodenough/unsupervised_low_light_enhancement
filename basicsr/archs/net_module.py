import torch
from torch import nn as nn
from torch.nn import functional as Func
import numpy as np
import random


def random_crop(img, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.

    It crops lists of lq and gt images with corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images. Note that all images
            should have the same shape. If the input is an ndarray, it will
            be transformed to a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: GT images and LQ images. If returned results
            only have one element, just return ndarray.
    """
    if not isinstance(img, list):
        img = [img]

    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img[0].size()[-2:]
    else:
        h_lq, w_lq = img[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale

    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    if input_type == 'Tensor':
        img = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img]
    else:
        img = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img]

    return img



class CFMLayer(nn.Module):
    def __init__(self, n_feat):
        super(CFMLayer, self).__init__()
        self.n_feat = n_feat
        spectral_norm = torch.nn.utils.spectral_norm
        conv2d = lambda x: spectral_norm(nn.Conv2d(*x))
        self.CFM_scale_conv0 = conv2d((n_feat, n_feat, 1))
        self.CFM_scale_conv1 = conv2d((n_feat, n_feat, 1))
        self.CFM_shift_conv0 = conv2d((n_feat, n_feat, 1))
        self.CFM_shift_conv1 = conv2d((n_feat, n_feat, 1))

    def forward(self, x):
        scale = self.CFM_scale_conv1(Func.leaky_relu(self.CFM_scale_conv0(x[:,self.n_feat:,:,:]), 0.1, inplace=True))
        shift = self.CFM_shift_conv1(Func.leaky_relu(self.CFM_shift_conv0(x[:,self.n_feat:,:,:]), 0.1, inplace=True))
        return torch.cat((x[:,:self.n_feat,:,:] * (scale + 1) + shift, x[:,self.n_feat:,:,:]), dim=1)


class AttenValueEnh(nn.Module):
    def __init__(self, c_in=3, c_hid=16) -> None:
        super().__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        spectral_norm = torch.nn.utils.spectral_norm
        conv2d = lambda x: spectral_norm(nn.Conv2d(**x))
        self.in_conv = nn.Sequential(
            conv2d(dict(in_channels=c_in, out_channels=c_hid, kernel_size=kernel_size, stride=1, padding=padding)),
            nn.ReLU()
        )

        self.hid_conv = nn.Sequential(
            CFMLayer(c_hid//2),
            nn.ReLU(),
            CFMLayer(c_hid//2),
            nn.ReLU(),
            CFMLayer(c_hid//2),
            nn.ReLU(),
        )

        self.out_conv = nn.Sequential(
            conv2d(dict(in_channels=c_hid, out_channels=c_in, kernel_size=kernel_size, stride=1, padding=padding)),
            nn.ReLU()
        )

        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x_in = x
        x = self.in_conv(x)
        x = self.hid_conv(x)
        x = self.out_conv(x)
        x = self.pooling(x)
        x = self.sigmoid(x)
        x = x * x_in + 1 - x
        return x


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet


import torch
from torch import nn

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=4, wf=64, slope=0.2):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNet, self).__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_channels, bias=True)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = Func.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.LeakyReLU(slope, inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, slope)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UnetDenoise(nn.Module):
    def __init__(self, in_channels, ckpt=None, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(UnetDenoise, self).__init__()
        self.in_channels = in_channels
        self.DNet = UNet(in_channels, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        if ckpt is not None:
            ckpt = torch.load(ckpt)
            # self.DNet.load_state_dict(ckpt)
            new_dict = dict()
            for key, item in ckpt.items():
                if 'DNet' in key:
                    new_key = '.'.join(key.split('.')[1:])
                    new_dict[new_key] = item
                    # print(new_key)

            self.DNet.load_state_dict(new_dict)
            print('DNet Loaded')



    def forward(self, x):
        _, _, h, w = x.shape
        if h % 8 != 0 or w % 8 != 0:
            h_ = h // 8 * 8
            w_ = w // 8 * 8
            x_ = Func.interpolate(x, (h_, w_))
            phi_z = self.DNet(x_)
            phi_z = Func.interpolate(phi_z, (h, w))
            # print(x.shape, phi_z.shape)
        else:
            phi_z = self.DNet(x)

        try:
            x = x - phi_z[:, :self.in_channels, ...]
        except:
            print(x.shape, phi_z.shape)
            import pdb
            pdb.set_trace()
        x = torch.clamp(x, 0, 1)
        return x


class MyUnetDenoise(nn.Module):
    def __init__(self, in_channels, ckpt=None, wf=64, dep_S=5, dep_U=4, slope=0.2):
        super(MyUnetDenoise, self).__init__()
        self.in_channels = in_channels
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        spectral_norm = torch.nn.utils.spectral_norm
        conv2d = lambda x: spectral_norm(nn.Conv2d(**x))
        self.noise_level_conv = nn.Sequential(
            conv2d(dict(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=kernel_size, stride=1, padding=padding)),
            nn.ReLU(),
            CFMLayer(in_channels),
            nn.ReLU(),
            conv2d(dict(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=padding)),
            nn.ReLU()
        )
        self.DNet = UNet(in_channels*2, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        self.alpha = np.sqrt(np.pi * 2)
        if ckpt is not None:
            ckpt = torch.load(ckpt)
            try:
                ckpt = ckpt['params']
            except:
                pass
            dict_key = self.state_dict()
            for key in dict_key.keys():
                for k in ckpt.keys():
                    # print(key, k)
                    if key in k:
                        # print(key, k)
                        dict_key[key] = ckpt[k]
                        break

            self.load_state_dict(dict_key)
            print('Model loaded!')
        print('\n\n\n\n\n\n\n\n')

    def nabla(self, x):
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    def get_noise_level_map(self, x):
        nabla_x = self.nabla(x)
        delta_x = torch.mean(abs(nabla_x - torch.mean(nabla_x)))
        noise_level = torch.sqrt(torch.pow(torch.mean(abs(delta_x) * self.alpha), 2/3) / 2)
        noise_level_map = torch.ones_like(x) * noise_level
        return noise_level_map

    def forward(self, x):
        n_map = self.get_noise_level_map(x)
        n_conv = self.noise_level_conv(n_map)
        x_inp = torch.cat([x, n_conv], dim=1)
        phi_z = self.DNet(x_inp)
        x = x - phi_z[:, :self.in_channels, ...]
        x = torch.clamp(x, 0, 1)
        return x


class MyUnetDenoise3(nn.Module):
    def __init__(self, in_channels, ckpt=None, wf=64, dep_S=5, dep_U=4, slope=0.2, noise_level=[0.01, 0.04]):
        super(MyUnetDenoise3, self).__init__()
        self.in_channels = in_channels
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        spectral_norm = torch.nn.utils.spectral_norm
        conv2d = lambda x: spectral_norm(nn.Conv2d(**x))
        self.noise_level_conv = nn.Sequential(
            conv2d(dict(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=kernel_size, stride=1, padding=padding)),
            nn.ReLU(),
            CFMLayer(in_channels),
            nn.ReLU(),
            conv2d(dict(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=kernel_size, stride=1, padding=padding)),
            nn.ReLU()
        )
        self.low = noise_level[0]
        self.up = noise_level[1]
        self.DNet = UNet(in_channels*2, in_channels*2, wf=wf, depth=dep_U, slope=slope)
        self.alpha = np.sqrt(np.pi) / 2
        if ckpt is not None:
            ckpt = torch.load(ckpt)
            try:
                ckpt = ckpt['params']
            except:
                pass
            dict_key = self.state_dict()
            for key in dict_key.keys():
                for k in ckpt.keys():
                    # print(key, k)
                    if key in k:
                        # print(key, k)
                        dict_key[key] = ckpt[k]
                        break

            self.load_state_dict(dict_key)
            print('Model loaded!')
        print('\n\n\n\n\n\n\n\n')

        self.noise_gate = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def nabla(self, x):
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    def get_noise_level_map(self, x):
        nabla_x = self.nabla(x) * 255.0
        delta_x = torch.mean(torch.abs(nabla_x).flatten(1), dim=1)[:, None, None, None]
        noise_level = delta_x * self.alpha / 255.0
        noise_level_map = torch.ones_like(x) * noise_level
        return noise_level_map

    def enh_show(self, x):
        alpha = 1 - np.mean(x)
        x = x / (x * alpha + 1 - alpha)
        return x


    def debug(self, x_in, x, n_map):
        # delta_x = (x - x_in)[0, ...].cpu().numpy().transpose(1, 2, 0)
        x_in = x_in[0, ...].cpu().numpy().transpose(1, 2, 0)
        x = x[0, ...].cpu().numpy().transpose(1, 2, 0)
        n_map = n_map[0, ...].cpu().numpy().transpose(1, 2, 0)
        import cv2
        x = self.enh_show(x)
        x_in = self.enh_show(x_in)
        delta_x = x - x_in
        other = self.enh_show(delta_x)
        # x = self.enh_show(x)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        x_in = cv2.cvtColor(x_in, cv2.COLOR_RGB2BGR)
        # n_map = cv2.cvtColor(n_map, cv2.COLOR_RGB2BGR)
        # delta_x = cv2.cvtColor(delta_x, cv2.COLOR_RGB2BGR)
        # cv2.imshow('x', x)
        # cv2.imshow('x_in', x_in)
        # cv2.imshow('delta_x', abs(delta_x))
        # cv2.imshow('n_map', n_map)
        show_up = np.concatenate([x_in, x], axis=1)
        show_do = np.concatenate([delta_x, other], axis=1)
        show = np.concatenate([show_up, show_do], axis=0)
        cv2.imshow('show', show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def forward(self, x, debug=False):
        x_in = x
        n_map = self.get_noise_level_map(x)
        n_conv = self.noise_level_conv(n_map)
        x_inp = torch.cat([x, n_conv], dim=1)
        phi_z = self.DNet(x_inp)
        if phi_z[:, :self.in_channels, ...].size() != x.size():
            phi_z = Func.interpolate(phi_z, x.size()[2:], mode='bicubic')
        x = x - phi_z[:, :self.in_channels, ...]
        x = torch.clamp(x, 0, 1)

        x_out = torch.where(n_map < self.low, x_in, x)
        if debug:
            print(torch.mean(n_map))
            self.debug(x_in, x_out, n_map)
        return x_out

