from abc import abstractmethod
import numpy as np

try:
    from .nn import (
        SiLU,
        conv_nd,
        linear,
        avg_pool_nd,
        zero_module,
        normalization,
        timestep_embedding,
        checkpoint,
    )
except:
    from nn import (
        SiLU,
        conv_nd,
        linear,
        avg_pool_nd,
        zero_module,
        normalization,
        timestep_embedding,
        checkpoint
    )

try:
    import basicsr.archs.basicblock as B
except Exception as e:
    print(e)
    import basicblock as B

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def caculate_factors():
    def C(x, y):
        if x > y // 2:
            x = y - x
        
        mul = 1
        div = 1
        for _ in range(x):
            mul *= (y - _)
            div *= (_ + 1)
        
        return int(mul / div)
    
    def factor(x):
        sum = 1
        for _ in range(x):
            sum += C(_, x) ** 2
        
        return sum
    
    factors = [factor(i) for i in range(1, 10)]
    factors = [np.sqrt(np.pi / 2 / fac) for fac in factors]
    factors = [1, ] + factors
    return factors


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TimestepConv(TimestepBlock):
    def __init__(
            self,
            conv_dict,
            emb_channels,
            use_scale_shift_norm=False,
            use_checkpoint=False,
            num_group=16
    ):
        super().__init__()
        # nn.Conv2d(in_channels=, out_channels, kernel_size, stride=, padding=)
        self.in_channels = conv_dict['in_channels']
        self.out_channels = conv_dict['out_channels']
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # self.conv = nn.Conv2d(**conv_dict)
        self.in_layers = nn.Sequential(
            normalization(self.in_channels, num_group),
            SiLU(),
            nn.Conv2d(**conv_dict)
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels
            )
        )
    
    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )
    
    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        # print(emb_out.shape)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
        return h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """
    
    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)
    
    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
    
    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
    
    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class TimestepIRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(TimestepIRCNN, self).__init__()
        embed_dim = nc * 4
        self.nc = nc
        self.embed = nn.Sequential(
            linear(nc, embed_dim),
            SiLU(),
            linear(embed_dim, embed_dim)
        )
        L = []
        L.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1,
                          bias=True),
                # nn.ReLU(inplace=True)
            )
        )
        L.append(
            TimestepEmbedSequential(
                TimestepConv(
                    dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                    embed_dim
                ),
                # nn.ReLU(inplace=True)
            )
        )
        L.append(
            TimestepEmbedSequential(
                TimestepConv(
                    dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
                    embed_dim
                ),
                # nn.ReLU(inplace=True)
            )
        )
        L.append(
            TimestepEmbedSequential(
                TimestepConv(
                    dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True),
                    embed_dim
                ),
                # nn.ReLU(inplace=True)
            )
        )
        L.append(
            TimestepEmbedSequential(
                TimestepConv(
                    dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True),
                    embed_dim
                ),
                # AttentionBlock(nc, num_heads=4),
                # nn.ReLU(inplace=True)
            )
        )
        L.append(
            TimestepEmbedSequential(
                TimestepConv(
                    dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                    embed_dim
                ),
                # nn.ReLU(inplace=True)
            )
        )
        L.append(
            TimestepEmbedSequential(
                nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1,
                          bias=True),
            )
        )
        self.model = nn.Sequential(*L)
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.model.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.model.apply(convert_module_to_f32)
    
    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.model.parameters()).dtype
    
    def forward(self, x, sigma=None):
        if sigma is None:
            print(self.inner_dtype)
            sigma = th.randn([1, ]).to(self.inner_dtype)
        emb = self.embed(timestep_embedding(sigma, self.nc))
        
        # n = self.model(x, emb)
        # inp = x
        for module in self.model:
            x = module(x, emb)
        # return inp - x
        return x


# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, trainable=True, ckpt=None):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = nn.Sequential(*L)
        
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False
        
        if ckpt is not None:
            ckpt = th.load(ckpt)
            self.load_state_dict(ckpt['params'])
    
    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# Noise-aware-IRCNN denoiser
# --------------------------------------------
class NAIRCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, trainable=True, ckpt=None):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(NAIRCNN, self).__init__()
        embed_dim = nc * 4
        self.nc = nc
        self.embed = nn.Sequential(
            linear(nc, embed_dim),
            SiLU(),
            linear(embed_dim, embed_dim)
        )
        self.in_conv = TimestepEmbedSequential(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
            TimestepConv(
                dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                embed_dim
            )
        )
        
        self.factors = caculate_factors()
        
        L = []
        # L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        # L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = nn.Sequential(*L)
        
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False
        
        if ckpt is not None:
            ckpt = th.load(ckpt)
            # print(ckpt['params'].keys())
            # print(self.state_dict().keys())
            self.load_state_dict(ckpt['params'])
    
    def estimate_sigma(self, x, t=2):
        for _ in range(t):
            x = x[:, :, :-1, :] - x[:, :, 1:, :]
        
        x = th.flatten(x, start_dim=1)
        x = th.mean(th.abs(x), dim=1)
        x = x * self.factors[t]
        return x
    
    def forward(self, x, t=2):
        inp = x
        sigma = self.estimate_sigma(x, t)
        emb = self.embed(timestep_embedding(sigma, self.nc))
        x = self.in_conv(x, emb)
        n = self.model(x)
        return inp - n


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return nn.Sequential(*L)


class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=17, act_mode='R', ckpt=None):
        super(DnCNN, self).__init__()
        bias = True
        
        m_head = conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)
        
        self.model = nn.Sequential(m_head, *m_body, m_tail)
        
        if ckpt is not None:
            ckpt = th.load(ckpt)
            self.load_state_dict(ckpt['params'])
    
    def forward(self, x):
        n = self.model(x)
        return x - n


class NADnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=17, act_mode='R', ckpt=None):
        super(NADnCNN, self).__init__()
        bias = True
        embed_dim = nc * 4
        self.nc = nc
        self.embed = nn.Sequential(
            linear(nc, embed_dim),
            SiLU(),
            linear(embed_dim, embed_dim)
        )
        self.in_conv = TimestepEmbedSequential(
            conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias),
            TimestepConv(
                dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                embed_dim
            ),
            nn.ReLU(inplace=True)
        )
        self.factors = caculate_factors()
        
        m_body = [conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 3)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)
        
        self.body = nn.Sequential(*m_body, m_tail)

        if ckpt is not None:
            ckpt = th.load(ckpt)
            self.load_state_dict(ckpt['params'])
    
    def estimate_sigma(self, x, t=1):
        for _ in range(t):
            x = x[:, :, :-1, :] - x[:, :, 1:, :]
        
        x = th.flatten(x, start_dim=1)
        x = th.mean(th.abs(x), dim=1)
        x = x * self.factors[t]
        return x
    
    def forward(self, x, t=2):
        inp = x
        sigma = self.estimate_sigma(x, t)
        emb = self.embed(timestep_embedding(sigma, self.nc))
        x = self.in_conv(x, emb)
        n = self.body(x)
        return inp - n



from basicsr.utils.registry import ARCH_REGISTRY
@ARCH_REGISTRY.register()
class NADnCNNDiff(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=17, act_mode='R', ckpt=None):
        super(NADnCNNDiff, self).__init__()
        bias = True
        embed_dim = nc * 4
        self.nc = nc
        self.embed = nn.Sequential(
            linear(nc, embed_dim),
            SiLU(),
            linear(embed_dim, embed_dim)
        )
        self.in_conv = TimestepEmbedSequential(
            conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias),
            TimestepConv(
                dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                embed_dim
            ),
            nn.ReLU(inplace=True)
        )
        self.factors = caculate_factors()
        
        m_body = [conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 3)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)
        
        self.body = nn.Sequential(*m_body, m_tail)

        if ckpt is not None:
            ckpt = th.load(ckpt)
            self.load_state_dict(ckpt['params'])
    
    def estimate_sigma(self, x, t=1):
        for _ in range(t):
            x = x[:, :, :-1, :] - x[:, :, 1:, :]
        
        x = th.flatten(x, start_dim=1)
        x = th.mean(th.abs(x), dim=1)
        x = x * self.factors[t]
        return x
    
    def forward(self, x, t=2):
        inp = x
        sigma = self.estimate_sigma(x, t)
        emb = self.embed(timestep_embedding(sigma, self.nc))
        x = self.in_conv(x, emb)
        n = self.body(x)

        alpha_bar = th.sqrt(1 - sigma**2)
        # print(alpha_bar.shape)
        denoised = (inp - sigma[:, None, None, None] * n) / alpha_bar[:, None, None, None]
        return denoised

"""
# --------------------------------------------
# FFDNet (15 or 12 conv layers)
# --------------------------------------------
Reference:
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
"""


# --------------------------------------------
# FFDNet
# --------------------------------------------
class FFDNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2
        
        self.m_down = B.PixelUnShuffle(upscale_factor=sf)
        
        m_head = B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = B.conv(nc, out_nc * sf * sf, mode='C', bias=bias)
        
        self.model = B.sequential(m_head, *m_body, m_tail)
        
        self.m_up = nn.PixelShuffle(upscale_factor=sf)
    
    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = th.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        
        x = self.m_down(x)
        # m = torch.ones(sigma.size()[0], sigma.size()[1], x.size()[-2], x.size()[-1]).type_as(x).mul(sigma)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = th.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        
        x = x[..., :h, :w]
        return x


class NAMIRNet_v2(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=80,
                 chan_factor=1.5,
                 n_RRG=4,
                 n_MRB=2,
                 height=3,
                 width=2,
                 scale=1,
                 bias=False,
                 task=None
                 ):
        super(NAMIRNet_v2, self).__init__()
        from .mirnet_v2_arch import RRG
        bias = True
        nc = n_feat
        embed_dim = nc
        self.nc = nc
        self.embed = nn.Sequential(
            linear(nc, embed_dim),
            SiLU(),
            linear(embed_dim, embed_dim)
        )
        self.in_conv = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        self.factors = caculate_factors()
        
        self.task = task
        
        # self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)
        
        modules_body = []
        
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=1))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=2))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        modules_body.append(RRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)
    
    def estimate_sigma(self, x, t=1):
        for _ in range(t):
            x = x[:, :, :-1, :] - x[:, :, 1:, :]
        
        x = th.flatten(x, start_dim=1)
        x = th.mean(th.abs(x), dim=1)
        x = x * self.factors[t]
        return x
    
    def forward(self, x, t=2):
        inp = x
        sigma = self.estimate_sigma(x, t)
        emb = self.embed(timestep_embedding(sigma, self.nc))
        while len(emb.shape) < len(x.shape):
            emb = emb[..., None]
        
        shallow_feats = self.in_conv(x) + emb
        
        deep_feats = self.body(shallow_feats)
        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            out_img = self.conv_out(deep_feats)
        else:
            out_img = self.conv_out(deep_feats)
            out_img += inp
        
        return out_img



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
        out = th.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
    

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


# print('\n\n\n\n\n\n\n\n\n\n registing')
@ARCH_REGISTRY.register()
class NAUnetDenoiser(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=6, depth=4, act_mode='R', wf=64, slope=0.2, ckpt=None, *args, **kwargs) -> None:
        super(NAUnetDenoiser, self).__init__(*args, **kwargs)
        bias = True
        embed_dim = nc * 4
        self.nc = nc
        self.factors = caculate_factors()
        self.embed = nn.Sequential(
            linear(nc, embed_dim),
            SiLU(),
            linear(embed_dim, embed_dim)
        )
        self.in_conv = TimestepEmbedSequential(
            conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias),
            TimestepConv(
                dict(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True),
                embed_dim, num_group=nc
            ),
            nn.ReLU(inplace=True)
        )

        self.depth = depth
        prev_channels = nc
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, out_nc, bias=True)

    def estimate_sigma(self, x, t=1):
        for _ in range(t):
            x = x[:, :, :-1, :] - x[:, :, 1:, :]
        
        x = th.flatten(x, start_dim=1)
        x = th.mean(th.abs(x), dim=1)
        x = x * self.factors[t]
        return x
    
    def forward(self, x, t=1):
        inp = x
        sigma = self.estimate_sigma(x, t)
        emb = self.embed(timestep_embedding(sigma, self.nc))
        # while len(emb.shape) < len(x.shape):
        #     emb = emb[..., None]
        
        x = self.in_conv(x, emb)
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        n = self.last(x)
        return n
        
        alpha_bar = th.sqrt(1 - sigma**2)
        # print(alpha_bar.shape)
        denoised = (inp - sigma[:, None, None, None] * n) / alpha_bar[:, None, None, None]
        return denoised


if __name__ == '__main__':
    net = NADnCNNDiff(3, 3, 64).cuda()
    
    import thop
    
    # from torchstat import stat
    input = th.randn(1, 3, 512, 512).cuda()
    sigma = th.randn(1, 1).cuda()
    # t = th.randn([1, ]).cuda()
    flops, params = thop.profile(net, inputs=(input, 2))
    print('flops:{}G'.format(flops / 1e9))
    print('params:{}M'.format(params / 1e6))
    
    # net = TimestepIRCNN(3, 3, 32)
    # stat(net, (3, 256, 64))