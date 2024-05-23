import torch
from torch import nn as nn
from torch.nn import functional as F
# import sys
# sys.path.append("../")
from basicsr.utils.registry import ARCH_REGISTRY
import higher
import numpy as np
from .net_module import *
# from os import path as osp

@ARCH_REGISTRY.register()
class BaseSCINetwork(nn.Module):
    def __init__(self, stage=3):
        super(BaseSCINetwork, self).__init__()
        self.stage = stage
        self.f = F(layers=1, channels=3)
        self.g = G(layers=3, channels=16)

        self.f.in_conv.apply(self.weights_init)
        self.f.conv.apply(self.weights_init)
        self.f.out_conv.apply(self.weights_init)
        self.g.in_conv.apply(self.weights_init)
        self.g.convs.apply(self.weights_init)
        self.g.out_conv.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.f(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.g(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        out_dict = self(x_in)
        result = out_dict['res'][0]
        return result


@ARCH_REGISTRY.register()
class BaseSCINetworkEnhOnly(nn.Module):
    def __init__(self, stage=3):
        super(BaseSCINetworkEnhOnly, self).__init__()
        self.stage = stage
        self.f = F(layers=1, channels=3)

        self.f.in_conv.apply(self.weights_init)
        self.f.conv.apply(self.weights_init)
        self.f.out_conv.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        inlist.append(input_op)
        i = self.f(input_op)
        r = input / i
        r = torch.clamp(r, 0, 1)
        ilist.append(i)
        rlist.append(r)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        out_dict = self(x_in)
        result = out_dict['res'][0]
        return result



@ARCH_REGISTRY.register()
class SCINetwork(nn.Module):
    def __init__(self, opt_config, stage=3, out_idx=0, test_patch_size=64, test_num_patches=4):
        super(SCINetwork, self).__init__()
        self.stage = stage
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        self.out_idx = out_idx
        self.f = F(layers=1, channels=3)
        self.g = G(layers=3, channels=16)

        self.f.in_conv.apply(self.weights_init)
        self.f.conv.apply(self.weights_init)
        self.f.out_conv.apply(self.weights_init)
        self.g.in_conv.apply(self.weights_init)
        self.g.convs.apply(self.weights_init)
        self.g.out_conv.apply(self.weights_init)
        # self.opt = torch.optim.SGD(self.f.parameters(), lr=1e-3)

        optim_type = opt_config.pop('type')
        self.opt = self.get_optimizer(optim_type, self.f.parameters(), **opt_config)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.f(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.g(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist
        )
        return out_dict

    def clip_patches(self, x_in):
        x_crop = []
        for _ in range(self.test_num_patches):
            x_crop += random_crop(x_in, self.test_patch_size, 1, None)
        x_crop = torch.cat(x_crop, dim=0)
        return x_crop

    def test_forward(self, x_in, loss_func=None):
        # with torch.
        test_crop = self.clip_patches(x_in)
        with higher.innerloop_ctx(self, self.opt) as (fmodel, fopt):
            out_dict = self(test_crop)
            if loss_func is not None:
                loss = loss_func(out_dict)
            else:
                in_list = out_dict['inp']
                loss = 0
                for i in range(self.stage - 1):
                    loss += torch.nn.functional.mse_loss(in_list[0], in_list[i+1])
            fopt.step(loss)
            with torch.no_grad():
                out_dict = fmodel(x_in)
                result = out_dict['res'][self.out_idx]

        return result



@ARCH_REGISTRY.register()
class SNNetwork(SCINetwork):
    def __init__(self, opt_config, stage=3, test_patch_size=64, test_num_patches=4):
        super().__init__(opt_config, stage, test_patch_size, test_num_patches)


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
            torch.nn.utils.spectral_norm(m)

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)


@ARCH_REGISTRY.register()
class ModuleChangeableSCINetwork(nn.Module):
    def __init__(self, stage=3, **opt) -> None:
        super().__init__()
        self.stage = stage
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        for m in self.g.modules():
            m.apply(self.weights_init)
        # self.f.in_conv.apply(self.weights_init)
        # self.f.conv.apply(self.weights_init)
        # self.f.out_conv.apply(self.weights_init)
        # self.g.in_conv.apply(self.weights_init)
        # self.g.convs.apply(self.weights_init)
        # self.g.out_conv.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.f(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.g(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        out_dict = self(x_in)
        result = out_dict['res'][0]
        return result



@ARCH_REGISTRY.register()
class ModuleChangeableMultiStageNetwork(nn.Module):
    def __init__(self, stage=3, **opt) -> None:
        super().__init__()
        self.stage = stage
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        for m in self.g.modules():
            m.apply(self.weights_init)
        # self.f.in_conv.apply(self.weights_init)
        # self.f.conv.apply(self.weights_init)
        # self.f.out_conv.apply(self.weights_init)
        # self.g.in_conv.apply(self.weights_init)
        # self.g.convs.apply(self.weights_init)
        # self.g.out_conv.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.f(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            input_op = self.g(r)
            ilist.append(i)
            rlist.append(r)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        out_dict = self(x_in)
        result = out_dict['res'][0]
        return result


@ARCH_REGISTRY.register()
class MineNetwork(nn.Module):
    def __init__(self, opt_config, stage=3, out_idx=0, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.stage = stage
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        self.out_idx = out_idx
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        for m in self.g.modules():
            m.apply(self.weights_init)

        optim_type = opt_config.pop('type')
        self.opt = self.get_optimizer(optim_type, self.f.parameters(), **opt_config)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist, attlist = [], [], [], []
        input_op = input
        for i in range(self.stage):
            inlist.append(input_op)
            i = self.f(input_op)
            r = input / i
            r = torch.clamp(r, 0, 1)
            att = self.g(r)
            input_op = input + att
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist
        )
        return out_dict

    def clip_patches(self, x_in):
        x_crop = []
        for _ in range(self.test_num_patches):
            x_crop += random_crop(x_in, self.test_patch_size, 1, None)
        x_crop = torch.cat(x_crop, dim=0)
        return x_crop

    def test_forward(self, x_in, loss_func=None):
        # with torch.
        test_crop = self.clip_patches(x_in)
        with higher.innerloop_ctx(self, self.opt) as (fmodel, fopt):
            out_dict = self(test_crop)
            if loss_func is not None:
                loss = loss_func(out_dict)
            else:
                in_list = out_dict['inp']
                loss = 0
                for i in range(self.stage - 1):
                    loss += torch.nn.functional.mse_loss(in_list[0], in_list[i+1])
            fopt.step(loss)
            with torch.no_grad():
                out_dict = fmodel(x_in)
                result = out_dict['res'][self.out_idx]

        return result


@ARCH_REGISTRY.register()
class MyNetwork2(nn.Module):
    def __init__(self, opt_config, stage=3, out_idx=0, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.stage = stage
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        self.out_idx = out_idx
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        n_config = opt.pop('n_config')
        n_type = n_config.pop('type')
        self.n = eval(n_type)(**n_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        for m in self.g.modules():
            m.apply(self.weights_init)
        for m in self.dn.modules():
            m.apply(self.weights_init)
        for m in self.n.modules():
            m.apply(self.weights_init)

        optim_type = opt_config.pop('type')
        self.opt = self.get_optimizer(optim_type, self.f.parameters(), **opt_config)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        x_noise = x
        b, c, h, w = x.shape
        for i in range(self.stage):
            inlist.append(x_noise)
            # enhance
            noise = self.dn(x_noise)
            x = x - noise
            dn_list.append(x)
            i = self.f(x) * x + 1 - self.f(x)
            y = x / i

            # degrade
            i_ = self.g(x) * y
            x = y * i_
            miu, std = self.n(noise)
            noise = torch.randn_like(x) * std + miu
            x_noise = x + noise

            ilist.append(i)
            rlist.append(y)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][0]
            return result


@ARCH_REGISTRY.register()
class MyNetwork3(nn.Module):
    def __init__(self, stage=3, out_idx=0, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.stage = stage
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        self.out_idx = out_idx
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        # n_config = opt.pop('n_config')
        # n_type = n_config.pop('type')
        # self.n = eval(n_type)(**n_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        for m in self.g.modules():
            m.apply(self.weights_init)
        for m in self.dn.modules():
            m.apply(self.weights_init)
        # for m in self.n.modules():
        #     m.apply(self.weights_init)

        # optim_type = opt_config.pop('type')
        # self.opt = self.get_optimizer(optim_type, self.f.parameters(), **opt_config)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        nlist = []
        input_op = x
        b, c, h, w = x.shape
        for i in range(self.stage):
            # enhance
            x = input_op
            nlist.append(x)
            x = self.dn(x)
            dn_list.append(x)
            inlist.append(x)
            i = self.f(x)
            r = x / i
            r = torch.clamp(r, 0, 1)
            # nlist.append(r)
            # r = self.dn(r)
            # dn_list.append(r)
            input_op = self.g(r)
            ilist.append(i)
            rlist.append(r)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            nx = nlist,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][self.out_idx]
            return result



@ARCH_REGISTRY.register()
class TestSingleStageEnhOnly(nn.Module):
    def __init__(self, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        for m in self.f.modules():
            m.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        enlist = []
        inlist.append(x)
        i = self.f(x)
        r = x / i
        r = torch.clamp(r, 0, 1)
        enlist.append(r)
        ilist.append(i)
        rlist.append(r)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            en=enlist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][0]
            return result



@ARCH_REGISTRY.register()
class TestSingleStageDenoiseOnly(nn.Module):
    def __init__(self, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        # for m in self.f.modules():
        #     m.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        inlist.append(x)
        n_list = [x, ]
        x = self.dn(x)
        rlist.append(x)
        dn_list.append(x)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            dnx = dn_list,
            nx = n_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][0]
            return result




@ARCH_REGISTRY.register()
class TestSingleStage(nn.Module):
    def __init__(self, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        # n_config = opt.pop('n_config')
        # n_type = n_config.pop('type')
        # self.n = eval(n_type)(**n_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        # for m in self.dn.modules():
        #     m.apply(self.weights_init)
        # for m in self.n.modules():
        #     m.apply(self.weights_init)

        # optim_type = opt_config.pop('type')
        # self.opt = self.get_optimizer(optim_type, self.f.parameters(), **opt_config)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        nlist = []
        nlist.append(x)
        x = self.dn(x)
        inlist.append(x)
        dn_list.append(x)
        i = self.f(x)
        r = x / i
        r = torch.clamp(r, 0, 1)
        ilist.append(i)
        rlist.append(r)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            nx = nlist,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][0]
            return result


@ARCH_REGISTRY.register()
class TestSingleStageED(nn.Module):
    def __init__(self, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        for m in self.f.modules():
            m.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)


    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        nlist = []
        en_list = []
        inlist.append(x)
        i = self.f(x)
        r = x / i
        r = torch.clamp(r, 0, 1)
        ilist.append(i)
        en_list.append(r)

        # dn_in = r.detach()
        dn_in = r
        nlist.append(dn_in)
        r = self.dn(dn_in)
        rlist.append(r)
        dn_list.append(r)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            nx = nlist,
            en = en_list,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][0]
            return result



@ARCH_REGISTRY.register()
class TestSingleStageDE(nn.Module):
    def __init__(self, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        for m in self.f.modules():
            m.apply(self.weights_init)

        if opt.get('small_init_dn', False):
            for m in self.dn.modules():
                m.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)


    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        nlist = []
        inlist.append(x)
        nlist.append(x)
        x = self.dn(x)
        dn_list.append(x)

        i = self.f(x)
        r = x / i
        r = torch.clamp(r, 0, 1)
        ilist.append(i)

        rlist.append(r)
        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            nx = nlist,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][0]
            return result



@ARCH_REGISTRY.register()
class ModuleChangableMultiStageModel(nn.Module):
    def __init__(self, stage=3, out_idx=0, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.stage = stage
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        self.out_idx = out_idx
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        dn_config = opt.pop('dn_config')
        dn_type = dn_config.pop('type')
        self.dn = eval(dn_type)(**dn_config)

        self.alpha = np.sqrt(np.pi * 2)

        # n_config = opt.pop('n_config')
        # n_type = n_config.pop('type')
        # self.n = eval(n_type)(**n_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        for m in self.g.modules():
            m.apply(self.weights_init)
        # for m in self.dn.modules():
        #     m.apply(self.weights_init)

        # for m in self.n.modules():
        #     m.apply(self.weights_init)

        # optim_type = opt_config.pop('type')
        # self.opt = self.get_optimizer(optim_type, self.f.parameters(), **opt_config)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def nabla(self, x):
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    def get_noise_level(self, x):
        nabla_x = self.nabla(x)
        delta_x = torch.mean(abs(nabla_x - torch.mean(nabla_x)))
        noise_level_estimate = torch.sqrt(torch.pow(torch.mean(abs(delta_x) * self.alpha), 2/3) / 2)
        return noise_level_estimate

    def random_mix_up(self, x_in, x):
        mask = torch.randn_like(x_in)
        mask = mask.masked_fill(mask > 0, 1)
        mask = mask.masked_fill(mask < 0, 0)
        return mask * x_in + (1 - mask) * x

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        nlist = []
        for i in range(self.stage):
            x_in = x
            inlist.append(x)
            nlist.append(x)

            # denoise
            noise = self.get_noise_level(x)
            x = self.dn(x)
            dn_list.append(x)

            # enhance
            i = self.f(x)
            r = x / i
            r = torch.clamp(r, 0, 1)
            ilist.append(i)
            rlist.append(r)

            # degrade
            x = self.g(r)
            x = torch.randn_like(x) * noise + x
            x = self.random_mix_up(x_in, x)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            nx = nlist,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][self.out_idx]
            return result


@ARCH_REGISTRY.register()
class ModuleChangableMultiStageModelWithoutDenoise(nn.Module):
    def __init__(self, stage=3, out_idx=0, test_patch_size=64, test_num_patches=4, **opt) -> None:
        super().__init__()
        self.stage = stage
        self.test_patch_size = test_patch_size
        self.test_num_patches = test_num_patches
        self.out_idx = out_idx
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        # dn_config = opt.pop('dn_config')
        # dn_type = dn_config.pop('type')
        # self.dn = eval(dn_type)(**dn_config)

        # self.alpha = np.sqrt(np.pi * 2)


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def nabla(self, x):
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    def get_noise_level(self, x):
        nabla_x = self.nabla(x)
        delta_x = torch.mean(abs(nabla_x - torch.mean(nabla_x)))
        noise_level_estimate = torch.sqrt(torch.pow(torch.mean(abs(delta_x) * self.alpha), 2/3) / 2)
        return noise_level_estimate

    def random_mix_up(self, x_in, x):
        mask = torch.randn_like(x_in)
        mask = mask.masked_fill(mask > 0, 1)
        mask = mask.masked_fill(mask < 0, 0)
        return mask * x_in + (1 - mask) * x

    def forward(self, x):
        ilist, rlist, inlist, dn_list = [], [], [], []
        nlist = []
        for i in range(self.stage):
            x_in = x
            inlist.append(x)
            nlist.append(x)

            # # denoise
            # noise = self.get_noise_level(x)
            # x = self.dn(x)
            # dn_list.append(x)

            # enhance
            i = self.f(x)
            r = x / i
            r = torch.clamp(r, 0, 1)
            ilist.append(i)
            rlist.append(r)

            # degrade
            x = self.g(r)
            # x = torch.randn_like(x) * noise + x
            # x = self.random_mix_up(x_in, x)

        out_dict = dict(
            inp = inlist,
            res = rlist,
            ill = ilist,
            nx = nlist,
            dnx = dn_list
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        # with torch.
        with torch.no_grad():
            out_dict = self(x_in)
            result = out_dict['res'][self.out_idx]
            return result


@ARCH_REGISTRY.register()
class ModuleChangeableAdvEnhNetwork(nn.Module):
    def __init__(self, stage=3, **opt) -> None:
        super().__init__()
        self.stage = stage
        # enhance: illumination estimation
        f_config = opt.pop('f_config')
        f_type = f_config.pop('type')
        self.f = eval(f_type)(**f_config)

        # denoise: noise removal
        g_config = opt.pop('g_config')
        g_type = g_config.pop('type')
        self.g = eval(g_type)(**g_config)

        for m in self.f.modules():
            m.apply(self.weights_init)
        # for m in self.g.modules():
        #     m.apply(self.weights_init)

        self.opt = opt
        self.adv_loss_init()
        self.atk_mtds_init()

    def atk_mtds_init(self):
        opt = self.opt['atk']
        atk_type = opt.pop('type', 'FGSM')
        atk_iter = opt.pop('iter', 1)
        self.atk_iter = atk_iter

        # FGSM
        def FGSM(net, inp, l_func, eps=0.06):
            adv_list = []
            inp = inp.clone().detach()
            inp.requires_grad = True
            for _ in range(self.atk_iter):
                oup = net(inp)
                res_dict = dict(
                    nx = [inp, ],
                    dnx = [oup, ]
                )
                loss = l_func(res_dict)
                grad = torch.autograd.grad(
                    loss, [inp, ], retain_graph=True
                )[0]
                adv = inp
                adv = adv + eps * torch.sign(grad)
                adv = torch.clamp(adv, 0, 1).clone().detach()
                adv_list.append(adv)
                inp = adv
                inp.requires_grad = True

            adv = torch.cat(adv_list, dim=0).detach()
            return adv
        
        if atk_type == 'FGSM':
            func = lambda net, inp, l_func: FGSM(net, inp, l_func, **opt)
        
        self.atk_mtd = func

    def caculate_adv_loss(self, out_dict):
        total_loss = 0
        for func, weight in zip(self.adv_losses['functions'], self.adv_losses['weights']):
            total_loss += weight * func(out_dict)
        return total_loss

    def adv_loss_init(self):
        from basicsr.losses import build_loss
        train_opt = self.opt
        adv_losses = train_opt['adv_losses']
        self.adv_losses = dict(
            functions=[],
            weights=[],
            names=[]
        )
        for loss in adv_losses:
            t_loss = loss['type']
            w_loss = loss.pop('weight', 1)
            self.adv_losses['functions'].append(
                build_loss(loss).cuda()
            )
            self.adv_losses['weights'].append(
                w_loss
            )
            self.adv_losses['names'].append(
                t_loss
            )
        pass

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        ilist, rlist, inlist = [], [], []
        nlist = []
        dnlist = []
        enlist = []
        input_op = input
        inlist.append(input_op)
        i = self.f(input_op)
        e = input / i
        ilist.append(i)
        e = torch.clamp(e, 0, 1)
        enlist.append(e)

        # adv example generate
        adv_example = self.atk_mtd(self.g, e.clone(), self.caculate_adv_loss)
        inp_example = torch.cat([e.clone(), adv_example], dim=0).detach()
        nlist.append(inp_example)
        res = self.g(inp_example)
        dnlist.append(res)
        rlist.append(res)

        out_dict = dict(
            en = enlist,
            inp = inlist,
            res = rlist,
            ill = ilist,
            dnx = dnlist,
            nx = nlist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        with torch.no_grad():
            i = self.f(x_in)
            e = x_in / i
            r = self.g(e)
            return r


# TODO: To be writed for segmenation guided low-light enhancement
# @ARCH_REGISTRY.register()
# class SegAssEnhNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         from .net_module import resnet_fe
#         self.seg_net = resnet_fe()
#         self.ill_net = 
#         pass


class TinyUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_convs = nn.ModuleList(
            [
                nn.Conv2d(48, 48, 3, 1, 1),
                nn.Conv2d(768, 768, 3, 1, 1),
            ]
        )
        self.up_convs = nn.ModuleList(
            [
                nn.Conv2d(768, 768, 3, 1, 1),
                nn.ConvTranspose2d(48, 48, 3, 1, 1),
            ]
        )

    def forward(self, x):
        b, c, h, w = x.shape
        flag = False
        if h % 16 != 0 or w % 16 != 0:
            h_ = h - h % 16
            w_ = w - w % 16
            x = torch.nn.functional.interpolate(x, (h_, w_), mode='bicubic')
            flag = True

        for down in self.down_convs:
            x = down(x)
            x = torch.nn.functional.pixel_unshuffle(x, 4)

        for up in self.up_convs:
            x = torch.nn.functional.pixel_shuffle(x, 4)
            x = up(x)

        if flag:
            x = torch.nn.functional.interpolate(x, (h, w), mode='bicubic')
        
        return x
    
    def enc(self, x):
        b, c, h, w = x.shape
        flag = False
        if h % 16 != 0 or w % 16 != 0:
            h_ = h - h % 16
            w_ = w - w % 16
            x = torch.nn.functional.interpolate(x, (h_, w_), mode='bicubic')
            flag = True

        for down in self.down_convs:
            x = torch.nn.functional.pixel_unshuffle(x, 4)
            x = down(x)
            # print(x.shape)
            # print(x.shape)

        return x, (h, w), flag
    
    def dec(self, x, shape, flag):
        for up in self.up_convs:
            x = up(x)
            x = torch.nn.functional.pixel_shuffle(x, 4)

        if flag:
            x = torch.nn.functional.interpolate(x, shape, mode='bicubic')

        return x


@ARCH_REGISTRY.register()
class SegDrivenEnh(nn.Module):
    def __init__(self, **opt) -> None:
        super().__init__()
        self.segnet = torch.hub.load('/home/zyyue/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local')
        print('Model Loaded')
        self.dn_net = TinyUnet()
        self.seg_conv = nn.Sequential(
            nn.Conv2d(384, 384, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(384, 768, 3, 1, 1)
        )
        self.down_convs = nn.ModuleList(
            [
                nn.Conv2d(48, 48, 3, 1, 1),
                nn.Conv2d(768, 768, 3, 1, 1),
            ]
        )
        self.up_convs = nn.ModuleList(
            [nn.ConvTranspose2d(768, 768, 3, 1, 1),
            nn.ConvTranspose2d(48, 48, 3, 1, 1),]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()

    def trans_shape(self, x):
        b, c, h, w = x.shape
        if h % 14 != 0 or w % 14 != 0:
            h = h - h % 14
            w = w - w % 14
            x = torch.nn.functional.interpolate(x, (h, w), mode='bicubic')
        
        return x
    
    def seg_atten_map(self, x):
        b, c, h, w = x.shape
        x = self.trans_shape(x)
        b, c, h_, w_ = x.shape
        x = self.segnet(x, is_training=True)['x_norm_patchtokens'].transpose(1, 2)
        x = x.view(b, -1, h_//14, w_//14)
        x = torch.nn.functional.interpolate(x, (h // 16, w // 16), mode='bicubic')
        x = self.seg_conv(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x
    
    def pixel_shuffle(self, x):
        x = torch.nn.functional.pixel_unshuffle(x, downscale_factor=16)
        return x        


    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)


    def forward(self, x):
        ilist, rlist, inlist, dnlist = [], [], [], []
        nlist, atlist, enlist = [], [], []
        inlist.append(x)
        nlist.append(x)

        b, c, h, w = x.shape
        flag = False
        if h % 16 != 0 or w % 16 != 0:
            h_ = h - h % 16
            w_ = w - w % 16
            x = torch.nn.functional.interpolate(x, (h_, w_), mode='bicubic')
            flag = True

        
        s = self.seg_atten_map(x)
        atlist.append(s)
        x, shape, flag = self.dn_net.enc(x)
        x = s * x
        x = self.dn_net.dec(x, shape, flag)

        dnlist.append(x)

        # s = self.seg_atten_map(x)
        # atlist.append(SegDrivenEnh)

        down_x = x
        for down in self.down_convs:
            # print(down_x.shape)
            down_x = torch.nn.functional.pixel_unshuffle(down_x, 4)
            down_x = down(down_x)
            down_x = self.relu(down_x)

        # alpha = 1 - self.pool(down_x)
        # ill = s * (alpha * down_x + 1 - alpha)
        # ill = s * down_x + 1 - s
        ill = s * down_x

        for up in self.up_convs:
            ill = up(ill)
            ill = self.relu(ill)
            ill = torch.nn.functional.pixel_shuffle(ill, 4)

        ill = x + ill

        x = x / torch.clip(ill, 0, 1)
        # print(torch.down)
        if flag:
            x = torch.nn.functional.interpolate(x, (h, w), mode='bicubic')
            # ill = torch.nn.functional.interpolate(ill, (h, w), mode='bicubic')

        enlist.append(x)
        rlist.append(x)
        ilist.append(ill)
        out_dict = dict(
            en = enlist,
            inp = inlist,
            res = rlist,
            ill = ilist,
            dnx = dnlist,
            nx = nlist
        )
        return out_dict

    def test_forward(self, x_in, *args, **kwargs):
        with torch.no_grad():
            b, c, h, w = x.shape
            flag = False
            if h % 16 != 0 or w % 16 != 0:
                h_ = h - h % 16
                w_ = w - w % 16
                x = torch.nn.functional.interpolate(x, (h_, w_), mode='bicubic')
                flag = True


            s = self.seg_atten_map(x)
            x, shape, flag = self.dn_net.enc(x)
            x = s * x
            x = self.dn_net.dec(x, shape, flag)
            s = self.seg_atten_map(x)
            down_x = x
            for down in self.down_convs:
                down_x = down(down_x)
                down_x = torch.nn.functional.pixel_unshuffle(down_x, 4)

            alpha = 1 - self.pool(down_x)
            ill = s * (alpha * down_x + 1 - alpha)

            for up in self.up_convs:
                ill = up(ill)
                ill = torch.nn.functional.pixel_shuffle(ill, 4)

            x = x / torch.clip(ill, 0, 1)
            if flag:
                x = torch.nn.functional.interpolate(x, (h, w), mode='bicubic')
                ill = torch.nn.functional.interpolate(ill, (h, w), mode='bicubic')
            return x
