import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import basicsr.losses.zero_dce_loss as zero_dce_loss
from basicsr.utils.registry import LOSS_REGISTRY


def nabla_x(x, iter):
    for _ in range(iter):
        x = x[:, :, :-1, :] - x[:, :, 1:, :]
    return x


@LOSS_REGISTRY.register()
class ZeroDCELoss(nn.Module):
    def __init__(self, use_col=True, use_spa=True, use_exp=True, use_tv=True) -> None:
        super().__init__()
        self.use_col = use_col
        self.use_spa = use_spa
        self.use_exp = use_exp
        self.use_tv = use_tv
        self.L_color = zero_dce_loss.L_color()
        self.L_spa = zero_dce_loss.L_spa()

        self.L_exp = zero_dce_loss.L_exp(16,0.6)
        self.L_TV = zero_dce_loss.L_TV()

    def forward(self, res_dict):
        ill_list = res_dict['ill']
        inp_list = res_dict['inp']
        res_list = res_dict['res']
        loss = 0
        for ill, inp, res in zip(ill_list, inp_list, res_list):
            if self.use_tv:
                loss_tv = 200*self.L_TV(ill)
            else:
                loss_tv = 0

            if self.use_spa:
                loss_spa = torch.mean(self.L_spa(res, inp))
            else:
                loss_spa = 0

            if self.use_exp:
                loss_exp = 10*torch.mean(self.L_exp(res))
            else:
                loss_exp = 0

            if self.use_col:
                loss_col = 5*torch.mean(self.L_color(res))
            else:
                loss_col = 0

            loss += loss_tv + loss_spa + loss_col + loss_exp
        return loss


@LOSS_REGISTRY.register()
class ColorLoss3(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> None:
        super().__init__()
        self.mean = torch.from_numpy(np.array(mean)).cuda()
        self.std = torch.from_numpy(np.array(std)).cuda()
        self.relu = nn.ReLU()

    def forward(self, res_dict):
        res_list = res_dict['res']
        loss = 0
        for res in res_list:
            avg = torch.mean(torch.flatten(res, 2), dim=-1)
            dif = abs(avg - self.mean)
            dif = self.relu(dif - self.std)
            avg_error = torch.norm(torch.exp(dif) - 1, p=1, keepdim=True)
            loss += torch.mean(avg_error)

        return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


@LOSS_REGISTRY.register()
class DenoiseLoss(nn.Module):
    def __init__(self, mse_weight=1, tv_weight=1) -> None:
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.mse_weight = mse_weight
        self.tv_weight = tv_weight

    def forward(self, res_dict):
        inp_list = res_dict['nx']
        dnx_list = res_dict['dnx']
        total_loss = 0

        for dnx, inp in zip(dnx_list, inp_list):
            total_loss += self.mse_weight * self.l2_loss(dnx, inp) + self.tv_weight * self.tv_loss(dnx)

        return total_loss


@LOSS_REGISTRY.register()
class AdaptiveDenoiseLoss(nn.Module):
    def __init__(self, mse_weight=1, tv_weight=1, sigma_weihgt=5) -> None:
        super().__init__()
        self.l2_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.mse_weight = mse_weight
        self.tv_weight = tv_weight
        self.coef = np.sqrt(np.pi) / 2
        self.sigma_weihgt = sigma_weihgt

    def nabla(self, x):
        return x[:, :, 1:, :] - x[:, :, :-1, :]

    def smooth_simga_weight(self, x):
        nabla_x = self.nabla(x) * 255.0
        sigma_x = torch.mean(torch.abs(nabla_x).flatten(1), dim=1) * self.coef
        return sigma_x[:, None, None, None].detach()

    def forward(self, res_dict):
        inp_list = res_dict['nx']
        dnx_list = res_dict['dnx']
        total_loss = 0

        # for dnx, inp in zip(dnx_list, inp_list):
        #     total_loss += self.mse_weight * self.l2_loss(dnx, inp) + \
        #         self.tv_weight * self.tv_loss(dnx * self.smooth_simga_weight(inp) / self.sigma_weihgt)

        for dnx, inp in zip(dnx_list, inp_list):
            total_loss += self.mse_weight * self.l2_loss(dnx, inp) + \
                self.tv_weight * self.tv_loss(dnx * self.smooth_simga_weight(inp) / self.sigma_weihgt)

        return total_loss

