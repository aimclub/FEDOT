import math

import torch
import torch.nn.functional as F
from fastai.layers import *
from fastcore.basics import snake2camel
from fastcore.meta import delegates
from torch.nn.init import normal_
from torch.nn.utils import spectral_norm, weight_norm

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import AddCoords1d
from fedot.industrial.core.models.nn.network_modules.layers.padding_layers import *


class Conv2dSame(Module):
    """
    Conv2d with padding='same'
    """

    def __init__(
            self, ni, nf, ks=(
                3, 3), stride=(
                1, 1), dilation=(
                1, 1), **kwargs):
        if isinstance(ks, Integral):
            ks = (ks, ks)
        if isinstance(stride, Integral):
            stride = (stride, stride)
        if isinstance(dilation, Integral):
            dilation = (dilation, dilation)
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv2d_same = nn.Conv2d(
            ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv2d_same.weight
        self.bias = self.conv2d_same.bias
        self.pad = Pad2d

    def forward(self, x):
        # stride=self.stride not used in padding calculation!
        self.padding = same_padding2d(
            x.shape[-2], x.shape[-1], self.ks, dilation=self.dilation)
        return self.conv2d_same(self.pad(self.padding)(x))


@delegates(nn.Conv2d.__init__)
def Conv2d(
        ni,
        nf,
        kernel_size=None,
        ks=None,
        stride=1,
        padding='same',
        dilation=1,
        init='auto',
        bias_std=0.01,
        **kwargs):
    """conv1d layer with padding='same', 'valid', or any integer (defaults to 'same')"""
    assert not (
        kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same':
        conv = Conv2dSame(ni, nf, kernel_size, stride=stride,
                          dilation=dilation, **kwargs)
    elif padding == 'valid':
        conv = nn.Conv2d(ni, nf, kernel_size, stride=stride,
                         padding=0, dilation=dilation, **kwargs)
    else:
        conv = nn.Conv2d(ni, nf, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, ni, nf, ks, stride=1, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__(ni,
                                           nf,
                                           kernel_size=ks,
                                           stride=stride,
                                           padding=0,
                                           dilation=dilation,
                                           groups=groups,
                                           bias=bias)
        self.__padding = (ks - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input,
                                                       (self.__padding, 0))
                                                 )


@delegates(nn.Conv1d.__init__)
def Conv1d(ni,
           nf,
           kernel_size=None,
           ks=None,
           stride=1,
           padding='same',
           dilation=1,
           init='auto',
           bias_std=0.01,
           **kwargs):
    """conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"""
    assert not (
        kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'

    kernel_size = kernel_size or ks
    if padding == 'same':
        if kernel_size % 2 == 1:
            conv = nn.Conv1d(
                ni,
                nf,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2 * dilation,
                dilation=dilation,
                **kwargs)
        else:
            conv = SameConv1d(ni, nf, kernel_size,
                              stride=stride, dilation=dilation, **kwargs)
    elif padding == 'causal':
        conv = CausalConv1d(ni, nf, kernel_size,
                            stride=stride, dilation=dilation, **kwargs)
    elif padding == 'valid':
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride,
                         padding=0, dilation=dilation, **kwargs)
    else:
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, **kwargs)
    init_linear(conv, None, init=init, bias_std=bias_std)
    return conv


class SeparableConv1d(Module):
    def __init__(
            self,
            ni,
            nf,
            ks,
            stride=1,
            padding='same',
            dilation=1,
            bias=True,
            bias_std=0.01):
        self.depthwise_conv = Conv1d(
            ni,
            ni,
            ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=ni,
            bias=bias)
        self.pointwise_conv = nn.Conv1d(
            ni, nf, 1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        if bias:
            if bias_std != 0:
                normal_(self.depthwise_conv.bias, 0, bias_std)
                normal_(self.pointwise_conv.bias, 0, bias_std)
            else:
                self.depthwise_conv.bias.data.zero_()
                self.pointwise_conv.bias.data.zero_()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


def SEModule1d(ni, reduction=16, act=nn.ReLU, act_kwargs={}):
    """Squeeze and excitation module for 1d"""
    nf = math.ceil(ni // reduction / 8) * 8
    assert nf != 0, 'nf cannot be 0'
    return SequentialEx(
        nn.AdaptiveAvgPool1d(1), ConvBlock(
            ni, nf, ks=1, norm=None, act=act, act_kwargs=act_kwargs), ConvBlock(
            nf, ni, ks=1, norm=None, act=nn.Sigmoid), ProdLayer())


class ConvBlock(nn.Sequential):
    """Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."""

    def __init__(
            self,
            ni,
            nf,
            kernel_size=None,
            ks=3,
            stride=1,
            padding='same',
            bias=None,
            bias_std=0.01,
            norm='Batch',
            zero_norm=False,
            batch_norm_1st=True,
            act=nn.ReLU,
            act_kwargs={},
            init='auto',
            dropout=0.,
            xtra=None,
            coord=False,
            separable=False,
            **kwargs):
        kernel_size = kernel_size or ks
        ndim = 1
        layers = [AddCoords1d()] if coord else []
        norm_type = getattr(
            NormType,
            f"{snake2camel(norm)}{'Zero' if zero_norm else ''}") if norm is not None else None
        batch_norm = norm_type in (NormType.Batch, NormType.BatchZero)
        inn = norm_type in (NormType.Instance, NormType.InstanceZero)
        if bias is None:
            bias = not (batch_norm or inn)
        if separable:
            conv = SeparableConv1d(
                ni + coord,
                nf,
                ks=kernel_size,
                bias=bias,
                stride=stride,
                padding=padding,
                **kwargs)
        else:
            conv = Conv1d(ni + coord, nf, ks=kernel_size, bias=bias,
                          stride=stride, padding=padding, **kwargs)
        act = None if act is None else act(**act_kwargs)
        if not separable:
            init_linear(conv, act, init=init, bias_std=bias_std)
        if norm_type == NormType.Weight:
            conv = weight_norm(conv)
        elif norm_type == NormType.Spectral:
            conv = spectral_norm(conv)
        layers += [conv]
        act_batch_norm = []
        if act is not None:
            act_batch_norm.append(act)
        if batch_norm:
            act_batch_norm.append(
                BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if inn:
            act_batch_norm.append(InstanceNorm(
                nf, norm_type=norm_type, ndim=ndim))
        if batch_norm_1st:
            act_batch_norm.reverse()
        if dropout:
            layers += [nn.Dropout(dropout)]
        layers += act_batch_norm
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


class MultiConv1d(Module):
    """Module that applies multiple convolutions with different kernel sizes"""

    def __init__(
            self,
            ni,
            nf=None,
            kss=[
                1,
                3,
                5,
                7],
            keep_original=False,
            separable=False,
            dim=1,
            **kwargs):
        kss = list(kss)
        n_layers = len(kss)
        if ni == nf:
            keep_original = False
        if nf is None:
            nf = ni * (keep_original + n_layers)
        nfs = [(nf - ni * keep_original) // n_layers] * n_layers
        while np.sum(nfs) + ni * keep_original < nf:
            for i in range(len(nfs)):
                nfs[i] += 1
                if np.sum(nfs) + ni * keep_original == nf:
                    break

        _conv = SeparableConv1d if separable else Conv1d
        self.layers = nn.ModuleList()
        for nfi, ksi in zip(nfs, kss):
            self.layers.append(_conv(ni, nfi, ksi, **kwargs))
        self.keep_original, self.dim = keep_original, dim

    def forward(self, x):
        output = [x] if self.keep_original else []
        for l in self.layers:
            output.append(l(x))
        x = torch.cat(output, dim=self.dim)
        return x
