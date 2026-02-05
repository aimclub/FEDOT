from numbers import Integral

import torch.nn as nn
from fastai.torch_core import Module


class Pad1d(nn.ConstantPad1d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)


class SameConv1d(Module):
    """Conv1d with padding='same'"""

    def __init__(self, ni, nf, ks=3, stride=1, dilation=1, **kwargs):
        self.ks, self.stride, self.dilation = ks, stride, dilation
        self.conv1d_same = nn.Conv1d(
            ni, nf, ks, stride=stride, dilation=dilation, **kwargs)
        self.weight = self.conv1d_same.weight
        self.bias = self.conv1d_same.bias
        self.pad = Pad1d

    def forward(self, x):
        # stride=self.stride not used in padding calculation!
        self.padding = same_padding1d(
            x.shape[-1], self.ks, dilation=self.dilation)
        return self.conv1d_same(self.pad(self.padding)(x))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def same_padding1d(seq_len, ks, stride=1, dilation=1):
    """Same padding formula as used in Tensorflow"""
    p = (seq_len - 1) * stride + (ks - 1) * dilation + 1 - seq_len
    return p // 2, p - p // 2


def same_padding2d(H, W, ks, stride=(1, 1), dilation=(1, 1)):
    """Same padding formula as used in Tensorflow"""
    if isinstance(ks, Integral):
        ks = (ks, ks)
    if ks[0] == 1:
        p_h = 0
    else:
        p_h = (H - 1) * stride[0] + (ks[0] - 1) * dilation[0] + 1 - H
    if ks[1] == 1:
        p_w = 0
    else:
        p_w = (W - 1) * stride[1] + (ks[1] - 1) * dilation[1] + 1 - W
    return p_w // 2, p_w - p_w // 2, p_h // 2, p_h - p_h // 2


class Pad2d(nn.ConstantPad2d):
    def __init__(self, padding, value=0.):
        super().__init__(padding, value)
