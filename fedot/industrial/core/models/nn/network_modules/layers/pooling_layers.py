import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import *
from fastai.torch_core import Module
from fastcore.basics import store_attr

from fedot.industrial.core.models.nn.network_modules.layers.conv_layers import Conv1d
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import init_lin_zero, LinLnDrop, Reshape, SoftMax


class PPV(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return torch.gt(x, 0).sum(dim=self.dim).float() / x.shape[self.dim]

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class PPAuc(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        x = F.relu(x).sum(self.dim) / (abs(x).sum(self.dim) + 1e-8)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class MaxPPVPool1d(Module):
    """Drop-in replacement for AdaptiveConcatPool1d - multiplies nf by 2"""

    def forward(self, x):
        _max = x.max(dim=-1).values
        _ppv = torch.gt(x, 0).sum(dim=-1).float() / x.shape[-1]
        return torch.cat((_max, _ppv), dim=-1).unsqueeze(2)


class AdaptiveWeightedAvgPool1d(Module):
    """Global Pooling layer that performs a weighted average along the temporal axis

    It can be considered as a channel-wise form of local temporal attention. Inspired by the paper:
    Hyun, J., Seong, H., & Kim, E. (2019). Universal Pooling--A New Pooling Method for Convolutional Neural Networks.
    arXiv preprint arXiv:1907.11440.
    """

    def __init__(self, n_in,
                 seq_len,
                 mult=2,
                 n_layers=2,
                 ln=False,
                 dropout=0.5,
                 act=nn.ReLU(),
                 zero_init=True):
        layers = nn.ModuleList()
        for i in range(n_layers):
            inp_mult = mult if i > 0 else 1
            out_mult = mult if i < n_layers - 1 else 1
            p = dropout[i] if isinstance(dropout, list) else dropout
            layers.append(
                LinLnDrop(
                    seq_len *
                    inp_mult,
                    seq_len *
                    out_mult,
                    ln=False,
                    p=p,
                    act=act if i < n_layers -
                    1 and n_layers > 1 else None))
        self.layers = layers
        self.softmax = SoftMax(-1)
        if zero_init:
            init_lin_zero(self)

    def forward(self, x):
        wap = x
        for l in self.layers:
            wap = l(wap)
        wap = self.softmax(wap)
        return torch.mul(x, wap).sum(-1)


class GAP1d(Module):
    """Global Adaptive Pooling + Flatten"""

    def __init__(self, output_size=1):
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()

    def forward(self, x):
        return self.flatten(self.gap(x))


class GACP1d(Module):
    "Global AdaptiveConcatPool + Flatten"

    def __init__(self, output_size=1):
        self.gacp = AdaptiveConcatPool1d(output_size)
        self.flatten = Reshape()

    def forward(self, x):
        return self.flatten(self.gacp(x))


class GAWP1d(Module):
    "Global AdaptiveWeightedAvgPool1d + Flatten"

    def __init__(self, n_in,
                 seq_len,
                 n_layers=2,
                 ln=False,
                 dropout=0.5,
                 act=nn.ReLU(),
                 zero_init=False):
        self.gacp = AdaptiveWeightedAvgPool1d(n_in,
                                              seq_len,
                                              n_layers=n_layers,
                                              ln=ln,
                                              dropout=dropout,
                                              act=act,
                                              zero_init=zero_init)
        self.flatten = Reshape()

    def forward(self, x):
        return self.flatten(self.gacp(x))


class GlobalWeightedAveragePool1d(Module):
    """ Global Weighted Average Pooling layer

    Inspired by Building Efficient CNN Architecture for Offline Handwritten Chinese Character Recognition
    https://arxiv.org/pdf/1804.01259.pdf
    """

    def __init__(self, n_in, seq_len):
        self.weight = nn.Parameter(torch.ones(1, n_in, seq_len))
        self.bias = nn.Parameter(torch.zeros(1, n_in, seq_len))

    def forward(self, x):
        α = F.softmax(torch.sigmoid(x * self.weight + self.bias), dim=-1)
        return (x * α).sum(-1)


GWAP1d = GlobalWeightedAveragePool1d


def gwa_pool_head(n_in, output_dim, seq_len, batch_norm=True, fc_dropout=0.):
    return nn.Sequential(
        GlobalWeightedAveragePool1d(
            n_in, seq_len), Reshape(), LinBnDrop(
            n_in, output_dim, p=fc_dropout, bn=batch_norm))


class AttentionalPool1d(Module):
    """Global Adaptive Pooling layer inspired by Attentional Pooling for Action
    Recognition https://arxiv.org/abs/1711.01467"""

    def __init__(self, n_in, output_dim, batch_norm=False):
        store_attr()
        self.batch_norm = nn.BatchNorm1d(n_in) if batch_norm else None
        self.conv1 = Conv1d(n_in, 1, 1)
        self.conv2 = Conv1d(n_in, output_dim, 1)

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return (self.conv1(x) @ self.conv2(x).transpose(1, 2)).transpose(1, 2)


class GAttP1d(nn.Sequential):
    def __init__(self, n_in, output_dim, batch_norm=False):
        super().__init__(
            AttentionalPool1d(
                n_in,
                output_dim,
                batch_norm=batch_norm),
            Reshape())


def attentional_pool_head(
        n_in,
        output_dim,
        seq_len=None,
        batch_norm=True,
        **kwargs):
    return nn.Sequential(
        AttentionalPool1d(
            n_in,
            output_dim,
            batch_norm=batch_norm,
            **kwargs),
        Reshape())


class PoolingLayer(Module):
    def __init__(self, method='cls', seq_len=None, token=True, seq_last=True):
        method = method.lower()
        assert method in ['cls', 'max', 'mean',
                          'max-mean', 'linear', 'conv1d', 'flatten']
        if method == 'cls':
            assert token, 'you can only choose method=cls if a token exists'
        self.method = method
        self.token = token
        self.seq_last = seq_last
        if method == 'linear' or method == 'conv1d':
            self.linear = nn.Linear(seq_len - token, 1)

    def forward(self, x):
        if self.method == 'cls':
            return x[..., 0] if self.seq_last else x[:, 0]
        if self.token:
            x = x[..., 1:] if self.seq_last else x[:, 1:]
        if self.method == 'max':
            return torch.max(x, -1)[0] if self.seq_last else torch.max(x, 1)[0]
        elif self.method == 'mean':
            return torch.mean(x, -1) if self.seq_last else torch.mean(x, 1)
        elif self.method == 'max-mean':
            return torch.cat([torch.max(x, - 1)[0] if self.seq_last else torch.max(x, 1)[0],
                             torch.mean(x, - 1) if self.seq_last else torch.mean(x, 1)], 1)
        elif self.method == 'flatten':
            return x.flatten(1)
        elif self.method == 'linear' or self.method == 'conv1d':
            return self.linear(x)[..., 0] if self.seq_last else self.linear(
                x.transpose(1, 2))[..., 0]

    def __repr__(self):
        return f"{self.__class__.__name__}(method={self.method}, token={self.token}, seq_last={self.seq_last})"
