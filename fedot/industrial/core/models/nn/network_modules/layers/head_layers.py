# from tensorflow.python.keras.layers.convolutional import Conv
from torch import tensor
from fedot.industrial.core.models.nn.network_modules.layers.conv_layers import Conv1d, ConvBlock
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import *
from fedot.industrial.core.models.nn.network_modules.layers.pooling_layers import (
    AdaptiveWeightedAvgPool1d,
    attentional_pool_head,
    GACP1d,
    GAP1d,
    gwa_pool_head
)
from fastai.layers import SigmoidRange, LinBnDrop, AdaptiveConcatPool1d, BatchNorm
from torch.nn import Conv3d


def create_pool_head(n_in, output_dim, seq_len=None, concat_pool=False,
                     fc_dropout=0., batch_norm=False, y_range=None, **kwargs):
    if kwargs:
        print(f'{kwargs}  not being used')
    if concat_pool:
        n_in *= 2
    layers = [GACP1d(1) if concat_pool else GAP1d(1)]
    layers += [LinBnDrop(n_in, output_dim,
                         bn=batch_norm, p=fc_dropout)]
    if y_range:
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


def max_pool_head(n_in, output_dim, seq_len, fc_dropout=0., batch_norm=False,
                  y_range=None, **kwargs):
    if kwargs:
        print(f'{kwargs}  not being used')
    layers = [nn.MaxPool1d(seq_len, **kwargs), Reshape()]
    layers += [LinBnDrop(n_in, output_dim,
                         bn=batch_norm, p=fc_dropout)]
    if y_range:
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


def create_pool_plus_head(
    *args,
    lin_ftrs=None,
    fc_dropout=0.,
    concat_pool=True,
    batch_norm_final=False,
    lin_first=False,
        y_range=None):
    nf = args[0]
    output_dim = args[1]
    if concat_pool:
        nf = nf * 2
    lin_ftrs = [nf, 512, output_dim] if lin_ftrs is None else [
        nf] + lin_ftrs + [output_dim]
    ps = list(fc_dropout)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    pool = AdaptiveConcatPool1d() if concat_pool else nn.AdaptiveAvgPool1d(1)
    layers = [pool, Reshape()]
    if lin_first:
        layers.append(nn.Dropout(ps.pop(0)))
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p,
                            act=actn, lin_first=lin_first)
    if lin_first:
        layers.append(nn.Linear(lin_ftrs[-2], output_dim))
    if batch_norm_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None:
        layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)


def create_conv_head(*args, adaptive_size=None, y_range=None):
    nf = args[0]
    output_dim = args[1]
    layers = [nn.AdaptiveAvgPool1d(
        adaptive_size)] if adaptive_size is not None else []
    for i in range(2):
        if nf > 1:
            layers += [ConvBlock(nf, nf // 2, 1)]
            nf = nf // 2
        else:
            break
    layers += [ConvBlock(nf, output_dim, 1), GAP1d(1)]
    if y_range:
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


def create_mlp_head(
        nf,
        output_dim,
        seq_len=None,
        flatten=True,
        fc_dropout=0.,
        batch_norm=False,
        lin_first=False,
        y_range=None):
    if flatten:
        nf *= seq_len
    layers = [Reshape()] if flatten else []
    layers += [LinBnDrop(nf, output_dim, bn=batch_norm,
                         p=fc_dropout, lin_first=lin_first)]
    if y_range:
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


def create_fc_head(nf, output_dim, seq_len=None, flatten=True, lin_ftrs=None,
                   y_range=None, fc_dropout=0., batch_norm=False,
                   batch_norm_final=False, act=nn.ReLU(inplace=True)):
    if flatten:
        nf *= seq_len
    layers = [Reshape()] if flatten else []
    lin_ftrs = [nf, 512, output_dim] if lin_ftrs is None else [
        nf] + lin_ftrs + [output_dim]
    if not isinstance(fc_dropout, list):
        fc_dropout = [fc_dropout] * (len(lin_ftrs) - 1)
    actns = [act for _ in range(len(lin_ftrs) - 2)] + [None]
    layers += [LinBnDrop(lin_ftrs[i],
                         lin_ftrs[i + 1],
                         bn=batch_norm and (
                             i != len(actns) - 1 or batch_norm_final),
                         p=p,
                         act=a) for i, (p, a) in enumerate(zip(fc_dropout + [0.], actns))]
    if y_range is not None:
        layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)


def create_rnn_head(*args, fc_dropout=0., batch_norm=False, y_range=None):
    nf = args[0]
    output_dim = args[1]
    layers = [LastStep()]
    layers += [LinBnDrop(nf, output_dim, bn=batch_norm, p=fc_dropout)]
    if y_range:
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


def imputation_head(
        input_dim,
        output_dim,
        seq_len=None,
        ks=1,
        y_range=None,
        fc_dropout=0.):
    layers = [nn.Dropout(fc_dropout), nn.Conv1d(output_dim, output_dim, ks)]
    if y_range is not None:
        y_range = (tensor(y_range[0]), tensor(y_range[1]))
        layers += [SigmoidRange(*y_range)]
    return nn.Sequential(*layers)


class CreateConvLinNDHead(nn.Sequential):
    """Module to create a nd output head"""

    def __init__(
            self,
            n_in,
            n_out,
            seq_len,
            d,
            conv_first=True,
            conv_batch_norm=False,
            lin_batch_norm=False,
            fc_dropout=0.,
            **kwargs):

        assert d, "you cannot use an nd head when d is None or 0"
        if isinstance(d, list):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1:
                shape.append(n_out)
        else:
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]

        conv = [BatchNorm(n_in, ndim=1)] if conv_batch_norm else []
        conv.append(Conv1d(n_in, n_out, 1, padding=0,
                           bias=not conv_batch_norm, **kwargs))
        l = [Transpose(-1, -2), BatchNorm(seq_len, ndim=1),
             Transpose(-1, -2)] if lin_batch_norm else []
        if fc_dropout != 0:
            l.append(nn.Dropout(fc_dropout))
        lin = [nn.Linear(seq_len, fd, bias=not lin_batch_norm)]
        lin_layers = l + lin
        layers = conv + lin_layers if conv_first else lin_layers + conv
        layers += [Transpose(-1, -2)]
        layers += [Reshape(*shape)]

        super().__init__(*layers)


class LinNDHead(nn.Sequential):
    """Module to create a nd output head with linear layers"""

    def __init__(
            self,
            n_in,
            n_out,
            seq_len=None,
            d=None,
            flatten=False,
            use_batch_norm=False,
            fc_dropout=0.):

        if seq_len is None:
            seq_len = 1
        if d is None:
            fd = 1
            shape = [n_out]
        elif isinstance(d, list):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1:
                shape.append(n_out)
        else:
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]

        layers = []
        if use_batch_norm:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        if d is None:
            if not flatten or seq_len == 1:
                layers += [nn.AdaptiveAvgPool1d(1),
                           Squeeze(-1), nn.Linear(n_in, n_out)]
                if n_out == 1:
                    layers += [Squeeze(-1)]
            else:
                layers += [Reshape(), nn.Linear(n_in * seq_len, n_out * fd)]
                if n_out * fd == 1:
                    layers += [Squeeze(-1)]
        else:
            if seq_len == 1:
                layers += [nn.AdaptiveAvgPool1d(1)]
            if not flatten and fd == seq_len:
                layers += [Transpose(1, 2), nn.Linear(n_in, n_out)]
            else:
                layers += [Reshape(), nn.Linear(n_in * seq_len, n_out * fd)]
            layers += [Reshape(*shape)]

        super().__init__(*layers)


class RocketNDHead(nn.Sequential):
    """Module to create a nd output head with linear layers for the rocket family of models"""

    def __init__(
            self,
            n_in,
            n_out,
            seq_len=None,
            d=None,
            use_batch_norm=False,
            fc_dropout=0.,
            zero_init=True):

        if d is None:
            fd = 1
            shape = [n_out]
        elif isinstance(d, list):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1:
                shape.append(n_out)
        else:
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]

        layers = [nn.Flatten()]
        if use_batch_norm:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        linear = nn.Linear(n_in, fd * n_out)
        if zero_init:
            nn.init.constant_(linear.weight.data, 0)
            nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        if d is None and n_out == 1:
            layers += [Squeeze(-1)]
        if d is not None:
            layers += [Reshape(*shape)]

        super().__init__(*layers)


class Xresnet1dNDHead(nn.Sequential):
    """Module to create a nd output head with linear layers for the xresnet family of models"""

    def __init__(
            self,
            n_in,
            n_out,
            seq_len=None,
            d=None,
            use_batch_norm=False,
            fc_dropout=0.,
            zero_init=True):

        if d is None:
            fd = 1
            shape = [n_out]
        elif isinstance(d, list):
            fd = 1
            shape = []
            for _d in d:
                fd *= _d
                shape.append(_d)
            if n_out > 1:
                shape.append(n_out)
        else:
            fd = d
            shape = [d, n_out] if n_out > 1 else [d]

        layers = [nn.AdaptiveAvgPool1d(1), nn.Flatten()]
        if use_batch_norm:
            layers += [nn.BatchNorm1d(n_in)]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        linear = nn.Linear(n_in, fd * n_out)
        if zero_init:
            nn.init.constant_(linear.weight.data, 0)
            nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        if d is None and n_out == 1:
            layers += [Squeeze(-1)]
        if d is not None:
            layers += [Reshape(*shape)]

        super().__init__(*layers)


class CreateConv3dHead(nn.Sequential):
    """Module to create a nd output head with a convolutional layer"""

    def __init__(
            self,
            n_in,
            n_out,
            seq_len,
            d,
            use_batch_norm=False,
            **kwargs):
        assert d, "you cannot use an 3d head when d is None or 0"
        assert d == seq_len, 'You can only use this head when learn.dls.len == learn.dls.d'
        layers = [nn.BatchNorm1d(n_in)] if use_batch_norm else []
        layers += [Conv3d(n_in, n_out, 1, **kwargs), Transpose(-1, -2)]
        # layers += [Conv(n_in, n_out, 1, **kwargs), Transpose(-1, -2)]
        if n_out == 1:
            layers += [Squeeze(-1)]
        super().__init__(*layers)


def universal_pool_head(
        n_in,
        output_dim,
        seq_len,
        mult=2,
        pool_n_layers=2,
        pool_ln=True,
        pool_dropout=0.5,
        pool_act=nn.ReLU(),
        zero_init=True,
        batch_norm=True,
        fc_dropout=0.):
    return nn.Sequential(
        AdaptiveWeightedAvgPool1d(
            n_in,
            seq_len,
            n_layers=pool_n_layers,
            mult=mult,
            ln=pool_ln,
            dropout=pool_dropout,
            act=pool_act),
        Reshape(),
        LinBnDrop(
            n_in,
            output_dim,
            p=fc_dropout,
            bn=batch_norm))


pool_head = create_pool_head
average_pool_head = partial(pool_head, concat_pool=False)
setattr(average_pool_head, "__name__", "average_pool_head")
concat_pool_head = partial(pool_head, concat_pool=True)
setattr(concat_pool_head, "__name__", "concat_pool_head")
pool_plus_head = create_pool_plus_head
conv_head = create_conv_head
mlp_head = create_mlp_head
fc_head = create_fc_head
rnn_head = create_rnn_head
conv_lin_nd_head = CreateConvLinNDHead
conv_lin_3d_head = CreateConvLinNDHead  # included for compatibility
create_conv_lin_3d_head = CreateConvLinNDHead  # included for compatibility
conv_3d_head = CreateConv3dHead
create_lin_nd_head = LinNDHead
lin_3d_head = LinNDHead  # included for backwards compatiblity
create_lin_3d_head = LinNDHead  # included for backwards compatiblity

heads = [
    mlp_head,
    fc_head,
    average_pool_head,
    max_pool_head,
    concat_pool_head,
    pool_plus_head,
    conv_head,
    rnn_head,
    conv_lin_nd_head,
    LinNDHead,
    conv_3d_head,
    attentional_pool_head,
    universal_pool_head,
    gwa_pool_head]
