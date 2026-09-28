from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.torch_core import Module
from fastcore.basics import snake2camel
from torch import Tensor

from fedot.industrial.core.architecture.settings.computational import default_device


def init_lin_zero(m):
    if isinstance(m, nn.Linear):
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 0)
    for l in m.children():
        init_lin_zero(l)


lin_zero_init = init_lin_zero


# class Flatten(nn.Module):
#     def __init__(self, out_features):
#         super(Flatten, self).__init__()
#         self.output_dim = out_features
#
#     def forward(self, x):
#         return x.view(-1, self.output_dim)
#

class Flatten(Module):

    def forward(self, x):
        if len(x.shape) < 4:
            return x
        else:
            bs, c, h, w = x.shape
            flattened_tensor = x.reshape(bs, c, h * w)
            return flattened_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}"


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, x):
        return x.view(-1, *self.out_shape)


def get_norm(nf, ndim=1, norm='Batch', zero_norm=False, init=True, **kwargs):
    """Norm layer with `nf` features and `ndim` with auto init."""
    assert 1 <= ndim <= 3
    nl = getattr(nn, f"{snake2camel(norm)}Norm{ndim}d")(nf, **kwargs)
    if nl.affine and init:
        nl.bias.data.fill_(1e-3)
        nl.weight.data.fill_(0. if zero_norm else 1.)
    return nl


BN1d = partial(get_norm, ndim=1, norm='Batch')
IN1d = partial(get_norm, ndim=1, norm='Instance')


class LinLnDrop(nn.Sequential):
    """Module grouping `LayerNorm1d`, `Dropout` and `Linear` layers"""

    def __init__(self, n_in, n_out, ln=True, p=0., act=None, lin_first=False):
        layers = [nn.LayerNorm(n_out if lin_first else n_in)] if ln else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not ln)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


class LambdaPlus(Module):
    def __init__(self, func, *args, **
                 kwargs): self.func, self.args, self.kwargs = func, args, kwargs

    def forward(self, x):
        return self.func(x, *self.args, **self.kwargs)


class Squeeze(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Unsqueeze(Module):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Add(Module):
    def forward(self, x, y):
        return x.add(y)

    def __repr__(self):
        return f'{self.__class__.__name__}'


class Concat(Module):
    def __init__(self, dim=1):
        self.dim = dim

    def forward(self, *x):
        return torch.cat(*x, dim=self.dim)

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim})'


class Unfold(Module):
    def __init__(self, dim, size,
                 step=1):
        self.dim, self.size, self.step = dim, size, step

    def forward(self, x: Tensor) -> Tensor:
        return x.unfold(dimension=self.dim, size=self.size, step=self.step)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, size={self.size}, step={self.step})"


class Permute(Module):
    def __init__(self, *dims):
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])})"


class Transpose(Module):
    def __init__(self, *dims, contiguous=False):
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

    def __repr__(self):
        if self.contiguous:
            return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else:
            return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"


class View(Module):
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], -1).contiguous() if not self.shape else x.view(-1).contiguous(
        ) if self.shape == (-1,) else x.view(x.shape[0], *self.shape).contiguous()

    def __repr__(
        self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"


class Reshape(Module):
    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(
            -1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"


class Max(Module):
    def __init__(self, dim=None, keepdim=False):
        self.dim, self.keepdim = dim, keepdim

    def forward(self, x):
        return x.max(self.dim, keepdim=self.keepdim)[0]

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.dim}, keepdim={self.keepdim})'


class LastStep(Module):
    def forward(self, x):
        return x[..., -1]

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class SoftMax(Module):
    """SoftMax layer"""

    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)

    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


class Clamp(Module):
    def __init__(self, min_=None, max_=None):
        self.min, self.max = min_, max_

    def forward(self, x):
        return x.clamp(min=self.min, max=self.max)

    def __repr__(self):
        return f'{self.__class__.__name__}(min={self.min}, max={self.max})'


class Clip(Module):
    def __init__(self, min_=None, max_=None):
        self.min, self.max = min_, max_

    def forward(self, x):
        if self.min is not None:
            x = torch.maximum(x, self.min)
        if self.max is not None:
            x = torch.minimum(x, self.max)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class ReZero(Module):
    def __init__(self, module_):
        self.module = module_
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.module(x)


class AddCoords1d(Module):
    """Add coordinates to ease position identification without modifying mean and std"""

    def forward(self, x):
        bs, _, seq_len = x.shape
        cc = torch.linspace(-1, 1, x.shape[-1],
                            device=x.device).repeat(bs, 1, 1)
        cc = (cc - cc.mean()) / cc.std()
        x = torch.cat([x, cc], dim=1)
        return x


class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, pred_dim):
        super().__init__()

        if isinstance(pred_dim, (tuple, list)):
            pred_dim = pred_dim[-1]
        self.individual = individual
        self.n = n_vars if individual else 1
        self.nf, self.pred_dim = nf, pred_dim

        if individual:
            self.layers = nn.ModuleList()
            for i in range(self.n):
                self.layers.append(nn.Sequential(nn.Flatten(
                    start_dim=-2), nn.Linear(nf, pred_dim)))
        else:
            self.layer = nn.Sequential(nn.Flatten(
                start_dim=-2), nn.Linear(nf, pred_dim))

    def forward(self, x: Tensor):
        """
        Args:
            x: [bs x nvars x d_model x n_patch]
        Returns:
            output: [bs x nvars x pred_dim]
        """
        if self.individual:
            x_out = []
            for i, layer in enumerate(self.layers):
                x_out.append(layer(x[:, i]))
            x = torch.stack(x_out, dim=1)
            return x
        else:
            try:
                return self.layer(x)
            except Exception:
                self.layer = nn.Sequential(
                    nn.Flatten(
                        start_dim=-2),
                    nn.Linear(
                        x.shape[3] * self.nf,
                        self.pred_dim,
                        device=default_device()))
                return self.layer(x)


Noop = nn.Sequential()
