from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import *
from fastai.torch_core import Module
from fastcore.basics import listify
from torch import Tensor

from fedot.industrial.core.models.nn.network_modules.activation import pytorch_act_names, pytorch_acts
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import Noop
from fedot.industrial.core.models.nn.network_modules.layers.pooling_layers import GAP1d


def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes


def pass_through(X):
    return X


def if_module_to_torchscript(
        m: torch.nn.Module,
        inputs: Tensor,
        trace: bool = True,
        script: bool = True,
        serialize: bool = True,
        verbose: bool = True,
):
    """Tests if a PyTorch module can be correctly traced or scripted and serialized

    Args:
        m (torch.nn.Module): The PyTorch module to be tested.
        inputs (Tensor): A tensor or tuple of tensors representing the inputs to the model.
        trace (bool, optional): If `True`, attempts to trace the model. Defaults to `True`.
        script (bool, optional): If `True`, attempts to script the model. Defaults to `True`.
        serialize (bool, optional): If `True`, saves and loads the traced/scripted module to ensure it can be serialized. Defaults to `True`.
        verbose (bool, optional): If `True`, prints detailed information about the tracing and scripting process. Defaults to `True`.

    """

    m = m.eval()
    m_name = m.__class__.__name__

    # Ensure inputs are in a tuple or list format
    inp_is_tuple = isinstance(inputs, (tuple, list))

    # Get the model's output
    output = m(*inputs) if inp_is_tuple else m(inputs)
    output_shapes = output.shape if not isinstance(output, (tuple, list)) else [
        o.shape for o in output]
    if verbose:
        print(f"output.shape: {output_shapes}")

    # Try tracing the model
    if trace:
        if verbose:
            print("Tracing...")
        try:
            traced_m = torch.jit.trace(m, inputs)
            if serialize:
                file_path = Path(f"test_traced_{m_name}.pt")
                torch.jit.save(traced_m, file_path)
                torch.jit.load(file_path)
                file_path.unlink()
            traced_output = traced_m(
                *inputs) if inp_is_tuple else traced_m(inputs)
            torch.testing.assert_close(traced_output, output)
            if verbose:
                print(f"...{m_name} has been successfully traced 😃\n")
            return True
        except Exception as e:
            if verbose:
                print(f"{m_name} cannot be traced 😔")
                print(e)
                print("\n")

    # Try scripting the model
    if script:
        if verbose:
            print("Scripting...")
        try:
            scripted_m = torch.jit.script(m)
            if serialize:
                file_path = Path(f"test_scripted_{m_name}.pt")
                torch.jit.save(scripted_m, file_path)
                torch.jit.load(file_path)
                file_path.unlink()
            scripted_output = scripted_m(
                *inputs) if inp_is_tuple else scripted_m(inputs)
            torch.testing.assert_close(scripted_output, output)
            if verbose:
                print(f"...{m_name} has been successfully scripted 😃\n")
            return True
        except Exception as e:
            if verbose:
                print(f"{m_name} cannot be scripted 😔")
                print(e)

    return False


def init_lin_zero(m):
    if isinstance(m, nn.Linear):
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 0)
    for l in m.children():
        init_lin_zero(l)


lin_zero_init = init_lin_zero


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    It's similar to Dropout but it drops individual connections instead of nodes.
    Original code in https://github.com/rwightman/pytorch-image-models (timm library)
    """

    def __init__(self, p=None):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        # output = x.div(random_tensor.mean()) * random_tensor # divide by the
        # actual mean to mantain the input mean?
        return output


class Sharpen(Module):
    """This is used to increase confidence in predictions - MixMatch paper"""

    def __init__(self, T=.5): self.T = T

    def forward(self, x):
        x = x ** (1. / self.T)
        return x / x.sum(dim=1, keepdims=True)


class Sequential(nn.Sequential):
    """Class that allows you to pass one or multiple inputs"""

    def forward(self, *x):
        for i, module in enumerate(self._modules.values()):
            x = module(*x) if isinstance(x, (list, tuple, L)) else module(x)
        return x


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))

        return y


class TempScale(Module):
    """Used to perform Temperature Scaling (dirichlet=False) or Single-parameter Dirichlet calibration
    (dirichlet=True)"""

    def __init__(self, temp=1., dirichlet=False):
        self.weight = nn.Parameter(Tensor(temp))
        self.bias = None
        self.log_softmax = dirichlet

    def forward(self, x):
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1)
        return x.div(self.weight)


class VectorScale(Module):
    """Used to perform Vector Scaling (dirichlet=False) or Diagonal Dirichlet calibration (dirichlet=True)"""

    def __init__(self, n_classes=1, dirichlet=False):
        self.weight = nn.Parameter(torch.ones(n_classes))
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.log_softmax = dirichlet

    def forward(self, x):
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1)
        return x.mul(self.weight).add(self.bias)


class MatrixScale(Module):
    """Used to perform Matrix Scaling (dirichlet=False) or Dirichlet calibration (dirichlet=True)"""

    def __init__(self, n_classes=1, dirichlet=False):
        self.ms = nn.Linear(n_classes, n_classes)
        self.ms.weight.data = nn.Parameter(torch.eye(n_classes))
        nn.init.constant_(self.ms.bias.data, 0.)
        self.weight = self.ms.weight
        self.bias = self.ms.bias
        self.log_softmax = dirichlet

    def forward(self, x):
        if self.log_softmax:
            x = F.log_softmax(x, dim=-1)
        return self.ms(x)


def get_calibrator(calibrator=None, n_classes=1, **kwargs):
    if calibrator is None or not calibrator:
        return Noop
    elif calibrator.lower() == 'temp':
        return TempScale(dirichlet=False, **kwargs)
    elif calibrator.lower() == 'vector':
        return VectorScale(n_classes=n_classes, dirichlet=False, **kwargs)
    elif calibrator.lower() == 'matrix':
        return MatrixScale(n_classes=n_classes, dirichlet=False, **kwargs)
    elif calibrator.lower() == 'dtemp':
        return TempScale(dirichlet=True, **kwargs)
    elif calibrator.lower() == 'dvector':
        return VectorScale(n_classes=n_classes, dirichlet=True, **kwargs)
    elif calibrator.lower() == 'dmatrix':
        return MatrixScale(n_classes=n_classes, dirichlet=True, **kwargs)
    else:
        assert False, f'please, select a correct calibrator instead of {calibrator}'


class LogitAdjustmentLayer(Module):
    """Logit Adjustment for imbalanced datasets"""

    def __init__(self, class_priors):
        self.class_priors = class_priors

    def forward(self, x):
        return x.add(self.class_priors)


LogitAdjLayer = LogitAdjustmentLayer


def get_act_fn(act, **act_kwargs):
    if act is None:
        return
    elif isinstance(act, nn.Module):
        return act
    elif callable(act):
        return act(**act_kwargs)
    idx = pytorch_act_names.index(act.lower())
    return pytorch_acts[idx](**act_kwargs)


class SqueezeExciteBlock(Module):
    def __init__(self, ni, reduction=16):
        self.avg_pool = GAP1d(1)
        self.fc = nn.Sequential(
            nn.Linear(
                ni,
                ni // reduction,
                bias=False),
            nn.ReLU(),
            nn.Linear(
                ni // reduction,
                ni,
                bias=False),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y).unsqueeze(2)
        return x * y.expand_as(x)


class GaussianNoise(Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value yours are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        self.sigma, self.is_relative_detach = sigma, is_relative_detach

    def forward(self, x):
        if self.training and self.sigma not in [0, None]:
            scale = self.sigma * (x.detach() if self.is_relative_detach else x)
            sampled_noise = torch.empty(
                x.size(), device=x.device).normal_() * scale
            x = x + sampled_noise
        return x


class PositionwiseFeedForward(nn.Sequential):
    def __init__(self, dim, dropout=0., act='reglu', mlp_ratio=1):
        act_mult = 2 if act.lower() in ["geglu", "reglu"] else 1
        super().__init__(nn.Linear(dim, dim * mlp_ratio * act_mult),
                         get_act_fn(act),
                         nn.Dropout(dropout),
                         nn.Linear(dim * mlp_ratio, dim),
                         nn.Dropout(dropout))


class TokenLayer(Module):
    def __init__(self, token=True):
        self.token = token

    def forward(self, x):
        return x[..., 0] if self.token is not None else x.mean(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class LSTMOutput(Module):
    def forward(self, x):
        return x[0]

    def __repr__(self):
        return f'{self.__class__.__name__}()'


def emb_sz_rule(n_cat):
    """Rule of thumb to pick embedding size corresponding to `n_cat` (original from fastai)"""
    return min(600, round(1.6 * n_cat ** 0.56))


class TSEmbedding(nn.Embedding):
    """Embedding layer with truncated normal initialization adapted from fastai"""

    def __init__(self, ni, nf, std=0.01, padding_idx=None):
        super().__init__(ni, nf)
        trunc_normal_(self.weight.data, std=std)
        if padding_idx is not None:
            nn.init.zeros_(self.weight.data[padding_idx])


class MultiEmbedding(Module):
    def __init__(
            self,
            c_in,
            n_cat_embeds,
            cat_embed_dims=None,
            cat_pos=None,
            std=0.01,
            cat_padding_idxs=None):
        cat_n_embeds = listify(n_cat_embeds)
        if cat_padding_idxs is None:
            cat_padding_idxs = [None]
        else:
            cat_padding_idxs = listify(cat_padding_idxs)
        if len(cat_padding_idxs) == 1 and len(
                cat_padding_idxs) < len(cat_n_embeds):
            cat_padding_idxs = cat_padding_idxs * len(cat_n_embeds)
        assert len(cat_n_embeds) == len(cat_padding_idxs)
        if cat_embed_dims is None:
            cat_embed_dims = [emb_sz_rule(s) for s in cat_n_embeds]
        else:
            cat_embed_dims = listify(cat_embed_dims)
            if len(cat_embed_dims) == 1:
                cat_embed_dims = cat_embed_dims * len(cat_n_embeds)
            assert len(cat_embed_dims) == len(cat_n_embeds)
        if cat_pos:
            cat_pos = torch.as_tensor(listify(cat_pos))
        else:
            cat_pos = torch.arange(len(cat_n_embeds))
        self.register_buffer("cat_pos", cat_pos)
        cont_pos = torch.tensor(
            [p for p in torch.arange(c_in) if p not in self.cat_pos])
        self.register_buffer("cont_pos", cont_pos)
        self.cat_embed = nn.ModuleList(
            [
                TSEmbedding(
                    n, d, std=std, padding_idx=p) for n, d, p in zip(
                    cat_n_embeds, cat_embed_dims, cat_padding_idxs)])

    def forward(self, x):
        if isinstance(x, tuple):
            x_cat, x_cont, *_ = x
        else:
            x_cat, x_cont = x[:, self.cat_pos], x[:, self.cont_pos]
        x_cat = torch.cat([e(torch.round(x_cat[:, i]).long()).transpose(
            1, 2) for i, e in enumerate(self.cat_embed)], 1)
        return torch.cat([x_cat, x_cont], 1)

# def build_ts_model(arch, c_in=None, c_out=None, seq_len=None, d=None, dls=None, device=None, verbose=False,
#                    s_cat_idxs=None, s_cat_embeddings=None, s_cat_embedding_dims=None, s_cont_idxs=None,
#                    o_cat_idxs=None, o_cat_embeddings=None, o_cat_embedding_dims=None, o_cont_idxs=None,
#                    patch_len=None, patch_stride=None, fusion_layers=128, fusion_act='relu', fusion_dropout=0.,
#                    fusion_use_bn=True,
#                    pretrained=False, weights_path=None, exclude_head=True, cut=-1, init=None, arch_config={}, **kwargs):
#     device = ifnone(device, default_device())
#     if dls is not None:
#         c_in = ifnone(c_in, dls.vars)
#         c_out = ifnone(c_out, dls.c)
#         seq_len = ifnone(seq_len, dls.len)
#         d = ifnone(d, dls.d)
#
#     if s_cat_idxs or s_cat_embeddings or s_cat_embedding_dims or s_cont_idxs or o_cat_idxs or o_cat_embeddings or o_cat_embedding_dims or o_cont_idxs:
#         from tsai.models.multimodal import MultInputWrapper
#         model = MultInputWrapper(arch, c_in=c_in, c_out=c_out, seq_len=seq_len, d=d,
#                                  s_cat_idxs=s_cat_idxs, s_cat_embeddings=s_cat_embeddings,
#                                  s_cat_embedding_dims=s_cat_embedding_dims, s_cont_idxs=s_cont_idxs,
#                                  o_cat_idxs=o_cat_idxs, o_cat_embeddings=o_cat_embeddings,
#                                  o_cat_embedding_dims=o_cat_embedding_dims, o_cont_idxs=o_cont_idxs,
#                                  patch_len=patch_len, patch_stride=patch_stride,
#                                  fusion_layers=fusion_layers, fusion_act=fusion_act, fusion_dropout=fusion_dropout,
#                                  fusion_use_bn=fusion_use_bn,
#                                  **kwargs)
#     else:
#         if d and arch.__name__ not in ["PatchTST", "PatchTSTPlus", 'TransformerRNNPlus', 'TransformerLSTMPlus',
#                                        'TransformerGRUPlus']:
#             if 'custom_head' not in kwargs.keys():
#                 if "rocket" in arch.__name__.lower():
#                     kwargs['custom_head'] = partial(rocket_nd_head, d=d)
#                 elif "xresnet1d" in arch.__name__.lower():
#                     kwargs["custom_head"] = partial(xresnet1d_nd_head, d=d)
#                 else:
#                     kwargs['custom_head'] = partial(lin_nd_head, d=d)
#             elif not isinstance(kwargs['custom_head'], nn.Module):
#                 kwargs['custom_head'] = partial(kwargs['custom_head'], d=d)
#         if 'ltsf_' in arch.__name__.lower() or 'patchtst' in arch.__name__.lower():
#             pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} seq_len={seq_len} pred_dim={d} arch_config={arch_config}, kwargs={kwargs})',
#                verbose)
#             model = (arch(c_in=c_in, c_out=c_out, seq_len=seq_len, pred_dim=d, **arch_config, **kwargs)).to(
#                 device=device)
#         elif arch.__name__ in ['TransformerRNNPlus', 'TransformerLSTMPlus', 'TransformerGRUPlus']:
#             pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} seq_len={seq_len} d={d} arch_config={arch_config}, kwargs={kwargs})',
#                verbose)
#             model = (arch(c_in=c_in, c_out=c_out, seq_len=seq_len, d=d, **arch_config, **kwargs)).to(device=device)
#         elif sum([1 for v in
#                   ['RNN_FCN', 'LSTM_FCN', 'RNNPlus', 'LSTMPlus', 'GRUPlus', 'InceptionTime', 'TSiT', 'Sequencer',
#                    'XceptionTimePlus',
#                    'GRU_FCN', 'OmniScaleCNN', 'mWDN', 'TST', 'XCM', 'MLP', 'MiniRocket', 'InceptionRocket',
#                    'ResNetPlus',
#                    'RNNAttention', 'LSTMAttention', 'GRUAttention', 'MultiRocket', 'MultiRocketPlus']
#                   if v in arch.__name__]):
#             pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} seq_len={seq_len} arch_config={arch_config} kwargs={kwargs})',
#                verbose)
#             model = arch(c_in, c_out, seq_len=seq_len, **arch_config, **kwargs).to(device=device)
#         elif 'xresnet' in arch.__name__ and not '1d' in arch.__name__:
#             pv(f'arch: {arch.__name__}(c_in={c_in} n_out={c_out} arch_config={arch_config} kwargs={kwargs})', verbose)
#             model = (arch(c_in=c_in, n_out=c_out, **arch_config, **kwargs)).to(device=device)
#         elif 'xresnet1d' in arch.__name__.lower():
#             pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} seq_len={seq_len} arch_config={arch_config} kwargs={kwargs})',
#                verbose)
#             model = (arch(c_in=c_in, c_out=c_out, seq_len=seq_len, **arch_config, **kwargs)).to(device=device)
#         elif 'minirockethead' in arch.__name__.lower():
#             pv(f'arch: {arch.__name__}(c_in={c_in} seq_len={seq_len} arch_config={arch_config} kwargs={kwargs})',
#                verbose)
#             model = (arch(c_in, c_out, seq_len=1, **arch_config, **kwargs)).to(device=device)
#         elif 'rocket' in arch.__name__.lower():
#             pv(f'arch: {arch.__name__}(c_in={c_in} seq_len={seq_len} arch_config={arch_config} kwargs={kwargs})',
#                verbose)
#             model = (arch(c_in=c_in, seq_len=seq_len, **arch_config, **kwargs)).to(device=device)
#         else:
#             pv(f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} arch_config={arch_config} kwargs={kwargs})', verbose)
#             model = arch(c_in, c_out, **arch_config, **kwargs).to(device=device)
#
#     try:
#         model[0], model[1]
#         subscriptable = True
#     except:
#         subscriptable = False
#     if hasattr(model, "head_nf"):
#         head_nf = model.head_nf
#     else:
#         try:
#             head_nf = get_nf(model)
#         except:
#             head_nf = None
#
#     if not subscriptable and 'Plus' in arch.__name__:
#         model = nn.Sequential(*model.children())
#         model.backbone = model[:cut]
#         model.head = model[cut:]
#
#     if pretrained and not ('xresnet' in arch.__name__ and not '1d' in arch.__name__):
#         assert weights_path is not None, "you need to pass a valid weights_path to use a pre-trained model"
#         transfer_weights(model, weights_path, exclude_head=exclude_head, device=device)
#
#     if init is not None:
#         apply_init(model[1] if pretrained else model, init)
#
#     setattr(model, "head_nf", head_nf)
#     setattr(model, "__name__", arch.__name__)
#
#     return model
