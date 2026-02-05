import torch
from torch import nn
from torch import Tensor

from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import FlattenHead
from fedot.industrial.core.models.nn.network_modules.layers.special import _TSTiEncoder, RevIN


class _PatchTST_backbone(nn.Module):
    def __init__(
            self,
            input_dim,
            seq_len,
            pred_dim,
            patch_len,
            stride,
            n_layers=3,
            d_model=128,
            n_heads=16,
            d_k=None,
            d_v=None,
            d_ff=256,
            norm='BatchNorm',
            attn_dropout=0.,
            dropout=0.,
            act="GELU",
            res_attention=True,
            pre_norm=False,
            store_attn=False,
            padding_patch=True,
            individual=False,
            revin=True,
            affine=True,
            subtract_last=False,
            preprocess_to_lagged=False):

        super().__init__()

        # RevIn
        self.revin = revin
        self.revin_layer = RevIN(
            input_dim, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((seq_len - patch_len) / stride + 1) + 1
        self.patch_num = patch_num
        self.padding_patch_layer = nn.ReplicationPad1d(
            (stride, 0))  # original padding at the end
        self.preprocess_to_lagged = preprocess_to_lagged

        # Unfold
        self.unfold = nn.Unfold(kernel_size=(1, patch_len), stride=stride)
        self.patch_len = patch_len

        # Backbone
        self.backbone = _TSTiEncoder(
            input_dim=input_dim,
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = input_dim
        self.individual = individual
        self.head = FlattenHead(
            self.individual, self.n_vars, self.head_nf, pred_dim)

    def forward(self, z: Tensor):
        """
        Args:
            z: [batch_size x input_dim x seq_len]
        """
        # norm
        if self.revin:
            z = self.revin_layer(z, torch.tensor(True, dtype=torch.bool))

        b, c, s = z.size()
        z = z.reshape(-1, 1, 1, s).permute(0, 1, 3, 2)

        # model
        z, scores = self.backbone(z)  # z: [bs x nvars x d_model x patch_num]
        z = self.head(z)  # z: [bs x nvars x pred_dim]

        # denorm
        if self.revin:
            z = self.revin_layer(z, torch.tensor(False, dtype=torch.bool))
        return z
