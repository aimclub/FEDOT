import math
from typing import Optional

import torch
from fastai.layers import SigmoidRange
from fastai.torch_core import Module
from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn, optim, Tensor

from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.activation import get_activation_fn
from fedot.industrial.core.models.nn.network_modules.layers.attention_layers import \
    MultiHeadAttention
from fedot.industrial.core.models.nn.network_modules.layers.conv_layers import Conv1d
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import Flatten, Transpose
from fedot.industrial.core.models.nn.network_modules.layers.padding_layers import Pad1d


class _TSTEncoderLayer(Module):
    def __init__(self,
                 q_len: int,
                 model_dim: int,
                 number_heads: int,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 activation: str = "GELU"):
        assert model_dim // number_heads, f"model_dim ({model_dim}) must be divisible by number_heads ({number_heads})"
        if d_k is None:
            d_k = model_dim // number_heads
        if d_v is None:
            d_v = model_dim // number_heads

        # Multi-Head attention
        self.self_attn = MultiHeadAttention(model_dim, number_heads, d_k, d_v)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        self.batchnorm_attn = nn.Sequential(Transpose(1, 2),
                                            nn.BatchNorm1d(model_dim),
                                            Transpose(1, 2))

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(model_dim, d_ff),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, model_dim))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        self.batchnorm_ffn = nn.Sequential(Transpose(1, 2),
                                           nn.BatchNorm1d(model_dim),
                                           Transpose(1, 2))

    def forward(self, src: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Multi-Head attention sublayer
        # Multi-Head attention
        src2, attn = self.self_attn(src, src, src, mask=mask)
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_attn(src2)
        src = self.batchnorm_attn(src)  # Norm: batchnorm

        # Feed-forward sublayer
        # Position-wise Feed-Forward
        src2 = self.ff(src)
        # Add & Norm
        # Add: residual connection with residual dropout
        src = src + self.dropout_ffn(src2)
        src = self.batchnorm_ffn(src)  # Norm: batchnorm

        return src


class _TSTEncoder(Module):
    def __init__(self,
                 q_len,
                 model_dim,
                 number_heads,
                 d_k=None,
                 d_v=None,
                 d_ff=None,
                 dropout=0.1,
                 activation='GELU',
                 n_layers=1):
        self.layers = nn.ModuleList(
            [_TSTEncoderLayer(q_len,
                              model_dim,
                              number_heads=number_heads,
                              d_k=d_k,
                              d_v=d_v,
                              d_ff=d_ff,
                              dropout=dropout,
                              activation=activation) for i in range(n_layers)])

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class TST(Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 seq_len: int,
                 max_seq_len: Optional[int] = None,
                 n_layers: int = 3,
                 model_dim: int = 128,
                 number_heads: int = 16,
                 d_k: Optional[int] = None,
                 d_v: Optional[int] = None,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 activation: str = "GELU",
                 fc_dropout: float = 0.,
                 y_range: Optional[tuple] = None,
                 verbose: bool = False, **kwargs):
        """TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.

        Args:
            input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset.
            output_dim: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            model_dim: total dimension of the model (number of features created by the model)
            number_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512.
            Default: None -> (model_dim/number_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512.
             Default: None -> (model_dim/number_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            dropout: amount of residual dropout applied in the encoder.
            activation: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.

        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)

        """
        self.output_dim, self.seq_len = output_dim, seq_len

        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if max_seq_len is not None and seq_len > max_seq_len:  # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = (tr_factor * q_len - seq_len)
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(
                Pad1d(padding),
                Conv1d(
                    input_dim,
                    model_dim,
                    kernel_size=tr_factor,
                    padding=0,
                    stride=tr_factor))
            print(
                f'temporal resolution modified: {seq_len} --> {q_len} time steps: kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n',
                verbose)
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(input_dim, model_dim, **kwargs)  # Eq 2
            print(
                f'Conv1d with kwargs={kwargs} applied to input to create input encodings\n',
                verbose)
        else:
            # Eq 1: projection of feature vectors onto a d-dim vector space
            self.W_P = nn.Linear(input_dim, model_dim)

        # Positional encoding
        W_pos = torch.empty((q_len, model_dim), device=default_device())
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = _TSTEncoder(q_len,
                                   model_dim,
                                   number_heads,
                                   d_k=d_k,
                                   d_v=d_v,
                                   d_ff=d_ff,
                                   dropout=dropout,
                                   activation=activation,
                                   n_layers=n_layers)
        self.flatten = Flatten()

        # Head
        self.head_nf = q_len * model_dim
        self.head = self.create_head(self.head_nf,
                                     output_dim,
                                     act=activation,
                                     fc_dropout=fc_dropout,
                                     y_range=y_range)

    def create_head(self,
                    number_of_filters,
                    output_dim,
                    activation="GELU",
                    fc_dropout=0.,
                    y_range=None,
                    **kwargs):
        layers = [get_activation_fn(activation), Flatten()]
        if fc_dropout:
            layers += [nn.Dropout(fc_dropout)]
        layers += [nn.Linear(number_of_filters, output_dim)]
        if y_range:
            layers += [SigmoidRange(*y_range)]
        return nn.Sequential(*layers)

    def forward(self,
                x: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:  # x: [bs x nvars x q_len]

        # Input encoding
        if self.new_q_len:
            u = self.W_P(x).transpose(2,
                                      1)  # Eq 2
            # u: [bs x model_dim x q_len] transposed to [bs x q_len x
            # model_dim]
        else:
            u = self.W_P(x.transpose(2,
                                     1))  # Eq 1
            # u: [bs x q_len x nvars] converted to [bs x q_len x model_dim]

        # Positional encoding
        u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)  # z: [bs x q_len x model_dim]
        z = z.transpose(2, 1).contiguous()  # z: [bs x model_dim x q_len]

        # Classification/ Regression head
        return self.head(z)


class TSTModel(BaseNeuralModel):
    """Class responsible for Time series transformer (TST) model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot.industrial.tools.loader import DataLoader
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels
            train_data, test_data = DataLoader(dataset_name='Lightning7').load_data()
            input_data = init_input_data(train_data[0], train_data[1])
            val_data = init_input_data(test_data[0], test_data[1])
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('inception_model', params={'epochs': 100,
                                                                                 'batch_size': 10}).build()
                pipeline.fit(input_data)
                target = pipeline.predict(val_data).predict
                metric = evaluate_metric(target=test_data[1], prediction=target)

    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.num_classes = self.params.get('num_classes', 1)
        self.epochs = self.params.get('epochs', 100)
        self.batch_size = self.params.get('batch_size', 32)

    def _init_model(self, ts):
        self.model = TST(input_dim=ts.features.shape[1],
                         output_dim=self.num_classes,
                         seq_len=ts.features.shape[2]).to(default_device())
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = self._get_loss_metric(ts)
        return loss_fn, optimizer
