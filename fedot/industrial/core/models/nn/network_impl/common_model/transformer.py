from typing import Optional

from fastai.torch_core import Module
from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn, optim
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.layers.linear_layers import Max, Permute, Transpose


class TransformerModule(Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 d_model=64,
                 n_head=1,
                 d_ffn=128,
                 dropout=0.1,
                 activation="relu",
                 n_layers=1):
        """
        Args:
            input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset
            output_dim: the number of target classes
            d_model: total dimension of the model.
            n_head: parallel attention layers.
            d_ffn: the dimension of the feedforward network model.
            dropout: a Dropout layer on attn_output_weights.
            activation: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.

        Input shape:
            bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)

        """
        self.permute = Permute(2, 0, 1)
        self.inlinear = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU()
        encoder_layer = TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation=activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, n_layers, norm=encoder_norm)
        self.transpose = Transpose(1, 0)
        self.max = Max(1)
        self.outlinear = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # bs x nvars x seq_len -> seq_len x bs x nvars
        x = self.permute(x.squeeze())
        x = self.inlinear(x)  # seq_len x bs x nvars -> seq_len x bs x d_model
        x = self.relu(x)
        x = self.transformer_encoder(x)
        x = self.transpose(x)
        x = self.max(x)
        x = self.relu(x)
        x = self.outlinear(x)
        return x


class TransformerModel(BaseNeuralModel):
    """Class responsible for Transformer model implementation.

       Attributes:
           self.num_features: int, the number of features.
           self.epoch: int, the number of epochs.
           self.batch_size: int, the batch size.

       """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.num_classes = self.params.get('num_classes', 1)
        self.epochs = self.params.get('epochs', 10)
        self.batch_size = self.params.get('batch_size', 20)

    def _init_model(self, ts):
        self.model = TransformerModule(
            input_dim=ts.features.shape[2],
            output_dim=self.num_classes).to(
            default_device())

        self.model_for_inference = self.model

        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = self._get_loss_metric(ts)
        return loss_fn, optimizer
