from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from torch import nn

from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.layers.forecasting.nbeats import _NBeatsStack, NBeatsNet
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix


class NBeatsModel(BaseNeuralModel):
    """Class responsible for NBeats model implementation.

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows:

    References:
        @inproceedings{
            Oreshkin2020:N-BEATS,
            title={{N-BEATS}: Neural basis expansion analysis for interpretable time series forecasting},
            author={Boris N. Oreshkin and Dmitri Carpov and Nicolas Chapados and Yoshua Bengio},
            booktitle={International Conference on Learning Representations},
            year={2020},
            url={https://openreview.net/forum?id=r1ecqn4YwB}
        }
        Original paper: https://arxiv.org/abs/1905.10437
        Original code:  https://github.com/ServiceNow/N-BEATS
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.is_generic_architecture = self.params.get(
            "is_generic_architecture", True)
        self.epochs = self.params.get("epochs", 10)
        self.batch_size = self.params.get("batch_size", 16)
        self.loss = self.params.get("loss", 'mse')
        self.optimizer = self.params.get("optimizer", 'adam')
        self.activation = 'None'

        self.n_stacks = self.params.get("n_stacks", 30)
        self.layers = self.params.get("layers", 4)
        self.layer_size = self.params.get("layer_size", 512)

        self.n_trend_blocks = self.params.get("n_trend_blocks", 3)
        self.n_trend_layers = self.params.get("n_trend_layers", 4)
        self.trend_layer_size = self.params.get("trend_layer_size", 2)
        self.degree_of_polynomial = self.params.get("degree_of_polynomial", 20)

        self.n_seasonality_blocks = self.params.get("n_seasonality_blocks", 3)
        self.n_seasonality_layers = self.params.get("n_seasonality_layers", 4)
        self.seasonality_layer_size = self.params.get(
            "seasonality_layer_size", 2048)
        self.n_of_harmonics = self.params.get("n_of_harmonics", 1)

    def _init_model(self, ts):
        self.forecast_length = ts.task.task_params.forecast_length
        self.backcast_length = 3 * ts.task.task_params.forecast_length
        self.model = NBeatsNet(
            stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK),
            device=default_device(),
            forecast_length=self.forecast_length,
            backcast_length=self.backcast_length,
            hidden_layer_units=128,
        )
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        self.is_training = True
        self.split_ratio = int(len(ts.features) * 0.8)

    def data_generator(self, x, y, size):
        assert len(x) == len(y)
        batches = []
        for ii in range(0, len(x), size):
            batches.append((x[ii:ii + size], y[ii:ii + size]))
        for batch in batches:
            yield batch

    def _save_and_clear_cache(self):
        pass

    def _create_predict_data(self, ts: np.ndarray):
        x_trajectory = HankelMatrix(time_series=ts,
                                    window_size=self.backcast_length,
                                    strides=1)
        return x_trajectory.trajectory_matrix.swapaxes(0, 1)

    def _fit_model(self, ts: InputData):

        self._init_model(ts)
        ts_val = deepcopy(ts)

        # data backcast/forecast generation.
        ts.features[:self.split_ratio], ts_val.features[:self.split_ratio] = \
            ts.features[:self.split_ratio], ts_val.features[:self.split_ratio]
        self.norm_constant = np.max(ts.features)
        _, train_transformed, train_target = transform_features_and_target_into_lagged(
            ts, self.forecast_length, self.backcast_length)
        x_train, y_train = train_transformed / \
            self.norm_constant, train_target / self.norm_constant

        _, val_transformed, val_target = transform_features_and_target_into_lagged(
            ts_val, self.forecast_length, self.backcast_length)
        val_transformed, val_target = val_transformed / \
            self.norm_constant, val_target / self.norm_constant

        self.is_training = False

        self.model.fit(x_train=x_train,
                       y_train=y_train,
                       validation_data=(val_transformed, val_target),
                       epochs=self.epochs,
                       batch_size=self.batch_size)

    def _predict_model(self, x_test, output_mode: str = 'default'):
        x_test_lagged = self._create_predict_data(x_test)
        x_predict_lagged = self.model.predict(x_test_lagged)
        forecast = x_predict_lagged[-1:, :].flatten()
        return forecast


class NBeats(nn.Module):
    """
    N-Beats Model proposed in https://arxiv.org/abs/1905.10437
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 is_generic_architecture: bool,
                 n_stacks: int,
                 n_trend_blocks: int,
                 n_trend_layers: int,
                 trend_layer_size: int,
                 degree_of_polynomial: int,
                 n_seasonality_blocks: int,
                 n_seasonality_layers: int,
                 seasonality_layer_size: int,
                 n_of_harmonics: int,
                 ):
        """
        Args:
            input_dim: the number of features (aka variables, dimensions, channels) in the time series dataset
            output_dim: the number of target classes
            is_generic_architecture: indicating whether the generic architecture of N-BEATS is used.
                If not, the interpretable architecture outlined in the paper (consisting of one trend
                and one seasonality stack with appropriate waveform generator functions)
            n_stacks: the number of: The number of stacks that make up the whole model. Only used if `is_generic_architecture` is set to `True`.
                The interpretable architecture always uses two stacks - one for trend and one for seasonality.
            n_blocks: the number of blocks making up every stack.
            n_layers: the number of fully connected layers preceding the final forking layers in each block of every stack.
            layer_size: Determines the number of neurons that make up each fully connected layer in each block of every stack.
            n_trend_blocks:
                Used if `is_generic_architecture` is set to `False`
            n_trend_layers:
                Used if `is_generic_architecture` is set to `False`
            degree_of_polynomial:
                Used if `is_generic_architecture` is set to `False`
            n_seasonality_blocks
                Used if `is_generic_architecture` is set to `False`
            n_seasonality_layers
                Used if `is_generic_architecture` is set to `False`
            seasonality_layer_size:
                Used if `is_generic_architecture` is set to `False`
            n_of_harmonics:
                Used if `is_generic_architecture` is set to `False`
            dropout: probability to be used in fully connected layers.
            activation: the activation function of intermediate layer, relu or gelu.
        """

        super().__init__()

        self.blocks = None

        if is_generic_architecture:
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_stacks=n_stacks,
                # n_blocks=n_blocks,
                # n_layers=n_layers,
                # layer_size=layer_size,
            )
        else:
            # The overall interpretable architecture consists of two stacks:
            # the trend stack is followed by the seasonality stack
            self.stack = _NBeatsStack(
                input_dim=input_dim,
                output_dim=output_dim,
                is_generic_architecture=is_generic_architecture,
                n_trend_blocks=n_trend_blocks,
                n_trend_layers=n_trend_layers,
                trend_layer_size=trend_layer_size,
                degree_of_polynomial=degree_of_polynomial,
                n_seasonality_blocks=n_seasonality_blocks,
                n_seasonality_layers=n_seasonality_layers,
                seasonality_layer_size=seasonality_layer_size,
                n_of_harmonics=n_of_harmonics,
            )

        self.blocks = nn.ModuleList(self.stacks)

    def forward(
            self,
            x: torch.Tensor,
            input_mask: torch.Tensor) -> torch.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

        return forecast
