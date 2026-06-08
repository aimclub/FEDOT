import pickle
import random
from time import time
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.functional import mse_loss, l1_loss, binary_cross_entropy, cross_entropy

from fedot.industrial.core.models.nn.network_modules.losses import SMAPELoss


class _NBeatsStack(nn.Module):
    def __init__(
            self,
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
        self.block = None

        if is_generic_architecture:
            self.block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=input_dim + output_dim,
                basis_function=_GenericBasis(
                    backcast_size=input_dim,
                    forecast_size=output_dim
                )
            )
            self.blocks = [self.block for _ in range(n_stacks)]

        else:
            trend_block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=2 * (degree_of_polynomial + 1),
                basis_function=_TrendBasis(
                    degree_of_polynomial=degree_of_polynomial,
                    backcast_size=input_dim,
                    forecast_size=output_dim),
                layers=n_trend_layers,
                layer_size=trend_layer_size,
            )

            seasonality_block = _NBeatsBlock(
                input_size=input_dim,
                theta_size=4 *
                int(np.ceil(n_of_harmonics / 2 * output_dim) - (n_of_harmonics - 1)),
                basis_function=_SeasonalityBasis(
                    harmonics=n_of_harmonics,
                    backcast_size=input_dim,
                    forecast_size=output_dim),
                layers=n_seasonality_layers,
                layer_size=seasonality_layer_size
            )

            self.blocks = [trend_block for _ in range(
                n_trend_blocks)] + [seasonality_block for _ in range(n_seasonality_blocks)]


class _NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
            self,
            input_size,
            theta_size: int,
            basis_function: nn.Module,
            layers: int,
            layer_size: int
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=input_size, out_features=layer_size)] +
            [nn.Linear(in_features=layer_size, out_features=layer_size)
             for _ in range(layers - 1)]
        )

        self.basis_parameters = nn.Linear(
            in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)

        return self.basis_function(basis_parameters)


class _GenericBasis(nn.Module):
    """
    Generic basis function.
    The generic architecture does not rely on TS-specific knowledge.
    Set g^b_l and g^f_l to be a linear projection of the previous layer output.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class _TrendBasis(nn.Module):
    """
    Polynomial function to model trend.
    Trend model. A typical characteristic of trend is that most of the time it is a monotonic function,
    or at least a slowly varying function. In order to mimic this behaviour constrain g^b_sl, and g^f_sl,
    to be a polynomial of small degree p, a function slowly varying across forecast window.
    """

    def __init__(
            self,
            degree_of_polynomial: int,
            backcast_size: int,
            forecast_size: int
    ):
        super().__init__()

        self.polynomial_size = degree_of_polynomial + \
            1  # degree of polynomial with constant term
        self.backcast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :] for i in
                 range(self.polynomial_size)]),
                dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_time = nn.Parameter(
            torch.tensor(np.concatenate(
                [np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :] for i in
                 range(self.polynomial_size)]),
                dtype=torch.float32),
            requires_grad=False
        )

    def forward(self, theta: torch.Tensor):
        backcast = torch.einsum(
            "bp,pt->bt",
            theta[:, self.polynomial_size:],
            self.backcast_time
        )

        forecast = torch.einsum(
            "bp,pt->bt",
            theta[:, :self.polynomial_size],
            self.forecast_time
        )

        return backcast, forecast


class _SeasonalityBasis(_NBeatsBlock):
    """
    Harmonic functions to model seasonality.
    Seasonality model. Typical characteristic of seasonality is that it is a regular, cyclical, recurring fluctuation.
    Therefore, to model seasonality, constrain g^b_sl, and g^f_sl, to be long to the class of periodic functions,
    i.e. y_t = y_t-∆, where ∆ is a seasonality period.
    """

    def __init__(
            self,
            harmonics: int,
            backcast_size: int,
            forecast_size: int
    ):
        super().__init__()

        self.frequency = np.append(
            np.zeros(
                1,
                dtype=np.float32),
            np.arange(
                harmonics,
                harmonics /
                2 *
                forecast_size,
                dtype=np.float32) /
            harmonics)[
                None,
            :]
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency

        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency

        self.backcast_cos_template = nn.Parameter(
            torch.tensor(
                np.transpose(
                    np.cos(backcast_grid)),
                dtype=torch.float32),
            requires_grad=False)

        self.backcast_sin_template = nn.Parameter(
            torch.tensor(
                np.transpose(
                    np.sin(backcast_grid)),
                dtype=torch.float32),
            requires_grad=False)

        self.forecast_cos_template = nn.Parameter(
            torch.tensor(
                np.transpose(
                    np.cos(forecast_grid)),
                dtype=torch.float32),
            requires_grad=False)

        self.forecast_sin_template = nn.Parameter(
            torch.tensor(
                np.transpose(
                    np.sin(forecast_grid)),
                dtype=torch.float32),
            requires_grad=False)

    def forward(self, theta: torch.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
            self.backcast_cos_template
        )

        backcast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, 3 * params_per_harmonic:],
            self.backcast_sin_template
        )

        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = torch.einsum(
            "bp,pt->bt",
            theta[:, :params_per_harmonic],
            self.forecast_cos_template
        )

        forecast_harmonics_sin = torch.einsum(
            "bp,pt->bt",
            theta[:, params_per_harmonic:2 * params_per_harmonic],
            self.forecast_sin_template
        )

        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


class NBeatsNet(nn.Module):

    SEASONALITY_BLOCK = "seasonality"
    TREND_BLOCK = "trend"
    GENERIC_BLOCK = "generic"

    def __init__(
            self,
            device=torch.device("cpu"),
            stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
            nb_blocks_per_stack=3,
            forecast_length=5,
            backcast_length=10,
            thetas_dim=(4, 8),
            share_weights_in_stack=False,
            hidden_layer_units=256,
            nb_harmonics=None
    ):
        super(NBeatsNet, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.nb_harmonics = nb_harmonics
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        print("| N-Beats")
        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)
        self.to(self.device)
        self._loss = None
        self._opt = None
        self._gen_intermediate_outputs = False
        self._intermediary_outputs = []

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(
            f"| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})")
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                # pick up the last one when we share weights.
                block = blocks[-1]
            else:
                block = block_init(
                    self.hidden_layer_units, self.thetas_dim[stack_id],
                    self.device, self.backcast_length, self.forecast_length,
                    self.nb_harmonics
                )
                self.parameters.extend(block.parameters())
            print(f"     | -- {block}")
            blocks.append(block)
        return blocks

    def disable_intermediate_outputs(self):
        self._gen_intermediate_outputs = False

    def enable_intermediate_outputs(self):
        self._gen_intermediate_outputs = True

    def save(self, filename: str):
        torch.save(self, filename)

    @staticmethod
    def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
        return torch.load(f, map_location, pickle_module, **pickle_load_args)

    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def compile(self, loss: str, optimizer: Union[str, Optimizer]):
        if loss == "mae":
            loss_ = l1_loss
        elif loss == "mse":
            loss_ = mse_loss
        elif loss == "cross_entropy":
            loss_ = cross_entropy
        elif loss == "binary_crossentropy":
            loss_ = binary_cross_entropy
        elif loss == "smape":
            loss_ = SMAPELoss
        else:
            raise ValueError(f"Unknown loss name: {loss}.")
        # noinspection PyArgumentList
        if isinstance(optimizer, str):
            if optimizer == "adam":
                opt_ = optim.Adam
            elif optimizer == "sgd":
                opt_ = optim.SGD
            elif optimizer == "rmsprop":
                opt_ = optim.RMSprop
            else:
                raise ValueError(f"Unknown opt name: {optimizer}.")
            opt_ = opt_(lr=1e-4, params=self.parameters())
        else:
            opt_ = optimizer
        self._opt = opt_
        self._loss = loss_

    def fit(
            self,
            x_train,
            y_train,
            validation_data=None,
            epochs=10,
            batch_size=32):
        def split(arr, size):
            arrays = []
            while len(arr) > size:
                slice_ = arr[:size]
                arrays.append(slice_)
                arr = arr[size:]
            arrays.append(arr)
            return arrays

        for epoch in range(epochs):
            x_train_list = split(x_train, batch_size)
            y_train_list = split(y_train, batch_size)
            assert len(x_train_list) == len(y_train_list)

            shuffled_indices = list(range(len(x_train_list)))
            random.shuffle(shuffled_indices)
            self.train()
            train_loss = []
            timer = time()

            for batch_id in shuffled_indices:
                batch_x, batch_y = x_train_list[batch_id], y_train_list[batch_id]
                self._opt.zero_grad()
                _, forecast = self(
                    torch.tensor(
                        batch_x, dtype=torch.float).to(
                        self.device))
                loss = self._loss(
                    forecast, squeeze_last_dim(
                        torch.tensor(
                            batch_y, dtype=torch.float).to(
                            self.device)))
                train_loss.append(loss.item())
                loss.backward()
                self._opt.step()

            elapsed_time = time() - timer
            train_loss = np.mean(train_loss)

            test_loss = '[undefined]'
            if validation_data is not None:
                x_test, y_test = validation_data
                self.eval()
                _, forecast = self(
                    torch.tensor(
                        x_test, dtype=torch.float).to(
                        self.device))
                test_loss = self._loss(
                    forecast, squeeze_last_dim(
                        torch.tensor(
                            y_test, dtype=torch.float).to(
                            self.device))).item()

            num_samples = len(x_train_list)
            time_per_step = int(elapsed_time / num_samples * 1000)
            print(f"Epoch {str(epoch + 1).zfill(len(str(epochs)))}/{epochs}")
            print(
                f"{num_samples}/{num_samples} [==============================] - "
                f"{int(elapsed_time)}s {time_per_step}ms/step - "
                f"loss: {train_loss:.4f} - val_loss: {test_loss:.4f}")

    def predict(self, x, return_backcast=False):
        self.eval()
        b, f = self(torch.tensor(x, dtype=torch.float).to(self.device))
        b, f = b.to('cpu').detach().numpy(), f.to('cpu').detach().numpy()
        if len(x.shape) == 3:
            b = np.expand_dims(b, axis=-1)
            f = np.expand_dims(f, axis=-1)
        if return_backcast:
            return b
        return f

    @staticmethod
    def name():
        return "NBeatsNetPytorch"

    def get_generic_and_interpretable_outputs(self):
        g_pred = sum([a["value"][0]
                     for a in self._intermediary_outputs if "generic" in a["layer"].lower()])
        i_pred = sum([a["value"][0]
                     for a in self._intermediary_outputs if "generic" not in a["layer"].lower()])
        outputs = {o["layer"]: o['value'][0]
                   for o in self._intermediary_outputs}
        return g_pred, i_pred, outputs

    def forward(self, backcast):
        self._intermediary_outputs = []
        backcast = squeeze_last_dim(backcast)
        # maybe batch size here.
        forecast = torch.zeros(
            size=(
                backcast.size()[0],
                self.forecast_length,
            ))

        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast.to(self.device) - b
                forecast = forecast.to(self.device) + f
                block_type = self.stacks[stack_id][block_id].__class__.__name__
                layer_name = f"stack_{stack_id}-{block_type}_{block_id}"

                if self._gen_intermediate_outputs:
                    self._intermediary_outputs.append(
                        {"value": f.detach().numpy(), "layer": layer_name})

        return backcast, forecast


def squeeze_last_dim(tensor):
    # (128, 10, 1) => (128, 10).
    if len(tensor.shape) == 3 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    return tensor


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], "thetas_dim is too big."
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor(np.array([np.cos(2 * np.pi * i * t)
                      for i in range(p1)])).float()  # H/2-1
    s2 = torch.tensor(np.array([np.sin(2 * np.pi * i * t)
                      for i in range(p2)])).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, "thetas_dim is too big."
    T = torch.tensor(np.array([t ** i for i in range(p)])).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length, is_forecast=True):
    horizon = forecast_length if is_forecast else backcast_length
    return np.arange(0, horizon) / horizon


class Block(nn.Module):
    """
    Class refers to a single layer of the model.
    Each block is responsible for predicting a specific aspect of the time series,
    such as trend, seasonality, or residuals.
    """

    def __init__(
            self,
            units,
            thetas_dim,
            device,
            backcast_length=10,
            forecast_length=5,
            share_thetas=False,
            nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.device = device
        self.backcast_linspace = linear_space(
            backcast_length, forecast_length, is_forecast=False)
        self.forecast_linspace = linear_space(
            backcast_length, forecast_length, is_forecast=True)

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(
                units, thetas_dim, bias=False)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim, bias=False)
            self.theta_f_fc = nn.Linear(units, thetas_dim, bias=False)

    def forward(self, x):
        x = squeeze_last_dim(x)
        x = F.relu(self.fc1(x.to(self.device)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):
    """
    Class refers to a flexible block that can capture more complex patterns in the time series data
    that may not fit into the trend or seasonality categories.
    It can be used to model residual patterns or other irregular fluctuations in the data.
    """

    def __init__(
            self,
            units,
            thetas_dim,
            device,
            backcast_length=10,
            forecast_length=5,
            nb_harmonics=None):
        super(
            GenericBlock,
            self).__init__(
            units,
            thetas_dim,
            device,
            backcast_length,
            forecast_length)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, forecast_length)

    def forward(self, x):
        # No constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)  # Generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # Generic. 3.3.

        return backcast, forecast


class SeasonalityBlock(Block):
    """
    Class refers to a block that captures the seasonal patterns in the time series data.
    It models the recurring patterns that occur at regular intervals, such as daily, weekly, or monthly cycles.
    """

    def __init__(
            self,
            units,
            thetas_dim,
            device,
            backcast_length=10,
            forecast_length=5,
            nb_harmonics=None):
        if nb_harmonics:
            super(
                SeasonalityBlock,
                self).__init__(
                units,
                nb_harmonics,
                device,
                backcast_length,
                forecast_length,
                share_thetas=True)
        else:
            super(
                SeasonalityBlock,
                self).__init__(
                units,
                forecast_length,
                device,
                backcast_length,
                forecast_length,
                share_thetas=True)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        backcast = seasonality_model(
            self.theta_b_fc(x),
            self.backcast_linspace,
            self.device)
        forecast = seasonality_model(
            self.theta_f_fc(x),
            self.forecast_linspace,
            self.device)
        return backcast, forecast


class TrendBlock(Block):
    """
    Class refers to a block that is designed to capture the trend component of the time series data.
    It models the overall increasing or decreasing pattern in the data.
    """

    def __init__(
            self,
            units,
            thetas_dim,
            device,
            backcast_length=10,
            forecast_length=5,
            nb_harmonics=None):
        super(
            TrendBlock,
            self).__init__(
            units,
            thetas_dim,
            device,
            backcast_length,
            forecast_length,
            share_thetas=True)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(
            self.theta_b_fc(x),
            self.backcast_linspace,
            self.device)
        forecast = trend_model(
            self.theta_f_fc(x),
            self.forecast_linspace,
            self.device)
        return backcast, forecast
