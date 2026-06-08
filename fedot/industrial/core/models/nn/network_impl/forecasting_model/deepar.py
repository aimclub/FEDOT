from copy import deepcopy
from typing import Optional, Tuple, Union

import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from torch.nn import LSTM, GRU, Linear, Module, RNN, Sequential
from tqdm import tqdm

from fedot.industrial.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.layers.special import EarlyStopping
from fedot.industrial.core.models.nn.network_modules.losses import (
    NormalDistributionLoss, CauchyDistributionLoss)
from fedot.industrial.core.operation.transformation.window_selector import WindowSizeSelector

__all__ = ['DeepAR']


class _TSScaler(Module):
    def __init__(self):
        super().__init__()
        self.factors = None
        self.eps = 1e-10
        self.fitted = False

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if normalize:
            self.fitted = True
            self.means = x.mean(dim=-1, keepdim=True)
            self.factors = torch.sqrt(
                x.std(dim=-1,
                      keepdim=True,  # True results in really strange behavior of affine transformer
                      unbiased=False)) + self.eps
            return (x - self.means) / self.factors
        else:
            assert self.fitted, 'Unknown scale! Fit this'
            factors, means = self.factors, self.means
            if len(x.size()) == 4:
                factors = factors[..., None]
                means = factors[..., None]
            return x * factors + means

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.means) / self.factors

    def upscale(self, x: torch.Tensor) -> torch.Tensor:
        return self(x, False)


class DeepARModule(Module):
    _loss_fns = {
        'normal': NormalDistributionLoss,
        'cauchy': CauchyDistributionLoss,

        # commented before real support added... now require ts > 0 which is incompatible with rnn normalization
        # 'inverse_gamma': InverseGaussDistributionLoss,
        # 'beta': BetaDistributionLoss,
        # 'lognorm': LogNormDistributionLoss,
    }

    def __init__(
            self,
            cell_type: str,
            input_size: int,
            hidden_size: int,
            rnn_layers: int,
            dropout: float,
            distribution: str,
            quantiles=None,
            prediction_averaging_factor=65):
        super().__init__()
        self.rnn = {'LSTM': LSTM, 'GRU': GRU, 'RNN': RNN}[cell_type](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.
        )
        self.hidden_size = hidden_size
        self.scaler = _TSScaler()
        self.__prediction_averaging_factor = prediction_averaging_factor
        self.quantiles = quantiles or torch.tensor([0.25, 0.5, 0.75])
        self.distribution = self._loss_fns[distribution]
        self.projector = Sequential(Linear(self.hidden_size, self.hidden_size),
                                    Linear(self.hidden_size, len(self.distribution.distribution_arguments)))

    def encode(
            self,
            ts: torch.Tensor,
            hidden_state: torch.Tensor = None,
            scaled: bool = True):
        """
        Encode sequence into hidden state
        ts.size = (length, hidden)
        """
        if not scaled:
            ts = self.scaler(ts)
        _, hidden_state = self.rnn(
            ts, hidden_state
        )
        return hidden_state

    def _decode_whole_seq(self,
                          ts: torch.Tensor,
                          hidden_state: torch.Tensor) -> Tuple[torch.Tensor,
                                                               torch.Tensor]:
        """ used for next value prediction"""
        output, hidden_state = self.rnn(
            ts, hidden_state
        )
        output = self.projector(output)
        return output, hidden_state

    def forecast(
            self,
            prefix: torch.Tensor,
            forecast_length: int,
            hidden_state=None,
            output_mode: str = 'quantiles',
            **mode_kw):
        self.eval()
        forecast = []
        with torch.no_grad():
            for i in range(forecast_length):
                output, hidden_state = self(prefix, hidden_state)
                forecast.append(
                    self._transform_params(
                        output,
                        mode=output_mode,
                        **mode_kw).detach().cpu())
                prediction = self._transform_params(output, mode='predictions')
                prefix = torch.roll(prefix, -1, dims=-1)
                prefix[..., [-1]] = prediction
            forecast = torch.stack(forecast, dim=1).squeeze(-1)
        return forecast

    def forward(self, x: torch.Tensor, hidden_state=None,
                mode='raw', **mode_kw) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        x.size == (nseries, length)
        """
        # encode
        x = self.scaler(x, normalize=True)
        hidden_state = self.encode(x, hidden_state=hidden_state, scaled=True)
        # decode
        if self.training:
            assert mode == 'raw', "cannot use another mode, but 'raw' while training"
            return self._decode_whole_seq(x, hidden_state)
        else:
            output, hidden_state = self._decode_whole_seq(x, hidden_state)
            return self._transform_params(output,
                                          mode=mode, **mode_kw), hidden_state

    def to_quantiles(
            self,
            params: torch.Tensor,
            quantiles=None) -> torch.Tensor:
        if quantiles is None:
            quantiles = self.quantiles
        distr = self.distribution.map_x_to_distribution(params)
        return distr.icdf(quantiles).unsqueeze(
            1)  # batch_size x 1 x n_quantiles

    def to_samples(self, params: torch.Tensor, n_samples=100) -> torch.Tensor:
        distr = self.distribution.map_x_to_distribution(params)
        return distr.sample(
            (n_samples,)).permute(
            1, 2, 0)  # batch_size x n_ts x n_samples

    def to_predictions(self, params: torch.Tensor) -> torch.Tensor:
        distr = self.distribution.map_x_to_distribution(params)
        return torch.median(
            distr.sample(
                (self.__prediction_averaging_factor,)).permute(
                1, 2, 0), dim=-1, keepdim=True).values  # batch_size x n_ts x 1

    def _transform_params(
            self,
            distr_params,
            mode='raw',
            **mode_kw) -> torch.Tensor:
        if mode == 'raw':
            return distr_params
        elif mode == 'quantiles':
            transformed = self.to_quantiles(distr_params, **mode_kw)
        elif mode == 'predictions':
            transformed = self.to_predictions(distr_params)
        elif mode == 'samples':
            transformed = self.to_samples(distr_params, **mode_kw)
        else:
            raise ValueError('Unexpected forecast mode!')
        if self.distribution.need_target_scale:
            transformed = self.scaler(transformed, False)
        return transformed


class DeepAR(BaseNeuralModel):
    """No exogenous variable support
    Variational Inference + Probable Anomaly detection"""

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        # training settings
        self.epochs = self.params.get('epochs', 150)
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.batch_size = self.params.get('batch_size', 16)
        self.device = self.params.get('device', 'cpu')
        self.device = default_device(self.device)
        # architecture settings
        self.activation = self.params.get('activation', 'tanh')
        self.cell_type = self.params.get(
            'cell_type', 'GRU')  # LSTM is unstable?
        self.hidden_size = self.params.get('hidden_size', 10)
        self.rnn_layers = self.params.get('rnn_layers', 1)
        self.dropout = self.params.get('dropout', 0.1)
        self.expected_distribution = self.params.get(
            'expected_distribution', 'normal')
        self.patch_len = self.params.get('patch_len', None)
        self.preprocess_to_lagged = False
        self.horizon = 1  # params.get('horizon', 1) for future extension
        self.task_type = 'ts_forecasting'

        # forecasting settings
        self.forecast_mode = self.params.get('forecast_mode', 'predictions')
        self.quantiles = torch.tensor(
            self.params.get('quantiles', [0.25, 0.5, 0.75]))
        self.n_samples = self.params.get('n_samples', 10)
        self.test_patch_len = None
        self.forecast_length = self.params.get('forecast_length', 1)

        # additional
        self._prediction_averaging_factor = self.params.get(
            'prediction_averaging_factor', 17)

    def _init_model(self, ts) -> tuple:
        self.loss_fn = DeepARModule._loss_fns[self.expected_distribution]()
        input_size = self.patch_len or ts.features.shape[-1]
        self.patch_len = input_size
        self.model = DeepARModule(
            input_size=input_size,
            hidden_size=self.hidden_size,
            cell_type=self.cell_type,
            dropout=self.dropout,
            rnn_layers=self.rnn_layers,
            distribution=self.expected_distribution,
            prediction_averaging_factor=self._prediction_averaging_factor).to(
            self.device)
        self._evaluate_num_of_epochs(ts)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        return self.loss_fn, self.optimizer

    def fit(self, input_data: InputData):
        if len(input_data.target.shape) >= 2:
            self.preprocess_to_lagged = True
        train_loader, val_loader = self._prepare_data(input_data,
                                                      split_data=False,
                                                      horizon=1,
                                                      is_train=True)
        loss_fn, optimizer = self._init_model(input_data)

        self.model = self._train_loop(model=self.model,
                                      train_loader=train_loader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      val_loader=val_loader,
                                      )
        return self

    def _prepare_data(
            self,
            input_data: InputData,
            split_data,
            horizon=None,
            is_train=True):
        val_loader = None
        if self.preprocess_to_lagged:
            self.patch_len = input_data.features.shape[-1]
            train_loader = self._create_torch_loader(input_data, is_train)
        else:
            if self.patch_len is None:
                dominant_window_size = WindowSizeSelector(
                    method='dff').get_window_size(input_data.features)
                self.patch_len = 2 * dominant_window_size
            train_loader, val_loader = self._get_train_val_loaders(input_data.features,
                                                                   self.patch_len,
                                                                   split_data,
                                                                   horizon=horizon,
                                                                   is_train=is_train)

        self.test_patch_len = self.patch_len
        return train_loader, val_loader

    def get_initial_hidden_state(self, data_loader: data.DataLoader, ):
        model = self.model  # place for model and model_for_inference switch if needed
        model.eval()
        hidden_state = None
        n = len(data_loader)
        for i, (x, *_) in enumerate(data_loader, 1):
            if i == n:
                break
            hidden_state = model.encode(x, hidden_state, scaled=True)
        initial_hidden_state = hidden_state

        if initial_hidden_state is not None and self.cell_type == 'LSTM':
            initial_hidden_state = (
                initial_hidden_state[0][:, -self.horizon:, :], initial_hidden_state[1][:, -self.horizon:, :])
        elif initial_hidden_state is not None:
            initial_hidden_state = initial_hidden_state[:, -self.horizon:, :]
        return initial_hidden_state

    def predict(self,
                test_data: InputData,
                output_mode: str = None):
        forecast_idx_predict = np.arange(start=test_data.idx[-1],
                                         stop=test_data.idx[-1] +
                                         self.forecast_length,
                                         step=1)
        # some logic to select needed ts
        if output_mode == 'quantiles':
            kwargs = dict(quantiles=self.quantiles)
        else:
            kwargs = dict(
                n_samples=self.n_samples) if output_mode == 'samples' else {}
        forecast = self._predict(test_data, output_mode, **kwargs)
        return OutputData(
            idx=forecast_idx_predict,
            task=test_data.task,
            predict=forecast[0, ...].squeeze().numpy(),
            target=test_data.target,
            data_type=DataTypesEnum.table)

    def _predict(self, test_data, output_mode, hidden_state=None, **output_kw):
        self.forecast_length = test_data.task.task_params.forecast_length or self.forecast_length
        test_loader, _ = self._prepare_data(test_data,
                                            split_data=False,
                                            horizon=1,
                                            is_train=False,
                                            )

        initial_hidden_state = hidden_state or self.get_initial_hidden_state(
            test_loader)
        last_patch, last_target = test_loader.dataset[[-1]]
        if len(last_target) == 1:  # rewrite for horizon != 1 in future
            last_patch, last_target = test_loader.dataset[[-1]]
        if len(last_target) == 1:  # rewrite for horizon != 1 in future
            last_patch = torch.roll(last_patch, -1, dims=-1)
            last_patch[..., -1] = last_target.squeeze()

        last_patch = last_patch.to(self.device)
        fc = self.model.forecast(last_patch, self.forecast_length,
                                 output_mode='predictions',
                                 hidden_state=initial_hidden_state,
                                 **output_kw)

        return fc

    def predict_for_fit(self, test_data, output_mode: str = None):
        forecast_idx_predict = np.arange(start=test_data.idx[-1],
                                         stop=test_data.idx[-1] +
                                         self.forecast_length,
                                         step=1)
        forecast = self._predict(test_data, output_mode)
        return OutputData(
            idx=forecast_idx_predict,
            task=test_data.task,
            predict=forecast.squeeze().numpy(),
            target=test_data.target,
            data_type=DataTypesEnum.table)

    def _train_loop(self, model,
                    train_loader,
                    loss_fn,
                    val_loader,
                    optimizer,
                    val_interval=10):
        def train_one_batch(iter, batch):
            batch_x, batch_y = batch[0], batch[1]
            iter += 1
            optimizer.zero_grad()
            batch_x = (batch_x.float()).to(self.device)
            batch_y = batch_y[:, ..., [0]].float().to(
                self.device)  # only first entrance
            outputs, *hidden_state = model(batch_x)
            loss = loss_fn(outputs, batch_y, self.model.scaler)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            return loss.item()

        def val_one_epoch(iter, batch):
            model.eval()
            val_loss = 0.0
            total = 0
            inputs, targets = batch
            output = model(inputs)
            loss = loss_fn(output, targets.float())
            val_loss += loss.data.item() * inputs.size(0)
            total += inputs.size(0)
            val_loss /= total
            if val_loss < val_loss:
                deepcopy(model)

        train_steps, early_stopping, best_model, best_val_loss = max(1, len(train_loader)), EarlyStopping(), \
            None, float('inf')
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            epochs=self.epochs,
                                            max_lr=self.learning_rate)
        for epoch in tqdm(range(self.epochs)):
            iter_count = 0
            model.train()
            train_loss = list(map(lambda batch_tuple: train_one_batch(
                iter_count, batch_tuple), train_loader))
            train_loss = np.average(train_loss)
            if val_loader is not None and epoch % val_interval == 0:
                valid_loss = list(map(lambda batch_tuple: val_one_epoch(
                    iter_count, batch_tuple), val_loader))
            last_lr = scheduler.get_last_lr()[0]
            if epoch % 25 == 0:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                print('Updating learning rate to {}'.format(last_lr))
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return model

    def _get_train_val_loaders(self,
                               ts,
                               patch_len=None,
                               split_data: bool = True,
                               val_size: float = 0.2,
                               horizon=None,
                               is_train=True,
                               ):
        train_data = self.__ts_to_input_data(ts)
        if horizon is None:
            horizon = self.horizon
        if patch_len is None:
            patch_len = self.patch_len

        if patch_len + horizon > train_data.features.shape[0]:
            self.patch_len = train_data.features.shape[0] - horizon
            patch_len = self.patch_len

        if not split_data:
            _, train_data.features, train_data.target = \
                transform_features_and_target_into_lagged(train_data,
                                                          horizon,
                                                          patch_len)
        else:
            raise NotImplementedError('Problem with lagged_data splitting')
        train_loader = self._create_torch_loader(train_data, is_train)
        return train_loader, None

    def __ts_to_input_data(self, input_data: Union[InputData, pd.DataFrame]):
        if isinstance(input_data, InputData):
            return input_data

        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.forecast_length))

        if isinstance(input_data, pd.DataFrame):
            time_series = input_data
            if 'datetime' in time_series.columns:
                idx = pd.to_datetime(time_series['datetime'].values)
            else:
                idx = np.arange(len(time_series.values.flatten()))
            time_series = time_series.value
        else:
            time_series = input_data
            idx = np.arange(len(time_series.flatten()))

        train_input = InputData(idx=idx,
                                features=time_series.flatten(),
                                target=time_series.flatten(),
                                task=task,
                                data_type=DataTypesEnum.ts)

        return train_input

    @convert_to_3d_torch_array
    def _create_torch_loader(self, train_data, is_train):
        # if is_train else self.forecast_length # for future development with
        # varying horizon
        batch_size = self.batch_size

        if not isinstance(train_data.features, torch.Tensor):
            features = torch.tensor(train_data.features).float()
            target = torch.tensor(train_data.target).float()
        else:
            features, target = train_data.features, train_data.target

        train_loader = torch.utils.data.DataLoader(data.TensorDataset(
            features, target), batch_size=batch_size, shuffle=False)
        return train_loader
