import math
import warnings
from typing import Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.split.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    transform_features_and_target_into_lagged
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from fedot.industrial.core.architecture.abstraction.decorators import convert_inputdata_to_torch_time_series_dataset
from fedot.industrial.core.architecture.preprocessing.data_convertor import DataConverter
from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.layers.special import adjust_learning_rate, EarlyStopping
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix
from fedot.industrial.core.operation.transformation.window_selector import WindowSizeSelector
from fedot.industrial.core.repository.constanst_repository import MSE

warnings.filterwarnings("ignore", category=UserWarning)


class _ResidualBlock(nn.Module):
    def __init__(self,
                 num_filters: int,
                 kernel_size: int,
                 dilation_base: int,
                 dropout_fn,
                 weight_norm: bool,
                 nr_blocks_below: int,
                 num_layers: int,
                 input_size: int,
                 target_size: int):
        super(_ResidualBlock, self).__init__()
        self.dilation_base = dilation_base
        self.kernel_size = kernel_size
        self.dropout_fn = dropout_fn
        self.num_layers = num_layers
        self.nr_blocks_below = nr_blocks_below

        input_dim = input_size if nr_blocks_below == 0 else num_filters
        output_dim = target_size if nr_blocks_below == num_layers - 1 else num_filters
        self.conv1 = nn.Conv1d(input_dim,
                               num_filters,
                               kernel_size,
                               dilation=(dilation_base ** nr_blocks_below))
        self.conv2 = nn.Conv1d(num_filters,
                               output_dim,
                               kernel_size,
                               dilation=(dilation_base ** nr_blocks_below))
        if weight_norm:
            self.conv1, self.conv2 = nn.utils.weight_norm(
                self.conv1), nn.utils.weight_norm(self.conv2)
        if nr_blocks_below == 0 or nr_blocks_below == num_layers - 1:
            self.conv3 = nn.Conv1d(input_dim,
                                   output_dim,
                                   kernel_size=1)

    def forward(self, x):
        residual = x

        # first step
        left_padding = (self.dilation_base **
                        self.nr_blocks_below) * (self.kernel_size - 1)
        x = F.pad(x, (left_padding, 0))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout_fn(x)

        # second step
        x = F.pad(x, (left_padding, 0))
        x = self.conv2(x)
        if self.nr_blocks_below < self.num_layers - 1:
            x = F.relu(x)
        x = self.dropout_fn(x)

        # add residual
        if self.nr_blocks_below in {0, self.num_layers - 1}:
            residual = self.conv3(residual)
        x += residual
        return x


class TCNModule(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_chunk_length: int,
                 kernel_size: int,
                 num_filters: int,
                 num_layers: Optional[int],
                 dilation_base: int,
                 weight_norm: bool,
                 target_size: int,
                 target_length: int,
                 dropout: float):
        super(TCNModule, self).__init__()

        # Defining parameters
        self.input_size = input_size
        self.input_chunk_length = input_chunk_length
        self.n_filters = num_filters
        self.kernel_size = kernel_size
        self.target_length = target_length
        self.target_size = target_size
        self.dilation_base = dilation_base
        self.dropout = nn.Dropout(p=dropout)

        # If num_layers is not passed, compute number of layers needed for full
        # history coverage
        if num_layers is None and dilation_base > 1:
            num_layers = math.ceil(math.log(
                (input_chunk_length - 1) * (dilation_base - 1) / (kernel_size - 1) / 2 + 1, dilation_base))
        elif num_layers is None:
            num_layers = math.ceil(
                (input_chunk_length - 1) / (kernel_size - 1) / 2)
        self.num_layers = num_layers

        # Building TCN module
        self.res_blocks_list = []
        for i in range(num_layers):
            res_block = _ResidualBlock(num_filters,
                                       kernel_size,
                                       dilation_base,
                                       self.dropout,
                                       weight_norm,
                                       i,
                                       num_layers,
                                       self.input_size,
                                       target_size)
            self.res_blocks_list.append(res_block)
        self.res_blocks = nn.ModuleList(self.res_blocks_list)

    def forward(self, x):
        # data is of size (batch_size, input_chunk_length, input_size)
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        for res_block in self.res_blocks_list:
            x = res_block(x)
        x = x.transpose(1, 2)
        x = x.view(batch_size, self.input_chunk_length, self.target_size)
        return x


class TCNModel(BaseNeuralModel):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.epochs = self.params.get("epochs", 100)
        self.batch_size = self.params.get("batch_size", 32)
        self.activation = self.params.get('activation', 'ReLU')
        self.learning_rate = self.params.get("learning_rate", 0.01)
        self.kernel_size = self.params.get("kernel_size", 3)
        self.num_filters = self.params.get("num_filters", 5)
        self.num_layers = self.params.get("num_layers", 3)
        self.dilation_base = self.params.get("dilation_base", 2)
        self.dropout = self.params.get("dropout", 0.2)
        self.weight_norm = self.params.get("weight_norm", False)
        self.split = self.params.get("split_data", False)
        self.patch_len = self.params.get("patch_len", None)
        self.horizon = self.params.get("horizon", None)
        self.train_log = self.params.get("train_log", False)
        self.model_list = []

    def _init_model(self, ts: InputData):
        print(f'tcn - {self.params.to_dict()}')
        input_size = self.patch_len or ts.features.shape[-1]
        self.patch_len = input_size
        model = TCNModule(input_size=input_size,
                          input_chunk_length=ts.features.shape[0],
                          target_size=self.horizon,
                          kernel_size=self.kernel_size,
                          num_filters=self.num_filters,
                          num_layers=self.num_layers,
                          dilation_base=self.dilation_base,
                          target_length=self.patch_len,
                          dropout=self.dropout,
                          weight_norm=self.weight_norm).to(default_device())
        self._evaluate_num_of_epochs(ts)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = MSE()
        return model, loss_fn, optimizer

    def _fit_model(self, input_data: InputData, split_data: bool):
        if self.patch_len is None:
            window_size = WindowSizeSelector(
                method='hac').get_window_size(input_data.features)
            self.patch_len = window_size * input_data.features.flatten().size // 100
        train_loader = self._prepare_data(input_data.features,
                                          self.patch_len,
                                          split_data)
        self.test_patch_len = self.patch_len
        model, loss_fn, optimizer = self._init_model(input_data)
        model = self._train_loop(model, train_loader, loss_fn, optimizer)
        self.model_list.append(model)

    def __preprocess_for_fedot(self, input_data: InputData):
        input_data.features = np.squeeze(input_data.features)
        if self.horizon is None:
            self.horizon = input_data.task.task_params.forecast_length
        if len(input_data.features.shape) == 1:
            input_data.features = input_data.features.reshape(1, -1)
        self.seq_len = input_data.features.shape[1]
        self.target = input_data.target
        self.task_type = input_data.task
        return input_data

    @convert_inputdata_to_torch_time_series_dataset
    def __create_torch_loader(self, train_data: tuple[np.ndarray]):
        return DataLoader(data.TensorDataset(
            train_data.x, train_data.y),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False)

    def fit(self, input_data: InputData):
        input_data = self.__preprocess_for_fedot(input_data)
        self._fit_model(input_data, split_data=self.split)

    def split_data(self, input_data):
        time_series = pd.DataFrame(input_data)
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=self.horizon))
        idx = pd.to_datetime(time_series['datetime'].values) if 'datetime' in time_series.columns \
            else time_series.columns.values
        time_series = time_series.values
        return InputData(idx=idx,
                         features=time_series.flatten(),
                         target=time_series.flatten(),
                         task=task,
                         data_type=DataTypesEnum.ts)

    def _prepare_data(self,
                      ts: np.ndarray,
                      patch_len: int,
                      split_data: bool,
                      validation_blocks: Optional[int] = None):
        train_data = self.split_data(ts)
        if split_data:
            validation_blocks = validation_blocks if validation_blocks is not None else 1
            train_data, val_data = train_test_data_setup(
                train_data, validation_blocks=validation_blocks)
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(
                train_data, self.horizon, patch_len)
            _, val_data.features, val_data.target = transform_features_and_target_into_lagged(
                val_data, self.horizon, patch_len)
        else:
            _, train_data.features, train_data.target = transform_features_and_target_into_lagged(
                train_data, self.horizon, patch_len)
        train_loader = self.__create_torch_loader(train_data)
        return train_loader

    def _train_loop(self,
                    model: TCNModule,
                    train_loader: DataLoader,
                    loss_fn: nn.MSELoss,
                    optimizer: torch.optim):
        train_steps = max(len(train_loader), 1)
        early_stopping = EarlyStopping()
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            epochs=self.epochs,
                                            max_lr=self.learning_rate)
        for epoch in tqdm(range(self.epochs)):
            train_loss = []
            model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(default_device())
                batch_y = batch_y.float().to(default_device())
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model.float()
                optimizer.step()
                adjust_learning_rate(optimizer,
                                     scheduler,
                                     epoch + 1,
                                     learning_rate=self.learning_rate,
                                     printout=False,
                                     lradj='constant')
                scheduler.step()
            train_loss = np.average(train_loss)
            if self.train_log:
                if epoch % 25 == 0:
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                        epoch + 1, train_steps, train_loss))
                    print('Updating learning rate to {}'.format(
                        scheduler.get_last_lr()[0]))
            if early_stopping.early_stop:
                print("Early stopping")
                break
        return model

    def _predict(self, model: TCNModule, test_loader: DataLoader):
        model.eval()
        with torch.no_grad():
            if self.forecast_mode == 'in_sample':
                outputs = []
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.float().to(default_device())
                    batch_y = batch_y.float().to(default_device())
                    outputs.append(model(batch_x))
                return torch.cat(outputs).cpu().numpy()
            else:
                last_patch = test_loader.dataset[0][-1]
                c, s = last_patch.size()
                last_patch = last_patch.reshape(1, c, s).to(default_device())
                outputs = model(last_patch)
                return outputs.cpu().numpy()

    def _predict_model(self,
                       test_data: np.ndarray,
                       test_idx: str = None):
        y_pred = []
        self.forecast_mode = 'out_of_sample'
        for model in self.model_list:
            y_pred.append(self._predict_loop(model, test_data))
        y_pred = np.array(y_pred).squeeze()
        # Workaround for prediction starting point shift
        # TODO: find out what triggers prediction starting point shift
        start_point = self.target[-1]
        shift = y_pred[0] - start_point
        y_pred -= shift
        if test_idx is None:
            forecast_idx_predict = np.arange(
                start=test_data.idx.shape[0],
                stop=test_data.idx.shape[0] +
                self.horizon,
                step=1)
        else:
            forecast_idx_predict = test_idx
        return OutputData(
            idx=forecast_idx_predict,
            task=self.task_type,
            predict=y_pred,
            target=self.target,
            data_type=DataTypesEnum.table)

    def _predict_loop(self,
                      model: TCNModule,
                      test_data: np.ndarray):
        if isinstance(test_data, InputData):
            test_data = test_data.features
        if len(test_data.shape) == 1:
            test_data = test_data.reshape(1, -1)
        features = HankelMatrix(
            time_series=test_data,
            window_size=self.test_patch_len).trajectory_matrix
        features = torch.from_numpy(
            DataConverter(
                data=features).convert_to_torch_format()).float().permute(2, 1, 0)
        target = torch.from_numpy(
            DataConverter(
                data=features).convert_to_torch_format()).float()
        test_loader = DataLoader(
            data.TensorDataset(features, target),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False)
        return self._predict(model, test_loader)
