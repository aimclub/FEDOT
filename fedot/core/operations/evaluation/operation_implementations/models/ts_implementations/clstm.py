from copy import copy

import numpy as np
from sklearn.preprocessing import StandardScaler

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import (
    prepare_target,
    ts_to_table, transform_features_and_target_into_lagged
)
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.ts_wrappers import _update_input, exception_if_not_ts_task
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.requirements_notificator import warn_requirement


class TorchMock:
    Module = list


try:
    import torch
    import torch.nn as nn

    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    warn_requirement('torch')
    torch = object()
    nn = TorchMock


class CLSTMImplementation(ModelImplementation):
    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.device = self._get_device()
        self.model = LSTMNetwork(
            hidden_size=int(params.get("hidden_size")),
            cnn1_kernel_size=int(params.get("cnn1_kernel_size")),
            cnn1_output_size=int(params.get("cnn1_output_size")),
            cnn2_kernel_size=int(params.get("cnn2_kernel_size")),
            cnn2_output_size=int(params.get("cnn2_output_size"))
        )

        self.optim_dict = {
            'adam': torch.optim.Adam(self.model.parameters(), lr=self.learning_rate),
            'sgd': torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        }

        self.loss_dict = {
            'mae': nn.L1Loss,
            'mse': nn.MSELoss
        }

        self.scaler = StandardScaler()
        self.optimizer = self.optim_dict[params.get("optimizer")]
        self.criterion = self.loss_dict[params.get("loss")]()

    @property
    def learning_rate(self) -> float:
        return self.params.get("learning_rate")

    @property
    def window_size(self) -> int:
        return self.params.get("window_size")

    def fit(self, train_data: InputData):
        """ Class fit ar model on data.

        Implementation uses the idea of teacher forcing. That means model learns
        to predict data when horizon != 1. It uses real values or previous model output
        to predict next value. self.teacher_forcing param is used to control probability
        of using real y values.

        :param train_data: data with features, target and ids to process
        """

        self.model = self.model.to(self.device)
        data_loader, forecast_length = self._create_dataloader(train_data)

        self.model.train()
        for epoch in range(self.params.get("num_epochs")):
            for x, y in data_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                final_output = self._apply_teacher_forcing(x, y, forecast_length)
                loss = self.criterion(final_output, y)
                loss.backward()
                self.optimizer.step()
        return self.model

    def _apply_teacher_forcing(self, x, y, forecast_length):
        final_output = None
        for i in range(forecast_length):
            self.model.init_hidden(x.shape[0], self.device)
            output = self.model(x.unsqueeze(1)).squeeze(0)
            if np.random.random_sample() > int(self.params.get("teacher_forcing")):
                x = torch.hstack((x[:, 1:], output))
            else:
                x = torch.hstack((x, y[:, i].unsqueeze(1)))

            if final_output is not None:
                final_output = torch.hstack((final_output, output))
            else:
                final_output = output
        return final_output

    def predict(self, input_data: InputData):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :return output_data: output data with smoothed time series
        """
        self.model.eval()
        input_data = copy(input_data)
        forecast_length = input_data.task.task_params.forecast_length

        input_data.features = input_data.features[-self.window_size:].reshape(1, -1)
        input_data.idx = input_data.idx[-forecast_length:]

        predict = self._out_of_sample_ts_forecast(input_data)

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData):
        self.model.eval()
        input_data = copy(input_data)
        forecast_length = input_data.task.task_params.forecast_length
        new_idx, lagged_table = ts_to_table(idx=input_data.idx,
                                            time_series=input_data.features,
                                            window_size=self.window_size,
                                            is_lag=True)

        final_idx, features_columns, final_target = prepare_target(all_idx=input_data.idx,
                                                                   idx=new_idx,
                                                                   features_columns=lagged_table,
                                                                   target=input_data.target,
                                                                   forecast_length=forecast_length)
        input_data.idx = final_idx
        input_data.features = features_columns
        input_data.target = final_target
        predict = self._out_of_sample_ts_forecast(input_data)

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def _predict(self, input_data: InputData):
        features_scaled = self._transform_scaler_features(input_data)
        x = torch.Tensor(features_scaled).to(self.device)
        self.model.init_hidden(x.shape[0], self.device)
        predict = self.model(x.unsqueeze(1)).squeeze(0).cpu().detach().numpy()
        return self._inverse_transform_scaler(predict)

    def _out_of_sample_ts_forecast(self, input_data: InputData) -> np.array:
        """ Method for out_of_sample CLSTM forecasting (use previous outputs as next inputs)

        :param input_data: data with features, target and ids to process
        :return np.array: np.array with predicted values to process it into output_data
        """

        input_data_new = copy(input_data)
        # Prepare data for time series forecasting
        task = input_data_new.task
        exception_if_not_ts_task(task)

        pre_history_ts = np.array(input_data_new.features)

        number_of_iterations = task.task_params.forecast_length

        final_forecast = None

        for _ in range(0, number_of_iterations):
            with torch.no_grad():
                iter_predict = self._predict(input_data_new)
            if final_forecast is not None:
                final_forecast = np.hstack((final_forecast, iter_predict))
            else:
                final_forecast = iter_predict

            # Add prediction to the historical data - update it
            pre_history_ts = np.hstack((pre_history_ts[:, 1:], iter_predict))
            # Prepare InputData for next iteration
            input_data_new = _update_input(pre_history_ts, number_of_iterations, task)

        return final_forecast

    def _fit_transform_scaler(self, data: InputData):
        f_scaled = self.scaler.fit_transform(data.features.reshape(-1, 1)).reshape(-1)
        t_scaled = self.scaler.transform(data.target.reshape(-1, 1)).reshape(-1)
        return f_scaled, t_scaled

    def _inverse_transform_scaler(self, data: np.ndarray):
        start_shape = data.shape
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(start_shape)

    def _transform_scaler_features(self, data: InputData):
        start_shape = data.features.shape
        return self.scaler.transform(data.features.reshape(-1, 1)).reshape(start_shape)

    def _transform_scaler_target(self, data: InputData):
        start_shape = data.features.shape
        return self.scaler.transform(data.target.reshape(-1, 1)).reshape(start_shape)

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device

    def _create_dataloader(self, input_data: InputData):
        """ Method for creating torch.utils.data.DataLoader object from input_data

        Generate lag tables and process it into DataLoader

        :param input_data: data with features, target and ids to process
        :return torch.utils.data.DataLoader: DataLoader with train data
        """
        forecast_length = input_data.task.task_params.forecast_length
        features_scaled, target_scaled = self._fit_transform_scaler(input_data)
        input_data.features = features_scaled
        input_data.target = target_scaled
        final_idx, features_columns, final_target = transform_features_and_target_into_lagged(input_data,
                                                                                              forecast_length,
                                                                                              self.window_size)
        x = torch.from_numpy(features_columns.copy()).float()
        y = torch.from_numpy(final_target.copy()).float()
        return DataLoader(TensorDataset(x, y), batch_size=self.params.get("batch_size")), forecast_length


class LSTMNetwork(nn.Module):
    def __init__(self,
                 hidden_size=200,
                 cnn1_kernel_size=5,
                 cnn1_output_size=16,
                 cnn2_kernel_size=3,
                 cnn2_output_size=32,
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn1_output_size, kernel_size=cnn1_kernel_size),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=cnn1_output_size, out_channels=cnn2_output_size, kernel_size=cnn2_kernel_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn2_output_size, self.hidden_size, dropout=0.1)
        self.hidden_cell = None
        self.linear = nn.Linear(self.hidden_size * 2, 1)

    def init_hidden(self, batch_size, device):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_size).to(device),
                            torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        if self.hidden_cell is None:
            raise Exception
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(2, 0, 1)
        out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        hidden_cat = torch.cat([self.hidden_cell[0], self.hidden_cell[1]], dim=2)
        predictions = self.linear(hidden_cat)

        return predictions
