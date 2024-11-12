import numpy as np
from golem.utilities.requirements_notificator import warn_requirement

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class TorchMock:
    Module = list


try:
    import torch
    import torch.nn as nn

    from torch.optim.lr_scheduler import MultiStepLR
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:
    warn_requirement('torch', 'fedot[extra]')
    torch = object()
    nn = TorchMock


class ConvolutionalNetworkImplementation(ModelImplementation):
    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.model = self.init_network(self.params)
        self.device = self._get_device()

        self.optim_dict = {
            'adamw': torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate),
            'sgd': torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        }

        self.loss_dict = {
            'mae': nn.L1Loss,
            'mse': nn.MSELoss
        }
        self.mu = None
        self.std = None
        self.optimizer = self.optim_dict[self.params.get("optimizer")]
        self.criterion = self.loss_dict[self.params.get("loss")]()
        self.scheduler = MultiStepLR(self.optimizer, milestones=[30, 80], gamma=0.5)
        self.seed = None

    @property
    def learning_rate(self) -> float:
        return self.params.get("learning_rate")

    @staticmethod
    def init_network(params: OperationParameters) -> nn.Module:
        raise NotImplementedError

    def fit(self, train_data: InputData):
        """ Class fit ar model on data.

        Implementation uses the idea of teacher forcing. That means model learns
        to predict data when horizon != 1. It uses real values or previous model output
        to predict next value. self.teacher_forcing param is used to control probability
        of using real y values.

        :param train_data: data with features, target and ids to process
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            self.model.seed = self.seed
        self.model.init_linear(train_data.task.task_params.forecast_length)
        self.model = self.model.to(self.device)
        data_loader = self._create_dataloader(train_data)

        self.model.train()
        for epoch in range(self.params.get("num_epochs")):
            for x, y in data_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.forward(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
        return self.model

    def forward(self, x):
        self.model.init_hidden(x.shape[0], self.device)
        output = self.model(x.unsqueeze(1)).squeeze(0)
        return output

    def predict(self, input_data: InputData):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :return output_data: output data with smoothed time series
        """
        self.model.eval()

        predict = self._predict(input_data)

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData):
        self.model.eval()
        predict = self._predict(input_data)

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def _predict(self, input_data: InputData):
        x = self._scale(input_data.features.copy())
        x = torch.Tensor(x).to(self.device)
        predict = self.forward(x).cpu().detach().numpy()
        predict = self._inverse_scale(predict)
        return predict

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
        x, y = self._fit_transform_scaler(input_data)
        x, y = (torch.from_numpy(np_data.astype(np.float32)) for np_data in (x, y))
        return DataLoader(TensorDataset(x, y), batch_size=self.params.get("batch_size"))

    def _fit_transform_scaler(self, data: InputData):
        f = data.features.copy()
        t = data.target.copy()
        self.mu = np.mean(f)
        self.std = np.std(f)
        f_scaled = self._scale(f)
        t_scaled = self._scale(t)
        return f_scaled, t_scaled

    def _scale(self, array: np.ndarray):
        return (array - self.mu) / (self.std + 1e-6)

    def _inverse_scale(self, array: np.ndarray):
        return array * self.std + self.mu


class CGRUNetwork(nn.Module):
    def __init__(self,
                 hidden_size=200,
                 cnn1_kernel_size=5,
                 cnn1_output_size=16,
                 cnn2_kernel_size=3,
                 cnn2_output_size=32
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
        self.gru = nn.GRU(cnn2_output_size, self.hidden_size, dropout=0.1)
        self.hidden_cell = None
        self.linear = None
        self.seed = None

    def init_linear(self, forecast_length):
        self.linear = nn.Linear(self.hidden_size, forecast_length)

    def init_hidden(self, batch_size, device):
        kwargs = dict()
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)
            kwargs['generator'] = g
        self.hidden_cell = torch.randn(1, batch_size, self.hidden_size, **kwargs).to(device)

    def forward(self, x):
        if self.hidden_cell is None:
            raise Exception
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(2, 0, 1)
        out, self.hidden_cell = self.gru(x, self.hidden_cell)
        predictions = self.linear(self.hidden_cell)

        return predictions


class CLSTMNetwork(nn.Module):
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
        self.linear = None
        self.seed = None

    def init_linear(self, forecast_length):
        self.linear = nn.Linear(self.hidden_size * 2, forecast_length)

    def init_hidden(self, batch_size, device):
        kwargs = dict()
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)
            kwargs['generator'] = g
        self.hidden_cell = (torch.randn(1, batch_size, self.hidden_size, **kwargs).to(device),
                            torch.randn(1, batch_size, self.hidden_size, **kwargs).to(device))

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


class CGRUImplementation(ConvolutionalNetworkImplementation):
    @staticmethod
    def init_network(params: OperationParameters):
        return CGRUNetwork(
            hidden_size=int(params.get("hidden_size")),
            cnn1_kernel_size=int(params.get("cnn1_kernel_size")),
            cnn1_output_size=int(params.get("cnn1_output_size")),
            cnn2_kernel_size=int(params.get("cnn2_kernel_size")),
            cnn2_output_size=int(params.get("cnn2_output_size"))
        )


class CLSTMImplementation(ConvolutionalNetworkImplementation):
    @staticmethod
    def init_network(params: OperationParameters):
        return CLSTMNetwork(
            hidden_size=int(params.get("hidden_size")),
            cnn1_kernel_size=int(params.get("cnn1_kernel_size")),
            cnn1_output_size=int(params.get("cnn1_output_size")),
            cnn2_kernel_size=int(params.get("cnn2_kernel_size")),
            cnn2_output_size=int(params.get("cnn2_output_size"))
        )

