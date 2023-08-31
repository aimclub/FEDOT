from typing import Optional

import numpy as np
from scipy.stats import ttest_rel

from fedot.core.data.data import InputData
from golem.utilities.requirements_notificator import warn_requirement

from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters


try:
    import torch
    from torch.nn import GRU, Linear, Dropout, MSELoss, BatchNorm1d
    from torch.optim import Adam
    from torch.utils.data import TensorDataset, DataLoader
except ModuleNotFoundError:
    warn_requirement('torch', 'fedot[extra]')
    torch = object()


class GRUImplementation(ModelImplementation):
    def __init__(self, params: OperationParameters):
        super().__init__(params)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.max_step = params.get('max_step') or 500
        self.seed = params.get('seed') or np.random.randint(0, np.iinfo(int).max)
        self.model = None
        self.batch_size = 50
        self.validation_size = 0.2
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.preprocessing_type = None
        self.preprocessing_mean = None
        self.preprocessing_std = None
        self.preprocessing_max = None
        self.preprocessing_min = None

    def preprocessing(self, x):
        if self.preprocessing_type == 'normalization':
            return (x - self.preprocessing_mean) / (self.preprocessing_std + 1e-6)
        elif self.preprocessing_type == 'minmax':
            return (x - self.preprocessing_min) / (self.preprocessing_max - self.preprocessing_min)
        else:
            raise ValueError((f"Unknown type of preprocessing: {self.preprocessing_type}."
                              f" Allowed types: normalization, minmax"))

    def postprocessing(self, y):
        if self.preprocessing_type == 'normalization':
            return y * self.preprocessing_std + self.preprocessing_mean
        elif self.preprocessing_type == 'minmax':
            return y * (self.preprocessing_max - self.preprocessing_min) + self.preprocessing_min
        else:
            raise ValueError((f"Unknown type of preprocessing: {self.preprocessing_type}."
                              f" Allowed types: normalization, minmax"))

    def numpy_to_torch(self, x, third_dimension=True):
        # (batch_size, num_timesteps or sequence_length, feature_size)
        x = x.astype(np.float32)
        x = self.preprocessing(x)
        if third_dimension:
            x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            x = x.reshape((x.shape[0], x.shape[1]))
        x = torch.from_numpy(x)
        return x

    def initialize_hidden(self):
        return torch.randn(*self.hidden_size, generator=self.generator)

    def fit(self, data: InputData):
        if self.model is None:
            dropout = self.params.get('dropout') or 0.2
            hidden_size = (self.params.get('hidden_size') or
                           round(data.features.shape[1] * 2 / (1 - dropout)))
            layer_count = self.params.get('layers_count') or 3

            self.model = GRUModel(input_data_length=data.features.shape[1],
                                  output_data_length=data.task.task_params.forecast_length,
                                  hidden_size=hidden_size,
                                  layer_count=layer_count,
                                  dropout=dropout)

        # prepare objects
        torch.manual_seed(self.seed)
        model = self.model.to(self.device)
        model.train()
        loss_fun = MSELoss(reduction="mean")
        opt_fun = Adam(model.parameters(), lr=1e-3)
        self.hidden_size = (model.gru.input_size * model.gru.num_layers, self.batch_size, model.gru.hidden_size)

        # prepare data
        x, y = data.features, data.target

        self.preprocessing_type = 'normalization'
        self.preprocessing_mean = np.mean(x[:, 0])
        self.preprocessing_std = np.std(x[:, 0])
        self.preprocessing_max = np.max(x[:, 0])
        self.preprocessing_min = np.min(x[:, 0])

        x = self.numpy_to_torch(x).to(self.device)
        y = self.numpy_to_torch(y, False).to(self.device)
        batch_count = int(x.shape[0] / self.batch_size)
        train_count = int(batch_count * (1 - self.validation_size))

        losses = []
        validations = []
        for epoch in range(self.max_step):

            # train
            _losses = []
            h = self.initialize_hidden().to(self.device)
            for batch_num in range(train_count):
                x_iter = x[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :, :]
                y_iter = y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :]
                y_pred, h = model(x_iter, h)
                loss = loss_fun(y_iter, y_pred)
                loss.backward()
                opt_fun.step()
                opt_fun.zero_grad()
                _losses.append(loss.item())
            losses.append(np.mean(_losses))

            # validation
            _validations = []
            h = self.initialize_hidden().to(self.device)
            for batch_num in range(train_count, batch_count):
                x_iter = x[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :, :]
                y_iter = y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :]
                y_pred, h = model(x_iter, h)
                loss = loss_fun(y_iter, y_pred)
                _validations.append(loss.item())
            validations.append(np.mean(_validations))

            if epoch > 5:
                # TODO: adaptive early stop
                if np.all(abs(np.diff(validations[-5:]) / validations[-4:]) < 0.05):
                    break

        # fit on validation data
        for epoch in range(2):
            h = self.initialize_hidden().to(self.device)
            for batch_num in range(train_count, batch_count):
                x_iter = x[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :, :]
                y_iter = y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size, :]
                y_pred, h = model(x_iter, h)
                loss = loss_fun(y_iter, y_pred)
                loss.backward()
                opt_fun.step()
                opt_fun.zero_grad()
                losses.append(loss.item())
        return self.model

    def predict(self, data: InputData):
        self.model.eval()
        with torch.no_grad():
            x = self.numpy_to_torch(data.features).to(self.device)
            return self.postprocessing(self.model(x)[0].to('cpu').numpy())


class GRUModel(torch.nn.Module):
    def __init__(self, input_data_length: int, output_data_length: int,
                 layer_count: int, dropout: float, hidden_size: int):
        super().__init__()

        self.gru = GRU(input_size=1,
                       hidden_size=hidden_size,
                       num_layers=layer_count,
                       dropout=dropout,
                       bias=True,
                       bidirectional=False,
                       batch_first=True)
        self.linear = Linear(in_features=input_data_length * hidden_size,
                             out_features=output_data_length)

    def forward(self, x, h=None):
        if h is None:
            x, h = self.gru(x, h)
        else:
            x, h = self.gru(x)
        x = self.linear(x.flatten(1))
        return x, h
