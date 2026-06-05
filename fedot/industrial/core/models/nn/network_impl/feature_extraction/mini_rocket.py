from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.base_extractor import BaseExtractor


class MiniRocketFeatures(nn.Module):
    """This is a Pytorch implementation of MiniRocket developed by Malcolm McLean and Ignacio Oguiza

    Args:
        input_dim (int): Number of input time series.
        seq_len (int): Length of input time series.
        num_features (int): Number of features to generate.
        max_dilations_per_kernel (int): Maximum number of dilations per kernel.
        random_state (int): Random state.

    References:
        @article{dempster_etal_2020,
          author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
          title   = {{MINIROCKET}: A Very Fast (Almost) Deterministic Transform for Time Series Classification},
          year    = {2020},
          journal = {arXiv:2012.08791}
        }
        Original paper: https://arxiv.org/abs/2012.08791
        Original code:  https://github.com/angus924/minirocket

    """

    kernel_size, num_kernels, fitting = 9, 84, False

    def __init__(self,
                 input_dim,
                 seq_len,
                 num_features=10_000,
                 max_dilations_per_kernel=32,
                 random_state=None):
        super(MiniRocketFeatures, self).__init__()
        self.input_dim, self.seq_len = input_dim, seq_len
        self.num_features = num_features // self.num_kernels * self.num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state

        # Convolution
        indices = torch.combinations(
            torch.arange(self.kernel_size), 3).unsqueeze(1)
        kernels = (-torch.ones(self.num_kernels, 1, self.kernel_size)
                   ).scatter_(2, indices, 2)
        self.kernels = nn.Parameter(kernels.repeat(
            input_dim, 1, 1), requires_grad=False)

        # Dilations & padding
        self._set_dilations(seq_len)

        # Channel combinations (multivariate)
        if input_dim > 1:
            self._set_channel_combinations(input_dim)

        # Bias
        for i in range(self.num_dilations):
            self.register_buffer(f'biases_{i}', torch.empty(
                (self.num_kernels, self.num_features_per_dilation[i])))
        self.register_buffer('prefit', torch.BoolTensor([False]))

    def fit(self, X, chunksize=None):
        num_samples = X.shape[0]
        if chunksize is None:
            chunksize = min(num_samples, self.num_dilations * self.num_kernels)
        else:
            chunksize = min(num_samples, chunksize)
        np.random.seed(self.random_state)
        idxs = np.random.choice(num_samples, chunksize, False)
        self.fitting = True
        if isinstance(X, np.ndarray):
            self(torch.from_numpy(X[idxs]).float().to(self.kernels.device))
        else:
            self(X[idxs].to(self.kernels.device))
        self.fitting = False
        return self

    def forward(self, x):
        _features = []
        for i, (dilation, padding) in enumerate(
                zip(self.dilations, self.padding)):
            _padding1 = i % 2

            # Convolution
            C = F.conv1d(x.float(), self.kernels, padding=padding,
                         dilation=dilation, groups=self.input_dim)
            if self.input_dim > 1:  # multivariate
                C = C.reshape(x.shape[0], self.input_dim, self.num_kernels, -1)
                channel_combination = getattr(
                    self, f'channel_combinations_{i}')
                C = torch.mul(C, channel_combination)
                C = C.sum(1)

            # Bias
            if not self.prefit or self.fitting:
                num_features_this_dilation = self.num_features_per_dilation[i]
                bias_this_dilation = self._get_bias(
                    C, num_features_this_dilation)
                setattr(self, f'biases_{i}', bias_this_dilation)
                if self.fitting:
                    if i < self.num_dilations - 1:
                        continue
                    else:
                        self.prefit = torch.BoolTensor([True])
                        return
                elif i == self.num_dilations - 1:
                    self.prefit = torch.BoolTensor([True])
            else:
                bias_this_dilation = getattr(self, f'biases_{i}')

            # Features
            _features.append(self._get_PPVs(
                C[:, _padding1::2], bias_this_dilation[_padding1::2]))
            _features.append(self._get_PPVs(
                C[:, 1 - _padding1::2, padding:-padding], bias_this_dilation[1 - _padding1::2]))
        return torch.cat(_features, dim=1)

    def _get_PPVs(self, C, bias):
        C = C.unsqueeze(-1)
        bias = bias.view(1, bias.shape[0], 1, bias.shape[1])
        return (C > bias).float().mean(2).flatten(1)

    def _set_dilations(self, input_length):
        num_features_per_kernel = self.num_features // self.num_kernels
        true_max_dilations_per_kernel = min(
            num_features_per_kernel, self.max_dilations_per_kernel)
        multiplier = num_features_per_kernel / true_max_dilations_per_kernel
        max_exponent = np.log2((input_length - 1) / (9 - 1))
        dilations, num_features_per_dilation = np.unique(
            np.logspace(
                0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
                np.int32), return_counts=True)
        num_features_per_dilation = (
            num_features_per_dilation * multiplier).astype(np.int32)
        remainder = num_features_per_kernel - num_features_per_dilation.sum()
        i = 0
        while remainder > 0:
            num_features_per_dilation[i] += 1
            remainder -= 1
            i = (i + 1) % len(num_features_per_dilation)
        self.num_features_per_dilation = num_features_per_dilation
        self.num_dilations = len(dilations)
        self.dilations = dilations
        self.padding = []
        for i, dilation in enumerate(dilations):
            self.padding.append((((self.kernel_size - 1) * dilation) // 2))

    def _set_channel_combinations(self, num_channels):
        num_combinations = self.num_kernels * self.num_dilations
        max_num_channels = min(num_channels, 9)
        max_exponent_channels = np.log2(max_num_channels + 1)
        np.random.seed(self.random_state)
        num_channels_per_combination = (
            2 ** np.random.uniform(0, max_exponent_channels, num_combinations)).astype(np.int32)
        channel_combinations = torch.zeros(
            (1, num_channels, num_combinations, 1))
        for i in range(num_combinations):
            channel_combinations[:, np.random.choice(
                num_channels, num_channels_per_combination[i], False), i] = 1
        channel_combinations = torch.split(
            channel_combinations, self.num_kernels, 2)  # split by dilation
        for i, channel_combination in enumerate(channel_combinations):
            self.register_buffer(
                f'channel_combinations_{i}',
                channel_combination)  # per dilation

    def _get_quantiles(self, n):
        return torch.tensor([(_ * ((np.sqrt(5) + 1) / 2)) %
                            1 for _ in range(1, n + 1)]).float()

    def _get_bias(self, C, num_features_this_dilation):
        np.random.seed(self.random_state)
        idxs = np.random.choice(C.shape[0], self.num_kernels)
        samples = C[idxs].diagonal().T
        biases = torch.quantile(samples, self._get_quantiles(
            num_features_this_dilation).to(C.device), dim=1).T
        return biases


MRF = MiniRocketFeatures


def get_minirocket_features(data,
                            model,
                            chunksize=1024,
                            use_cuda=None,
                            convert_to_numpy=True):
    """Function used to split a large dataset into chunks, avoiding OOM error."""
    use = torch.cuda.is_available() if use_cuda is None else use_cuda
    device = torch.device(torch.cuda.current_device()
                          ) if use else torch.device('cpu')
    model = model.to(device)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float().to(device)
    _features = []
    for oi in torch.split(data, chunksize):
        _features.append(model(oi))
    features = torch.cat(_features).unsqueeze(-1)
    if convert_to_numpy:
        return features.cpu().detach().numpy()
    else:
        return features


class MiniRocketHead(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seq_len=1,
                 batch_norm=True,
                 dropout=0.):
        layers = [nn.Flatten()]
        if batch_norm:
            layers += [nn.BatchNorm1d(input_dim)]
        if dropout:
            layers += [nn.Dropout(dropout)]
        linear = nn.Linear(input_dim, output_dim)
        nn.init.constant_(linear.weight.data, 0)
        nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        head = nn.Sequential(*layers)
        super().__init__(OrderedDict(
            [('backbone', nn.Sequential()), ('head', head)]))


class MiniRocket(nn.Sequential):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seq_len,
                 num_features=10_000,
                 max_dilations_per_kernel=32,
                 random_state=None,
                 batch_norm=True,
                 dropout=0.):

        # Backbone
        backbone = MiniRocketFeatures(
            input_dim,
            seq_len,
            num_features=num_features,
            max_dilations_per_kernel=max_dilations_per_kernel,
            random_state=random_state)
        num_features = backbone.num_features

        # Head
        self.head_number_filters = num_features
        layers = [nn.Flatten()]

        if batch_norm:
            layers += [nn.BatchNorm1d(num_features)]
        if dropout:
            layers += [nn.Dropout(dropout)]

        linear = nn.Linear(num_features, output_dim)
        nn.init.constant_(linear.weight.data, 0)
        nn.init.constant_(linear.bias.data, 0)
        layers += [linear]
        head = nn.Sequential(*layers)

        super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))

    def fit(self, X, chunksize=None):
        self.backbone.fit(X, chunksize=chunksize)


class MiniRocketExtractor(BaseExtractor):
    """Class responsible for MiniRocketmodel feature generator .

    Attributes:
        self.num_features: int, the number of features.

    Example:
        To use this operation you can create pipeline as follows::
            from fedot.core.pipelines.pipeline_builder import PipelineBuilder
            from examples.fedot.fedot_ex import init_input_data
            from fedot.industrial.tools.loader import DataLoader
            from fedot.industrial.core.repository.initializer_industrial_models import IndustrialModels

            train_data, test_data = DataLoader(dataset_name='Ham').load_data()
            with IndustrialModels():
                pipeline = PipelineBuilder().add_node('minirocket_features').add_node(
                    'rf').build()
                input_data = init_input_data(train_data[0], train_data[1])
                pipeline.fit(input_data)
                features = pipeline.predict(input_data)
                print(features)
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.num_features = params.get('num_features', 10000)

    def __repr__(self):
        return 'LargeFeatureSpace'

    def _save_and_clear_cache(self, model_list):
        del model_list
        with torch.no_grad():
            torch.cuda.empty_cache()

    def _generate_features_from_ts(
            self,
            ts: np.array,
            mode: str = 'multivariate'):

        if ts.shape[1] > 1 and mode == 'chanel_independent':
            mrf = MiniRocketFeatures(
                input_dim=1,
                seq_len=ts.shape[2],
                num_features=self.num_features).to(
                default_device())

            n_dim = range(ts.shape[1])
            ts_converted = [ts[:, i, :] for i in n_dim]
            ts_converted = [x.reshape(x.shape[0], 1, x.shape[1])
                            for x in ts_converted]
            model_list = [mrf for i in n_dim]
        else:
            mrf = MiniRocketFeatures(
                input_dim=ts.shape[1],
                seq_len=ts.shape[2],
                num_features=self.num_features).to(
                default_device())

            ts_converted = [ts]
            model_list = [mrf]

        fitted_model = [model.fit(data)
                        for model, data in zip(model_list, ts_converted)]
        features = [get_minirocket_features(
            data, model) for model, data in zip(fitted_model, ts_converted)]
        minirocket_features = [feature_by_dim.swapaxes(
            1, 2) for feature_by_dim in features]
        minirocket_features = np.concatenate(minirocket_features, axis=1)
        minirocket_features = OutputData(
            idx=np.arange(
                minirocket_features.shape[2]),
            task=self.task,
            predict=minirocket_features,
            data_type=DataTypesEnum.image)
        self._save_and_clear_cache(model_list)
        return minirocket_features

    def generate_minirocket_features(self, ts: np.array) -> InputData:
        return self._generate_features_from_ts(ts)

    def generate_features_from_ts(self, ts_data: np.array,
                                  dataset_name: str = None):
        return self.generate_minirocket_features(ts=ts_data)

    def _transform(self,
                   input_data: InputData) -> np.array:
        """
        Method for feature generation for all series
        """
        self.task = input_data.task
        self.task.task_params = self.__repr__()
        feature_matrix = self.generate_features_from_ts(input_data.features)
        feature_matrix.predict = self._clean_predict(feature_matrix.predict)
        return feature_matrix
