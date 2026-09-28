from typing import Optional

import torch
from fedot.core.operations.operation_parameters import OperationParameters
from torch import Tensor
from torch import nn
from torch import optim
from torchvision.models import ResNet, resnet101, resnet152, resnet18, resnet34, resnet50

from fedot.industrial.core.architecture.abstraction.decorators import convert_to_4d_torch_array
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel


def resnet18_one_channel(**kwargs) -> ResNet:
    """ResNet18 for one input channel"""
    model = resnet18(**kwargs)
    model.conv1 = nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
    return model


def resnet34_one_channel(**kwargs) -> ResNet:
    """ResNet34 for one input channel"""
    model = resnet34(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def resnet50_one_channel(**kwargs) -> ResNet:
    """ResNet50 for one input channel"""
    model = resnet50(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def resnet101_one_channel(**kwargs) -> ResNet:
    """ResNet101 for one input channel"""
    model = resnet101(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


def resnet152_one_channel(**kwargs) -> ResNet:
    """ResNet152 for one input channel"""
    model = resnet152(**kwargs)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
    return model


CLF_MODELS = {
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'ResNet101': resnet101,
    'ResNet152': resnet152,
    'ResNet18one': resnet18_one_channel,
    'ResNet34one': resnet34_one_channel,
    'ResNet50one': resnet50_one_channel,
    'ResNet101one': resnet101_one_channel,
    'ResNet152one': resnet152_one_channel,
}

CLF_MODELS_ONE_CHANNEL = {
    'ResNet18one': resnet18_one_channel,
    'ResNet34one': resnet34_one_channel,
    'ResNet50one': resnet50_one_channel,
    'ResNet101one': resnet101_one_channel,
    'ResNet152one': resnet152_one_channel,
}


class ResNet:
    def __init__(self,
                 input_dim,
                 output_dim,
                 model_name: str = 'ResNet18one'):
        model_list = {**CLF_MODELS, **CLF_MODELS_ONE_CHANNEL}
        self.model = model_list[model_name](num_classes=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Implements the forward method of the model and returns predictions."""
        x = x.to(default_device())
        return self.model(x)


class ResNetModel(BaseNeuralModel):
    """Class responsible for ResNet model implementation.

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
        self.epochs = self.params.get('epochs', 25)
        self.batch_size = self.params.get('batch_size', 64)
        self.model_name = self.params.get('model_name', 'ResNet18')

    def __repr__(self):
        return self.model_name

    def _init_model(self, ts):
        loss_fn = self._get_loss_metric(ts)
        self.model_name = f'{self.model_name}one' if self.is_regression_task else self.model_name
        self.model = ResNet(input_dim=ts.features.shape[1],
                            output_dim=self.num_classes,
                            model_name=self.model_name)
        self.model_for_inference = ResNet(input_dim=ts.features.shape[1],
                                          output_dim=self.num_classes,
                                          model_name=self.model_name).model
        self.model = self.model.model
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        return loss_fn, optimizer

    @convert_to_4d_torch_array
    def _predict_model(self, x_test, output_mode: str = 'default'):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device('cpu'))
        pred = self.model(x_test)
        return self._convert_predict(pred, output_mode)
