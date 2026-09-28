from typing import Optional

from fastai.torch_core import Module
from fastcore.meta import delegates
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from torch import Tensor
from torch import nn, optim

from fedot.industrial.core.architecture.abstraction.decorators import convert_to_3d_torch_array
from fedot.industrial.core.architecture.settings.computational import default_device
from fedot.industrial.core.models.nn.network_impl.base_nn_model import BaseNeuralModel
from fedot.industrial.core.models.nn.network_modules.layers.pooling_layers import GAP1d
from fedot.industrial.core.models.nn.network_modules.layers.special import InceptionBlock, InceptionModule


@delegates(InceptionModule.__init__)
class InceptionTime(Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            seq_len=None,
            number_of_filters=32,
            nb_filters=None,
            **kwargs):
        super().__init__()
        if number_of_filters is None:
            number_of_filters = nb_filters
        self.inception_block = InceptionBlock(
            input_dim, number_of_filters, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(number_of_filters * 4, output_dim)

    def forward(self, x):
        x = self.inception_block(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


class InceptionTimeModel(BaseNeuralModel):
    """Class responsible for InceptionTime model implementation.

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
        self.num_classes = self.params.get('num_classes', 1)

    def __repr__(self):
        return "InceptionNN"

    def _init_model(self, ts):
        loss_fn = self._get_loss_metric(ts)
        self.model = InceptionTime(
            input_dim=ts.features.shape[1],
            output_dim=self.num_classes).to(
            default_device())
        self.model_for_inference = InceptionTime(
            input_dim=ts.features.shape[1], output_dim=self.num_classes)
        self._evaluate_num_of_epochs(ts)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return loss_fn, optimizer

    def _fit_model(self, ts: InputData, split_data: bool = False):
        loss_fn, optimizer = self._init_model(ts)
        train_loader, val_loader = self._prepare_data(ts, split_data)

        self._train_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

    @convert_to_3d_torch_array
    def _predict_model(self, x_test, output_mode: str = 'default'):
        self.model.eval()
        x_test = Tensor(x_test).to(default_device('cpu'))
        pred = self.model(x_test)
        return self._convert_predict(pred, output_mode)
