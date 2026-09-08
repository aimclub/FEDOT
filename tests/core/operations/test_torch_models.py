import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.operation_implementations.models.torch import (
    TorchLinearClassifier,
    TorchLinearRegressor,
    TorchMLPClassifier,
    TorchMLPRegressor,
)
from fedot.core.operations.model import Model
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _classification_data() -> TensorData:
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(120, 6, generator=generator)
    target = (features[:, 0] + 0.7 * features[:, 1] > 0).float()
    return TensorData(
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.tabular,
        idx=list(range(len(features))),
        features=features,
        target=target[:, None],
        dataloader_kwargs={'batch_size': 32},
    )


def _regression_data() -> TensorData:
    generator = torch.Generator().manual_seed(42)
    features = torch.randn(120, 6, generator=generator)
    target = 2.5 * features[:, 0] - 1.2 * features[:, 1] + 0.3
    return TensorData(
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.tabular,
        idx=list(range(len(features))),
        features=features,
        target=target[:, None],
        dataloader_kwargs={'batch_size': 32},
    )


@pytest.mark.unit
@pytest.mark.parametrize('model_cls', [TorchLinearClassifier, TorchMLPClassifier])
def test_torch_classifiers_fit_and_keep_predictions_on_input_device(model_cls):
    data = _classification_data()
    params = OperationParameters(
        device='cpu',
        hidden_layer_sizes=[32] if model_cls is TorchMLPClassifier else [],
        epochs=80,
        learning_rate=0.02,
        validation_fraction=0,
        random_state=42,
    )

    model = model_cls(params).fit(data)
    prediction = model.predict_labels(data)
    probabilities = model.predict_proba(data)

    assert prediction.device == data.features.device
    assert probabilities.shape == (len(data.features), 2)
    assert (prediction == data.target[:, 0]).float().mean() > 0.9


@pytest.mark.unit
@pytest.mark.parametrize('model_cls', [TorchLinearRegressor, TorchMLPRegressor])
def test_torch_regressors_fit_and_restore_target_scale(model_cls):
    data = _regression_data()
    params = OperationParameters(
        device='cpu',
        hidden_layer_sizes=[32] if model_cls is TorchMLPRegressor else [],
        epochs=100,
        learning_rate=0.02,
        validation_fraction=0,
        random_state=42,
    )

    prediction = model_cls(params).fit(data).predict(data)
    rmse = torch.sqrt(torch.mean((prediction - data.target[:, 0]) ** 2))

    assert prediction.device == data.features.device
    assert rmse < 0.25


@pytest.mark.unit
@pytest.mark.parametrize(
    ('operation', 'data'),
    [
        ('torch_linear', _classification_data),
        ('torch_mlp', _classification_data),
        ('torch_linear_reg', _regression_data),
        ('torch_mlp_reg', _regression_data),
    ],
)
def test_torch_operation_strategy_returns_tensordata_prediction(operation, data):
    input_data = data()
    params = OperationParameters(
        device='cpu',
        hidden_layer_sizes=[16] if 'mlp' in operation else [],
        epochs=2,
        validation_fraction=0,
    )
    model = Model(operation)

    fitted_operation, _ = model.fit(params, input_data)
    output = model.predict(fitted_operation, input_data, params=params, output_mode='full_probs')

    assert isinstance(output, TensorData)
    assert output is not input_data
    assert len(output.predict) == len(input_data.features)


@pytest.mark.unit
def test_torch_model_prediction_is_used_as_next_model_features():
    input_data = _classification_data()
    parent = PipelineNode('torch_linear')
    parent.parameters = {
        'epochs': 5,
        'validation_fraction': 0,
        'batch_size': 64,
    }
    root = PipelineNode('torch_mlp', nodes_from=[parent])
    root.parameters = {
        'hidden_layer_sizes': [8],
        'epochs': 5,
        'validation_fraction': 0,
        'batch_size': 64,
    }
    pipeline = Pipeline(root)

    pipeline.fit(input_data)
    prediction = pipeline.predict(input_data, output_mode='full_probs').predict

    assert prediction.shape == (len(input_data.features), 2)
    assert root.fitted_operation.n_features_in_ == 1


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_torch_mlp_keeps_cuda_tensordata_on_cuda():
    input_data = _classification_data().to('cuda')
    model = TorchMLPClassifier(OperationParameters(
        device='auto',
        hidden_layer_sizes=[16],
        epochs=2,
        validation_fraction=0,
    )).fit(input_data)

    prediction = model.predict_proba(input_data)

    assert model.device.type == 'cuda'
    assert prediction.device.type == 'cuda'
