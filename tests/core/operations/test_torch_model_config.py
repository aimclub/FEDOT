import pytest
import torch

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.models.neural import (
    TorchMLPRegressor,
    normalize_hidden_layer_sizes,
    resolve_torch_device,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


def _data(samples_count=50, batch_size=16):
    return TensorData(
        features=torch.zeros(samples_count, 2),
        target=torch.zeros(samples_count, 1),
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.tabular,
        dataloader_kwargs={'batch_size': batch_size},
    )


@pytest.mark.unit
def test_auto_torch_device_preserves_tensordata_device():
    assert resolve_torch_device('auto', 'cpu', cuda_available=True) == 'cpu'
    assert resolve_torch_device('auto', 'cuda', cuda_available=True) == 'cuda'
    assert resolve_torch_device(
        'auto', 'cuda:1', cuda_available=True) == 'cuda:1'
    assert resolve_torch_device(
        'cuda', 'cuda:1', cuda_available=True) == 'cuda:1'


@pytest.mark.unit
def test_explicit_unavailable_cuda_fails_at_model_boundary():
    with pytest.raises(ValueError, match='CUDA is unavailable'):
        resolve_torch_device('cuda', 'cpu', cuda_available=False)


@pytest.mark.unit
def test_torch_training_config_uses_model_and_dataloader_contract():
    model = TorchMLPRegressor(OperationParameters(hidden_layer_sizes=[64, 32], batch_size=128, device='cpu'))
    model._configure_training(_data())

    assert model.hidden_layer_sizes == (64, 32)
    assert model.batch_size == 50
    assert model.device == torch.device('cpu')


@pytest.mark.unit
def test_torch_training_config_uses_dataloader_batch_size_when_model_has_none():
    model = TorchMLPRegressor()
    model._configure_training(_data(batch_size=16))

    assert model.batch_size == 16


@pytest.mark.unit
def test_hidden_layer_sizes_are_immutable_and_positive():
    assert normalize_hidden_layer_sizes(32) == (32,)
    assert normalize_hidden_layer_sizes([32, 16]) == (32, 16)
    with pytest.raises(ValueError, match='positive'):
        normalize_hidden_layer_sizes([32, 0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ('params', 'message'),
    [
        ({'epochs': 0}, 'epochs must be positive'),
        ({'patience': 0}, 'patience must be positive'),
        ({'learning_rate': 0}, 'learning_rate must be positive'),
        ({'batch_size': 0}, 'batch_size must be positive'),
        ({'validation_fraction': 1}, 'validation_fraction must be in'),
    ],
)
def test_torch_fit_rejects_invalid_training_values(params, message):
    with pytest.raises(ValueError, match=message):
        TorchMLPRegressor(OperationParameters(**params)).fit(_data(samples_count=10, batch_size=4))
