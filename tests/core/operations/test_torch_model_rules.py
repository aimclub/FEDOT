import pytest

from fedot.core.operations.evaluation.operation_implementations.models.torch_rules import (
    build_torch_tabular_fit_plan,
    normalize_hidden_layer_sizes,
    resolve_torch_device,
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
def test_build_torch_fit_plan_uses_model_and_dataloader_contract():
    plan = build_torch_tabular_fit_plan(
        params={'hidden_layer_sizes': [64, 32],
                'batch_size': 128, 'device': 'cpu'},
        samples_count=50,
        input_device='cpu',
        cuda_available=False,
        dataloader_kwargs={'batch_size': 16},
    )

    assert plan.hidden_layer_sizes == (64, 32)
    assert plan.batch_size == 50
    assert plan.device == 'cpu'


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
    ],
)
def test_torch_fit_plan_rejects_non_positive_training_values(params, message):
    with pytest.raises(ValueError, match=message):
        build_torch_tabular_fit_plan(
            params=params,
            samples_count=10,
            input_device='cpu',
            cuda_available=False,
            dataloader_kwargs={'batch_size': 4},
        )
