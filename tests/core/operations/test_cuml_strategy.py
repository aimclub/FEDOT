import pytest
import torch

from fedot.core.operations.strategies.models import cuml as common
from fedot.core.operations.strategies.models.cuml import (
    CuMLModelStrategy,
    _validate_cuml_environment,
    adapt_cuml_parameters,
    validate_cuml_tensor_operation,
)


@pytest.mark.unit
def test_cuml_forest_parameters_are_adapted_without_mutating_source():
    source = {'criterion': 'entropy', 'n_jobs': 4, 'n_estimators': 200}

    adapted = adapt_cuml_parameters('rf', source)

    assert source == {'criterion': 'entropy', 'n_jobs': 4, 'n_estimators': 200}
    assert adapted == {
        'split_criterion': 'entropy',
        'n_estimators': 200,
        'n_bins': 256,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_svc_always_fits_probability_model():
    assert adapt_cuml_parameters('svc', {}) == {
        'probability': True,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_logit_gets_conservative_convergence_defaults():
    assert adapt_cuml_parameters('logit', {}) == {
        'max_iter': 10000,
        'tol': 1e-7,
        'linesearch_max_iter': 200,
        'output_type': 'cupy',
    }


@pytest.mark.unit
def test_cuml_logit_uses_stable_precision_without_penalizing_other_models(monkeypatch):
    monkeypatch.setattr(common, '_validate_cuml_environment', lambda: None)
    monkeypatch.setattr(CuMLModelStrategy, '_operations_by_types', {'logit': object, 'rf': object})

    assert CuMLModelStrategy('logit').dtype == torch.float64
    assert CuMLModelStrategy('rf').dtype == torch.float32


@pytest.mark.unit
def test_cuml_kmeans_keeps_historical_two_cluster_default():
    assert adapt_cuml_parameters('kmeans', {}) == {
        'n_clusters': 2,
        'output_type': 'cupy',
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ('installed', 'system', 'release', 'cuda_available', 'message'),
    [
        (False, 'Linux', '6.0', True, 'cuML is not installed'),
        (True, 'Linux', '6.0', False, 'requires an available CUDA device'),
        (True, 'Linux', '5.15-microsoft-standard-WSL2', True, 'only on native Linux'),
        (True, 'Linux', '5.15-wsl', True, 'only on native Linux'),
        (True, 'Windows', '10', True, 'only on native Linux'),
        (True, 'Darwin', '22', True, 'only on native Linux'),
    ],
)
def test_cuml_environment_rejects_unsupported_runtime(monkeypatch, installed, system, release,
                                                     cuda_available, message):
    monkeypatch.setattr(common, 'cuml', object() if installed else None)
    monkeypatch.setattr(common.platform, 'system', lambda: system)
    monkeypatch.setattr(common.platform, 'release', lambda: release)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)

    with pytest.raises(RuntimeError, match=message):
        _validate_cuml_environment()


@pytest.mark.unit
def test_cuml_environment_accepts_native_linux_with_cuda(monkeypatch):
    monkeypatch.setattr(common, 'cuml', object())
    monkeypatch.setattr(common.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(common.platform, 'release', lambda: '6.0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    _validate_cuml_environment()


@pytest.mark.unit
def test_low_level_sgd_solver_is_not_exposed_as_tensordata_model():
    with pytest.raises(ValueError, match='low-level solver'):
        validate_cuml_tensor_operation('sgd')

    validate_cuml_tensor_operation('logit')
