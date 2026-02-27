import numpy as np
import pytest

from fedot_ind.core.metrics.loss.basis_loss import basis_approximation_metric


@pytest.fixture
def sample_data():
    derivation_coef = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    metric_values = np.array([10, 20, 30])
    return derivation_coef, metric_values


def test_basis_approximation_metric_with_default_regularization(sample_data):
    derivation_coef, metric_values = sample_data
    result = basis_approximation_metric(derivation_coef, metric_values)
    assert isinstance(result, np.int64)
    assert result >= 0
    assert result < len(derivation_coef)


def test_basis_approximation_metric_with_custom_regularization(sample_data):
    derivation_coef, metric_values = sample_data
    result = basis_approximation_metric(
        derivation_coef, metric_values, regularization_coef=0.5)
    assert isinstance(result, np.int64)
    assert result >= 0
    assert result < len(derivation_coef)


def test_basis_approximation_metric_with_invalid_input(sample_data):
    derivation_coef, metric_values = sample_data
    with pytest.raises(ValueError):
        basis_approximation_metric(np.array([]), np.array([]))


def test_basis_approximation_metric_with_large_input():
    # Test with large input data (performance test)
    derivation_coef_large = np.random.rand(1000, 1000)
    metric_values_large = np.random.rand(1000)
    result = basis_approximation_metric(
        derivation_coef_large, metric_values_large)
    assert isinstance(result, np.int64)
    assert result >= 0
    assert result < len(derivation_coef_large)
