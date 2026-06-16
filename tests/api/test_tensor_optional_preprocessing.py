import numpy as np
import pytest

from fedot import Fedot
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.data.tensor_data.tensor_data_creator import TensorDataCreator
from fedot.preprocessing.service.tensor_optional_runtime import get_optional_runtime_spec_for_tensor_data


def _tabular_data_with_nan() -> TensorData:
    """Same fixture shape as tests/preprocessing/test_optional_preprocessing.py."""
    features = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9],
    ], dtype=np.float32)
    return TensorDataCreator.create(features, backend_name='cpu')


def _nan_value(tensor_data: TensorData) -> float:
    return float(tensor_data.features[1, 1])


def _expected_auto_preprocessed(tensor_data: TensorData) -> TensorData:
    runtime_spec = get_optional_runtime_spec_for_tensor_data(tensor_data)
    service = runtime_spec.service_cls(use_cache=False)
    return service.fit_transform(tensor_data, runtime_spec.default_steps)


@pytest.mark.unit
def test_fedot_fit_transform_tensor_optional_applies_defaults_when_auto_enabled():
    model = Fedot(problem='classification', use_auto_preprocessing=True, use_cache=False)
    tensor_data = _tabular_data_with_nan()
    reference = _tabular_data_with_nan()
    expected = _expected_auto_preprocessed(reference)

    preprocessed = model.fit_transform_tensor_optional(tensor_data)

    assert isinstance(preprocessed, TensorData)
    assert np.allclose(preprocessed.features, expected.features)
    assert not np.isnan(_nan_value(preprocessed))


@pytest.mark.unit
def test_fedot_fit_transform_tensor_optional_skips_when_auto_disabled():
    model = Fedot(problem='classification', use_auto_preprocessing=False, use_cache=False)
    tensor_data = _tabular_data_with_nan()

    preprocessed = model.fit_transform_tensor_optional(tensor_data)

    assert preprocessed is tensor_data
    assert np.isnan(_nan_value(preprocessed))


@pytest.mark.unit
def test_fedot_fit_transform_tensor_optional_uses_user_strategy_without_auto():
    model = Fedot(
        problem='classification',
        use_auto_preprocessing=False,
        use_cache=False,
        tensor_data_config={
            'optional_strategy': {
                'imputation': None,
            },
        },
    )
    tensor_data = _tabular_data_with_nan()

    preprocessed = model.fit_transform_tensor_optional(tensor_data)

    assert not np.isnan(_nan_value(preprocessed))
    assert _nan_value(preprocessed) == 5


@pytest.mark.unit
def test_fedot_prepare_fit_context_td_applies_optional_preprocessing_when_auto_enabled():
    model = Fedot(problem='classification', use_auto_preprocessing=True, use_cache=False)
    features = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9],
    ], dtype=np.float32)
    target = np.array([0, 1, 0])

    _, train_data = model._prepare_fit_context_td(features=features, target=target)

    assert isinstance(train_data, TensorData)
    assert not np.isnan(float(train_data.features[1, 1]))


@pytest.mark.unit
def test_fedot_prepare_fit_context_td_skips_optional_preprocessing_when_auto_disabled():
    model = Fedot(problem='classification', use_auto_preprocessing=False, use_cache=False)
    features = np.array([
        [1, 2, 3],
        [4, np.nan, 6],
        [7, 8, 9],
    ], dtype=np.float32)
    target = np.array([0, 1, 0])

    _, train_data = model._prepare_fit_context_td(features=features, target=target)

    assert np.isnan(float(train_data.features[1, 1]))
