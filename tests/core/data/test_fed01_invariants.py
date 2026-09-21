"""Runtime validation and supported bridge conservation laws for FED-01."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.core.backend.backend import Backend
from fedot.core.data.bridges.input_to_tensor import input_data_to_tensordata
from fedot.core.data.bridges.tensor_to_input import tensordata_to_input_data
from fedot.core.data.common.enums import StateEnum
from fedot.core.data.tensor_data.contracts import TensorDataContractError
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum


pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('isolated_cache_dir')]


@pytest.mark.parametrize('outputs', [1, 3])
@pytest.mark.parametrize('one_hot', [False, True])
@pytest.mark.parametrize('label_kind', ['strings', 'tensor'])
def test_supported_bridge_roundtrip_conserves_multioutput_data_and_owns_arrays(outputs, one_hot, label_kind):
    """Repeated bridges preserve prepared values and never share mutable array storage."""
    frame = pd.DataFrame(
        {'category': ['b', 'a', 'b', 'c'], 'value': [7., 2., 9., 4.]})
    options = {}
    if one_hot:
        options['encoding_strategy'] = [
            {'method': EncodingMethodEnum.ohe, 'features_idx': ['category']}]
        expected = np.array([[7, 0, 1, 0], [2, 1, 0, 0],
                            [9, 0, 1, 0], [4, 0, 0, 1]])
    else:
        expected = np.array([[1, 7], [0, 2], [1, 9], [2, 4]])
    target = np.arange(4 * outputs, dtype=np.float32).reshape(4, outputs)
    labels = np.array(
        ['z', 'a', 'z', 'm']) if label_kind == 'strings' else torch.tensor([90, -1, 90, 7])
    original = create_data(frame, target=target, idx=labels,
                           task='regression', use_cache=False, **options)
    expected_names = list(original.features_names)
    current = original
    for _ in range(2):
        legacy = tensordata_to_input_data(current)
        np.testing.assert_array_equal(legacy.features, expected)
        np.testing.assert_array_equal(legacy.target, target)
        np.testing.assert_array_equal(legacy.idx, labels)
        assert list(legacy.features_names) == expected_names
        assert legacy.task.task_type == original.task.task_type
        restored = input_data_to_tensordata(legacy, 'cpu')
        np.testing.assert_array_equal(restored.features, expected)
        np.testing.assert_array_equal(restored.target, target)
        np.testing.assert_array_equal(restored.idx, labels)
        assert restored.features_names == expected_names
        assert restored.task.task_type == original.task.task_type
        assert restored.state == StateEnum.FIT

        legacy.features[0, 0] = -10
        legacy.target[0, 0] = -20
        legacy.idx[0] = 'changed' if label_kind == 'strings' else -30
        legacy.features_names[0] = 'changed'
        np.testing.assert_array_equal(current.features, expected)
        np.testing.assert_array_equal(restored.features, expected)
        np.testing.assert_array_equal(current.target, target)
        np.testing.assert_array_equal(restored.target, target)
        np.testing.assert_array_equal(current.idx, labels)
        np.testing.assert_array_equal(restored.idx, labels)
        assert list(current.features_names) == expected_names
        assert restored.features_names == expected_names
        current = restored

    current.features[0, 0] = -100
    current.target[0, 0] = -200
    current.idx[0] = 'other' if label_kind == 'strings' else -300
    current.features_names[0] = 'other'
    np.testing.assert_array_equal(original.features, expected)
    np.testing.assert_array_equal(original.target, target)
    np.testing.assert_array_equal(original.idx, labels)
    assert original.features_names == expected_names


def test_bridge_task_metadata_is_not_borrowed_from_tensor_container():
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=2))
    source = np.arange(18, dtype=np.float32).reshape(3, 6)
    data = create_data(source, task=task,
                       ts_forecast_horizon=2, use_cache=False)
    legacy = tensordata_to_input_data(data)
    assert legacy.task == task
    legacy.task.task_params.forecast_length = 3
    assert data.task.task_params.forecast_length == 2
    assert task.task_params.forecast_length == 2


@pytest.mark.parametrize('shape', [(4, 3), (4, 2, 5)])
def test_validate_is_observationally_pure_and_deepcopy_is_independent(shape):
    """Validation preserves identities; copying isolates tensors and nested metadata."""
    features = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    target = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    predict = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    data = TensorData('regression', 'tabular' if len(shape) == 2 else 'time_series', features,
                      target=target, predict=predict, idx=torch.tensor(
                          [8, -1, 8, 3]),
                      features_names=[f'column_{i}' for i in range(shape[1])],
                      idx_mapping=dict(enumerate(range(shape[1]))),
                      dataloader_kwargs={'nested': {'batch_size': 4}})
    identities = {name: id(getattr(data, name)) for name in
                  ('features', 'target', 'predict', 'idx', 'features_names', 'idx_mapping', 'dataloader_kwargs')}
    backend = Backend()
    backend_before = (backend.name, backend.device, backend.xp, backend.pd)
    snapshot = deepcopy(data)
    for _ in range(3):
        assert data.validate() is data
        assert {name: id(getattr(data, name))
                for name in identities} == identities
        assert (backend.name, backend.device,
                backend.xp, backend.pd) == backend_before
        for name in ('features', 'target', 'predict', 'idx'):
            torch.testing.assert_close(
                getattr(data, name), getattr(snapshot, name))
        assert data.features_names == snapshot.features_names
        assert data.idx_mapping == snapshot.idx_mapping
        assert data.dataloader_kwargs == snapshot.dataloader_kwargs

    snapshot.features.flatten()[0] = -10
    snapshot.target[0, 0] = -20
    snapshot.predict[0, 0] = -30
    snapshot.idx[0] = -40
    snapshot.features_names[0] = 'changed'
    snapshot.idx_mapping[0] = 99
    snapshot.dataloader_kwargs['nested']['batch_size'] = 999
    assert data.features.flatten()[0] == 0
    assert data.target[0, 0] == 0
    assert data.predict[0, 0] == 0
    assert data.idx[0] == 8
    assert data.features_names[0] == 'column_0'
    assert data.idx_mapping[0] == 0
    assert data.dataloader_kwargs == {'nested': {'batch_size': 4}}


@pytest.mark.parametrize('field', ['target', 'predict', 'idx'])
@pytest.mark.parametrize('violation', ['rows', 'device'])
def test_failed_validate_does_not_repair_or_replace_runtime_fields(field, violation):
    """Post-construction corruption is diagnosed without silently moving or trimming data."""
    data = TensorData('regression', 'tabular', torch.ones((4, 2)),
                      target=torch.ones((4, 1)), predict=torch.ones((4, 1)), idx=torch.arange(4))
    shape = (3,) if field == 'idx' else (3, 1)
    if violation == 'rows':
        invalid = torch.ones(shape)
    else:
        shape = (4,) if field == 'idx' else (4, 1)
        invalid = torch.empty(shape, device='meta')
    setattr(data, field, invalid)
    references = {name: getattr(data, name) for name in (
        'features', 'target', 'predict', 'idx')}
    backend_before = (Backend().name, Backend().device)
    with pytest.raises(TensorDataContractError) as error:
        data.validate()
    expected_code = 'row_alignment' if violation == 'rows' else 'device_mismatch'
    assert (error.value.code, error.value.field) == (expected_code, field)
    assert all(getattr(data, name) is reference for name,
               reference in references.items())
    assert (Backend().name, Backend().device) == backend_before
