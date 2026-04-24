import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.data.input_data_descriptor import build_input_data_descriptor
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.mark.unit
def test_build_input_data_descriptor_preserves_legacy_surface_and_tensor_view():
    input_data = InputData(
        idx=np.array([10, 11]),
        features=np.array([[1, 2], [3, 4]]),
        target=np.array([0, 1]),
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.text,
        features_names=np.array(['text', 'meta']),
        categorical_idx=np.array([0]),
    )

    descriptor = build_input_data_descriptor(input_data)

    assert descriptor.original_data_type == DataTypesEnum.text
    assert descriptor.tensor_canonical_data_type == DataTypesEnum.tabular
    assert descriptor.input_compatible_data_type == DataTypesEnum.text
    assert np.array_equal(descriptor.idx, np.array([10, 11]))
    assert np.array_equal(descriptor.features_names, np.array(['text', 'meta']))
    assert np.array_equal(descriptor.categorical_idx, np.array([0]))
    assert descriptor.target_is_present is True


@pytest.mark.unit
def test_build_input_data_descriptor_copies_arrays_to_avoid_hidden_mutation():
    idx = np.array([1, 2])
    features_names = np.array(['a', 'b'])
    categorical_idx = np.array([1])
    input_data = InputData(
        idx=idx,
        features=np.array([[1, 2], [3, 4]]),
        target=None,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.image,
        features_names=features_names,
        categorical_idx=categorical_idx,
    )

    descriptor = build_input_data_descriptor(input_data)

    idx[0] = 999
    features_names[0] = 'changed'
    categorical_idx[0] = 0

    assert np.array_equal(descriptor.idx, np.array([1, 2]))
    assert np.array_equal(descriptor.features_names, np.array(['a', 'b']))
    assert np.array_equal(descriptor.categorical_idx, np.array([1]))
    assert descriptor.tensor_canonical_data_type == DataTypesEnum.ts
    assert descriptor.input_compatible_data_type == DataTypesEnum.image


@pytest.mark.unit
def test_build_input_data_descriptor_handles_missing_optional_fields():
    input_data = InputData(
        idx=np.array([0]),
        features=np.array([[1]]),
        target=None,
        task=Task(TaskTypesEnum.regression),
        data_type=DataTypesEnum.table,
        features_names=None,
        categorical_idx=None,
    )

    descriptor = build_input_data_descriptor(input_data)

    assert descriptor.features_names is None
    assert descriptor.categorical_idx is None
    assert descriptor.target_is_present is False
    assert descriptor.tensor_canonical_data_type == DataTypesEnum.tabular
