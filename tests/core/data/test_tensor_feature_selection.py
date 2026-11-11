from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import EncodingMethodEnum, ImputationMethodEnum, PreprocessingStepEnum


def categorical_data():
    return TensorData(
        features=torch.tensor([[10., 1., 0., 0., 20., 0.], [11., 0., 1., 0., 21., 1.],
                               [12., 0., 0., 1., 22., 0.], [13., 1., 0., 0., 23., 1.]]),
        target=torch.tensor([0, 1, 0, 1]), predict=torch.zeros(4), idx=['a', 'b', 'c', 'd'],
        task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.tabular,
        numerical_idx=[0, 4], categorical_idx=[1, 2, 3, 5],
        idx_mapping={0: 10, 1: 20, 2: 20, 3: 20, 4: 30, 5: 40},
        categorical_features=np.array([['red', 'yes'], ['green', 'no'], ['blue', 'yes'], ['red', 'no']], dtype=object),
        categorical_source_idx=[20, 40],
        features_names=['age', 'color_red', 'color_green', 'color_blue', 'value', 'flag'], fingerprint='input',
    )


@pytest.mark.parametrize('columns', [[2, 4, 5], [0, 4], [2], [5, 4, 2]])
def test_subset_keeps_column_metadata_and_only_surviving_categorical_sources(columns):
    source = categorical_data()
    original = deepcopy(source)
    result = source.subset_features(torch.tensor(columns))
    torch.testing.assert_close(result.features, source.features[:, columns])
    assert result.idx_mapping == {pos: source.idx_mapping[column] for pos, column in enumerate(columns)}
    assert result.features_names == [source.features_names[column] for column in columns]
    assert result.categorical_idx == [pos for pos, column in enumerate(columns) if column in [1, 2, 3, 5]]
    assert result.numerical_idx == [pos for pos, column in enumerate(columns) if column in [0, 4]]
    expected_sources = [s for s in source.categorical_source_idx
                        if s in {result.idx_mapping[pos] for pos in result.categorical_idx}]
    assert result.categorical_source_idx == expected_sources
    restored = result.get_not_encoded_data()
    if expected_sources:
        raw_columns = [source.categorical_source_idx.index(s) for s in expected_sources]
        np.testing.assert_array_equal(restored.categorical_features, source.categorical_features[:, raw_columns])
    else:
        assert result.categorical_features is None
        assert restored.categorical_features.shape == (4, 0)
    torch.testing.assert_close(restored.numerical_features, result.features[:, result.numerical_idx])
    assert result.predict is None
    assert result.fingerprint is None
    assert result.idx == source.idx
    torch.testing.assert_close(result.target, source.target)
    assert source == original
    result.features.zero_()
    result.features_names[0] = 'changed'
    if result.categorical_features is not None:
        result.categorical_features[0, 0] = 'changed'
    assert source == original


def test_subset_without_explicit_source_mapping_keeps_direct_tensor_columns():
    source = TensorData(features=torch.randn(4, 3), task=Task(TaskTypesEnum.regression),
                        data_type=DataTypesEnum.tabular)
    result = source.subset_features([2, 0])
    assert result.idx_mapping == {0: 2, 1: 0}
    assert result.numerical_idx == [0, 1]


def test_optional_preprocessing_after_selection_uses_original_names_and_source_indices():
    frame = pd.DataFrame({'label': [0, 1, 0, 1], 'color': ['red', 'green', 'blue', 'red'],
                          'value': [10., 20., 30., 40.], 'unused': ['a', 'b', 'a', 'b']})
    source = create_data(frame, target='label', use_cache=False, encoding_strategy=[{
        'method': EncodingMethodEnum.ohe, 'features_idx': ['color', 'unused']}])
    color_column = next(pos for pos, original in source.idx_mapping.items() if original == 1)
    value_column = next(pos for pos, original in source.idx_mapping.items() if original == 2)
    selected = source.subset_features([color_column, value_column])
    selected.features[0, 1] = float('nan')
    service = OptionalTabularService(use_cache=False).fit(selected, {PreprocessingStepEnum.imputation: [{
        'method': ImputationMethodEnum.constant, 'features_idx': ['value'], 'step_args': {'constant': 7},
    }]})
    result = service.predict(selected)
    assert result.features[0, 1] == 7
    assert torch.isnan(selected.features[0, 1])
    assert result.idx_mapping == {0: 1, 1: 2}
    assert result.categorical_source_idx == [1]
    restored = result.get_not_encoded_data()
    assert restored.features_names == ['value', 'color']
    np.testing.assert_array_equal(restored.categorical_features[:, 0], frame['color'].to_numpy())


def test_mode_imputation_after_partial_ohe_selection_updates_both_representations():
    source = categorical_data()
    source.categorical_features[-1, 0] = np.nan
    source.features[-1, 1:4] = float('nan')
    selected = source.subset_features([2, 4])
    service = OptionalTabularService(use_cache=False).fit(selected, {PreprocessingStepEnum.imputation: [{
        'method': ImputationMethodEnum.mode, 'features_idx': [20], 'step_args': None,
    }]})
    result = service.predict(selected)
    # All three observed colors occur once; mode uses the sorted tie-break (blue).
    assert result.categorical_features[-1, 0] == 'blue'
    assert result.features[-1, 0] == source.features[2, 2]
    assert np.isnan(selected.categorical_features[-1, 0])
    assert torch.isnan(selected.features[-1, 0])


def test_unseen_constant_after_partial_ohe_selection_keeps_explicit_imputation_restriction():
    frame = pd.DataFrame({'value': [1., 2., 3., 4.], 'color': ['red', 'blue', 'green', np.nan]})
    source = create_data(frame, target=torch.tensor([0, 1, 0, 1]), use_cache=False, encoding_strategy=[{
        'method': EncodingMethodEnum.ohe, 'features_idx': ['color']}])
    column = next(pos for pos, original in source.idx_mapping.items() if original == 1)
    selected = source.subset_features([column])
    with pytest.raises(ValueError, match='complete unmodified encoded block'):
        OptionalTabularService(use_cache=False).fit(selected, {PreprocessingStepEnum.imputation: [{
            'method': ImputationMethodEnum.constant, 'features_idx': ['color'],
            'step_args': {'constant': '__MISSING__'},
        }]})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_categorical_subset_keeps_features_on_gpu_and_raw_strings_on_host():
    source = categorical_data().to('cuda')
    result = source.subset_features(torch.tensor([2, 4], device='cuda'))
    assert result.features.is_cuda
    assert result.get_not_encoded_data().numerical_features.is_cuda
    assert result.categorical_source_idx == [20]
    np.testing.assert_array_equal(result.categorical_features, source.categorical_features[:, :1])
