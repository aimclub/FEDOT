"""Original categories, optional imputation and encoded tensors agree by source and row."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.core.caching.hasher import Hasher
from fedot.core.caching.evaluation_context import preparation_identity
from fedot.core.caching.normalization import build_tensor_data_payload, restore_tensor_data_payload
from fedot.core.data.merge.data_merger import TensorDataMerger
from fedot.core.data.split.data_split import _split_tensor_data_by_indexes
from fedot.core.data.tensor_data.contracts import TensorDataContractError
from fedot.core.data.tensor_data.preparation import PreparationRuntime
from fedot.core.operations.implementations.data_operations.tools import replace_projected_features
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import (
    EncodingMethodEnum, FilteringMethodEnum, ImputationMethodEnum, PreprocessingStepEnum, ScalingMethodEnum,
)


pytestmark = pytest.mark.unit


def categorical_data(method=EncodingMethodEnum.ohe, colors=None, **kwargs):
    colors = ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', np.nan] if colors is None else colors
    frame = pd.DataFrame({'value': np.arange(len(colors), dtype=float), 'color': colors})
    return create_data(frame, target=np.arange(len(colors)) % 2, use_cache=False,
                       encoding_strategy=[{'method': method, 'features_idx': ['color']}], **kwargs)


@pytest.mark.parametrize('method', [EncodingMethodEnum.ohe, EncodingMethodEnum.label])
def test_creation_preserves_values_and_predict_uses_its_own_categories(method):
    train = categorical_data(method)
    test = create_data(pd.DataFrame({'value': [10., 11., 12.], 'color': ['blue', 'unseen', np.nan]}),
                       from_data=train, use_cache=False)
    assert train.categorical_source_idx == [1]
    assert train.categorical_features[:, 0].tolist()[:3] == ['red'] * 3
    assert test.categorical_features[:2, 0].tolist() == ['blue', 'unseen']
    assert pd.isna(test.categorical_features[2, 0])
    view = test.get_not_encoded_data()
    assert view.numerical_features.device == test.device
    assert view.features_names == ['value', 'color']
    assert view.numerical_features[1].tolist() == [11.]
    assert view.categorical_features[1].tolist() == ['unseen']


@pytest.mark.parametrize('missing', [None, pd.NA, np.nan])
def test_source_boundary_normalizes_missing_categories_before_tensor_preprocessing(missing):
    data = categorical_data(colors=['red', 'red', 'blue', 'green', missing])
    assert np.isnan(data.categorical_features[-1, 0])
    service = OptionalTabularService(use_cache=False).fit(data, {PreprocessingStepEnum.imputation: None})
    result = service.predict(deepcopy(data))
    assert result.categorical_features[-1, 0] == 'red'


@pytest.mark.parametrize('method', [EncodingMethodEnum.ohe, EncodingMethodEnum.label])
def test_categorical_mode_preserves_sorted_tie_breaking_without_pandas(method):
    data = categorical_data(method, colors=['red', 'blue', 'green', 'red', 'blue', 'green', np.nan])
    service = OptionalTabularService(use_cache=False).fit(data, {PreprocessingStepEnum.imputation: None})
    result = service.predict(deepcopy(data))
    assert result.categorical_features[-1, 0] == 'blue'
    torch.testing.assert_close(result.features[-1, result.categorical_idx], data.features[1, data.categorical_idx])


def test_target_extraction_and_missing_targets_preserve_category_source_ids_and_rows():
    frame = pd.DataFrame({'value': [1., 2., 3.], 'target': [0., np.nan, 1.], 'color': ['red', 'blue', 'green']})
    data = create_data(frame, target='target', idx=[9, 2, 7], use_cache=False)
    assert data.categorical_source_idx == [2]
    assert data.categorical_features[:, 0].tolist() == ['red', 'green']
    assert list(data.idx) == [9, 7]
    assert data.get_not_encoded_data().features_names == ['value', 'color']


@pytest.mark.parametrize('method', [EncodingMethodEnum.ohe, EncodingMethodEnum.label])
@pytest.mark.parametrize('use_cache', [False, True])
def test_mode_imputation_updates_both_representations_and_survives_serialization(method, use_cache):
    train = categorical_data(method)
    original = deepcopy(train)
    service = OptionalTabularService(use_cache=use_cache).fit(train, {PreprocessingStepEnum.imputation: None})
    service = pickle.loads(pickle.dumps(service))
    result = service.predict(deepcopy(train))
    assert result.categorical_features[-1, 0] == 'red'
    torch.testing.assert_close(result.features[-1, result.categorical_idx],
                               train.features[0, train.categorical_idx])
    assert train == original  # Fitting must not fill the caller's data.
    test = create_data(pd.DataFrame({'value': [10., 11.], 'color': [np.nan, 'unseen']}),
                       from_data=train, use_cache=False)
    predicted = service.predict(test)
    assert predicted.categorical_features[:, 0].tolist() == ['red', 'unseen']
    assert predicted.validate() is predicted


def test_without_imputation_scaling_preserves_missing_categories():
    data = categorical_data()
    service = OptionalTabularService(use_cache=False).fit(data, {PreprocessingStepEnum.scaling: None})
    result = service.predict(deepcopy(data))
    assert pd.isna(result.categorical_features[-1, 0])
    assert torch.isnan(result.features[-1, result.categorical_idx]).all()
    assert pd.isna(result.get_not_encoded_data().categorical_features[-1, 0])
    assert result.features[-1, 0] == 1.


@pytest.mark.parametrize('method', [EncodingMethodEnum.ohe, EncodingMethodEnum.label])
@pytest.mark.parametrize('constant', ['blue', '__MISSING__'])
def test_explicit_constant_uses_fitted_encoding_without_changing_width(method, constant):
    data = categorical_data(method)
    service = OptionalTabularService(use_cache=False).fit(data, {
        PreprocessingStepEnum.imputation: [{'method': ImputationMethodEnum.constant,
                                           'features_idx': ['color'], 'step_args': {'constant': constant}}]})
    result = service.predict(deepcopy(data))
    expected = create_data(pd.DataFrame({'value': [7.], 'color': [constant]}), from_data=data, use_cache=False)
    assert result.categorical_features[-1, 0] == constant
    assert result.features.shape == data.features.shape
    torch.testing.assert_close(result.features[-1], expected.features[0])


def test_imputation_rejects_partial_ohe_source_and_numeric_statistics_on_strings():
    data = categorical_data()
    for method, selectors, match in [(ImputationMethodEnum.mode, ['color__0'], 'whole encoded'),
                                      (ImputationMethodEnum.mean, ['color'], 'Original categorical')]:
        with pytest.raises(ValueError, match=match):
            OptionalTabularService(use_cache=False).fit(data, {
                PreprocessingStepEnum.imputation: [{'method': method, 'features_idx': selectors, 'step_args': None}]})


def test_unseen_constant_after_explicit_category_scaling_is_not_encoded_as_unscaled_bits():
    data = categorical_data()
    with pytest.raises(ValueError, match='unmodified encoded source'):
        OptionalTabularService(use_cache=False).fit(data, {
            PreprocessingStepEnum.scaling: [{'method': ScalingMethodEnum.standard,
                                            'features_idx': ['color'], 'step_args': None}],
            PreprocessingStepEnum.imputation: [{'method': ImputationMethodEnum.constant,
                                               'features_idx': ['color'],
                                               'step_args': {'constant': '__MISSING__'}}],
        })


def test_row_deletion_uses_predict_rows_and_slices_categories_target_and_idx():
    train = categorical_data(idx=[9, 2, 7, 8, 1, 4, 5, 6])
    service = OptionalTabularService(use_cache=False).fit(train, {
        PreprocessingStepEnum.imputation: ImputationMethodEnum.delete_raw})
    test = create_data(pd.DataFrame({'value': [10., 11., 12.], 'color': ['blue', np.nan, 'green']}),
                       target=np.array([1, 0, 1]), idx=[20, 30, 40], from_data=train, use_cache=False)
    result = service.predict(test)
    assert list(result.idx) == [20, 40]
    assert result.categorical_features[:, 0].tolist() == ['blue', 'green']
    assert result.target.flatten().tolist() == [1., 1.]
    assert result.features.shape[0] == 2
    result.validate()


def test_split_and_cache_preserve_original_categories():
    data = categorical_data(idx=[9, 2, 7, 8, 1, 4, 5, 6])
    subset = _split_tensor_data_by_indexes(data, torch.tensor([4, 1, 6]))
    assert subset.categorical_features[:, 0].tolist() == ['blue', 'red', 'green']
    assert list(subset.idx) == [1, 2, 5]
    restored = restore_tensor_data_payload(build_tensor_data_payload(subset))
    assert restored == subset
    restored.categorical_features[0, 0] = 'changed'
    assert data.categorical_features[4, 0] == 'blue'


def test_same_encoded_values_with_different_raw_categories_have_different_fingerprints():
    data = categorical_data()
    other = deepcopy(data)
    other.categorical_features[0, 0] = 'different'
    assert torch.equal(data.features[:-1], other.features[:-1])
    assert Hasher.hash(data) != Hasher.hash(other)


def test_dropped_categories_and_projections_are_not_restored():
    data = categorical_data(colors=['red'] * 8)
    service = OptionalTabularService(use_cache=False).fit(data, {
        PreprocessingStepEnum.filtering: [{'method': FilteringMethodEnum.variance,
                                          'features_idx': ['color'], 'step_args': None}]})
    result = service.predict(deepcopy(data))
    assert result.features.shape[1] == 1
    assert result.categorical_idx == []
    assert result.get_not_encoded_data().categorical_features.shape == (8, 0)
    projected = replace_projected_features(data, torch.ones((8, 2)))
    assert projected.get_not_encoded_data().categorical_features.shape == (8, 0)


def test_merger_aligns_categories_with_reordered_rows_and_keeps_branch_specific_values():
    data = categorical_data(idx=[9, 2, 7, 8, 1, 4, 5, 6])
    left = _split_tensor_data_by_indexes(data, torch.tensor([0, 1, 7]))
    right = _split_tensor_data_by_indexes(data, torch.tensor([7, 0, 1]))
    right.categorical_features[-3, 0] = 'filled'
    merged = TensorDataMerger([left, right], [False, False]).merge()
    assert list(merged.idx) == [2, 6, 9]
    assert merged.categorical_source_idx == [1, 3]
    assert merged.categorical_features[1, 1] == 'filled'
    assert pd.isna(merged.categorical_features[1, 0])
    assert merged.get_not_encoded_data().categorical_features.shape == (3, 2)
    torch.testing.assert_close(merged.features[:, 0], torch.tensor([1., 7., 0.]))
    predicted = replace(left, predict=torch.arange(3, dtype=left.features.dtype))
    model_input = TensorDataMerger([predicted], [True]).merge()
    assert model_input.categorical_features is None
    assert model_input.categorical_idx == []
    assert model_input.numerical_idx == [0]


@pytest.mark.parametrize('values, sources', [(np.ones((7, 1)), [1]), (np.ones((8, 2)), [1, 1])])
def test_category_sidecar_contract_rejects_misalignment(values, sources):
    with pytest.raises(TensorDataContractError):
        replace(categorical_data(), categorical_features=values, categorical_source_idx=sources)


def test_all_missing_category_remains_addressable_without_inventing_a_mode():
    frame = pd.DataFrame({'value': np.arange(8, dtype=float), 'color': [np.nan] * 8})
    data = create_data(frame, target=np.arange(8) % 2, categorical_idx=['color'], use_cache=False,
                       encoding_strategy=[{'method': EncodingMethodEnum.ohe, 'features_idx': ['color']}])
    assert data.categorical_idx == [1]
    assert data.get_not_encoded_data().categorical_features.shape == (8, 1)
    service = OptionalTabularService(use_cache=False).fit(data, {PreprocessingStepEnum.imputation: None})
    result = service.predict(deepcopy(data))
    assert pd.isna(result.categorical_features).all()
    assert torch.isnan(result.features[:, 1]).all()


def test_two_encoders_keep_each_source_and_imputation_pattern_separate():
    frame = pd.DataFrame({'first': ['a', 'a', 'b', np.nan], 'value': [0., 1., 2., 3.],
                          'second': ['x', 'y', 'y', np.nan]})
    data = create_data(frame, target=np.array([0, 1, 0, 1]), use_cache=False, encoding_strategy=[
        {'method': EncodingMethodEnum.ohe, 'features_idx': ['first']},
        {'method': EncodingMethodEnum.label, 'features_idx': ['second']},
    ])
    assert data.categorical_source_idx == [0, 2]
    service = OptionalTabularService(use_cache=False).fit(data, {PreprocessingStepEnum.imputation: None})
    result = service.predict(deepcopy(data))
    assert result.categorical_features[-1].tolist() == ['a', 'y']
    for source, row in [(0, 0), (2, 1)]:
        positions = [pos for pos, origin in result.idx_mapping.items() if origin == source]
        torch.testing.assert_close(result.features[-1, positions], data.features[row, positions])


def test_merged_encoding_metadata_supports_unseen_constants_and_retains_target_decoder():
    frame = pd.DataFrame({'value': np.arange(8, dtype=float),
                          'color': ['red', 'red', 'red', 'blue', 'blue', 'green', 'green', np.nan]})
    data = create_data(frame, target=np.array(['yes', 'no'] * 4), use_cache=False)
    merged = TensorDataMerger([deepcopy(data), deepcopy(data)], [False, False]).merge()
    state = merged.preparation_state
    assert preparation_identity(merged) != preparation_identity(data)
    assert PreparationRuntime.inverse_target(torch.tensor([0, 1]), state).tolist() == ['no', 'yes']
    service = OptionalTabularService(use_cache=False).fit(merged, {
        PreprocessingStepEnum.imputation: ImputationMethodEnum.constant})
    result = service.predict(deepcopy(merged))
    assert result.categorical_features[-1].tolist() == [0., 0.]
    assert not torch.isnan(result.features[-1]).any()
    assert PreparationRuntime.inverse_target(torch.tensor([0, 1]), result.preparation_state).tolist() == ['no', 'yes']
