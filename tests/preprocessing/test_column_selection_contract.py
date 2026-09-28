"""Source-coordinate regressions for optional transforms after preparation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from fedot import create_data
from fedot.core.data.tensor_data.contracts import FeatureSchema, TensorDataContractError
from fedot.preprocessing.planner.column_selection import (
    compose_source_mapping, resolve_optional_columns,
)
from fedot.preprocessing.service.tabular_optional_service import OptionalTabularService
from fedot.preprocessing.tools.preprocessor_types import (
    EncodingMethodEnum, ImputationMethodEnum, PreprocessingStepEnum,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def expanded_columns():
    schema = FeatureSchema(
        shape=(3,), names=('id', 'category', 'value'),
        mapping=((0, 0), (1, 1), (2, 2)), categorical=(1,))
    return SimpleNamespace(
        preparation_state=SimpleNamespace(schema=schema),
        features=torch.zeros((4, 5)),
        features_names=['id', 'value', 'category__0',
                        'category__1', 'category__2'],
        idx_mapping={0: 0, 1: 2, 2: 1, 3: 1, 4: 1})


@pytest.mark.parametrize('selector, expected', [
    (['value'], [1]), (['category'], [2, 3, 4]),
    (['category__1'], [3]), (['category__1', 'category'], [3, 2, 4]),
    ([0], [0]), ([1], [2, 3, 4]), ([2], [1]),
])
def test_selectors_keep_source_and_generated_name_coordinates(
        expanded_columns, selector, expected):
    before = deepcopy(expanded_columns)
    assert resolve_optional_columns(expanded_columns, selector) == expected
    assert expanded_columns.idx_mapping == before.idx_mapping
    assert expanded_columns.features_names == before.features_names
    assert expanded_columns.preparation_state.schema == before.preparation_state.schema


def test_unknown_generated_name_fails_before_handler_fit(expanded_columns):
    with pytest.raises(TensorDataContractError) as error:
        resolve_optional_columns(expanded_columns, ['category__42'])
    assert error.value.code == 'unknown_column'
    assert error.value.field == 'features_idx'


def test_legacy_source_name_metadata_keeps_old_resolution(expanded_columns):
    expanded_columns.preparation_state = None
    expanded_columns.features_names = ['id', 'category', 'value']
    assert resolve_optional_columns(expanded_columns, ['value']) == [1]
    assert resolve_optional_columns(
        expanded_columns, ['category']) == [2, 3, 4]


@pytest.mark.parametrize('parent, transformed, expected', [
    ({0: 1, 1: 3}, {0: 0, 1: 1}, {0: 1, 1: 3}),
    ({0: 1, 1: 3}, {0: 1, 1: 1, 2: 0}, {0: 3, 1: 3, 2: 1}),
    ({0: 1, 1: 3}, {0: 1}, {0: 3}),
    ({}, {0: 0, 1: 1}, {0: 0, 1: 1}),
])
def test_mapping_composition_preserves_source_ownership(parent, transformed, expected):
    before = deepcopy((parent, transformed))
    assert compose_source_mapping(parent, transformed) == expected
    assert (parent, transformed) == before


@pytest.mark.parametrize('selector', [['value'], [2]])
def test_optional_imputation_after_target_removal_and_expansion_preserves_mapping(selector):
    frame = pd.DataFrame({
        'label': [0, 1, 0], 'category': ['A', 'B', 'C'], 'value': [1., np.nan, 3.],
    })
    data = create_data(
        frame, target='label', use_cache=False,
        encoding_strategy=[{'method': EncodingMethodEnum.ohe, 'features_idx': ['category']}])
    mapping = dict(data.idx_mapping)
    assert mapping == {0: 2, 1: 1, 2: 1, 3: 1}
    service = OptionalTabularService(use_cache=False)
    service.fit(data, {PreprocessingStepEnum.imputation: [{
        'method': ImputationMethodEnum.constant,
        'features_idx': selector, 'step_args': {'constant': 7},
    }]})
    assert torch.isnan(data.features[1, 0])
    result = service.predict(deepcopy(data))
    torch.testing.assert_close(result.features, torch.tensor([
        [1., 1., 0., 0.], [7., 0., 1., 0.], [3., 0., 0., 1.],
    ]))
    assert result.idx_mapping == mapping
    assert result.features_names == data.features_names
    assert torch.isnan(data.features[1, 0])
