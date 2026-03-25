import types

import numpy as np
import pandas as pd
import pytest

from fedot.api.api_utils.api_data_rules import (
    DataDefinitionResolutionError,
    iter_shared_index_assignments,
    normalize_features_for_definition,
    plan_fit_preprocessing,
    plan_prediction,
    plan_predict_preprocessing,
    resolve_strategy,
)
from fedot.core.repository.tasks import TaskTypesEnum


class _StrategyA:
    pass


class _StrategyB:
    pass


def test_normalize_features_for_definition_extracts_shared_index_without_mutation():
    original_features = {'idx': np.array([0, 1]), 'table': np.array([[1], [2]])}

    normalized = normalize_features_for_definition(original_features)

    assert 'idx' in original_features
    assert 'idx' not in normalized.features
    assert np.array_equal(normalized.shared_index, np.array([0, 1]))


def test_iter_shared_index_assignments_returns_pairs_for_multimodal_mapping():
    assignments = iter_shared_index_assignments({'first': object(), 'second': object()}, [1, 2])
    assert assignments == (('first', [1, 2]), ('second', [1, 2]))


def test_plan_preprocessing_steps_are_explicit_and_stable():
    assert plan_fit_preprocessing().steps == (
        'obligatory_prepare_for_fit',
        'optional_prepare_for_fit',
        'convert_indexes_for_fit',
        'reduce_memory_size',
    )
    assert plan_predict_preprocessing().steps == (
        'obligatory_prepare_for_predict',
        'optional_prepare_for_predict',
        'convert_indexes_for_predict',
        'update_indices_for_time_series',
        'reduce_memory_size',
    )


@pytest.mark.parametrize('task_type,in_sample,validation_blocks,forecast_length,expected', [
    (TaskTypesEnum.classification, False, None, None, ('labels', False, False, None)),
    (TaskTypesEnum.ts_forecasting, True, 2, 3, (None, True, False, 6)),
    (TaskTypesEnum.ts_forecasting, False, None, 3, (None, False, True, None)),
    (TaskTypesEnum.regression, False, None, None, (None, False, False, None)),
])
def test_plan_prediction_returns_typed_branching_decision(
        task_type, in_sample, validation_blocks, forecast_length, expected):
    plan = plan_prediction(task_type, in_sample, validation_blocks, forecast_length)
    assert (plan.output_mode, plan.use_in_sample_forecast, plan.flatten_prediction, plan.horizon) == expected


def test_resolve_strategy_finds_matching_factory():
    resolution = resolve_strategy(pd.DataFrame([[1]]), [(np.ndarray, _StrategyA), (pd.DataFrame, _StrategyB)])
    assert resolution.strategy_factory is _StrategyB


def test_resolve_strategy_raises_typed_error_for_unsupported_source():
    with pytest.raises(DataDefinitionResolutionError):
        resolve_strategy(123, [(np.ndarray, _StrategyA), (pd.DataFrame, _StrategyB)])
