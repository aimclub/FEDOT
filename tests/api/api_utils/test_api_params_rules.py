import pytest

from fedot.validation.errors import FedotValidationError
from fedot.api.api_utils.api_params_rules import (
    build_label_encoded_preset_name,
    merge_param_recommendations,
    normalize_timeout_and_generations,
    resolve_task,
    should_update_available_operations,
)
from fedot.core.constants import AUTO_PRESET_NAME, DEFAULT_FORECAST_LENGTH
from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams


def test_resolve_task_adds_default_ts_params_and_warning():
    resolution = resolve_task('ts_forecasting', None)

    assert resolution.task.task_type == TaskTypesEnum.ts_forecasting
    assert isinstance(resolution.task.task_params, TsForecastingParams)
    assert resolution.task.task_params.forecast_length == DEFAULT_FORECAST_LENGTH
    assert resolution.warning_message is not None


def test_resolve_task_rejects_unknown_problem():
    with pytest.raises(FedotValidationError, match='Wrong type name of the given task'):
        resolve_task('clustering', None)


def test_normalize_timeout_and_generations_handles_infinite_and_default_cases():
    infinite_resolution = normalize_timeout_and_generations(-1, 5)
    finite_resolution = normalize_timeout_and_generations(10, None)

    assert infinite_resolution.timeout is None
    assert infinite_resolution.num_of_generations == 5
    assert finite_resolution.timeout == 10
    assert finite_resolution.num_of_generations == 10000


def test_normalize_timeout_and_generations_rejects_invalid_values():
    with pytest.raises(FedotValidationError, match='num_of_generations'):
        normalize_timeout_and_generations(None, None)

    with pytest.raises(FedotValidationError, match='invalid "timeout" value'):
        normalize_timeout_and_generations(0, 5)


def test_small_preset_and_recommendation_helpers_are_deterministic():
    assert build_label_encoded_preset_name('fast_train') == 'fast_train*tree'
    assert build_label_encoded_preset_name(None) == '*tree'
    assert should_update_available_operations(AUTO_PRESET_NAME) is False
    assert should_update_available_operations('fast_train') is True
    assert merge_param_recommendations(
        {'a': 1}, {'b': 2, 'a': 3}) == {'a': 3, 'b': 2}
