import numpy as np
import pytest
from sklearn.datasets import make_classification

from fedot.api.sampling_stage.config import validate_sampling_config
from fedot.api.sampling_stage.executor import SamplingStageExecutor
from fedot.api.sampling_stage.providers import SamplingProvider, SamplingSubsetResult
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum


class FirstKProvider(SamplingProvider):
    def sample(self,
               features: np.ndarray,
               target: np.ndarray,
               strategy: str,
               strategy_params,
               random_state,
               budget_seconds,
               strategy_kind='subset',
               injectable_params=None):
        del target, strategy, strategy_params, random_state, budget_seconds
        k = max(1, int(round(features.shape[0] * injectable_params['ratio'])))
        return SamplingSubsetResult(
            sample_indices=np.arange(k, dtype=int),
            sample_scores=np.linspace(1.0, 0.0, num=k),
            meta={'provider': 'stub'}
        )


class DuplicateProvider(SamplingProvider):
    def sample(self,
               features: np.ndarray,
               target: np.ndarray,
               strategy: str,
               strategy_params,
               random_state,
               budget_seconds,
               strategy_kind='subset',
               injectable_params=None):
        del features, target, strategy, strategy_params, random_state, budget_seconds, strategy_kind, injectable_params
        return SamplingSubsetResult(sample_indices=np.array([0, 0], dtype=int),
                                    sample_scores=None,
                                    meta={})


def _classification_input(n_samples: int = 120, n_features: int = 8) -> InputData:
    x, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=5,
                               random_state=42)
    return InputData(idx=np.arange(n_samples),
                     features=x,
                     target=y,
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)


def test_sampling_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match='Unknown keys'):
        validate_sampling_config({'strategy_kind': 'subset', 'unknown_key': 1})


def test_sampling_config_rejects_non_fail_fast_mode():
    with pytest.raises(ValueError, match='fail_fast'):
        validate_sampling_config({'strategy_kind': 'subset', 'error_policy': 'fallback'})


def test_dynamic_cap_budget_and_timeout_update():
    config = {
        'strategy_kind': 'subset',
        'strategy': 'random',
        'candidate_ratios': [0.5],
        'delta_metric_threshold': 1.0,
        'cap_max_timeout_share': 0.4,
        'min_automl_time_minutes': 2.0,
    }
    validated = validate_sampling_config(config)

    budget = SamplingStageExecutor._compute_budget_seconds(validated, total_timeout_minutes=10.0)
    # min(10m * 0.4, 10m - 2m) = min(240s, 480s)
    assert budget == pytest.approx(240.0)

    updated_timeout = SamplingStageExecutor._compute_updated_timeout(elapsed_seconds=120.0,
                                                                     total_timeout_minutes=10.0,
                                                                     min_automl_time_minutes=2.0)
    assert updated_timeout == pytest.approx(8.0)


def test_sampling_provider_contract_checks_indices_uniqueness():
    data = _classification_input()
    config = {
        'strategy_kind': 'subset',
        'strategy': 'random',
        'candidate_ratios': [0.5],
        'delta_metric_threshold': 1.0,
    }
    executor = SamplingStageExecutor(sampling_config=config,
                                     task_type=TaskTypesEnum.classification,
                                     total_timeout_minutes=5.0,
                                     provider=DuplicateProvider())

    with pytest.raises(ValueError, match='must be unique'):
        executor.execute(data)


def test_effective_size_selection_on_deterministic_scores(monkeypatch):
    data = _classification_input()
    config = {
        'strategy_kind': 'subset',
        'strategy': 'random',
        'candidate_ratios': [0.2, 0.5, 0.9],
        'delta_metric_threshold': 0.05,
    }
    executor = SamplingStageExecutor(sampling_config=config,
                                     task_type=TaskTypesEnum.classification,
                                     total_timeout_minutes=5.0,
                                     provider=FirstKProvider())

    def fake_score(train_data, valid_data, task_type, random_state):
        del valid_data, task_type, random_state
        size = len(train_data.idx)
        if size >= 70:
            return 1.0
        if size >= 40:
            return 0.97
        return 0.8

    monkeypatch.setattr(SamplingStageExecutor, '_score_light_model', staticmethod(fake_score))

    result = executor.execute(data)
    assert result.metadata['selected_ratio'] == pytest.approx(0.5)
    assert result.metadata['rows_after'] < result.metadata['rows_before']


def test_fail_fast_when_optional_dependency_is_missing(monkeypatch):
    config = {
        'strategy_kind': 'subset',
        'provider': 'sampling_zoo',
        'strategy': 'random',
        'candidate_ratios': [0.5],
        'delta_metric_threshold': 1.0,
    }

    def missing_provider(*args, **kwargs):
        raise ModuleNotFoundError('sampling zoo not installed')

    monkeypatch.setattr(SamplingStageExecutor, '_create_provider', missing_provider)

    with pytest.raises(ModuleNotFoundError):
        SamplingStageExecutor(sampling_config=config,
                              task_type=TaskTypesEnum.classification,
                              total_timeout_minutes=5.0)


def test_sampling_config_respects_heavy_parameter_guards():
    with pytest.raises(ValueError, match='guard_max_sample_size'):
        validate_sampling_config({
            'strategy_kind': 'subset',
            'strategy_params': {'sample_size': 1000},
            'guard_max_sample_size': 100,
        })


def test_sampling_config_rejects_unsorted_candidate_ratios():
    with pytest.raises(ValueError, match='sorted in ascending order'):
        validate_sampling_config({'strategy_kind': 'subset', 'candidate_ratios': [0.5, 0.2]})


def test_dynamic_cap_for_infinite_timeout_uses_absolute_stage_cap():
    config = validate_sampling_config({
        'strategy_kind': 'subset',
        'strategy': 'random',
        'candidate_ratios': [0.5],
        'delta_metric_threshold': 1.0,
        'infinite_timeout_cap_minutes': 7.0,
    })

    assert SamplingStageExecutor._compute_budget_seconds(config, total_timeout_minutes=None) == pytest.approx(420.0)


def test_sampling_config_rejects_non_dict_value():
    with pytest.raises(ValueError, match='dictionary or None'):
        validate_sampling_config('not_a_dict')


def test_sampling_config_rejects_invalid_validation_size_range():
    with pytest.raises(ValueError, match='validation_size'):
        validate_sampling_config({'strategy_kind': 'subset', 'validation_size': 1.0})
