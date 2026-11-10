from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fedot.api.main import Fedot
from fedot.api.api_utils.api_params_repository import ApiParamsRepository
from fedot.core.repository.tasks import TaskTypesEnum


@pytest.mark.parametrize('elapsed,enabled,expected', [
    (600, True, 2478), (3000, True, 78), (600, False, 324),
])
def test_final_refit_budget_is_separate_from_cv(monkeypatch, elapsed, enabled, expected):
    model = Fedot.__new__(Fedot)
    requirements = SimpleNamespace(max_graph_fit_time=timedelta(seconds=324))
    model.params = SimpleNamespace(timeout=54, n_jobs=8, composer_requirements=requirements,
                                   data={'final_fit_use_remaining_time': enabled})
    model._fit_started_at = 100
    model.log = Mock()
    model.current_pipeline = Mock()
    monkeypatch.setattr('fedot.api.main.monotonic', lambda: 100 + elapsed)
    data = object()
    model._train_pipeline_on_full_dataset(None, data)
    model.current_pipeline.fit.assert_called_once_with(data, time_constraint=timedelta(seconds=expected), n_jobs=8)
    assert requirements.max_graph_fit_time == timedelta(seconds=324)


def test_exhausted_refit_budget_does_not_start_training(monkeypatch):
    model = Fedot.__new__(Fedot)
    model.params = SimpleNamespace(timeout=54, data={'final_fit_use_remaining_time': True})
    model._fit_started_at = 0
    model.current_pipeline = Mock()
    monkeypatch.setattr('fedot.api.main.monotonic', lambda: 3200)
    with pytest.raises(TimeoutError, match='No AutoML budget'):
        model._train_pipeline_on_full_dataset(None, object())
    model.current_pipeline.fit.assert_not_called()


@pytest.mark.parametrize('value,expected', [(True, True), (False, False), ('true', True), ('false', False)])
def test_final_refit_flag_parses_harness_strings(value, expected):
    params = ApiParamsRepository(TaskTypesEnum.classification).check_and_set_default_params(
        {'final_fit_use_remaining_time': value})
    assert params['final_fit_use_remaining_time'] is expected


def test_invalid_final_refit_flag_is_rejected():
    with pytest.raises(ValueError, match='must be boolean'):
        ApiParamsRepository(TaskTypesEnum.classification).check_and_set_default_params(
            {'final_fit_use_remaining_time': 'maybe'})
