from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fedot.api.api_utils.assumptions.tuning_budget import (
    measured_tuning_resources, selected_pipeline_fold_seconds,
)
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def test_selected_cost_matches_exact_graph_and_uses_cv_seconds():
    pipeline = PipelineBuilder().add_node('scaling').add_node('logit').build()
    graph = PipelineAdapter().adapt(pipeline)
    different = PipelineAdapter().adapt(PipelineBuilder().add_node('rf').build())

    def individual(graph, seconds, valid=True):
        return SimpleNamespace(graph=graph, fitness=SimpleNamespace(valid=valid),
                               metadata={'computation_time_in_seconds': seconds})

    history = SimpleNamespace(generations=[[
        individual(graph, 444), individual(different, 9000),
        individual(graph, 9999, False), individual(graph, float('nan')),
        individual(graph, -1), individual(graph, True)]])
    assert selected_pipeline_fold_seconds(history, pipeline, 3) == 148
    assert selected_pipeline_fold_seconds(None, pipeline, 3) is None


@pytest.mark.parametrize('available,fold_seconds,folds', [(1600, 148, 5), (2400, 148, 3)])
def test_measured_budget_can_complete_initial_trial_and_final(available, fold_seconds, folds):
    search, evaluation, selected_folds = measured_tuning_resources(available, fold_seconds, folds)
    assert selected_folds >= 2
    assert evaluation >= 1.3 * fold_seconds * selected_folds
    assert search >= evaluation
    assert search + 2 * evaluation == pytest.approx(available)


def test_infeasible_budget_must_skip_not_fall_back_to_unbounded_tuning():
    assert measured_tuning_resources(800, 148, 3) is None
    assert measured_tuning_resources(0, 148, 3) is None


def test_composer_skips_infeasible_measured_cv(monkeypatch):
    from fedot.api.api_utils import api_composer
    from fedot.api.time import ApiTime

    params = Mock()
    params.get.return_value = False
    params.composer_requirements.cv_folds = 3
    composer = api_composer.ApiComposer(params, ['roc_auc'])
    composer.timer = ApiTime(time_for_automl=54, with_tuning=True)
    composer.timer.composing_spend_time = timedelta(seconds=2300)
    composer.timer.assumption_fit_spend_time_single_fold = timedelta(seconds=69)
    composer.timer.assumption_fit_spend_time = timedelta(seconds=207)
    monkeypatch.setattr(api_composer, 'selected_pipeline_fold_seconds', lambda *args: 148)
    builder = Mock()
    monkeypatch.setattr(api_composer, 'TunerBuilder', builder)
    pipeline = PipelineBuilder().add_node('logit').build()
    assert composer.tune_final_pipeline(Mock(), pipeline, Mock()) is pipeline
    assert composer.was_tuned is False
    # No tuner evaluation was started.
    assert not any(call[0].endswith('.build') for call in builder.mock_calls)
