from datetime import timedelta
from unittest.mock import Mock

from fedot.api.api_utils.assumptions.tuning_budget import bounded_tuning_resources
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate


def test_large_workload_reserves_two_bounded_cv_evaluations():
    search, evaluation, folds = bounded_tuning_resources(960, 43, 5)
    assert (search, evaluation, folds) == (480, 240, 3)
    assert search + 2 * evaluation == 960
    assert bounded_tuning_resources(2640, 81.5, 5) == (1320, 660, 3)


def test_small_or_unbounded_tuning_preserves_original_cv():
    assert bounded_tuning_resources(960, 2, 5) is None
    assert bounded_tuning_resources(0, 43, 5) is None
    assert bounded_tuning_resources(960, 43, None) is None


def test_cv_evaluation_stops_before_the_next_fold_at_its_deadline(monkeypatch):
    from fedot.core.optimisers.objective import data_objective_eval

    graph = Mock()
    graph.root_node.descriptive_id = 'test_graph'
    objective = Mock()
    evaluator = PipelineObjectiveEvaluate(
        objective, lambda: iter([(Mock(), Mock()) for _ in range(3)]),
        time_constraint=timedelta(seconds=30),
        evaluation_time_constraint=timedelta(seconds=10))
    evaluator.prepare_graph = Mock(return_value=graph)
    evaluator._objective = Mock(return_value=Mock(valid=True, values=(0.5,)))
    monkeypatch.setattr(data_objective_eval, 'monotonic', Mock(side_effect=[0, 0, 11]))
    monkeypatch.setattr(data_objective_eval, 'to_fitness', Mock(return_value=None))

    evaluator.evaluate(graph)

    assert evaluator.prepare_graph.call_count == 1
    assert evaluator.prepare_graph.call_args.kwargs['fit_time_constraint'] == timedelta(seconds=10)


def test_api_composer_keeps_tuning_enabled_and_applies_cv_limits(monkeypatch):
    from fedot.api.api_utils import api_composer
    from fedot.api.time import ApiTime

    params = Mock()
    params.get.return_value = False
    params.composer_requirements.cv_folds = 5
    params.composer_requirements.max_graph_fit_time = timedelta(seconds=324)
    composer = api_composer.ApiComposer(params, ['roc_auc'])
    composer.timer = ApiTime(time_for_automl=54, with_tuning=True)
    composer.timer.composing_spend_time = timedelta(minutes=33)
    composer.timer.assumption_fit_spend_time_single_fold = timedelta(seconds=43)
    composer.timer.assumption_fit_spend_time = timedelta(seconds=215)

    builder = Mock()
    builder.with_tuner.return_value = builder
    builder.with_metric.return_value = builder
    builder.with_iterations.return_value = builder
    builder.with_timeout.return_value = builder
    builder.with_eval_time_constraint.return_value = builder
    builder.with_requirements.return_value = builder
    builder.with_cv_folds.return_value = builder
    builder.with_evaluation_time_constraint.return_value = builder
    tuner = builder.build.return_value
    monkeypatch.setattr(api_composer, 'TunerBuilder', Mock(return_value=builder))

    pipeline = Mock()
    assert composer.tune_final_pipeline(Mock(), pipeline) is tuner.tune.return_value
    tuner.tune.assert_called_once_with(pipeline)
    builder.with_cv_folds.assert_called_once_with(3)
    builder.with_evaluation_time_constraint.assert_called_once()
    assert builder.with_timeout.call_args_list[-1].args[0] < timedelta(minutes=16)
