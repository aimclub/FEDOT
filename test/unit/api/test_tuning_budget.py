from contextlib import contextmanager
from datetime import timedelta
from unittest.mock import MagicMock, Mock

from fedot.api.api_utils.assumptions.tuning_budget import bounded_composition_resources, bounded_tuning_resources
from fedot.core.optimisers.objective.data_objective_eval import PipelineObjectiveEvaluate


def test_expensive_composition_reserves_tuning_and_bounds_cv_overrun():
    from fedot.api.time import ApiTime

    composing, evaluation = bounded_composition_resources(3240, 1944, 388.4)
    assert composing == 1296
    assert round(evaluation, 1) == 776.8
    assert bounded_composition_resources(3240, 1944, 10) is None
    assert bounded_composition_resources(3240, 1944, 600) is None

    timer = ApiTime(time_for_automl=54, with_tuning=True)
    timer.assumption_fit_spend_time_single_fold = timedelta(seconds=77.7)
    timer.assumption_fit_spend_time = timedelta(seconds=388.4)
    timer.composing_spend_time = timedelta(seconds=composing + evaluation)
    assert timer.have_time_for_tuning()
    timer.composing_spend_time = timedelta(minutes=40)
    assert not timer.have_time_for_tuning()


def test_gp_composer_passes_cv_evaluation_deadline(monkeypatch):
    from fedot.core.composer.gp_composer import gp_composer

    requirements = Mock(cv_folds=5, parallelization_mode='sequential', n_jobs=8,
                        max_graph_fit_time=timedelta(seconds=324),
                        evaluation_time_constraint=timedelta(seconds=778),
                        collect_intermediate_metric=False)
    optimizer = Mock()
    composer = gp_composer.GPComposer(optimizer, requirements)
    splitter = Mock()
    monkeypatch.setattr(gp_composer, 'DataSourceSplitter', Mock(return_value=splitter))
    evaluator_class = Mock()
    monkeypatch.setattr(gp_composer, 'PipelineObjectiveEvaluate', evaluator_class)
    monkeypatch.setattr(composer, '_convert_opt_results_to_pipeline', Mock(return_value=(Mock(), [])))

    composer.compose_pipeline(Mock())

    assert evaluator_class.call_args.kwargs['evaluation_time_constraint'] == timedelta(seconds=778)


def test_large_workload_reserves_two_bounded_cv_evaluations():
    search, evaluation, folds = bounded_tuning_resources(960, 43, 5)
    assert (search, evaluation, folds) == (480, 240, 3)
    assert search + 2 * evaluation == 960
    assert bounded_tuning_resources(2640, 81.5, 5) == (1320, 660, 3)


def test_small_or_unbounded_tuning_preserves_original_cv():
    assert bounded_tuning_resources(960, 2, 5) is None
    assert bounded_tuning_resources(0, 43, 5) is None
    assert bounded_tuning_resources(960, 43, None) is None


def test_fast_initial_fit_does_not_allow_heavy_tuning_candidates_to_overrun():
    # The initial assumption may fit quickly even when evolution chooses a
    # costly model. Limit entire tuning trials rather than assuming its fit time.
    search, evaluation, folds = bounded_tuning_resources(1860, 22.2, 5)
    assert (search, evaluation, folds) == (930, 465, 3)
    assert search + 2 * evaluation == 1860
    assert bounded_tuning_resources(2040, 12.2, 5) == (1020, 510, 3)


def test_complex_evolved_pipeline_gets_a_cv_deadline_even_with_fast_baseline():
    assert bounded_tuning_resources(1560, 8.6, 5) is None
    assert bounded_tuning_resources(1560, 8.6, 5, complex_pipeline=True) == (780, 390, 3)
    assert bounded_tuning_resources(960, 2, 5, complex_pipeline=True) == (480, 240, 3)


def test_composer_bounds_tuning_of_complex_pipeline_after_fast_initial_fit(monkeypatch):
    from fedot.api.api_utils import api_composer
    from fedot.api.time import ApiTime
    from fedot.core.pipelines.pipeline import Pipeline

    params = Mock()
    params.get.return_value = False
    params.composer_requirements.cv_folds = 5
    params.composer_requirements.max_graph_fit_time = timedelta(seconds=324)
    composer = api_composer.ApiComposer(params, ['roc_auc'])
    composer.timer = ApiTime(time_for_automl=54, with_tuning=True)
    composer.timer.composing_spend_time = timedelta(minutes=27)
    composer.timer.assumption_fit_spend_time_single_fold = timedelta(seconds=8.6)
    composer.timer.assumption_fit_spend_time = timedelta(seconds=43)

    builder = Mock()
    for method in ('with_tuner', 'with_metric', 'with_iterations', 'with_timeout',
                   'with_eval_time_constraint', 'with_requirements', 'with_cv_folds',
                   'with_evaluation_time_constraint'):
        getattr(builder, method).return_value = builder
    monkeypatch.setattr(api_composer, 'TunerBuilder', Mock(return_value=builder))

    pipeline = Mock(spec=Pipeline)
    pipeline.nodes = [Mock() for _ in range(6)]
    composer.tune_final_pipeline(Mock(), pipeline)

    builder.with_cv_folds.assert_called_once_with(3)
    builder.with_evaluation_time_constraint.assert_called_once()
    builder.build.return_value.tune.assert_called_once_with(pipeline)


def test_evolution_without_valid_graph_falls_back_to_fitted_assumption(monkeypatch):
    from fedot.api.api_utils import api_composer
    from fedot.api.time import ApiTime

    params = Mock()
    params.get.side_effect = lambda key: 10 if key == 'pop_size' else False
    params.n_jobs = 8
    params.composer_requirements = Mock()
    params.composer_requirements.evaluation_time_constraint = None
    params.graph_generation_params = Mock()
    composer = api_composer.ApiComposer(params, ['roc_auc'])
    composer.timer = ApiTime(time_for_automl=54, with_tuning=True)
    composer.timer.assumption_fit_spend_time = timedelta(seconds=200)

    gp_composer = Mock()
    gp_composer.compose_pipeline.return_value = None
    gp_composer.best_models = []
    builder = Mock()
    builder.with_requirements.return_value = builder
    builder.with_initial_pipelines.return_value = builder
    builder.with_optimizer.return_value = builder
    builder.with_optimizer_params.return_value = builder
    builder.with_metrics.return_value = builder
    builder.with_cache.return_value = builder
    builder.with_graph_generation_param.return_value = builder
    builder.build.return_value = gp_composer
    monkeypatch.setattr(api_composer, 'ComposerBuilder', Mock(return_value=builder))

    fitted = Mock()
    pipeline, candidates, result_composer = composer.compose_pipeline(Mock(), [fitted], fitted)
    assert pipeline is fitted
    assert candidates == [fitted]
    assert result_composer is gp_composer


def test_expensive_evolution_keeps_cv_but_reduces_folds(monkeypatch):
    from fedot.api.api_utils import api_composer
    from fedot.api.time import ApiTime

    values = {'cv_folds': 5, 'pop_size': 10}
    params = MagicMock()
    params.data = values
    params.n_jobs = 8
    params.get.side_effect = lambda key: {
        'available_operations': ['logit', 'lgbm'], 'preset': 'best_quality',
        'with_tuning': True,
    }.get(key)
    params.__getitem__.side_effect = values.__getitem__
    params.__setitem__.side_effect = values.__setitem__
    composer = api_composer.ApiComposer(params, ['roc_auc'])
    composer.timer = ApiTime(time_for_automl=54, with_tuning=True)

    @contextmanager
    def timed_assumption_fit(n_folds):
        yield
        composer.timer.assumption_fit_spend_time_single_fold = timedelta(seconds=81)
        composer.timer.assumption_fit_spend_time = timedelta(seconds=81 * n_folds)

    composer.timer.launch_assumption_fit = timed_assumption_fit
    assumptions = Mock()
    assumptions.propose_assumptions.return_value = [Mock()]
    assumptions.propose_preset.return_value = 'best_quality'
    monkeypatch.setattr(api_composer, 'AssumptionsHandler', Mock(return_value=assumptions))
    monkeypatch.setattr(api_composer, 'memory_safe_operations', lambda data, operations: operations)

    composer.propose_and_fit_initial_assumption(Mock())

    assert values['cv_folds'] == 3
    assert composer.timer.assumption_fit_spend_time == timedelta(seconds=243)
    assert composer.composition_evaluation_timeout == timedelta(seconds=777.6)


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
