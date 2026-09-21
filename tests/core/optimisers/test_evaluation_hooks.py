from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters

from fedot import create_data
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposer
from fedot.core.optimisers.evaluation_hooks import EvaluationRequest, EvolutionHooks, build_evaluator
from fedot.core.optimisers.objective.data_source_context import build_external_holdout_composer_tensor_data_source_context
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.optimisers.population import ReproductionPolicy, UniqueEvaluationDispatcher
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_composer_requirements import PipelineComposerRequirements


def training_data():
    return create_data(np.array([[0., 1.], [1., 0.], [0., 2.], [2., 0.]], dtype=np.float32),
                       target=np.array([0, 1, 0, 1]), use_cache=False)


def test_composer_factory_receives_typed_request_and_releases_evaluator_on_failure():
    train = training_data()
    requests = []
    service = Mock()

    def factory(request):
        requests.append(request)
        return service

    optimizer = SimpleNamespace(objective=MetricsObjective([lambda *a, **kw: 1.]),
                                optimise=Mock(side_effect=ValueError('optimization failed')))
    requirements = PipelineComposerRequirements(n_jobs=1, parallelization_mode='sequential')
    composer = GPComposer(optimizer, requirements, evolution_hooks=EvolutionHooks(evaluator_factory=factory))
    with pytest.raises(ValueError, match='optimization failed'):
        composer.compose_pipeline(build_external_holdout_composer_tensor_data_source_context(train, train))
    assert len(requests) == 1 and isinstance(requests[0], EvaluationRequest)
    service.clear_results.assert_called_once_with()


@pytest.mark.integration
def test_builder_and_actual_golem_optimizer_use_current_cpu_evaluator():
    train = training_data()
    node = PrimaryNode('torch_linear')
    node.parameters = {'epochs': 1, 'learning_rate': 0.01}
    pipeline = Pipeline(node)
    evaluations = []

    def metric(model, reference_data, **kwargs):
        prediction = model.predict(reference_data).predict
        evaluations.append(len(prediction))
        return float(np.mean(np.asarray(prediction) != np.asarray(reference_data.target)))

    requirements = PipelineComposerRequirements(
        primary=['torch_linear'], secondary=['torch_linear'], n_jobs=1,
        parallelization_mode='sequential', num_of_generations=0, cv_folds=None,
        timeout=timedelta(seconds=20), show_progress=False, keep_history=False)
    composer = (ComposerBuilder(train.task).with_requirements(requirements)
                .with_optimizer_params(GPAlgorithmParameters(pop_size=1))
                .with_metrics([metric]).with_initial_pipelines([pipeline])
                .with_evolution_hooks(EvolutionHooks(reproduction=ReproductionPolicy(seed=4)))
                .build())
    best = composer.compose_pipeline(build_external_holdout_composer_tensor_data_source_context(train, train))
    assert isinstance(best, Pipeline)
    assert evaluations == [4]
    assert isinstance(composer.optimizer.eval_dispatcher, UniqueEvaluationDispatcher)
    assert not best.is_fitted


def test_public_request_factory_applies_retry_and_count_contract():
    from fedot.core.optimisers.objective.evaluation_contracts import RetryPolicy
    request = EvaluationRequest(MetricsObjective([lambda *a, **kw: 1.]), lambda: (),
                                retry_policy=RetryPolicy(3), expected_folds=2)
    evaluator = build_evaluator(request)
    assert evaluator.retry_policy == RetryPolicy(3)
    assert evaluator.expected_folds == 2


def test_builder_rejects_untyped_hook_configuration():
    with pytest.raises(TypeError, match='EvolutionHooks'):
        ComposerBuilder(training_data().task).with_evolution_hooks({'retry': 3})


def test_hooks_reject_boolean_fold_count():
    with pytest.raises(ValueError, match='expected_folds'):
        EvolutionHooks(expected_folds=True)
