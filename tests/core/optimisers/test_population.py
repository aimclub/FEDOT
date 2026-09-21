from copy import deepcopy
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from hypothesis import given, strategies as st
from golem.core.optimisers.fitness import null_fitness
from golem.core.optimisers.genetic.evaluation import SequentialDispatcher
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective.objective import to_fitness
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError

from fedot.core.optimisers.evaluation_hooks import EvolutionHooks, configure_golem_evaluation
from fedot.core.optimisers.population import (
    BoundedReproduction, ContractEvaluationDispatcher, ReproductionPolicy, UniqueEvaluationDispatcher,
    candidate_identity, plan_reproduction, plan_unique_population, reproduction_random_state,
)
from fedot.core.pipelines.adapters import PipelineAdapter


def individual(name='torch_linear', valid=False):
    return Individual(OptGraph(OptNode(name)), fitness=to_fitness((1.,)) if valid else null_fitness())


@given(st.lists(st.text(alphabet='abc', min_size=1, max_size=3), max_size=25))
def test_unique_population_is_stable_idempotent_and_complete(ids):
    original = tuple(ids)
    plan = plan_unique_population(original)
    unique = tuple(original[i] for i in plan.selected)
    assert unique == tuple(dict.fromkeys(original))
    assert set(plan.selected).isdisjoint(plan.duplicates)
    assert sorted(plan.selected + plan.duplicates) == list(range(len(ids)))
    assert plan_unique_population(unique).selected == tuple(range(len(unique)))
    assert tuple(ids) == original


def test_unique_population_prefers_an_already_evaluated_duplicate():
    plan = plan_unique_population(('same', 'same', 'other'), preferred=frozenset({1}))

    assert plan.selected == (1, 2)
    assert plan.duplicates == (0,)


def test_unique_dispatcher_keeps_evaluated_duplicate_instead_of_pending_one():
    calls = []
    delegate = SimpleNamespace(dispatch=lambda *args: lambda pop: calls.append(pop) or pop)
    dispatcher = UniqueEvaluationDispatcher(delegate)
    pending, evaluated = individual(), individual(valid=True)

    result = dispatcher.dispatch(None)([pending, evaluated])

    assert result == [evaluated]
    assert calls == [[evaluated]]


@given(st.integers(), st.integers(min_value=0, max_value=100), st.integers(min_value=1, max_value=10))
def test_reproduction_plan_is_deterministic_bounded_and_does_not_touch_rng(seed, generation, attempt):
    random_state, np_state = random.getstate(), np.random.get_state()
    policy = ReproductionPolicy(seed, max_attempts=10)
    first = plan_reproduction(policy, generation, attempt, 10, 4, 20)
    assert first == plan_reproduction(policy, generation, attempt, 10, 4, 20)
    assert first.requested == 6
    assert random.getstate() == random_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])


def test_rng_scope_restores_state_even_on_operator_failure():
    state, np_state = random.getstate(), np.random.get_state()
    with pytest.raises(ValueError, match='operator failed'):
        with reproduction_random_state(42):
            random.random()
            np.random.random()
            raise ValueError('operator failed')
    assert random.getstate() == state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])


def test_unique_dispatcher_filters_structural_duplicates_with_different_uids():
    calls = []
    delegate = SimpleNamespace(dispatch=lambda *args: lambda pop: calls.append(pop) or pop)
    dispatcher = UniqueEvaluationDispatcher(delegate)
    population = [individual(), individual(), individual('ridge')]
    result = dispatcher.dispatch(None)(population)
    assert result == [population[0], population[2]]
    assert len(calls) == 1 and len(population) == 3
    assert dispatcher.last_plan.duplicates == (1,)


def test_contract_dispatcher_does_not_repeat_failed_evaluation():
    dispatcher = ContractEvaluationDispatcher(PipelineAdapter(), n_jobs=1)
    calls = []
    result = dispatcher.dispatch(lambda graph: calls.append(graph) or null_fitness())([individual()])
    assert not result
    assert len(calls) == 1
    assert dispatcher.evaluation_cache == {}


def test_contract_dispatcher_does_not_retry_expired_deadline():
    dispatcher = ContractEvaluationDispatcher(PipelineAdapter(), n_jobs=1)
    timer = SimpleNamespace(is_time_limit_reached=lambda: True)
    calls = []
    assert dispatcher.dispatch(lambda graph: calls.append(graph) or to_fitness((1.,)), timer)([individual()]) == []
    assert not calls


@pytest.mark.integration
def test_contract_dispatcher_preserves_golem_parallel_transport():
    delegate = ContractEvaluationDispatcher(PipelineAdapter(), n_jobs=2)
    dispatcher = UniqueEvaluationDispatcher(delegate)
    population = [individual(), individual(), individual('ridge')]
    result = dispatcher.dispatch(lambda graph: to_fitness((float(len(graph.nodes)),)))(population)
    assert len(result) == 2
    assert all(ind.fitness.valid and ind.fitness.value == 1 for ind in result)
    assert len(set(map(candidate_identity, result))) == 2
    assert delegate.evaluation_cache == {}


@pytest.mark.integration
def test_parallel_workers_receive_and_release_scoped_extensions():
    from fedot.extensions import (ExtensionManifest, ExternalModelSpec, ModelCapabilities,
                                  extension_scope, get_registered_extensions)
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import TaskTypesEnum
    manifest = ExtensionManifest('fed03_parallel', '1', (ExternalModelSpec(
        'fed03_external', lambda: object(), ModelCapabilities(
            (TaskTypesEnum.classification,), (DataTypesEnum.table,))),))
    before = get_registered_extensions()
    delegate = ContractEvaluationDispatcher(PipelineAdapter(), n_jobs=2)

    def metric(graph):
        return to_fitness((float(len(get_registered_extensions())),))

    with extension_scope(manifest):
        result = delegate.dispatch(metric)([individual('fed03_external'), individual('torch_linear')])
        assert len(result) == 2 and all(ind.fitness.value == len(before) + 1 for ind in result)
    assert get_registered_extensions() == before
    fresh = delegate.dispatch(metric)([individual('torch_linear')])
    assert fresh[0].fitness.value == len(before)


def test_contract_dispatcher_clears_remote_cache_on_unexpected_exception():
    dispatcher = ContractEvaluationDispatcher(PipelineAdapter(), n_jobs=1)

    def fail(graph):
        dispatcher.evaluation_cache['retained'] = object()
        raise KeyError('unexpected')

    with pytest.raises(KeyError):
        dispatcher.dispatch(fail)([individual()])
    assert dispatcher.evaluation_cache == {}


def test_configuration_uses_public_dispatcher_api_and_is_idempotent():
    adapter = PipelineAdapter()
    optimizer = SimpleNamespace(
        eval_dispatcher=SequentialDispatcher(adapter),
        graph_generation_params=SimpleNamespace(adapter=adapter, remote_evaluator=None),
        requirements=SimpleNamespace(n_jobs=3),
    )
    configure_golem_evaluation(optimizer, EvolutionHooks())
    installed = optimizer.eval_dispatcher
    assert isinstance(installed, UniqueEvaluationDispatcher)
    assert installed.delegate.jobs == 1
    configure_golem_evaluation(optimizer, EvolutionHooks())
    assert optimizer.eval_dispatcher is installed


def test_reproduction_uses_actual_golem_controller_and_is_replayable():
    parents = [individual('a', True), individual('b', True)]
    identities = tuple(map(candidate_identity, parents))
    params = GPAlgorithmParameters(pop_size=2)

    def mutate(population):
        return [individual(f'child-{random.randrange(100000)}', True) for _ in population]

    def make():
        golem = ReproductionController(params, lambda pop, n: list(pop)[:n], mutate, lambda pop: pop)
        return BoundedReproduction(golem, ReproductionPolicy(seed=42), lambda: params.pop_size)

    first, second = make(), make()
    result_a = first.reproduce(parents, lambda pop: pop)
    result_b = second.reproduce(parents, lambda pop: pop)
    assert tuple(map(candidate_identity, result_a)) == tuple(map(candidate_identity, result_b))
    assert first.last_report == second.last_report
    assert first.last_report.complete
    assert tuple(map(candidate_identity, parents)) == identities
    assert all(child.graph is not parent.graph for child in result_a for parent in parents)


def test_reproduction_repeated_candidate_never_counts_twice():
    calls = []

    class DuplicateGenerator:
        def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
            return [individual('duplicate', True), individual('duplicate', True)]

    reproduce = BoundedReproduction(DuplicateGenerator(), ReproductionPolicy(max_attempts=3), lambda: 2)
    with pytest.raises(EvaluationAttemptsError, match='1/2 unique valid'):
        reproduce.reproduce([individual('a'), individual('b')], lambda pop: calls.append(pop) or pop)
    assert len(calls) == 1 and len(calls[0]) == 1
    assert len(reproduce.last_report.steps) == 3
    assert reproduce.last_report.duplicate_count == 5
    assert not reproduce.last_report.complete


def test_reproduction_does_not_mutate_parent_graph_even_with_mutating_operator():
    class MutatingGenerator:
        def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
            population[0].graph.root_node.content['name'] = 'changed'
            return [individual('success', True)]

    parent = individual('original')
    before = deepcopy(parent.graph)
    reproduce = BoundedReproduction(MutatingGenerator(), ReproductionPolicy(), lambda: 1)
    reproduce.reproduce([parent], lambda pop: pop)
    assert parent.graph.descriptive_id == before.descriptive_id


def test_reproduction_accepts_worker_parameter_normalization_by_requested_uid():
    class Generator:
        def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
            return [individual('requested')]

    def evaluate(population):
        child = population[0]
        child.graph.root_node.content['name'] = 'normalized'
        child.set_evaluation_result(to_fitness((1.,)))
        return population

    reproduce = BoundedReproduction(Generator(), ReproductionPolicy(), lambda: 1)
    result = reproduce.reproduce([individual('parent')], evaluate)
    assert candidate_identity(result[0]) != reproduce.last_report.candidate_ids[0]
    assert reproduce.last_report.complete


@pytest.mark.parametrize('policy', [dict(max_attempts=0), dict(max_attempts=True),
                                    dict(required_valid_ratio=0), dict(required_valid_ratio=1.1),
                                    dict(required_valid_ratio=float('nan')), dict(seed=True)])
def test_invalid_reproduction_policy_is_rejected(policy):
    with pytest.raises((ValueError, TypeError)):
        ReproductionPolicy(**policy)


def test_reproduction_rejects_boolean_plan_and_population_sizes():
    policy = ReproductionPolicy()
    with pytest.raises(ValueError, match='invalid reproduction state'):
        plan_reproduction(policy, False, 1, 1, 0, 1)

    reproduction = BoundedReproduction(Mock(), policy, lambda: True)
    with pytest.raises(ValueError, match='population size'):
        reproduction.reproduce([], Mock())
