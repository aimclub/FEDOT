"""Independent population and reproduction invariants."""
import random
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from golem.core.optimisers.fitness import null_fitness
from golem.core.optimisers.genetic.evaluation import SequentialDispatcher
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective.objective import to_fitness
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.populational_optimizer import EvaluationAttemptsError
from hypothesis import given, strategies as st

from fedot.core.optimisers.evaluation_hooks import EvolutionHooks, configure_golem_evaluation
from fedot.core.optimisers.population import (
    BoundedReproduction,
    ContractEvaluationDispatcher,
    ReproductionPolicy,
    UniqueEvaluationDispatcher,
    candidate_identity,
    plan_unique_population,
)
from fedot.core.pipelines.adapters import PipelineAdapter


def individual(name='torch_linear', valid=False):
    fitness = to_fitness((1.0,)) if valid else null_fitness()
    return Individual(OptGraph(OptNode(name)), fitness=fitness)


@given(
    ids=st.lists(st.text(alphabet='abc', min_size=1, max_size=3), max_size=30),
    excluded=st.sets(
        st.text(alphabet='abc', min_size=1, max_size=3), max_size=6),
)
def test_unique_plan_matches_stable_reference_model_without_mutating_input(ids, excluded):
    original = tuple(ids)
    plan = plan_unique_population(original, frozenset(excluded))
    expected = []
    seen = set(excluded)
    for index, candidate_id in enumerate(original):
        if candidate_id not in seen:
            seen.add(candidate_id)
            expected.append(index)

    assert plan.selected == tuple(expected)
    assert sorted(plan.selected +
                  plan.duplicates) == list(range(len(original)))
    assert tuple(ids) == original


@pytest.mark.parametrize('evaluated_first', [False, True])
def test_deduplication_preserves_existing_success_independent_of_duplicate_order(evaluated_first):
    pending = individual('same-candidate')
    complete = individual('same-candidate', valid=True)
    population = [complete, pending] if evaluated_first else [
        pending, complete]
    calls = []
    delegate = ContractEvaluationDispatcher(PipelineAdapter(), n_jobs=1)
    dispatcher = UniqueEvaluationDispatcher(delegate)

    result = dispatcher.dispatch(lambda graph: calls.append(
        graph) or null_fitness())(population)

    assert result == [complete]
    assert calls == []
    assert len(population) == 2


def test_failed_logical_candidate_is_not_evaluated_again_across_reproduction_attempts():
    evaluation_calls = []

    class RepeatedFailureGenerator:
        def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
            return [individual('same-failed-candidate')]

    def evaluate(items):
        evaluation_calls.extend(map(candidate_identity, items))
        return items

    reproduction = BoundedReproduction(
        RepeatedFailureGenerator(), ReproductionPolicy(max_attempts=5), lambda: 1)

    with pytest.raises(EvaluationAttemptsError, match='0/1 unique valid'):
        reproduction.reproduce([individual('parent')], evaluate)

    assert evaluation_calls == [candidate_identity(
        individual('same-failed-candidate'))]
    assert len(reproduction.last_report.steps) == 5
    assert reproduction.last_report.duplicate_count == 4


def test_reproduction_tracks_requested_identity_when_fit_normalizes_graph():
    requested = individual('requested')
    requested_id = candidate_identity(requested)

    class NormalizingGenerator:
        def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
            return [requested]

    def evaluate(items):
        child = items[0]
        child.graph.root_node.content['name'] = 'normalized-after-fit'
        child.set_evaluation_result(to_fitness((1.0,)))
        return items

    parent = individual('parent')
    parent_before = deepcopy(parent.graph)
    reproduction = BoundedReproduction(
        NormalizingGenerator(), ReproductionPolicy(seed=19), lambda: 1)

    result = reproduction.reproduce([parent], evaluate)

    assert reproduction.last_report.candidate_ids == (requested_id,)
    assert candidate_identity(result[0]) != requested_id
    assert candidate_identity(parent) == parent_before.descriptive_id


def test_reproduction_is_replayable_without_consuming_global_random_streams():
    class RandomGenerator:
        def reproduce_uncontrolled(self, population, evaluator, pop_size=None):
            suffix = f'{random.randrange(1_000_000)}-{np.random.randint(1_000_000)}'
            return [individual(f'child-{suffix}', valid=True)]

    def run():
        reproduction = BoundedReproduction(
            RandomGenerator(), ReproductionPolicy(seed=91), lambda: 1)
        result = reproduction.reproduce(
            [individual('parent')], lambda items: items)
        return tuple(map(candidate_identity, result)), reproduction.last_report

    random.seed(1234)
    np.random.seed(5678)
    expected_python = random.random()
    expected_numpy = np.random.random()
    random.seed(1234)
    np.random.seed(5678)

    first = run()
    observed_python = random.random()
    observed_numpy = np.random.random()
    second = run()

    assert first == second
    assert observed_python == expected_python
    assert observed_numpy == expected_numpy


def test_public_dispatcher_callback_survives_fedot_configuration():
    adapter = PipelineAdapter()
    optimizer = SimpleNamespace(
        eval_dispatcher=SequentialDispatcher(adapter),
        graph_generation_params=SimpleNamespace(
            adapter=adapter, remote_evaluator=None),
        requirements=SimpleNamespace(n_jobs=1),
    )
    callback = Mock()

    configure_golem_evaluation(optimizer, EvolutionHooks())
    optimizer.eval_dispatcher.set_graph_evaluation_callback(callback)
    result = optimizer.eval_dispatcher.dispatch(
        lambda graph: to_fitness((1.0,)))([individual()])

    assert len(result) == 1 and result[0].fitness.valid
    callback.assert_called_once()
