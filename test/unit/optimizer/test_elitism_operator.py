from copy import deepcopy

import pytest

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.evaluation import SimpleDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism, ElitismTypesEnum
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.optimizer.test_evaluation import prepared_objective
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth, \
    pipeline_fifth


@pytest.fixture()
def set_up():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]
    best_individual = [Individual(adapter.adapt(pipeline_fourth())), Individual(adapter.adapt(pipeline_fifth()))]
    task = Task(TaskTypesEnum.classification)
    operations = get_operations_for_task(task, mode='model')
    requirements = PipelineComposerRequirements(primary=operations, secondary=operations)

    dispatcher = SimpleDispatcher(adapter)
    objective = prepared_objective
    evaluator = dispatcher.dispatch(objective)
    evaluated_population = evaluator(population)
    evaluated_best_individuals = evaluator(best_individual)
    return requirements, evaluated_best_individuals, evaluated_population


def test_keep_n_best_elitism(set_up):
    requirements, best_individuals, population = set_up
    elitism = Elitism(ElitismTypesEnum.keep_n_best, requirements, is_multi_objective=False)
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert best_ind in new_population
    assert len(population) == len(new_population)


def test_replace_worst(set_up):
    requirements, best_individuals, population = set_up
    elitism = Elitism(ElitismTypesEnum.replace_worst, requirements, is_multi_objective=False)
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert any(best_ind.fitness > ind.fitness for ind in population) == \
               (best_ind in new_population)
    assert len(new_population) == len(population)


def test_elitism_not_applicable(set_up):
    requirements, best_individuals, population = set_up
    modified_requirements = deepcopy(requirements)
    modified_requirements.pop_size = 4
    elitisms = [Elitism(ElitismTypesEnum.replace_worst, requirements, is_multi_objective=True),
                Elitism(ElitismTypesEnum.replace_worst, modified_requirements, is_multi_objective=False,
                        min_population_size_with_elitism=5),
                Elitism(ElitismTypesEnum.none, requirements, is_multi_objective=False)]
    for elitism in elitisms:
        new_population = elitism(best_individuals, population)
        for best_ind in best_individuals:
            assert best_ind not in new_population
        assert new_population == population
