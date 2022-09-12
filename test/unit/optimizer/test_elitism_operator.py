from typing import TYPE_CHECKING

import pytest

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.gp_comp.evaluation import SimpleDispatcher
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism, ElitismTypesEnum
from test.unit.optimizer.test_evaluation import prepared_objective
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth, \
    pipeline_fifth

if TYPE_CHECKING:
    from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters


@pytest.fixture()
def set_up():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]
    best_individual = [Individual(adapter.adapt(pipeline_fourth())), Individual(adapter.adapt(pipeline_fifth()))]

    dispatcher = SimpleDispatcher(adapter)
    objective = prepared_objective
    evaluator = dispatcher.dispatch(objective)
    evaluated_population = evaluator(population)
    evaluated_best_individuals = evaluator(best_individual)
    return evaluated_best_individuals, evaluated_population


def test_keep_n_best_elitism(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.keep_n_best), is_multi_objective=False)
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert best_ind in new_population
    assert len(population) == len(new_population)


def test_replace_worst(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.replace_worst), is_multi_objective=False)
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert any(best_ind.fitness > ind.fitness for ind in population) == \
               (best_ind in new_population)
    assert len(new_population) == len(population)


def test_elitism_not_applicable(set_up):
    best_individuals, population = set_up
    elitisms = [Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.replace_worst),
                        is_multi_objective=True),
                Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                                   pop_size=4),
                        is_multi_objective=False,
                        min_population_size_with_elitism=5),
                Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.none),
                        is_multi_objective=False)]
    for elitism in elitisms:
        new_population = elitism(best_individuals, population)
        for best_ind in best_individuals:
            assert best_ind not in new_population
        assert new_population == population
