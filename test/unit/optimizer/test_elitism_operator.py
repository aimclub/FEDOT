import pytest

from fedot.core.optimisers.gp_comp.evaluation import SequentialDispatcher
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.elitism import Elitism, ElitismTypesEnum
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.pipelines.adapters import PipelineAdapter
from test.unit.optimizer.test_evaluation import prepared_objective
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third, pipeline_fourth, \
    pipeline_fifth


@pytest.fixture()
def set_up():
    adapter = PipelineAdapter()
    pipelines = [pipeline_first(), pipeline_second(), pipeline_third(), pipeline_fourth()]
    population = [Individual(adapter.adapt(pipeline)) for pipeline in pipelines]
    best_individual = [Individual(adapter.adapt(pipeline_fourth())), Individual(adapter.adapt(pipeline_fifth()))]

    dispatcher = SequentialDispatcher(adapter)
    objective = prepared_objective
    evaluator = dispatcher.dispatch(objective)
    evaluated_population = evaluator(population)
    evaluated_best_individuals = evaluator(best_individual)
    return evaluated_best_individuals, evaluated_population


def test_keep_n_best_elitism(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.keep_n_best))
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert best_ind in new_population
    assert len(population) == len(new_population)


def test_replace_worst(set_up):
    best_individuals, population = set_up
    elitism = Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.replace_worst))
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert any(best_ind.fitness > ind.fitness for ind in population) == \
               (best_ind in new_population)
    assert len(new_population) == len(population)


def test_elitism_not_applicable(set_up):
    best_individuals, population = set_up
    elitisms = [
        Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                           multi_objective=True)),
        Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                           pop_size=4, min_pop_size_with_elitism=5)),
        Elitism(GPGraphOptimizerParameters(elitism_type=ElitismTypesEnum.none)),
    ]
    for elitism in elitisms:
        new_population = elitism(best_individuals, population)
        for best_ind in best_individuals:
            assert best_ind not in new_population
        assert new_population == population
