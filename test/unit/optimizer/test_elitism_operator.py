import pytest
from golem.core.optimisers.genetic.evaluation import SequentialDispatcher
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.elitism import Elitism, ElitismTypesEnum
from golem.core.optimisers.opt_history_objects.individual import Individual

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
    elitism = Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.keep_n_best))
    new_population = elitism(best_individuals, population)
    for best_ind in best_individuals:
        assert best_ind in new_population
    assert len(population) == len(new_population)


def test_elitism_not_applicable(set_up):
    best_individuals, population = set_up
    elitisms = [
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                      multi_objective=True)),
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.replace_worst,
                                      pop_size=4, min_pop_size_with_elitism=5)),
        Elitism(GPAlgorithmParameters(elitism_type=ElitismTypesEnum.none)),
    ]
    for elitism in elitisms:
        new_population = elitism(best_individuals, population)
        for best_ind in best_individuals:
            assert best_ind not in new_population
        assert new_population == population
