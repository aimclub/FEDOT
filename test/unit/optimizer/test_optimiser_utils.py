from copy import deepcopy

from deap import tools

from fedot.core.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.utils.pareto import ParetoFront as FedotParetoFront
from fedot.core.optimisers.utils.population_utils import is_equal_archive
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_third


def test_is_eq_archive():
    archive_pipeline_first = pipeline_third()
    archive_pipeline_second = pipeline_first()
    archive_pipeline_third = pipeline_third()
    archive_pipeline_forth = pipeline_third()
    population_first = [archive_pipeline_first, archive_pipeline_second, archive_pipeline_third]
    population_second = deepcopy(population_first)
    population_first.append(archive_pipeline_forth)
    eval_fitness = [(-0.9821, 0.8), (-0.8215, 0.6), (-0.21111, 0.4), (-0.92, 0.9)]
    for population in (population_first, population_second):
        for pipeline_num, pipeline in enumerate(population):
            fitness = MultiObjFitness(values=eval_fitness[pipeline_num],
                                      weights=tuple([-1 for _ in range(len(eval_fitness[pipeline_num]))]))
            pipeline.fitness = fitness
    archive_first = tools.ParetoFront()
    archive_first.update(population_first)
    assert len(archive_first.items) == 3
    archive_second = tools.ParetoFront()
    archive_second.update(population_second)
    assert is_equal_archive(archive_first, archive_second)
    new_fitness = (-0.9821, 0.80001)
    population_second[0].fitness.values = new_fitness
    archive_third = tools.ParetoFront()
    archive_third.update(population_second)
    assert not is_equal_archive(archive_first, archive_third)


def test_pareto_front():
    archive_pipeline_first = pipeline_third()
    archive_pipeline_second = pipeline_first()
    archive_pipeline_third = pipeline_third()
    population = [archive_pipeline_first, archive_pipeline_second, archive_pipeline_third]

    eval_fitness = [(-0.9821, 0.8), (-0.8215, 0.6), (-0.9821, 0.8)]

    for pipeline_num, pipeline in enumerate(population):
        fitness = MultiObjFitness(values=eval_fitness[pipeline_num],
                                  weights=tuple([-1 for _ in range(len(eval_fitness[pipeline_num]))]))
        pipeline.fitness = fitness

    front = tools.ParetoFront()
    front.__class__ = FedotParetoFront
    front.update(population)

    assert len(front) == 2
