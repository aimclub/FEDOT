from typing import Sequence

import pytest

from fedot.core.optimisers.gp_comp.generation_keeper import GenerationKeeper
from fedot.core.optimisers.gp_comp.individual import Individual

from fedot.core.optimisers.fitness import Fitness, SingleObjFitness, MultiObjFitness, null_fitness
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum, ComplexityMetricsEnum
from fedot.core.utils import DEFAULT_PARAMS_STUB


def create_individual(fitness: Fitness = None) -> Individual:
    first = OptNode(content={'name': 'logit', 'params': DEFAULT_PARAMS_STUB})
    graph = OptGraph(first)
    individual = Individual(graph)
    individual.fitness = fitness or null_fitness()
    return individual


def create_population(fitness: Sequence[Fitness]) -> PopulationT:
    return tuple(map(create_individual, fitness))


def generation_keeper(init_population=None):
    return GenerationKeeper(init_population,
                            is_multi_objective=True,
                            metrics=[RegressionMetricsEnum.RMSE, ComplexityMetricsEnum.structural])


def population1():
    return create_population([
        MultiObjFitness([2, 4]),
        MultiObjFitness([3, 2]),
    ])


def population2():
    return create_population([
        MultiObjFitness([1, 5]),
        MultiObjFitness([3, 3]),
    ])


def test_archive_no_improvement():
    archive = generation_keeper(population1())
    assert archive.stagnation_length == 0
    assert archive.last_improved
    assert archive.quality_improved and archive.complexity_improved
    assert archive.generation_num == 0

    archive.append(population1())
    assert archive.stagnation_length == 1
    assert not archive.last_improved
    assert not archive.quality_improved and not archive.complexity_improved
    assert archive.generation_num == 1


def test_archive_multiobj_one_improvement():
    archive = generation_keeper(population1())
    previous_size = len(archive.best_individuals)

    # second population has dominating individuals
    assert any(new_ind.fitness.dominates(population1()[1].fitness)
               for new_ind in population2())
    archive.append(population2())

    assert archive.stagnation_length == 0
    assert archive.last_improved
    assert archive.generation_num == 1
    # plus one non-dominated individual
    # minus one strongly dominated individual (substituted by better one)
    assert len(archive.best_individuals) == previous_size + 1
    assert archive.complexity_improved
    assert not archive.quality_improved
