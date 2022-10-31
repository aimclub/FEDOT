from typing import Sequence

from fedot.core.optimisers.archive import GenerationKeeper
from fedot.core.optimisers.fitness import Fitness, MultiObjFitness, null_fitness
from fedot.core.optimisers.gp_comp.operators.operator import PopulationT
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.objective.objective import Objective, MetricsObjective
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.repository.quality_metrics_repository import ComplexityMetricsEnum, RegressionMetricsEnum


def create_individual(fitness: Fitness = None) -> Individual:
    first = OptNode(content={'name': 'logit'})
    graph = OptGraph(first)
    individual = Individual(graph)
    individual.set_evaluation_result(fitness or null_fitness())
    return individual


def create_population(fitness: Sequence[Fitness]) -> PopulationT:
    return tuple(map(create_individual, fitness))


def generation_keeper(init_population=None, multi_objective=True):
    metrics = (RegressionMetricsEnum.RMSE, ComplexityMetricsEnum.structural)
    objective = MetricsObjective(metrics, multi_objective)
    return GenerationKeeper(objective, initial_generation=init_population)


def population1():
    return create_population([
        MultiObjFitness([2, 4], weights=-1),
        MultiObjFitness([3, 2], weights=-1),
    ])


def population2():
    return create_population([
        MultiObjFitness([1, 5], weights=-1),
        MultiObjFitness([3, 3], weights=-1),
    ])


def test_archive_no_improvement():
    archive = generation_keeper(population1())
    assert archive.stagnation_duration == 0
    assert archive.is_any_improved
    assert archive.is_quality_improved and archive.is_complexity_improved
    assert archive.generation_num == 1

    archive.append(population1())
    assert archive.stagnation_duration == 1
    assert not archive.is_any_improved
    assert not archive.is_quality_improved and not archive.is_complexity_improved
    assert archive.generation_num == 2


def test_archive_multiobj_one_improvement():
    archive = generation_keeper(population1())
    previous_size = len(archive.best_individuals)

    # second population has dominating individuals
    assert any(new_ind.fitness.dominates(population1()[1].fitness)
               for new_ind in population2())
    archive.append(population2())

    assert archive.stagnation_duration == 0
    assert archive.is_any_improved
    assert archive.generation_num == 2
    # plus one non-dominated individual
    # minus one strongly dominated individual (substituted by better one)
    assert len(archive.best_individuals) == previous_size + 1
    assert archive.is_complexity_improved
    assert not archive.is_quality_improved
