from functools import partial

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.debug.metrics import RandomMetric
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.fitness.fitness import SingleObjFitness
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.selection import (
    SelectionTypesEnum,
    individuals_selection,
    random_selection,
    selection,
    tournament_selection
)
from fedot.core.optimisers.optimizer import GraphGenerationParams


def rand_population_gener_and_eval(pop_size=4):
    models_set = ['knn', 'logit', 'rf']
    requirements = PipelineComposerRequirements(primary=models_set,
                                                secondary=models_set, max_depth=1)
    pipeline_gener_params = GraphGenerationParams(advisor=PipelineChangeAdvisor(), adapter=PipelineAdapter())
    random_pipeline_function = partial(random_graph, pipeline_gener_params.validator, requirements)
    population = []
    while len(population) != pop_size:
        # to ensure uniqueness
        ind = Individual(random_pipeline_function())
        if ind not in population:
            population.append(ind)

    # evaluation
    for ind in population:
        ind.fitness = SingleObjFitness(obj_function())
    return population


def obj_function() -> float:
    metric_function = RandomMetric.get_value
    return metric_function()


def test_tournament_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = tournament_selection(individuals=population,
                                                pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_random_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = random_selection(individuals=population,
                                            pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    graph_params = GraphGenerationParams(advisor=PipelineChangeAdvisor(), adapter=PipelineAdapter())

    selected_individuals = selection(types=[SelectionTypesEnum.tournament],
                                     population=population,
                                     pop_size=num_of_inds,
                                     params=graph_params)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_random_individuals():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    types = [SelectionTypesEnum.tournament]
    graph_params = GraphGenerationParams(advisor=PipelineChangeAdvisor(), adapter=PipelineAdapter())
    selected_individuals = individuals_selection(types=types,
                                                 individuals=population,
                                                 pop_size=num_of_inds,
                                                 graph_params=graph_params)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(set(selected_individuals_ref)) == len(selected_individuals) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_equality_individuals():
    num_of_inds = 4
    population = rand_population_gener_and_eval(pop_size=1)
    types = [SelectionTypesEnum.tournament]
    population = [population[0] for _ in range(4)]
    graph_params = GraphGenerationParams(advisor=PipelineChangeAdvisor(), adapter=PipelineAdapter())
    selected_individuals = individuals_selection(types=types,
                                                 individuals=population,
                                                 pop_size=num_of_inds, graph_params=graph_params)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(selected_individuals) == num_of_inds and
            len(set(selected_individuals_ref)) == 1)
