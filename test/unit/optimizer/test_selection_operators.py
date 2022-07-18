from functools import partial

from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.debug.metrics import RandomMetric
from fedot.core.optimisers.fitness.fitness import SingleObjFitness
from fedot.core.optimisers.gp_comp.gp_operators import random_graph
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum, Selection
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params


def rand_population_gener_and_eval(pop_size=4):
    models_set = ['knn', 'logit', 'rf']
    requirements = PipelineComposerRequirements(primary=models_set,
                                                secondary=models_set, max_depth=1)
    pipeline_gener_params = get_pipeline_generation_params(requirements=requirements)
    random_pipeline_function = partial(random_graph, pipeline_gener_params, requirements)
    population = []
    while len(population) != pop_size:
        # to ensure uniqueness
        ind = Individual(random_pipeline_function())
        if ind not in population:
            population.append(ind)

    # evaluation
    for ind in population:
        ind.set_evaluation_result(SingleObjFitness(obj_function()))
    return population


def obj_function() -> float:
    metric_function = RandomMetric.get_value
    return metric_function()


def test_tournament_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = Selection.tournament_selection(individuals=population, pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_random_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    selected_individuals = Selection.random_selection(individuals=population, pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_selection():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    types = [SelectionTypesEnum.tournament]
    selection = Selection(types)
    selected_individuals = selection(population=population, pop_size=num_of_inds)
    assert (all([ind in population for ind in selected_individuals]) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_random_individuals():
    num_of_inds = 2
    population = rand_population_gener_and_eval(pop_size=4)
    types = [SelectionTypesEnum.tournament]
    selection = Selection(types)
    selected_individuals = selection.individuals_selection(individuals=population, pop_size=num_of_inds)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(set(selected_individuals_ref)) == len(selected_individuals) and
            len(selected_individuals) == num_of_inds)


def test_individuals_selection_equality_individuals():
    num_of_inds = 4
    population = rand_population_gener_and_eval(pop_size=1)
    types = [SelectionTypesEnum.tournament]
    population = [population[0] for _ in range(4)]
    selection = Selection(types)
    selected_individuals = selection.individuals_selection(individuals=population, pop_size=num_of_inds)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert (len(selected_individuals) == num_of_inds and
            len(set(selected_individuals_ref)) == 1)
