import os
import pytest
from functools import partial
from core.utils import project_root
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import ModelTypesIdsEnum
from core.composer.gp_composer.gp_composer import GPComposerRequirements
from core.composer.optimisers.gp_operators import random_chain
from core.composer.optimisers.selection import SelectionTypesEnum
from core.composer.optimisers.selection import tournament_selection, \
    individuals_selection, random_selection, selection
from core.models.data import InputData
from core.repository.quality_metrics_repository import MetricsRepository, \
    ClassificationMetricsEnum


def train_test_data_setup():
    train_file_path = 'cases/data/scoring/scoring_train.csv'
    train_file_path = os.path.join(str(project_root()), train_file_path)
    test_file_path = 'cases/data/scoring/scoring_test.csv'
    test_file_path = os.path.join(str(project_root()), test_file_path)
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)
    return train_data, test_data


@pytest.fixture()
def rand_population_gener_and_eval(pop_size=4):
    models_set = [ModelTypesIdsEnum.knn, ModelTypesIdsEnum.logit,
                  ModelTypesIdsEnum.rf]
    requirements = GPComposerRequirements(primary=models_set,
                                          secondary=models_set, max_depth=1)
    secondary_node_func = NodeGenerator.secondary_node
    primary_node_func = NodeGenerator.primary_node
    random_chain_function = partial(random_chain, chain_class=Chain,
                                    secondary_node_func=secondary_node_func,
                                    primary_node_func=primary_node_func,
                                    requirements=requirements)
    population = [random_chain_function() for _ in range(pop_size)]
    # evaluation
    train_data, test_data = train_test_data_setup()
    objective_function = partial(obj_function, train_data, test_data)
    for ind in population:
        ind.fitness = objective_function(ind)
    return population


def obj_function(train_data: InputData, test_data: InputData, chain: Chain) \
        -> float:
    metric_function = MetricsRepository().metric_by_id(
        ClassificationMetricsEnum.ROCAUC)
    chain.fit(input_data=train_data)
    return metric_function(chain, test_data)


@pytest.mark.parametrize('pop_fixture', ['rand_population_gener_and_eval'])
def test_tournament_selection(pop_fixture, request):
    num_of_inds = 2
    population = request.getfixturevalue(pop_fixture)
    selected_individuals = tournament_selection(individuals=population,
                                                pop_size=num_of_inds)
    assert all([ind in population for ind in selected_individuals]) and \
           len(selected_individuals) == num_of_inds


@pytest.mark.parametrize('pop_fixture', ['rand_population_gener_and_eval'])
def test_random_selection(pop_fixture, request):
    num_of_inds = 2
    population = request.getfixturevalue(pop_fixture)
    selected_individuals = random_selection(individuals=population,
                                            pop_size=num_of_inds)
    assert all([ind in population for ind in selected_individuals]) and \
           len(selected_individuals) == num_of_inds


@pytest.mark.parametrize('pop_fixture', ['rand_population_gener_and_eval'])
def test_selection(pop_fixture, request):
    num_of_inds = 2
    population = request.getfixturevalue(pop_fixture)
    selected_individuals = selection(types=[SelectionTypesEnum.tournament],
                                     population=population,
                                     pop_size=num_of_inds)
    assert all([ind in population for ind in selected_individuals]) and \
           len(selected_individuals) == num_of_inds


@pytest.mark.parametrize('pop_fixture', ['rand_population_gener_and_eval'])
def test_individuals_selection(pop_fixture, request):
    num_of_inds = 2
    population = request.getfixturevalue(pop_fixture)
    types = [SelectionTypesEnum.tournament]
    selected_individuals = individuals_selection(types=types,
                                                 individuals=population,
                                                 pop_size=num_of_inds)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert len(set(selected_individuals_ref)) == len(selected_individuals) \
           and len(selected_individuals) == num_of_inds
