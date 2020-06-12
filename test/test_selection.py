import os
from functools import partial
from core.utils import project_root
from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.model import ModelTypesIdsEnum
from core.composer.gp_composer.gp_composer import GPComposerRequirements
from core.composer.optimisers.gp_operators import random_chain
from core.models.data import InputData
from core.repository.quality_metrics_repository import MetricsRepository, ClassificationMetricsEnum
from core.composer.optimisers.selection import tournament_selection, individuals_selection, SelectionTypesEnum


def random_population(pop_size):
    models_set = [ModelTypesIdsEnum.knn, ModelTypesIdsEnum.logit, ModelTypesIdsEnum.rf]
    requirements = GPComposerRequirements(primary=models_set, secondary=models_set, max_depth=1)
    random_chain_function = partial(random_chain, chain_class=Chain, secondary_node_func=NodeGenerator.secondary_node,
                                    primary_node_func=NodeGenerator.primary_node, requirements=requirements)
    return [random_chain_function() for _ in range(pop_size)]


def obj_function(metric_function, train_data: InputData, test_data: InputData, chain: Chain) -> float:
    chain.fit(input_data=train_data)
    return metric_function(chain, test_data)


def test_selection():
    pop_size = 5
    population = random_population(pop_size=pop_size)
    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)
    train_file_path = 'cases/data/scoring/scoring_train.csv'
    train_file_path = os.path.join(str(project_root()), train_file_path)
    test_file_path = 'cases/data/scoring/scoring_test.csv'
    test_file_path = os.path.join(str(project_root()), test_file_path)
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)
    objective_function = partial(obj_function, metric_function, train_data, test_data)
    for ind in population:
        ind.fitness = objective_function(ind)

    best_fit_in_population = min([ind.fitness for ind in population])
    selected_individuals = tournament_selection(individuals=population, pop_size=1, fraction=1)
    assert best_fit_in_population == selected_individuals[0].fitness
    selected_individuals = tournament_selection(individuals=population, pop_size=3, fraction=0.5)
    assert len(selected_individuals) == 3
    assert all([ind in population for ind in selected_individuals])
    selected_individuals = individuals_selection(types=[SelectionTypesEnum.tournament], individuals=population,
                                                 pop_size=3)
    selected_individuals_ref = [str(ind) for ind in selected_individuals]
    assert len(set(selected_individuals_ref)) == len(selected_individuals)
