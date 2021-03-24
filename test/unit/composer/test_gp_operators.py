import datetime
import os
from functools import partial

from deap import tools

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.constraint import constraint_function
from fedot.core.composer.gp_composer.gp_composer import ChainGenerationParams, GPComposerBuilder, \
    GPComposerRequirements, sample_split_ration_for_tasks
from fedot.core.composer.optimisers.gp_comp.gp_operators import evaluate_individuals, filter_duplicates, \
    nodes_from_height
from fedot.core.composer.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum, mutation
from fedot.core.composer.optimisers.utils.multi_objective_fitness import MultiObjFitness
from fedot.core.composer.timer import CompositionTimer
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.log import default_log
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import project_root
from test.unit.chains.test_node_cache import chain_fifth, chain_first, chain_fourth, chain_second, chain_third


def chain_example():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [SecondaryNode(model) for model in ('xgboost', 'xgboost',
                                            'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = PrimaryNode(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def test_nodes_from_height():
    chain = chain_example()
    found_nodes = nodes_from_height(chain, 1)
    true_nodes = [node for node in chain.root_node.nodes_from]
    assert all([node_model == found_node for node_model, found_node in
                zip(true_nodes, found_nodes)])


def test_evaluate_individuals():
    project_root_path = str(project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    full_path_train = os.path.join(str(project_root()), file_path_train)

    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(full_path_train, task=task)
    available_model_types, _ = OperationTypesRepository().suitable_operation(task_type=task.task_type)

    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    composer_requirements = GPComposerRequirements(primary=available_model_types,
                                                   secondary=available_model_types)

    builder = GPComposerBuilder(task=task).with_requirements(composer_requirements). \
        with_metrics(metric_function)

    composer = builder.build()

    train_data, test_data = train_test_data_setup(dataset_to_compose,
                                                  sample_split_ration_for_tasks[dataset_to_compose.task.task_type])
    metric_function_for_nodes = partial(composer.composer_metric, composer.metrics, train_data, test_data)
    population = [chain_first(), chain_second(), chain_third(), chain_fourth()]
    max_lead_time = datetime.timedelta(minutes=0.001)
    with CompositionTimer(max_lead_time=max_lead_time) as t:
        evaluate_individuals(individuals_set=population, objective_function=metric_function_for_nodes,
                             is_multi_objective=False, timer=t)
    assert len(population) == 1
    assert population[0].fitness is not None

    population = [chain_first(), chain_second(), chain_third(), chain_fourth()]
    max_lead_time = datetime.timedelta(minutes=5)
    with CompositionTimer(max_lead_time=max_lead_time) as t:
        evaluate_individuals(individuals_set=population, objective_function=metric_function_for_nodes,
                             is_multi_objective=False, timer=t)
    assert len(population) == 4
    assert all([ind.fitness is not None for ind in population])


def test_filter_duplicates():
    archive = tools.ParetoFront()
    archive_items = [chain_first(), chain_second(), chain_third()]
    population = [chain_first(), chain_second(), chain_third(), chain_fourth()]
    archive_items_fitness = ((-0.80001, 0.25), (-0.7, 0.1), (-0.9, 0.7))
    population_fitness = ((-0.8, 0.25), (-0.59, 0.25), (-0.9, 0.7), (-0.7, 0.1))
    weights = tuple([-1 for _ in range(len(population_fitness[0]))])
    for ind_num in range(len(archive_items)):
        archive_items[ind_num].fitness = MultiObjFitness(values=archive_items_fitness[ind_num], weights=weights)
    for ind_num in range(len(population)):
        population[ind_num].fitness = MultiObjFitness(values=population_fitness[ind_num], weights=weights)
    archive.update(archive_items)
    filtered_archive = filter_duplicates(archive, population)
    assert len(filtered_archive) == 1
    assert filtered_archive[0].fitness.values[0] == -0.80001
    assert filtered_archive[0].fitness.values[1] == 0.25


def test_crossover():
    chain_example_first = chain_first()
    chain_example_second = chain_second()
    log = default_log(__name__)
    crossover_types = [CrossoverTypesEnum.none]
    new_chains = crossover(crossover_types, chain_example_first, chain_example_second, max_depth=3, log=log,
                           crossover_prob=1)
    assert new_chains[0] == chain_example_first
    assert new_chains[1] == chain_example_second
    crossover_types = [CrossoverTypesEnum.subtree]
    new_chains = crossover(crossover_types, chain_example_first, chain_example_second, max_depth=3, log=log,
                           crossover_prob=0)
    assert new_chains[0] == chain_example_first
    assert new_chains[1] == chain_example_second


def test_mutation():
    chain = chain_first()
    mutation_types = [MutationTypesEnum.none]
    log = default_log(__name__)
    chain_gener_params = ChainGenerationParams()
    task = Task(TaskTypesEnum.classification)
    primary_model_types, _ = OperationTypesRepository().suitable_operation(task_type=task.task_type)
    secondary_model_types = ['xgboost', 'knn', 'lda', 'qda']
    composer_requirements = GPComposerRequirements(primary=primary_model_types,
                                                   secondary=secondary_model_types, mutation_prob=1)
    new_chain = mutation(mutation_types, chain_gener_params, chain, composer_requirements, log=log, max_depth=3)
    assert new_chain == chain
    mutation_types = [MutationTypesEnum.growth]
    composer_requirements = GPComposerRequirements(primary=primary_model_types,
                                                   secondary=secondary_model_types, mutation_prob=0)
    new_chain = mutation(mutation_types, chain_gener_params, chain, composer_requirements, log=log, max_depth=3)
    assert new_chain == chain
    chain = chain_fifth()
    assert not constraint_function(chain)
    new_chain = mutation(mutation_types, chain_gener_params, chain, composer_requirements, log=log, max_depth=3)
    assert new_chain == chain
