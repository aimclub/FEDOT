import datetime
import os

import numpy as np

from fedot.core.composer.advisor import PipelineChangeAdvisor
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.optimisers.objective.data_objective_builder import DataObjectiveBuilder
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation
from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.data.data import InputData
from fedot.core.log import default_log, DEFAULT_LOG_PATH
from fedot.core.optimisers.adapters import DirectAdapter, PipelineAdapter
from fedot.core.optimisers.gp_comp.gp_operators import filter_duplicates
from fedot.core.optimisers.gp_comp.individual import Individual
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum, crossover
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, _adapt_and_apply_mutations
from fedot.core.optimisers.gp_comp.operators.mutation import mutation, reduce_mutation, single_drop_mutation
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.core.optimisers.archive import ParetoFront
from fedot.core.optimisers.objective.objective import Objective

from test.unit.composer.test_composer import to_numerical
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third
from test.unit.pipelines.test_node_cache import pipeline_fourth, pipeline_fifth
from test.unit.tasks.test_regression import get_synthetic_regression_data
from test.unit.tasks.test_forecasting import get_ts_data
from test.unit.test_logger import release_log


def file_data():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/simple_classification.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = to_numerical(categorical_ids=input_data.idx)
    return input_data


def graph_example():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    graph = OptGraph()

    root_of_tree, root_child_first, root_child_second = \
        [OptNode({'name': model}) for model in ('xgboost', 'xgboost', 'knn')]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in ('logit', 'lda'):
            new_node = OptNode({'name': requirement_model})
            root_node_child.nodes_from.append(new_node)
            graph.add_node(new_node)
        graph.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    graph.add_node(root_of_tree)
    return graph


def generate_pipeline_with_single_node():
    pipeline = Pipeline()
    pipeline.add_node(PrimaryNode('knn'))

    return pipeline


def generate_so_complex_pipeline():
    node_imp = PrimaryNode('simple_imputation')
    node_lagged = SecondaryNode('lagged', nodes_from=[node_imp])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_ridge])
    node_pca = SecondaryNode('pca', nodes_from=[node_decompose])
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge, node_pca])
    pipeline = Pipeline(node_final)
    return pipeline


def pipeline_with_custom_parameters(alpha_value):
    node_scaling = PrimaryNode('scaling')
    node_norm = PrimaryNode('normalization')
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_scaling])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_norm])
    node_final = SecondaryNode('ridge', nodes_from=[node_dtreg, node_lasso])
    node_final.custom_params = {'alpha': alpha_value}
    pipeline = Pipeline(node_final)

    return pipeline


def test_nodes_from_height():
    graph = graph_example()
    found_nodes = graph.operator.nodes_from_layer(1)
    true_nodes = [node for node in graph.root_node.nodes_from]
    assert all([node_model == found_node for node_model, found_node in
                zip(true_nodes, found_nodes)])


def test_evaluate_individuals():
    project_root_path = str(fedot_project_root())
    file_path_train = os.path.join(project_root_path, 'test/data/simple_classification.csv')
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)

    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(full_path_train, task=task)
    pipelines_to_evaluate = [pipeline_first(), pipeline_second(),
                             pipeline_third(), pipeline_fourth()]

    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    objective_builder = DataObjectiveBuilder(Objective([metric_function]))
    objective_eval = objective_builder.build(dataset_to_compose)
    adapter = PipelineAdapter()

    population = [Individual(adapter.adapt(c)) for c in pipelines_to_evaluate]
    timeout = datetime.timedelta(minutes=0.001)
    params = GraphGenerationParams(adapter=PipelineAdapter(), advisor=PipelineChangeAdvisor())
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(params.adapter, timer=t).dispatch(objective_eval)
        evaluated = evaluator(population)
    assert len(evaluated) == 1
    assert evaluated[0].fitness is not None
    assert evaluated[0].fitness.valid
    assert evaluated[0].metadata['computation_time_in_seconds'] is not None

    population = [Individual(adapter.adapt(c)) for c in pipelines_to_evaluate]
    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(params.adapter, timer=t).dispatch(objective_eval)
        evaluated = evaluator(population)
    assert len(evaluated) == 4
    assert all([ind.fitness.valid for ind in evaluated])


def test_filter_duplicates():
    archive = ParetoFront()
    archive_items = [pipeline_first(), pipeline_second(), pipeline_third()]
    adapter = PipelineAdapter()

    population = [Individual(adapter.adapt(c)) for c in [pipeline_first(), pipeline_second(),
                                                         pipeline_third(), pipeline_fourth()]]
    archive_items_fitness = ((0.80001, 0.25), (0.7, 0.1), (0.9, 0.7))
    population_fitness = ((0.8, 0.25), (0.59, 0.25), (0.9, 0.7), (0.7, 0.1))
    weights = (-1, 1)
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
    adapter = PipelineAdapter()
    graph_example_first = adapter.adapt(pipeline_first())
    graph_example_second = adapter.adapt(pipeline_second())
    log = default_log(__name__)
    crossover_types = [CrossoverTypesEnum.none]
    new_graphs = crossover(crossover_types, Individual(graph_example_first),
                           Individual(graph_example_second),
                           max_depth=3, log=log, crossover_prob=1)
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second
    crossover_types = [CrossoverTypesEnum.subtree]
    new_graphs = crossover(crossover_types, Individual(graph_example_first),
                           Individual(graph_example_second),
                           max_depth=3, log=log, crossover_prob=0)
    assert new_graphs[0].graph == graph_example_first
    assert new_graphs[1].graph == graph_example_second


def test_mutation():
    adapter = PipelineAdapter()
    ind = Individual(adapter.adapt(pipeline_first()))
    mutation_types = [MutationTypesEnum.none]
    log = default_log(__name__)
    graph_gener_params = GraphGenerationParams()
    task = Task(TaskTypesEnum.classification)
    primary_model_types, _ = OperationTypesRepository().suitable_operation(task_type=task.task_type)
    secondary_model_types = ['xgboost', 'knn', 'lda', 'qda']
    composer_requirements = PipelineComposerRequirements(primary=primary_model_types,
                                                         secondary=secondary_model_types, mutation_prob=1)
    new_ind = mutation(mutation_types, graph_gener_params, ind,
                       composer_requirements, log=log, max_depth=3)
    assert new_ind.graph == ind.graph
    mutation_types = [MutationTypesEnum.growth]
    composer_requirements = PipelineComposerRequirements(primary=primary_model_types,
                                                         secondary=secondary_model_types, mutation_prob=0)
    new_ind = mutation(mutation_types, graph_gener_params, ind,
                       composer_requirements, log=log, max_depth=3)
    assert new_ind.graph == ind.graph
    ind = Individual(adapter.adapt(pipeline_fifth()))
    new_ind = mutation(mutation_types, graph_gener_params, ind,
                       composer_requirements, log=log, max_depth=3)
    assert new_ind.graph == ind.graph


def test_intermediate_add_mutation_for_linear_graph():
    """
    Tests single_add mutation can add node between two existing nodes
    """

    linear_two_nodes = OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'scaling'})]))
    nodes_from = [OptNode({'name': 'one_hot_encoding'}, [OptNode({'name': 'scaling'})])]
    linear_three_nodes_inner = OptGraph(OptNode({'name': 'logit'}, nodes_from))

    composer_requirements = PipelineComposerRequirements(primary=['scaling'],
                                                         secondary=['one_hot_encoding'], mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=DirectAdapter(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_inner = False

    for _ in range(100):
        graph_after_mutation = mutation(types=[MutationTypesEnum.single_add],
                                        params=graph_params,
                                        ind=Individual(linear_two_nodes),
                                        requirements=composer_requirements,
                                        log=default_log(__name__), max_depth=3).graph
        if not successful_mutation_inner:
            successful_mutation_inner = \
                graph_after_mutation.root_node.descriptive_id == linear_three_nodes_inner.root_node.descriptive_id
        else:
            break

    assert successful_mutation_inner


def test_parent_add_mutation_for_linear_graph():
    """
    Tests single_add mutation can add node before existing node
    """

    linear_one_node = OptGraph(OptNode({'name': 'logit'}))

    linear_two_nodes = OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'scaling'})]))

    composer_requirements = PipelineComposerRequirements(primary=['scaling'],
                                                         secondary=['logit'], mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=DirectAdapter(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_outer = False
    for _ in range(200):  # since add mutations has a lot of variations
        graph_after_mutation = mutation(types=[MutationTypesEnum.single_add],
                                        params=graph_params,
                                        ind=Individual(linear_one_node),
                                        requirements=composer_requirements,
                                        log=default_log(__name__), max_depth=2).graph
        if not successful_mutation_outer:
            successful_mutation_outer = \
                graph_after_mutation.root_node.descriptive_id == linear_two_nodes.root_node.descriptive_id
        else:
            break
    assert successful_mutation_outer


def test_edge_mutation_for_graph():
    """
    Tests edge mutation can add edge between nodes
    """
    graph_without_edge = \
        OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'one_hot_encoding'}, [OptNode({'name': 'scaling'})])]))

    primary = OptNode({'name': 'scaling'})
    graph_with_edge = \
        OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'one_hot_encoding'}, [primary]), primary]))

    composer_requirements = PipelineComposerRequirements(primary=['scaling', 'one_hot_encoding'],
                                                         secondary=['logit', 'scaling'], mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=DirectAdapter(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_edge = False
    for _ in range(100):
        graph_after_mutation = mutation(types=[MutationTypesEnum.single_edge],
                                        params=graph_params,
                                        ind=Individual(graph_without_edge),
                                        requirements=composer_requirements,
                                        log=default_log(__name__), max_depth=graph_with_edge.depth).graph
        if not successful_mutation_edge:
            successful_mutation_edge = \
                graph_after_mutation.root_node.descriptive_id == graph_with_edge.root_node.descriptive_id
        else:
            break
    assert successful_mutation_edge


def test_replace_mutation_for_linear_graph():
    """
    Tests single_change mutation can change node to another
    """
    linear_two_nodes = OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'scaling'})]))

    linear_changed = OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'one_hot_encoding'})]))

    composer_requirements = PipelineComposerRequirements(primary=['scaling', 'one_hot_encoding'],
                                                         secondary=['logit'], mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=DirectAdapter(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_replace = False
    for _ in range(100):
        graph_after_mutation = mutation(types=[MutationTypesEnum.single_change],
                                        params=graph_params,
                                        ind=Individual(linear_two_nodes),
                                        requirements=composer_requirements,
                                        log=default_log(__name__), max_depth=2).graph
        if not successful_mutation_replace:
            successful_mutation_replace = \
                graph_after_mutation.root_node.descriptive_id == linear_changed.root_node.descriptive_id
        else:
            break
    assert successful_mutation_replace


def test_drop_mutation_for_linear_graph():
    """
    Tests single_drop mutation can remove node
    """

    linear_two_nodes = OptGraph(OptNode({'name': 'logit'}, [OptNode({'name': 'scaling'})]))

    linear_one_node = OptGraph(OptNode({'name': 'logit'}))

    composer_requirements = PipelineComposerRequirements(primary=['scaling'],
                                                         secondary=['logit'], mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=DirectAdapter(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_drop = False
    for _ in range(100):
        graph_after_mutation = mutation(types=[MutationTypesEnum.single_drop],
                                        params=graph_params,
                                        ind=Individual(linear_two_nodes),
                                        requirements=composer_requirements,
                                        log=default_log(__name__), max_depth=2).graph
        if not successful_mutation_drop:
            successful_mutation_drop = \
                graph_after_mutation.root_node.descriptive_id == linear_one_node.root_node.descriptive_id
        else:
            break
    assert successful_mutation_drop


def test_boosting_mutation_for_linear_graph():
    """
    Tests boosting mutation can add correct boosting cascade
    """

    linear_one_node = OptGraph(OptNode({'name': 'knn'}, [OptNode({'name': 'scaling'})]))

    init_node = OptNode({'name': 'scaling'})
    model_node = OptNode({'name': 'knn'}, [init_node])

    boosting_graph = \
        OptGraph(
            OptNode({'name': 'logit'},
                    [model_node, OptNode({'name': 'linear', },
                                         [OptNode({'name': 'class_decompose'},
                                                  [model_node, init_node])])]))

    available_operations = [node.content['name'] for node in boosting_graph.nodes]
    composer_requirements = PipelineComposerRequirements(primary=available_operations,
                                                         secondary=available_operations, mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=PipelineAdapter(),
                                         advisor=PipelineChangeAdvisor(task=Task(TaskTypesEnum.classification)),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_boosting = False
    for _ in range(100):
        if not successful_mutation_boosting:
            graph_after_mutation = mutation(types=[boosting_mutation],
                                            params=graph_params,
                                            ind=Individual(linear_one_node),
                                            requirements=composer_requirements,
                                            log=default_log(__name__), max_depth=2).graph
            successful_mutation_boosting = \
                graph_after_mutation.root_node.descriptive_id == boosting_graph.root_node.descriptive_id
        else:
            break
    assert successful_mutation_boosting

    # check that obtained pipeline can be fitted
    pipeline = PipelineAdapter().restore(graph_after_mutation)
    data = file_data()
    pipeline.fit(data)
    result = pipeline.predict(data)
    assert result is not None


def test_boosting_mutation_for_non_lagged_ts_model():
    """
    Tests boosting mutation can add correct boosting cascade for ts forecasting with non-lagged model
    """
    linear_two_nodes = OptGraph(OptNode({'name': 'clstm'},
                                        nodes_from=[OptNode({'name': 'smoothing'})]))

    init_node = OptNode({'name': 'smoothing'})
    model_node = OptNode({'name': 'clstm'}, nodes_from=[init_node])
    lagged_node = OptNode({'name': 'lagged'}, nodes_from=[init_node])

    boosting_graph = \
        OptGraph(
            OptNode({'name': 'ridge'},
                    [model_node, OptNode({'name': 'ridge', },
                                         [OptNode({'name': 'decompose'},
                                                  [model_node, lagged_node])])]))
    adapter = PipelineAdapter()
    # to ensure hyperparameters of custom models
    boosting_graph = adapter.adapt(adapter.restore(boosting_graph))

    available_operations = [node.content['name'] for node in boosting_graph.nodes]
    composer_requirements = PipelineComposerRequirements(primary=available_operations,
                                                         secondary=available_operations, mutation_prob=1)

    graph_params = GraphGenerationParams(adapter=adapter,
                                         advisor=PipelineChangeAdvisor(
                                             task=Task(TaskTypesEnum.ts_forecasting)),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    successful_mutation_boosting = False
    for _ in range(100):
        if not successful_mutation_boosting:
            graph_after_mutation = mutation(types=[boosting_mutation],
                                            params=graph_params,
                                            ind=Individual(linear_two_nodes),
                                            requirements=composer_requirements,
                                            log=default_log(__name__), max_depth=2).graph
            successful_mutation_boosting = \
                graph_after_mutation.root_node.descriptive_id == boosting_graph.root_node.descriptive_id
        else:
            break
    assert successful_mutation_boosting

    # check that obtained pipeline can be fitted
    pipeline = PipelineAdapter().restore(graph_after_mutation)
    data_train, data_test = get_ts_data()
    pipeline.fit(data_train)
    result = pipeline.predict(data_test)
    assert result is not None


def test_pipeline_adapters_params_correct():
    """ Checking the correct conversion of hyperparameters in nodes when nodes
    are passing through adapter
    """
    init_alpha = 12.1
    pipeline = pipeline_with_custom_parameters(init_alpha)

    # Convert into OptGraph object
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    # Get Pipeline object back
    restored_pipeline = adapter.restore(opt_graph)
    # Get hyperparameter value after pipeline restoration
    restored_alpha = restored_pipeline.root_node.custom_params['alpha']
    assert np.isclose(init_alpha, restored_alpha)


def test_preds_before_and_after_convert_equal():
    """ Check if the pipeline predictions change before and after conversion
    through the adapter
    """
    init_alpha = 12.1
    pipeline = pipeline_with_custom_parameters(init_alpha)

    # Generate data
    input_data = get_synthetic_regression_data(n_samples=10, n_features=2,
                                               random_state=2021)
    # Init fit
    pipeline.fit(input_data)
    init_preds = pipeline.predict(input_data)

    # Convert into OptGraph object
    adapter = PipelineAdapter()
    opt_graph = adapter.adapt(pipeline)
    restored_pipeline = adapter.restore(opt_graph)

    # Restored pipeline fit
    restored_pipeline.fit(input_data)
    restored_preds = restored_pipeline.predict(input_data)

    assert np.array_equal(init_preds.predict, restored_preds.predict)


def test_crossover_with_single_node():
    adapter = PipelineAdapter()
    graph_example_first = adapter.adapt(generate_pipeline_with_single_node())
    graph_example_second = adapter.adapt(generate_pipeline_with_single_node())
    log = default_log(__name__)
    graph_params = GraphGenerationParams(adapter=adapter, advisor=PipelineChangeAdvisor(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)

    for crossover_type in CrossoverTypesEnum:
        new_graphs = crossover([crossover_type], Individual(graph_example_first), Individual(graph_example_second),
                               params=graph_params, max_depth=3, log=log, crossover_prob=1)

        assert new_graphs[0].graph == graph_example_first
        assert new_graphs[1].graph == graph_example_second


def test_mutation_with_single_node():
    adapter = PipelineAdapter()
    graph = adapter.adapt(generate_pipeline_with_single_node())
    task = Task(TaskTypesEnum.classification)
    available_model_types, _ = OperationTypesRepository().suitable_operation(task_type=task.task_type)

    graph_params = GraphGenerationParams(adapter=adapter, advisor=PipelineChangeAdvisor(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)

    composer_requirements = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                                         max_arity=3, max_depth=3, pop_size=5, num_of_generations=4,
                                                         crossover_prob=.8, mutation_prob=1)
    new_graph = reduce_mutation(graph, composer_requirements)
    assert graph == new_graph

    new_graph = single_drop_mutation(graph, graph_params)
    assert graph == new_graph


def test_no_opt_or_graph_nodes_after_mutation():
    test_file_path = str(os.path.dirname(__file__))
    test_log = default_log('test_no_opt_or_graph_nodes_after_mutation')

    adapter = PipelineAdapter()
    graph = adapter.adapt(generate_pipeline_with_single_node())
    task = Task(TaskTypesEnum.classification)
    mutation_types = [MutationTypesEnum.growth]
    mutation_prob = 1
    available_model_types, _ = OperationTypesRepository().suitable_operation(task_type=task.task_type)
    composer_requirements = PipelineComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                                         max_arity=3, max_depth=3, pop_size=5, num_of_generations=4,
                                                         crossover_prob=.8, mutation_prob=1)
    graph_params = GraphGenerationParams(adapter=adapter, advisor=PipelineChangeAdvisor(),
                                         rules_for_constraint=DEFAULT_DAG_RULES)
    _adapt_and_apply_mutations(new_graph=graph,
                               mutation_prob=mutation_prob,
                               types=mutation_types,
                               num_mut=1,
                               requirements=composer_requirements,
                               params=graph_params,
                               max_depth=2)

    if os.path.exists(DEFAULT_LOG_PATH):
        with open(DEFAULT_LOG_PATH, 'r') as file:
            content = file.readlines()

    # Is there a required message in the logs
    assert not any('Unexpected: GraphNode found in PipelineAdapter instead' in log_message for log_message in content)
    # assert not any('Unexpected: OptNode found in PipelineAdapter instead' in log_message for log_message in content)


def test_no_opt_or_graph_nodes_after_adapt_so_complex_graph():
    test_file_path = str(os.path.dirname(__file__))
    test_log = default_log('test_no_opt_in_complex_graph')

    adapter = PipelineAdapter()
    pipeline = generate_so_complex_pipeline()
    adapter.adapt(pipeline)

    if os.path.exists(DEFAULT_LOG_PATH):
        with open(DEFAULT_LOG_PATH, 'r') as file:
            content = file.readlines()

    # Is there a required message in the logs
    assert not any('Unexpected: GraphNode found in PipelineAdapter instead' in log_message for log_message in content)
    # assert not any('Unexpected: OptNode found in PipelineAdapter instead' in log_message for log_message in content)
