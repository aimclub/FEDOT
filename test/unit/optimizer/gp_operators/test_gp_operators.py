import datetime
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Optional

from golem.core.dag.graph_utils import nodes_from_layer
from golem.core.optimisers.archive import ParetoFront
from golem.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from golem.core.optimisers.genetic.evaluation import MultiprocessingDispatcher
from golem.core.optimisers.genetic.gp_operators import filter_duplicates, replace_subtrees
from golem.core.optimisers.genetic.gp_params import GPGraphOptimizerParameters
from golem.core.optimisers.genetic.operators.mutation import MutationTypesEnum, Mutation, MutationStrengthEnum
from golem.core.optimisers.genetic.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.timer import OptimisationTimer

from fedot.core.data.data import InputData
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from test.unit.composer.test_composer import to_numerical
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_second, pipeline_third
from test.unit.pipelines.test_node_cache import pipeline_fourth


def get_mutation_operator(mutation_types: Sequence[MutationTypesEnum],
                          requirements: Optional[PipelineComposerRequirements] = None,
                          task: Optional[Task] = None,
                          mutation_prob: float = 1.0,
                          mutation_strength: Optional[MutationStrengthEnum] = MutationStrengthEnum.mean):
    if not requirements:
        operations = get_operations_for_task(task)
        requirements = PipelineComposerRequirements(primary=operations, secondary=operations)
    graph_params = get_pipeline_generation_params(requirements=requirements, task=task)
    parameters = GPGraphOptimizerParameters(mutation_types=mutation_types,
                                            mutation_prob=mutation_prob,
                                            mutation_strength=mutation_strength)
    mutation = Mutation(parameters, requirements, graph_params)
    return mutation


def file_data():
    test_file_path = Path(__file__).parents[3].joinpath('data', 'simple_classification.csv')
    input_data = InputData.from_csv(test_file_path)
    input_data.idx = to_numerical(categorical_ids=input_data.idx)
    return input_data


def graph_example():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    graph = OptGraph()

    root_of_tree, root_child_first, root_child_second = [OptNode({'name': model}) for model in
                                                         ('xgboost', 'xgboost', 'knn')]

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
    pipeline.add_node(PipelineNode('knn'))

    return pipeline


def generate_so_complex_pipeline():
    node_imp = PipelineNode('simple_imputation')
    node_lagged = PipelineNode('lagged', nodes_from=[node_imp])
    node_ridge = PipelineNode('ridge', nodes_from=[node_lagged])
    node_decompose = PipelineNode('decompose', nodes_from=[node_lagged, node_ridge])
    node_pca = PipelineNode('pca', nodes_from=[node_decompose])
    node_final = PipelineNode('ridge', nodes_from=[node_ridge, node_pca])
    pipeline = Pipeline(node_final)
    return pipeline


def pipeline_with_custom_parameters(alpha_value):
    node_scaling = PipelineNode('scaling')
    node_norm = PipelineNode('normalization')
    node_dtreg = PipelineNode('dtreg', nodes_from=[node_scaling])
    node_lasso = PipelineNode('lasso', nodes_from=[node_norm])
    node_final = PipelineNode('ridge', nodes_from=[node_dtreg, node_lasso])
    node_final.parameters = {'alpha': alpha_value}
    pipeline = Pipeline(node_final)

    return pipeline


def get_requirements_and_params_for_task(task: TaskTypesEnum):
    ops = get_operations_for_task(Task(task))
    req = PipelineComposerRequirements(primary=ops, secondary=ops, max_depth=2)
    gen_params = get_pipeline_generation_params(requirements=req, task=Task(task))
    return req, gen_params


def test_nodes_from_height():
    graph = graph_example()
    found_nodes = nodes_from_layer(graph, 1)
    true_nodes = [node for node in graph.root_node.nodes_from]
    assert all([node_model == found_node for node_model, found_node in
                zip(true_nodes, found_nodes)])


def test_evaluate_individuals():
    file_path_train = Path(fedot_project_root(), 'test', 'data', 'simple_classification.csv')
    full_path_train = Path(fedot_project_root(), file_path_train)

    task = Task(TaskTypesEnum.classification)
    dataset_to_compose = InputData.from_csv(full_path_train, task=task)
    pipelines_to_evaluate = [pipeline_first(), pipeline_second(),
                             pipeline_third(), pipeline_fourth()]

    metric_function = ClassificationMetricsEnum.ROCAUC_penalty
    objective = MetricsObjective(metric_function)
    data_source = DataSourceSplitter().build(dataset_to_compose)
    objective_eval = PipelineObjectiveEvaluate(objective, data_source)
    adapter = PipelineAdapter()

    population = [Individual(adapter.adapt(c)) for c in pipelines_to_evaluate]
    timeout = datetime.timedelta(minutes=0.001)
    adapter = PipelineAdapter()
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(adapter).dispatch(objective_eval, timer=t)
        evaluated = evaluator(population)
    assert len(evaluated) == 1
    assert evaluated[0].fitness is not None
    assert evaluated[0].fitness.valid
    assert evaluated[0].metadata['computation_time_in_seconds'] is not None

    population = [Individual(adapter.adapt(c)) for c in pipelines_to_evaluate]
    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(adapter).dispatch(objective_eval, timer=t)
        evaluated = evaluator(population)
    assert len(evaluated) == 4
    assert all([ind.fitness.valid for ind in evaluated])


def test_filter_duplicates():
    archive = ParetoFront()
    adapter = PipelineAdapter()

    archive_items = [Individual(adapter.adapt(p)) for p in [pipeline_first(), pipeline_second(), pipeline_third()]]
    population = [Individual(adapter.adapt(p)) for p in [pipeline_first(), pipeline_second(),
                                                         pipeline_third(), pipeline_fourth()]]
    archive_items_fitness = ((0.80001, 0.25), (0.7, 0.1), (0.9, 0.7))
    population_fitness = ((0.8, 0.25), (0.59, 0.25), (0.9, 0.7), (0.7, 0.1))
    weights = (-1, 1)
    for ind_num in range(len(archive_items)):
        archive_items[ind_num].set_evaluation_result(
            MultiObjFitness(values=archive_items_fitness[ind_num], weights=weights))
    for ind_num in range(len(population)):
        population[ind_num].set_evaluation_result(MultiObjFitness(values=population_fitness[ind_num], weights=weights))
    archive.update(archive_items)
    filtered_archive = filter_duplicates(archive, population)
    assert len(filtered_archive) == 1
    assert filtered_archive[0].fitness.values[0] == -0.80001
    assert filtered_archive[0].fitness.values[1] == 0.25


def test_replace_subtree():
    # graph with depth = 3
    pipeline_1 = pipeline_first()
    passed_pipeline_1 = deepcopy(pipeline_1)
    # graph with depth = 2
    pipeline_2 = pipeline_third()

    # choose the first layer of the first graph
    layer_in_first = pipeline_1.depth - 1
    # choose the last layer of the second graph
    layer_in_second = 0
    max_depth = 3

    node_from_graph_first = nodes_from_layer(pipeline_1, layer_in_first)[0]
    node_from_graph_second = nodes_from_layer(pipeline_2, layer_in_second)[0]

    # replace_subtrees must not replace subgraph in the first graph and its depth must be <= max_depth
    replace_subtrees(pipeline_1, pipeline_2, node_from_graph_first, node_from_graph_second,
                     layer_in_first, layer_in_second, max_depth)
    assert pipeline_1.depth <= max_depth
    assert pipeline_1 == passed_pipeline_1
    assert pipeline_2.depth <= max_depth
