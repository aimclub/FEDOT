import datetime
from pathlib import Path
from typing import Sequence, Optional

from fedot.core.dag.graph_utils import nodes_from_layer
from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.data.data import InputData
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.archive import ParetoFront
from fedot.core.optimisers.fitness.multi_objective_fitness import MultiObjFitness
from fedot.core.optimisers.gp_comp.evaluation import MultiprocessingDispatcher
from fedot.core.optimisers.gp_comp.gp_operators import filter_duplicates
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum, Mutation, MutationStrengthEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.objective import PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.optimisers.objective.metrics_objective import MetricsObjective
from fedot.core.optimisers.opt_history_objects.individual import Individual
from fedot.core.optimisers.timer import OptimisationTimer
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
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
    graph_params = get_pipeline_generation_params(requirements=requirements,
                                                  rules_for_constraint=DEFAULT_DAG_RULES,
                                                  task=task)
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
    node_final.parameters = {'alpha': alpha_value}
    pipeline = Pipeline(node_final)

    return pipeline


def get_requirements_and_params_for_task(task: TaskTypesEnum):
    ops = get_operations_for_task(Task(task))
    return (PipelineComposerRequirements(primary=ops, secondary=ops, max_depth=2),
            get_pipeline_generation_params(rules_for_constraint=DEFAULT_DAG_RULES, task=Task(task)))


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
    params = get_pipeline_generation_params()
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(params.adapter).dispatch(objective_eval, timer=t)
        evaluated = evaluator(population)
    assert len(evaluated) == 1
    assert evaluated[0].fitness is not None
    assert evaluated[0].fitness.valid
    assert evaluated[0].metadata['computation_time_in_seconds'] is not None

    population = [Individual(adapter.adapt(c)) for c in pipelines_to_evaluate]
    timeout = datetime.timedelta(minutes=5)
    with OptimisationTimer(timeout=timeout) as t:
        evaluator = MultiprocessingDispatcher(params.adapter).dispatch(objective_eval, timer=t)
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
