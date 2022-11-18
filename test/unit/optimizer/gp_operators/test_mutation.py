from copy import deepcopy

import pytest
from golem.core.dag.graph_node import GraphNode
from golem.core.dag.verification_rules import DEFAULT_DAG_RULES
from golem.core.optimisers.genetic.gp_params import GPGraphOptimizerParameters
from golem.core.optimisers.genetic.operators.mutation import Mutation, MutationStrengthEnum, MutationTypesEnum
from golem.core.optimisers.genetic.pipeline_composer_requirements import PipelineComposerRequirements
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimizer import GraphGenerationParams

from fedot.core.composer.gp_composer.specific_operators import boosting_mutation
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.unit.dag.test_graph_utils import find_first
from test.unit.optimizer.gp_operators.test_gp_operators import get_requirements_and_params_for_task, file_data, \
    get_mutation_operator
from test.unit.pipelines.test_node_cache import pipeline_first, pipeline_fifth
from test.unit.tasks.test_forecasting import get_ts_data


def get_mutation_obj() -> Mutation:
    """
    Function for initializing mutation interface
    """
    task = Task(TaskTypesEnum.classification)
    operations = ['logit', 'normalization']

    requirements = PipelineComposerRequirements(primary=operations, secondary=operations)

    graph_params = get_pipeline_generation_params(requirements=requirements,
                                                  rules_for_constraint=DEFAULT_DAG_RULES,
                                                  task=task)
    parameters = GPGraphOptimizerParameters(mutation_strength=MutationStrengthEnum.strong,
                                            mutation_prob=1)

    mutation = Mutation(parameters, requirements, graph_params)
    return mutation


def get_simple_linear_graph() -> OptGraph:
    """
    Returns simple linear graph
    """
    pipeline = PipelineBuilder().add_node('scaling').add_node('poly_features').add_node('rf').build()
    return PipelineAdapter().adapt(pipeline)


def get_simple_linear_boosting_pipeline() -> Pipeline:
    """
    Returns simple linear graph with boosting
    """
    node_scaling = PipelineNode('scaling')
    node_pf = PipelineNode('poly_features', nodes_from=[node_scaling])
    node_rf = PipelineNode('rf', nodes_from=[node_pf])
    node_decompose = PipelineNode('class_decompose', nodes_from=[node_pf, node_rf])
    node_linear = PipelineNode('linear', nodes_from=[node_decompose])
    final_node = PipelineNode('logit', nodes_from=[node_linear, node_rf])
    pipeline = Pipeline(final_node)
    return pipeline


def get_ts_forecasting_graph() -> OptGraph:
    """
    Returns simple linear graph for ts forecasting
    """
    pipeline = PipelineBuilder().add_node('smoothing').add_node('ar').build()

    return PipelineAdapter().adapt(pipeline)


def get_ts_forecasting_graph_with_boosting() -> Pipeline:
    """
    Returns simple linear graph for ts forecasting with boosting
    """
    node_init = PipelineNode('smoothing')
    node_model = PipelineNode('ar', nodes_from=[node_init])
    node_lagged = PipelineNode('lagged', nodes_from=[node_init])
    node_decompose = PipelineNode('decompose',
                                   [node_model, node_lagged])
    node_ridge = PipelineNode('ridge', nodes_from=[node_decompose])
    node_final = PipelineNode('ridge', nodes_from=[node_ridge, node_model])
    pipeline = Pipeline(node_final)
    return pipeline


def get_tree_graph() -> OptGraph:
    """
    Returns tree graph
    scaling->poly_features->rf
    pca_____/
    """
    pipeline = PipelineBuilder().add_node('scaling') \
        .add_branch('pca', branch_idx=1) \
        .join_branches('poly_features').add_node('rf').build()
    return PipelineAdapter().adapt(pipeline)


def test_mutation_none():
    mutation = get_mutation_obj()
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    new_graph = mutation._no_mutation(new_graph)
    assert new_graph == graph


def test_simple_mutation():
    """
    Test correctness of simple mutation
    """
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    mutation = get_mutation_obj()
    new_graph = mutation._simple_mutation(new_graph)
    for i in range(len(graph.nodes)):
        assert graph.nodes[i] != new_graph.nodes[i]


def test_drop_node():
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    mutation = get_mutation_obj()
    for _ in range(5):
        new_graph = mutation._single_drop_mutation(new_graph)
    assert len(new_graph) < len(graph)


def test_add_as_parent_node_linear():
    """
    Test correctness of adding as a parent in simple case
    """
    # 1-poly_features
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_separate_parent_node(new_graph, node_to_mutate)
    assert len(node_to_mutate.nodes_from) > len(graph.nodes[1].nodes_from)


def test_add_as_parent_node_tree():
    """
    Test correctness of adding as a parent in complex case
    """
    graph = get_tree_graph()
    new_graph = deepcopy(graph)
    # 1-poly_features
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_separate_parent_node(new_graph, node_to_mutate)
    assert len(node_to_mutate.nodes_from) > len(graph.nodes[1].nodes_from)


def test_add_as_child_node_linear():
    """
    Test correctness of adding as a child in simple case
    """
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    # 1-poly_features
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_as_child(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert new_graph.node_children(node_to_mutate) != graph.node_children(node_to_mutate)


def test_add_as_child_node_tree():
    """
    Test correctness of adding as a child in complex case
    """

    graph = get_tree_graph()
    new_graph = deepcopy(graph)
    # 2-scaling
    node_to_mutate = new_graph.nodes[2]
    mutation = get_mutation_obj()
    mutation._add_as_child(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert new_graph.node_children(node_to_mutate) != graph.node_children(node_to_mutate)


def test_add_as_intermediate_node_linear():
    """
    Test correctness of adding as an intermediate node in simple case
    """
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    # 1-poly_features
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_intermediate_node(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert node_to_mutate.nodes_from[0] != graph.nodes[1].nodes_from[0]


def test_add_as_intermediate_node_tree():
    """
    Test correctness of adding as intermediate node in complex case
    """
    graph = get_tree_graph()
    new_graph = deepcopy(graph)
    # 1-poly_features
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_intermediate_node(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert node_to_mutate.nodes_from[0] != graph.nodes[1].nodes_from[0]


def test_edge_mutation_for_graph():
    """
    Tests edge mutation can add edge between nodes
    """

    graph_without_edge = get_simple_linear_graph()
    mutation = get_mutation_obj()
    graph_with_edge = mutation._single_edge_mutation(graph_without_edge)
    # 0-rf
    assert graph_with_edge.nodes[0].nodes_from == graph_with_edge.nodes[1:]


def test_replace_mutation_for_linear_graph():
    """
    Tests single_change mutation can change node to another
    """
    graph = get_simple_linear_graph()
    new_graph = deepcopy(graph)
    mutation = get_mutation_obj()

    new_graph = mutation._single_change_mutation(new_graph)
    assert graph.descriptive_id != new_graph.descriptive_id


def test_boosting_mutation_for_linear_graph():
    """
    Tests boosting mutation can add correct boosting cascade
    """

    graph = PipelineAdapter().restore(get_simple_linear_graph())
    boosting_graph = get_simple_linear_boosting_pipeline()
    requirements = PipelineComposerRequirements(primary=['logit'],
                                                secondary=['logit'])
    pipeline = boosting_mutation(graph,
                                 requirements,
                                 get_pipeline_generation_params(requirements=requirements,
                                                                rules_for_constraint=DEFAULT_DAG_RULES,
                                                                task=Task(TaskTypesEnum.classification)))
    data = file_data()
    pipeline.fit(data)
    result = pipeline.predict(data)
    assert pipeline.descriptive_id == boosting_graph.descriptive_id
    assert result is not None


def test_boosting_mutation_for_non_lagged_ts_model():
    """
    Tests boosting mutation can add correct boosting cascade for ts forecasting with non-lagged model
    """

    graph = PipelineAdapter().restore(get_ts_forecasting_graph())

    boosting_graph = get_ts_forecasting_graph_with_boosting()
    requirements = PipelineComposerRequirements(primary=['ridge'],
                                                secondary=['ridge'])
    pipeline = boosting_mutation(graph,
                                 requirements,
                                 get_pipeline_generation_params(requirements=requirements,
                                                                rules_for_constraint=DEFAULT_DAG_RULES,
                                                                task=Task(TaskTypesEnum.ts_forecasting)))
    data_train, data_test = get_ts_data()
    pipeline.fit(data_train)
    result = pipeline.predict(data_test)
    assert boosting_graph.descriptive_id == pipeline.descriptive_id
    assert result is not None


@pytest.mark.parametrize('pipeline, requirements, params',
                         [(PipelineBuilder().add_node('scaling').add_node('rf').build(),
                           *get_requirements_and_params_for_task(TaskTypesEnum.classification)),
                          (PipelineBuilder().add_node('smoothing').add_node('ar').build(),
                           *get_requirements_and_params_for_task(TaskTypesEnum.ts_forecasting))
                          ])
def test_boosting_mutation_changes_pipeline(pipeline: Pipeline, requirements: PipelineComposerRequirements,
                                            params: GraphGenerationParams):
    new_pipeline = deepcopy(pipeline)
    new_pipeline = boosting_mutation(new_pipeline, requirements, params)
    assert new_pipeline.descriptive_id != pipeline.descriptive_id
    assert 'class_decompose' in new_pipeline.descriptive_id or 'decompose' in new_pipeline.descriptive_id


def test_mutation_with_single_node():
    adapter = PipelineAdapter()
    graph = adapter.adapt(PipelineBuilder().add_node('rf').build())
    new_graph = deepcopy(graph)
    mutation = get_mutation_obj()
    new_graph = mutation._reduce_mutation(new_graph)

    assert graph == new_graph
    new_graph = mutation._single_drop_mutation(new_graph)
    assert graph == new_graph


def test_mutation_with_zero_prob():
    adapter = PipelineAdapter()
    ind = Individual(adapter.adapt(pipeline_first()))
    task = Task(TaskTypesEnum.classification)
    primary_model_types = OperationTypesRepository().suitable_operation(task_type=task.task_type)
    secondary_model_types = ['xgboost', 'knn', 'lda', 'qda']
    composer_requirements = PipelineComposerRequirements(primary=primary_model_types,
                                                         secondary=secondary_model_types)
    for mutation_type in MutationTypesEnum:
        mutation = get_mutation_operator([mutation_type], composer_requirements, mutation_prob=0)
        new_ind = mutation(ind)
        assert new_ind.graph == ind.graph
        ind = Individual(adapter.adapt(pipeline_fifth()))
        new_ind = mutation(ind)
        assert new_ind.graph == ind.graph


def test_no_opt_or_graph_nodes_after_mutation():
    adapter = PipelineAdapter()
    graph = get_simple_linear_graph()
    mutation = get_mutation_obj()
    new_graph, _ = mutation._adapt_and_apply_mutations(new_graph=graph, num_mut=1)
    new_pipeline = adapter.restore(new_graph)

    assert not find_first(new_pipeline, lambda n: type(n) in (GraphNode, OptNode))
