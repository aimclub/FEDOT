from copy import deepcopy

from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.graph import OptGraph
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.mutation import Mutation, MutationStrengthEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.pipelines.pipeline_graph_generation_params import get_pipeline_generation_params
from fedot.core.repository.operation_types_repository import get_operations_for_task
from fedot.core.repository.tasks import Task, TaskTypesEnum


def get_mutation_obj() -> Mutation:
    """
    Function for initializing mutation interface
    """
    task = Task(TaskTypesEnum.classification)
    operations = get_operations_for_task(task)
    operations.remove('rf')
    operations.remove('poly_features')
    operations.remove('scaling')
    requirements = PipelineComposerRequirements(primary=operations, secondary=operations)

    graph_params = get_pipeline_generation_params(requirements=requirements,
                                                  rules_for_constraint=DEFAULT_DAG_RULES,
                                                  task=task)
    parameters = GPGraphOptimizerParameters(mutation_strength=MutationStrengthEnum.strong,
                                            mutation_prob=1)

    mutation = Mutation(parameters, requirements, graph_params)
    return mutation


def get_simple_linear_pipeline() -> OptGraph:
    """
    Returns simple linear graph
    """
    pipeline = PipelineBuilder().add_node('scaling').add_node('poly_features').add_node('rf').to_pipeline()
    return PipelineAdapter().adapt(pipeline)


def get_tree_pipeline() -> OptGraph:
    """
    Returns tree graph
    scaling->poly_features->rf
    pca_____/
    """
    pipeline = PipelineBuilder().add_node('scaling') \
        .add_branch('pca', branch_idx=1) \
        .join_branches('poly_features').add_node('rf').to_pipeline()
    return PipelineAdapter().adapt(pipeline)


def test_simple_mutation():
    """
    Test correctness of simple mutation
    """
    graph = get_simple_linear_pipeline()
    new_graph = deepcopy(graph)
    mutation = get_mutation_obj()
    new_graph = mutation._simple_mutation(new_graph)
    for i in range(len(graph.nodes)):
        assert graph.nodes[i] != new_graph.nodes[i]


def test_drop_node():
    graph = get_simple_linear_pipeline()
    new_graph = deepcopy(graph)
    mutation = get_mutation_obj()
    new_graph = mutation._single_drop_mutation(new_graph)
    assert len(new_graph) < len(graph)


def test_add_as_parent_node_linear():
    """
    Test correctness of adding as a parent in simple case
    """
    graph = get_simple_linear_pipeline()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_separate_parent_node(new_graph, node_to_mutate)
    assert len(node_to_mutate.nodes_from) > len(graph.nodes[1].nodes_from)


def test_add_as_parent_node_tree():
    """
    Test correctness of adding as a parent in complex case
    """
    graph = get_tree_pipeline()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_separate_parent_node(new_graph, node_to_mutate)
    assert len(node_to_mutate.nodes_from) > len(graph.nodes[1].nodes_from)


def test_add_as_child_node_linear():
    """
    Test correctness of adding as a child in simple case
    """
    graph = get_simple_linear_pipeline()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_as_child(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert new_graph.node_children(node_to_mutate) != graph.node_children(node_to_mutate)


def test_add_as_child_node_tree():
    """
    Test correctness of adding as a child in complex case
    """

    graph = get_tree_pipeline()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[2]
    mutation = get_mutation_obj()
    mutation._add_as_child(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert new_graph.node_children(node_to_mutate) != graph.node_children(node_to_mutate)


def test_add_as_intermediate_node_linear():
    """
    Test correctness of adding as an intermediate node in simple case
    """
    graph = get_simple_linear_pipeline()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_intermediate_node(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert node_to_mutate.nodes_from[0] != graph.nodes[1].nodes_from[0]


def test_add_as_intermediate_node_tree():
    """
    Test correctness of adding as intermediate node in complex case
    """
    graph = get_tree_pipeline()
    new_graph = deepcopy(graph)
    node_to_mutate = new_graph.nodes[1]
    mutation = get_mutation_obj()
    mutation._add_intermediate_node(new_graph, node_to_mutate)
    assert len(new_graph) > len(graph)
    assert node_to_mutate.nodes_from[0] != graph.nodes[1].nodes_from[0]
