import pytest

from fedot.core.dag.graph_verifier import GraphVerifier
from fedot.core.dag.verification_rules import DEFAULT_DAG_RULES
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.graph import OptNode
from fedot.core.pipelines.node import Node, SecondaryNode, PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_builder import PipelineBuilder


def get_pipelines():
    one_node_pipeline = PipelineBuilder() \
        .add_sequence('logit') \
        .to_pipeline()
    linear_pipeline = PipelineBuilder() \
        .add_sequence('logit', 'logit', 'logit') \
        .to_pipeline()
    branching_structure = PipelineBuilder() \
        .add_node('operation_a') \
        .add_branch('operation_a', 'operation_f') \
        .join_branches('operation_f') \
        .to_pipeline()
    branching_structure2 = PipelineBuilder() \
        .add_node('operation_a') \
        .add_branch('operation_b', 'operation_c') \
        .grow_branches('operation_d', None) \
        .join_branches('operation_f') \
        .add_node('operation_a') \
        .to_pipeline()
    node_a = PrimaryNode('logit')
    node_b = SecondaryNode('logit', nodes_from=[node_a])
    node_c = SecondaryNode('logit', nodes_from=[node_b, node_a])
    skip_connection_structure = Pipeline(node_c)

    return [one_node_pipeline, linear_pipeline,
            branching_structure, branching_structure2,
            skip_connection_structure]


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_pipelines_adapt_properly(pipeline):
    adapter = PipelineAdapter()
    verifier = GraphVerifier(DEFAULT_DAG_RULES)

    assert all(isinstance(node, Node) for node in pipeline.nodes)
    assert _check_nodes_references_correct(pipeline)
    assert verifier(pipeline)

    opt_graph = adapter.adapt(pipeline)

    assert all(type(node) is OptNode for node in opt_graph.nodes)  # checking strict type equality!
    assert _check_nodes_references_correct(opt_graph)
    assert verifier(opt_graph)


@pytest.mark.parametrize('pipeline', get_pipelines())
def test_adapted_and_restored_are_equal(pipeline):
    adapter = PipelineAdapter()

    opt_graph = adapter.adapt(pipeline)
    restored_pipeline = adapter.restore(opt_graph)

    assert pipeline.descriptive_id == restored_pipeline.descriptive_id
    assert id(pipeline) != id(restored_pipeline)


def _check_nodes_references_correct(graph):
    for node in graph.nodes:
        if node.nodes_from:
            for parent_node in node.nodes_from:
                if parent_node not in graph.nodes:
                    return False
    return True
