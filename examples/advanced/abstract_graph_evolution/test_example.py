import networkx as nx
import pytest

from examples.advanced.abstract_graph_evolution.graph_utils import optgraph_to_nx, nx_to_optgraph
from fedot.core.dag.graph import Graph
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
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


@pytest.mark.parametrize('graph', [
    *get_pipelines()
])
def test_transform_to_and_from_nx(graph: Graph):
    nx_graph = optgraph_to_nx(graph)

    for node in graph.nodes:
        assert node.uid in nx_graph.nodes

    retranslated_graph = nx_to_optgraph(nx_graph)

    assert retranslated_graph.descriptive_id == graph.descriptive_id


def test_transform_to(graph: Graph, nxgraph: nx.DiGraph):
    assert True