from copy import copy, deepcopy
from random import seed

import numpy as np

from fedot.core.dag.graph import Graph
from fedot.core.dag.graph_node import GraphNode
from test.unit.pipelines.test_pipeline_tuning import classification_dataset

seed(1)
np.random.seed(1)

tmp = classification_dataset


def test_graph_id():
    right_id = '((/n1;)/n2;;(/n1;)/n3;)/n4'
    first = GraphNode(content='n1')
    second = GraphNode(content='n2', nodes_from=[first])
    third = GraphNode(content='n3', nodes_from=[first])
    final = GraphNode(content='n4', nodes_from=[second, third])

    assert final.descriptive_id == right_id


def test_graph_str():
    # given
    first = GraphNode(content='n1')
    second = GraphNode(content='n2')
    third = GraphNode(content='n3')
    final = GraphNode(content='n4',
                      nodes_from=[first, second, third])
    graph = Graph(final)

    expected_graph_description = "{'depth': 2, 'length': 4, 'nodes': [n4, n1, n2, n3]}"

    # when
    actual_graph_description = str(graph)

    # then
    assert actual_graph_description == expected_graph_description


def test_pipeline_repr():
    first = GraphNode(content='n1')
    second = GraphNode(content='n2')
    third = GraphNode(content='n3')
    final = GraphNode(content='n4',
                      nodes_from=[first, second, third])
    graph = Graph(final)

    expected_graph_description = "{'depth': 2, 'length': 4, 'nodes': [n4, n1, n2, n3]}"

    assert repr(graph) == expected_graph_description


def test_delete_primary_node():
    # given
    first = GraphNode(content='n1')
    second = GraphNode(content='n2')
    third = GraphNode(content='n3', nodes_from=[first])
    final = GraphNode(content='n4', nodes_from=[second, third])
    graph = Graph(final)

    # when
    graph.delete_node(first)

    new_primary_node = [node for node in graph.nodes if node.content['name'] == 'n2'][0]

    # then
    assert len(graph.nodes) == 3
    assert isinstance(new_primary_node, GraphNode)


def test_graph_copy():
    pipeline = Graph(GraphNode(content='n1'))
    pipeline_copy = copy(pipeline)
    assert pipeline.uid != pipeline_copy.uid


def test_pipeline_deepcopy():
    pipeline = Graph(GraphNode(content='n1'))
    pipeline_copy = deepcopy(pipeline)
    assert pipeline.uid != pipeline_copy.uid
