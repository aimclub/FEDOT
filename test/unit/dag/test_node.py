from fedot.core.dag.graph_utils import node_depth
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode


def get_nodes():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    return [root, third_node, first_node, second_node]


def test_node_depth():
    # given
    root = get_nodes()[0]

    distance = node_depth(root)

    assert distance == 3
