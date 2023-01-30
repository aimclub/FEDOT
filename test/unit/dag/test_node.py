from fedot.core.dag.graph_utils import distance_to_primary_level
from fedot.core.pipelines.node import PipelineNode


def get_nodes():
    first_node = PipelineNode('knn')
    second_node = PipelineNode('knn')
    third_node = PipelineNode('lda', nodes_from=[first_node, second_node])
    root = PipelineNode('logit', nodes_from=[third_node])

    return [root, third_node, first_node, second_node]


def test_distance_to_primary_level():
    # given
    root = get_nodes()[0]

    distance = distance_to_primary_level(root)

    assert distance == 2
