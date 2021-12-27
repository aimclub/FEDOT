from fedot.core.composer.constraint import constraint_function
from fedot.core.optimisers.adapters import PipelineAdapter
from fedot.core.optimisers.graph import OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline


def get_nodes():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    return [root, third_node, first_node, second_node]


def test_constraint_validation_with_opt_node():
    first_node = PrimaryNode('ridge')
    second_node = OptNode({'name': "opt"})
    root = SecondaryNode('ridge', nodes_from=[first_node, second_node])
    graph = PipelineAdapter().adapt(Pipeline(root))
    graph_gener_params = GraphGenerationParams()
    assert constraint_function(graph, graph_gener_params)


def test_node_operator_ordered_subnodes_hierarchy():
    # given
    root = get_nodes()[0]

    # when
    ordered_nodes = root._operator.ordered_subnodes_hierarchy()

    # then
    assert len(ordered_nodes) == 4


def test_node_operator_distance_to_primary_level():
    # given
    root = get_nodes()[0]

    distance = root._operator.distance_to_primary_level()

    assert distance == 2
