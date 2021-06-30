from fedot.core.chains.node import PrimaryNode, SecondaryNode


def get_nodes():
    first_node = PrimaryNode('knn')
    second_node = PrimaryNode('knn')
    third_node = SecondaryNode('lda', nodes_from=[first_node, second_node])
    root = SecondaryNode('logit', nodes_from=[third_node])

    return [root, third_node, first_node, second_node]


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
