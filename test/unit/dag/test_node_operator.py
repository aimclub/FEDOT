from data.pipeline_manager import get_nodes


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
