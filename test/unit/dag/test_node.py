from fedot.core.optimisers.graph import OptNode
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.verification import common_rules
from fedot.core.dag.graph_verifier import GraphVerifier


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
    graph = Pipeline(root)
    verifier = GraphVerifier(common_rules)
    assert verifier(graph)


def test_node_operator_ordered_subnodes_hierarchy():
    # given
    root = get_nodes()[0]

    # when
    ordered_nodes = root.ordered_subnodes_hierarchy()

    # then
    assert len(ordered_nodes) == 4


def test_node_operator_distance_to_primary_level():
    # given
    root = get_nodes()[0]

    distance = root.distance_to_primary_level()

    assert distance == 2
