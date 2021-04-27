from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.graph_operator import GraphOperator


def get_chain():
    third_level_one = PrimaryNode('lda')

    second_level_one = SecondaryNode('qda', nodes_from=[third_level_one])
    second_level_two = PrimaryNode('qda')

    first_level_one = SecondaryNode('knn', nodes_from=[second_level_one, second_level_two])

    root = SecondaryNode(operation_type='logit', nodes_from=[first_level_one])
    chain = Chain(root)

    return chain


def test_chain_operator_init():
    chain = get_chain()
    assert type(chain.operator) is GraphOperator


def test_distance_to_root_level():
    # given
    chain = get_chain()
    selected_node = chain.nodes[2]

    # when
    height = chain.operator.distance_to_root_level(selected_node)

    # then
    assert height == 2


def test_nodes_from_layer():
    # given
    chain = get_chain()
    desired_layer = 2

    # when
    nodes_from_desired_layer = chain.operator.nodes_from_layer(desired_layer)

    # then
    assert len(nodes_from_desired_layer) == 2


def test_actualise_old_node_children():
    # given
    chain = get_chain()
    selected_node = chain.nodes[2]
    new_node = PrimaryNode('knnreg')

    # when
    chain.operator.actualise_old_node_children(old_node=selected_node,
                                               new_node=new_node)
    updated_parent = chain.nodes[1]

    # then
    assert new_node in updated_parent.nodes_from


def test_sort_nodes():
    # given
    chain = get_chain()
    selected_node = chain.nodes[2]
    original_length = chain.length
    new_node = PrimaryNode('knnreg')
    new_subroot = SecondaryNode('knnreg', nodes_from=[new_node])

    # when
    selected_node.nodes_from.append(new_subroot)
    chain.operator.sort_nodes()

    # then
    assert chain.length == original_length + 2
    assert chain.nodes[4] is new_subroot
    assert chain.nodes[5] is new_node


def test_node_children():
    # given
    chain = get_chain()
    selected_node = chain.nodes[2]

    # when
    children = chain.operator.node_children(selected_node)

    # then
    assert len(children) == 1
    assert children[0] is chain.nodes[1]
