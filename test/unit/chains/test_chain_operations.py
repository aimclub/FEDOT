from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode


def get_chain():
    third_level_one = PrimaryNode('lda')

    second_level_one = SecondaryNode('qda', nodes_from=[third_level_one])
    second_level_two = PrimaryNode('qda')
    second_level_three = PrimaryNode('qda')
    second_level_four = PrimaryNode('qda')

    first_level_one = SecondaryNode('knn', nodes_from=[second_level_one, second_level_two])
    first_level_two = SecondaryNode('knn', nodes_from=[second_level_three, second_level_four])

    root = SecondaryNode(operation_type='logit', nodes_from=[first_level_one, first_level_two])
    chain = Chain(root)

    return chain


def test_distance_to_root_level():
    # given
    chain = get_chain()

    # when
    new_height = chain.actions.distance_to_root_level(node=chain.nodes[2])

    # then
    assert new_height == 2
