import itertools
from copy import deepcopy

import pytest

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.composer.optimisers.gp_operators import equivalent_subtree
from core.repository.model_types_repository import ModelTypesIdsEnum


def chain_first():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()

    root_of_tree, root_child_first, root_child_second = \
        [NodeGenerator.secondary_node(model) for model in (ModelTypesIdsEnum.xgboost, ModelTypesIdsEnum.xgboost,
                                                           ModelTypesIdsEnum.knn)]

    for root_node_child in (root_child_first, root_child_second):
        for requirement_model in (ModelTypesIdsEnum.logit, ModelTypesIdsEnum.lda):
            new_node = NodeGenerator.primary_node(requirement_model)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree)
    return chain


def chain_second():
    #      XG
    #   |      \
    #  XG      KNN
    #  | \      |  \
    # LR XG   LR   LDA
    #    |  \
    #   KNN  LDA
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.xgboost)
    for model_type in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda):
        new_node.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain = chain_first()
    chain.replace_node_with_parents(chain.root_node.nodes_from[0].nodes_from[1], new_node)

    return chain


def chain_third():
    #      XG
    #   |  |  \
    #  KNN LDA KNN
    root_of_tree = NodeGenerator.secondary_node(ModelTypesIdsEnum.xgboost)
    for model_type in (ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda, ModelTypesIdsEnum.knn):
        root_of_tree.nodes_from.append(NodeGenerator.primary_node(model_type))
    chain = Chain()

    for node in root_of_tree.nodes_from:
        chain.add_node(node)
    chain.add_node(root_of_tree)

    return chain


def chain_fourth():
    #      XG
    #   |  \  \
    #  KNN  XG  KNN
    #      |  \
    #    KNN   KNN

    chain = chain_third()
    new_node = NodeGenerator.secondary_node(ModelTypesIdsEnum.xgboost)
    [new_node.nodes_from.append(NodeGenerator.primary_node(ModelTypesIdsEnum.knn)) for _ in range(2)]
    chain.replace_node_with_parents(chain.root_node.nodes_from[1], new_node)

    return chain


@pytest.fixture()
def equality_cases():
    pairs = [[chain_first(), chain_first()], [chain_third(), chain_third()], [chain_fourth(), chain_fourth()]]

    # the following changes don't affect to chains equality:
    for node_num, type in enumerate([ModelTypesIdsEnum.knn, ModelTypesIdsEnum.lda]):
        pairs[1][1].root_node.nodes_from[node_num].model.model_type = type

    for node_num in ((2, 1), (1, 2)):
        old_node = pairs[2][1].root_node.nodes_from[node_num[0]]
        new_node = deepcopy(pairs[2][0].root_node.nodes_from[node_num[1]])
        pairs[2][1].replace_node_with_parents(old_node, new_node)

    return pairs


@pytest.fixture()
def non_equality_cases():
    return list(itertools.combinations([chain_first(), chain_second(), chain_third()], 2))


@pytest.mark.parametrize('chain_fixture', ['equality_cases'])
def test_equality_cases(chain_fixture, request):
    list_chains_pairs = request.getfixturevalue(chain_fixture)
    for pair in list_chains_pairs:
        assert pair[0] == pair[1]
        assert pair[1] == pair[0]


@pytest.mark.parametrize('chain_fixture', ['non_equality_cases'])
def test_non_equality_cases(chain_fixture, request):
    list_chains_pairs = request.getfixturevalue(chain_fixture)
    for pair in list_chains_pairs:
        assert not pair[0] == pair[1]
        assert not pair[1] == pair[0]


def test_chains_equivalent_subtree():
    c_first = chain_first()
    c_second = chain_second()
    c_third = chain_third()

    similar_nodes_first_and_second = equivalent_subtree(c_first, c_second)
    assert len(similar_nodes_first_and_second) == 6

    similar_nodes_first_and_third = equivalent_subtree(c_first, c_third)
    assert not similar_nodes_first_and_third

    similar_nodes_second_and_third = equivalent_subtree(c_second, c_third)
    assert not similar_nodes_second_and_third

    similar_nodes_third = equivalent_subtree(c_third, c_third)
    assert len(similar_nodes_third) == 4
