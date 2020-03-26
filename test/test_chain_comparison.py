from copy import deepcopy

import pytest

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.composer.node import equivalent_subtree
from core.models.model import LogRegression, KNN, LDA, XGBoost


def chain_first():
    #    XG
    #  |     \
    # XG     KNN
    # |  \    |  \
    # LR LDA LR  LDA
    chain = Chain()
    root_of_tree = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())

    for node in (root_of_tree, root_child1, root_child2):
        node.nodes_from = []

    for root_node_child in (root_child1, root_child2):
        for requirement_model in (LogRegression(), LDA()):
            new_node = NodeGenerator.primary_node(requirement_model, input_data=None)
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
    chain = chain_first()
    new_node = NodeGenerator.secondary_node(XGBoost())
    new_node.nodes_from = []
    new_node.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    new_node.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))
    chain.replace_node(chain.root_node.nodes_from[0].nodes_from[1], new_node)

    return chain


def chain_third():
    #      XG
    #   |  |  \
    #  KNN LDA KNN
    chain = Chain()
    root_of_tree = NodeGenerator.secondary_node(XGBoost())
    root_of_tree.nodes_from = []
    root_of_tree.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    root_of_tree.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))
    root_of_tree.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
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
    new_node = NodeGenerator.secondary_node(XGBoost())
    new_node.nodes_from = []
    new_node.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    new_node.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    chain.replace_node(chain.root_node.nodes_from[1], new_node)

    return chain


@pytest.fixture()
def case_first():
    c1 = chain_first()
    c2 = chain_first()
    return c1, c2


@pytest.fixture()
def case_second():
    c1 = chain_third()
    c2 = chain_third()
    c2.root_node.nodes_from[1].eval_strategy.model = KNN()
    c2.root_node.nodes_from[2].eval_strategy.model = LDA()
    return c1, c2


@pytest.fixture()
def case_third():
    c1 = chain_fourth()
    c2 = chain_fourth()
    c2.replace_node(c2.root_node.nodes_from[2], deepcopy(c1.root_node.nodes_from[1]))
    c2.replace_node(c2.root_node.nodes_from[1], deepcopy(c1.root_node.nodes_from[2]))
    return c1, c2


@pytest.fixture()
def case_fourth():
    c1 = chain_first()
    c2 = chain_second()
    return c1, c2


@pytest.fixture()
def case_fifth():
    c1 = chain_first()
    c2 = chain_third()
    return c1, c2


@pytest.fixture()
def case_sixth():
    c1 = chain_second()
    c2 = chain_third()
    return c1, c2


@pytest.mark.parametrize('chain_fixture', ['case_first', 'case_second', 'case_third'])
def test_chain_comparison_equals(chain_fixture, request):
    c1, c2 = request.getfixturevalue(chain_fixture)
    assert c1 == c2


@pytest.mark.parametrize('chain_fixture', ['case_fourth', 'case_fifth', 'case_sixth'])
def test_chain_comparison_different(chain_fixture, request):
    c1, c2 = request.getfixturevalue(chain_fixture)
    assert not c1 == c2


def test_chains_equivalent_subtree():
    c1 = chain_first()

    c2 = chain_second()

    c3 = chain_third()

    similar_nodes_c1_c2 = equivalent_subtree(c1.root_node, c2.root_node)
    assert len(similar_nodes_c1_c2) == 6

    similar_nodes_c1_c3 = equivalent_subtree(c1.root_node, c3.root_node)
    assert len(similar_nodes_c1_c3) == 0 and similar_nodes_c1_c3 == []

    similar_nodes_c2_c3 = equivalent_subtree(c2.root_node, c3.root_node)
    assert len(similar_nodes_c2_c3) == 0 and similar_nodes_c2_c3 == []

    simular_nodes_c3_c3 = equivalent_subtree(c3.root_node, c3.root_node)
    assert len(simular_nodes_c3_c3) == 4
