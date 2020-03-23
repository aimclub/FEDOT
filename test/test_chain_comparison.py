import pytest

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.composer.node import equivalent_subtree
from core.models.model import LogRegression, KNN, LDA, XGBoost


def chain1():
    # XG
    # |   \
    # XG   KNN
    # | \   |  \
    # LR LDA LR  LDA
    chain = Chain()
    root_of_tree1 = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())

    for node in (root_of_tree1, root_child1, root_child2):
        node.nodes_from = []

    for root_node_child in (root_child1, root_child2):
        for requirement_model in (LogRegression(), LDA()):
            new_node = NodeGenerator.primary_node(requirement_model, input_data=None)
            root_node_child.nodes_from.append(new_node)
            chain.add_node(new_node)
        chain.add_node(root_node_child)
        root_of_tree1.nodes_from.append(root_node_child)

    chain.add_node(root_of_tree1)
    return chain


def chain2():
    chain = Chain()
    root_of_tree2 = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())

    for node in (root_of_tree2, root_child1, root_child2):
        node.nodes_from = []

    new_node = NodeGenerator.primary_node(LogRegression(), input_data=None)
    root_child1.nodes_from.append(new_node)
    chain.add_node(new_node)

    new_node = NodeGenerator.secondary_node(XGBoost())
    new_node.nodes_from = []
    new_node.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    new_node.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))

    chain.add_node(new_node.nodes_from[0])
    chain.add_node(new_node.nodes_from[1])
    chain.add_node(new_node)
    chain.add_node(root_child1)

    root_child1.nodes_from.append(new_node)
    root_of_tree2.nodes_from.append(root_child1)
    root_child2.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))
    root_child2.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))

    chain.add_node(root_child2.nodes_from[0])
    chain.add_node(root_child2.nodes_from[1])
    chain.add_node(root_child2)
    root_of_tree2.nodes_from.append(root_child2)
    chain.add_node(root_of_tree2)

    return chain


def chain3():
    chain = Chain()
    root_of_tree3 = NodeGenerator.secondary_node(XGBoost())
    root_of_tree3.nodes_from = []
    root_of_tree3.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    root_of_tree3.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))
    root_of_tree3.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    for node in root_of_tree3.nodes_from:
        chain.add_node(node)
    chain.add_node(root_of_tree3)
    return chain


def chain4():
    chain = Chain()
    root_of_tree2 = NodeGenerator.secondary_node(XGBoost())
    root_of_tree2.nodes_from = []
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))
    for node in root_of_tree2.nodes_from:
        chain.add_node(node)
    chain.add_node(root_of_tree2)
    return chain


def chain5():
    chain = Chain()
    root_of_tree2 = NodeGenerator.secondary_node(XGBoost())
    root_of_tree2.nodes_from = []
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))

    new_node = NodeGenerator.secondary_node(XGBoost())

    new_node.nodes_from = []
    new_node.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))
    new_node.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))

    root_of_tree2.nodes_from.append(new_node)
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))

    for node in new_node.nodes_from:
        chain.add_node(node)

    chain.add_node(root_of_tree2.nodes_from[0])
    chain.add_node(new_node)
    chain.add_node(root_of_tree2.nodes_from[2])

    chain.add_node(root_of_tree2)
    return chain


def chain6():
    chain = Chain()
    root_of_tree2 = NodeGenerator.secondary_node(XGBoost())

    root_of_tree2.nodes_from = []
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))
    root_of_tree2.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))

    new_node = NodeGenerator.secondary_node(XGBoost())

    new_node.nodes_from = []
    new_node.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))
    new_node.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))

    root_of_tree2.nodes_from.append(new_node)

    for node1, node2 in zip(root_of_tree2.nodes_from, new_node.nodes_from):
        chain.add_node(node1)
        chain.add_node(node2)

    chain.add_node(new_node)
    chain.add_node(root_of_tree2)
    return chain


@pytest.fixture()
def case1():
    c1 = chain1()
    c2 = chain1()
    return c1, c2


@pytest.fixture()
def case2():
    c1 = chain1()
    c2 = chain2()
    return c1, c2


@pytest.fixture()
def case3():
    c1 = chain1()
    c2 = chain3()
    return c1, c2


@pytest.fixture()
def case4():
    c1 = chain2()
    c2 = chain3()
    return c1, c2


@pytest.fixture()
def case5():
    c1 = chain3()
    c2 = chain4()
    return c1, c2


@pytest.fixture()
def case6():
    c1 = chain5()
    c2 = chain6()
    return c1, c2


@pytest.mark.parametrize('chain_fixture', ['case1', 'case5', 'case6'])
def test_chain_comparison_equal_case(chain_fixture, request):
    c1, c2 = request.getfixturevalue(chain_fixture)
    assert c1 == c2


@pytest.mark.parametrize('chain_fixture', ['case2', 'case3', 'case4'])
def test_chain_comparison_different_case(chain_fixture, request):
    c1, c2 = request.getfixturevalue(chain_fixture)
    assert not c1 == c2


def test_equivalent_subtree():
    # the threes с1 ana c2 have 6 similar nodes
    c1 = chain1()

    c2 = chain2()

    c3 = chain3()

    similar_nodes_c1_c2 = equivalent_subtree(c1.root_node, c2.root_node)
    assert len(similar_nodes_c1_c2) == 6

    similar_nodes_c1_c3 = equivalent_subtree(c1.root_node, c3.root_node)
    assert len(similar_nodes_c1_c3) == 0 and similar_nodes_c1_c3 == []

    similar_nodes_c2_c3 = equivalent_subtree(c2.root_node, c3.root_node)
    assert len(similar_nodes_c2_c3) == 0 and similar_nodes_c2_c3 == []

    simular_nodes_c3_c3 = equivalent_subtree(c3.root_node, c3.root_node)
    assert len(simular_nodes_c3_c3) == 4
