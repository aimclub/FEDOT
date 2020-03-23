import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData
from core.models.model import LogRegression, KNN, LDA, XGBoost


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100))
    return data


@pytest.fixture()
def file_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = 'data/test_dataset.csv'
    input_data = InputData.from_csv(
        os.path.join(test_file_path, file))
    input_data.idx = _to_numerical(categorical_ids=input_data.idx)
    return input_data


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


def _to_numerical(categorical_ids: np.ndarray):
    encoded = pd.factorize(categorical_ids)[0]
    return encoded


@pytest.mark.parametrize('data_fixture', ['data_setup', 'file_data_setup'])
def test_models_sequence(data_fixture, request):
    data = request.getfixturevalue(data_fixture)

    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y3])
    results = y4.apply()
    assert y4.cached_result.cached_output.size == data.target.size
    assert len(y4.cached_result.last_parents_ids) == 2
    assert len(results.predict) == len(data.target)


def test_models_chain_nested(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    results = chain.predict(data)
    assert chain.length == 4
    assert chain.depth == 3
    assert len(results.predict) == len(data.target)


def test_models_chain_seq(data_setup):
    data = data_setup
    chain = Chain()
    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y3])

    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    results = chain.predict(data)
    assert chain.length == 4
    assert chain.depth == 4
    assert len(results.predict) == len(data.target)


@pytest.mark.parametrize('chain_fixture', ['case1', 'case5', 'case6'])
def test_chain_comparison_equal_case(chain_fixture, request):
    c1, c2 = request.getfixturevalue(chain_fixture)
    assert c1 == c2


@pytest.mark.parametrize('chain_fixture', ['case2', 'case3', 'case4'])
def test_chain_comparison_different_case(chain_fixture, request):
    c1, c2 = request.getfixturevalue(chain_fixture)
    assert not c1 == c2
