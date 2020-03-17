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


def test_chain_comparison_equals_case():
    # the threes are equlas
    chain1 = Chain()
    root_of_tree1 = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())
    for node in (root_of_tree1, root_child1, root_child2):
        node.nodes_from = []
    for root_node_child in (root_child1, root_child2):
        for requirement_model in (LogRegression(), LDA()):
            new_node = NodeGenerator.primary_node(requirement_model, input_data=None)
            root_node_child.nodes_from.append(new_node)
            chain1.add_node(new_node)
        chain1.add_node(root_node_child)
        root_of_tree1.nodes_from.append(root_node_child)
    chain1.add_node(root_of_tree1)

    chain2 = Chain()
    root_of_tree2 = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())
    for node in (root_of_tree2, root_child1, root_child2):
        node.nodes_from = []

    for root_node_child in (root_child1, root_child2):
        for requirement_model in (LogRegression(), LDA()):
            new_node = NodeGenerator.primary_node(requirement_model, input_data=None)
            root_node_child.nodes_from.append(new_node)
            chain2.add_node(new_node)
        chain2.add_node(root_node_child)
        root_of_tree2.nodes_from.append(root_node_child)
    chain2.add_node(root_of_tree2)

    assert chain1.equals(chain2)


def test_chain_comparison_diffenrent_case():
    #the threes have 6 similar nodes
    chain1 = Chain()
    root_of_tree1 = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())
    for node in (root_of_tree1, root_child1, root_child2):
        node.nodes_from = []
    for root_node_child in (root_child1, root_child2):
        for requirement_model in (LogRegression(), LDA()):
            new_node = NodeGenerator.primary_node(requirement_model, input_data=None)
            root_node_child.nodes_from.append(new_node)
            chain1.add_node(new_node)
        chain1.add_node(root_node_child)
        root_of_tree1.nodes_from.append(root_node_child)
    chain1.add_node(root_of_tree1)

    chain2 = Chain()
    root_of_tree2 = NodeGenerator.secondary_node(XGBoost())
    root_child1 = NodeGenerator.secondary_node(XGBoost())
    root_child2 = NodeGenerator.secondary_node(KNN())
    for node in (root_of_tree2, root_child1, root_child2):
        node.nodes_from = []
    new_node = NodeGenerator.primary_node(LogRegression(), input_data=None)
    root_child1.nodes_from.append(new_node)
    chain2.add_node(new_node)
    new_node = NodeGenerator.secondary_node(XGBoost())
    new_node.nodes_from = []
    new_node.nodes_from.append(NodeGenerator.primary_node(KNN(), input_data=None))
    new_node.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))
    chain2.add_node(new_node.nodes_from[0])
    chain2.add_node(new_node.nodes_from[1])
    chain2.add_node(new_node)
    chain2.add_node(root_child1)
    root_child1.nodes_from.append(new_node)
    root_of_tree2.nodes_from.append(root_child1)
    root_child2.nodes_from.append(NodeGenerator.primary_node(LogRegression(), input_data=None))
    root_child2.nodes_from.append(NodeGenerator.primary_node(LDA(), input_data=None))
    chain2.add_node(root_child2.nodes_from[0])
    chain2.add_node(root_child2.nodes_from[1])
    chain2.add_node(root_child2)
    root_of_tree2.nodes_from.append(root_child2)
    chain2.add_node(root_of_tree2)

    similar_nodes = chain1.root_node.get_similar_nodes(chain2.root_node)

    assert (not chain1.equals(chain2)) and len(similar_nodes) == 6
