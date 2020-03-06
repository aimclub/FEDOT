import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData
from core.models.data import normalize
from core.models.evaluation import EvaluationStrategy
from core.models.model import LogRegression


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = normalize(predictors[:100])
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

    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    results = y4.apply()
    assert y4.cached_result.cached_output.size == data.target.size
    assert len(y4.cached_result.last_parents_ids) == 2
    assert len(results.predict) == len(data.target)


def test_models_chain_nested(data_setup):
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    results = chain.evaluate()
    assert chain.length == 4
    assert chain.depth == 3
    assert len(results.predict) == len(data.target)


def test_models_chain_seq(data_setup):
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    results = chain.evaluate()
    assert chain.length == 4
    assert chain.depth == 4
    assert len(results.predict) == len(data.target)


@pytest.mark.xfail(raises=ValueError)
def test_chain_has_cycles(data_setup):
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    y5 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    y2.nodes_from.append(y4)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain._has_no_cycle()


def test_chain_has_no_cycles(data_setup):
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain._has_no_cycle()


@pytest.mark.xfail(raises=ValueError)
def test_has_isolated_nodes(data_setup):
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[])
    y5 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y4])
    y6 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y5])
    y4.nodes_from.append(y6)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain.add_node(y6)
    chain._has_isolated_nodes()


@pytest.mark.xfail(raises=AssertionError)
def test_multi_root_node(data_setup):
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1, y2])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    y5 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    assert isinstance(chain.root_node, SecondaryNode)


def test_has_primary_ndoes():
    data = data_setup
    eval_strategy = EvaluationStrategy(model=LogRegression())
    chain = Chain()
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    chain.add_node(y1)
    chain.add_node(y2)
    assert chain._has_primary_nodes()


@pytest.mark.xfail(raises=ValueError)
def test_has_self_cycled_nodes():
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2])
    y2.nodes_from.append(y2)
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain._has_no_self_cycled_nodes()


def test_validate_chain():
    data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y2 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y3 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y4 = PrimaryNode(input_data=data, eval_strategy=eval_strategy)
    y5 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1, y2])
    y6 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3, y4])
    y7 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y5, y6])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    chain.add_node(y5)
    chain.add_node(y6)
    chain.add_node(y7)
    chain.validate_chain()
