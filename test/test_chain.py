import numpy as np
import pytest
from sklearn.datasets import load_iris

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import Data, normalize, split_train_test
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
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = Data(features=train_data_x, target=train_data_y,
                      idx=np.arange(0, len(train_data_y)))
    test_data = Data(features=test_data_x, target=test_data_y,
                     idx=np.arange(0, len(test_data_y)))
    data = Data(features=predictors, target=response, idx=np.arange(0, 100))
    return train_data, test_data, data


def test_models_sequence(data_setup):
    _, _, data = data_setup

    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data_stream=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    result = y4.apply()
    assert y4.cached_result.cached_output.size == data.target.size
    assert len(y4.cached_result.last_parents_ids) == 2
    assert result.size == data.target.size


def test_models_chain_nested(data_setup):
    _, _, data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data_stream=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    result = chain.evaluate()
    assert chain.length == 4
    assert chain.depth == 3
    assert result.size == data.target.size


def test_models_chain_seq(data_setup):
    _, _, data = data_setup
    chain = Chain()
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data_stream=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)
    result = chain.evaluate()
    assert chain.length == 4
    assert chain.depth == 4
    assert result.size == data.target.size
