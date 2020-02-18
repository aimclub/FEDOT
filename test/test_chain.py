import numpy as np
import pytest
from sklearn.datasets import load_iris

from core.data import Data, normalize, split_train_test
from core.model import LogRegression
from core.node import PrimaryNode, SecondaryNode
from core.evaluation import EvaluationStrategy


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


def test_model_chain(data_setup):
    train_data, test_data, data = data_setup
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(data_stream=None, eval_strategy=eval_strategy, nodes_to=None)
    y2 = SecondaryNode(data_stream=None, eval_strategy=eval_strategy, nodes_to=None,
                       nodes_from=[y1])
    y3 = SecondaryNode(data_stream=None, eval_strategy=eval_strategy, nodes_to=None,
                       nodes_from=[y1])
    y4 = SecondaryNode(data_stream=data, eval_strategy=eval_strategy, nodes_to=None,
                       nodes_from=[y2, y3])
    y4.apply()
    assert y1.cached_result is data
