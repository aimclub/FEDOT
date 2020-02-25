import numpy as np
import pytest
from sklearn.datasets import load_iris

from core.models.data import InputData, normalize, split_train_test
from core.models.model import LogRegression
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.evaluation import EvaluationStrategy


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
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=np.arange(0, len(train_data_y)))
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)))
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100))
    return train_data, test_data, data


def test_model_chain(data_setup):
    _, _, data = data_setup
    eval_strategy = EvaluationStrategy(model=LogRegression())
    y1 = PrimaryNode(input_data_stream=data, eval_strategy=eval_strategy)
    y2 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y3 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y1])
    y4 = SecondaryNode(eval_strategy=eval_strategy, nodes_from=[y2, y3])
    y4.apply()
    assert y4.cached_result.cached_output.size == data.target.size
    assert len(y4.cached_result.last_parents_ids) == 2
    assert y4.data_stream.target.all() == data.target.all()
