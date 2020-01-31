import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from core.evaluation import (LinRegression, LogRegression, XGBoost, splitted_train_test)
from core.node import (NodeFactory, OperationNode, ModelNode, )
from core.datastream import DataStream


@pytest.fixture()
def data_setup():
    test_data = load_iris()
    predictors = test_data['data']
    response = test_data['target']
    train_data_x, test_data_x = splitted_train_test(predictors)
    train_data_y, test_data_y = splitted_train_test(response)
    data_stream = DataStream(x=predictors, y=response)
    return train_data_x, train_data_y, test_data_x, test_data_y, data_stream


def test_node_log_reg():
    test_node = NodeFactory().log_reg()
    assert test_node.eval_strategy.__class__ == LogRegression


def test_node_lin_log():
    test_node = NodeFactory().lin_reg()
    assert test_node.eval_strategy.__class__ == LinRegression
    assert test_node.__class__ == OperationNode


def test_node_xgboost():
    test_node = NodeFactory().default_xgb()
    assert test_node.eval_strategy.__class__ == XGBoost


def test_eval_strategy_log(data_setup):
    train_data_x, train_data_y, test_data_x, expected_y, data_stream = data_setup
    test_skl_model = LogisticRegression(random_state=1)
    test_skl_model.fit(train_data_x, train_data_y)
    expected_result = test_skl_model.predict(test_data_x)

    test_model_node = ModelNode(nodes_from=None, nodes_to=None, data_stream=data_stream,
                                eval_strategy=LogRegression(seed=1))
    actual_result = test_model_node.apply()
    assert actual_result.all() == expected_result.all()
