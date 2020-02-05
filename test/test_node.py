import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from core.evaluation import (LinRegression, LogRegression, XGBoost, split_train_test,
                             normalize)
from core.node import (NodeFactory, OperationNode, ModelNode, )
from core.datastream import DataStream
import numpy as np


@pytest.fixture()
def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    data_stream = DataStream(x=predictors, y=response)
    return normalize(train_data_x), train_data_y, normalize(
        test_data_x), test_data_y, data_stream


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


def test_eval_strategy_logreg(data_setup):
    train_data_x, train_data_y, test_data_x, true_y, data_stream = data_setup
    test_skl_model = LogisticRegression(C=10., random_state=1, solver='liblinear',
                                        max_iter=10000, verbose=0)
    test_skl_model.fit(train_data_x, train_data_y)
    expected_result = test_skl_model.predict(test_data_x)

    test_model_node = ModelNode(nodes_from=None, nodes_to=None, data_stream=data_stream,
                                eval_strategy=LogRegression(seed=1))
    actual_result = test_model_node.apply()
    print('\n', '#test_eval_strategy_logreg_log')
    print(classification_report(true_y, actual_result,
                                target_names=['Class 0', 'Class 1', 'Class 2']))
    print('Test model accuracy: ', accuracy_score(true_y, actual_result))
    print('Test model score', test_skl_model.score(train_data_x, train_data_y))
    assert actual_result.all() == expected_result.all()
