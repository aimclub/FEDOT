import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from core.data import (
    Data,
    split_train_test,
    normalize
)
from core.evaluation import EvaluationStrategy
from core.model import LogRegression
from core.node import PrimaryNode, NodeGenerator


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
    data = Data(features=predictors, target=response, idx=np.arange(0, 100))
    return train_data_x, train_data_y, test_data_x, test_data_y, data


def model_metrics_info(class_name, y_true, y_pred):
    print('\n', f'#test_eval_strategy_{class_name}')
    print(classification_report(y_true, y_pred))
    print('Test model accuracy: ', accuracy_score(y_true, y_pred))


def test_node_factory_log_reg_correct():
    node = NodeGenerator().get_primary_mode(LogRegression())

    expected_model = LogRegression
    actual_model = node.eval_strategy.model.__class__

    assert node.__class__ == PrimaryNode
    assert expected_model == actual_model


def test_eval_strategy_logreg(data_setup):
    print_metrics = False
    train_data_x, train_data_y, test_data_x, true_y, data_stream = data_setup
    test_skl_model = LogisticRegression(C=10., random_state=1, solver='liblinear',
                                        max_iter=10000, verbose=0)
    test_skl_model.fit(train_data_x, train_data_y)
    expected_result = test_skl_model.predict(test_data_x)

    eval_strategy = EvaluationStrategy(model=LogRegression())
    test_model_node = PrimaryNode(nodes_to=None, data_stream=data_stream,
                                  eval_strategy=eval_strategy)
    actual_result = test_model_node.apply()
    if print_metrics:
        model_metrics_info(test_skl_model.__class__.__name__, true_y, actual_result)
    assert actual_result.all() == expected_result.all()
