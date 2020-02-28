import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, split_train_test
from core.models.model import LogRegression
from core.repository.quality_metrics_repository import MetricsRepository, ComplexityMetricsEnum, \
    ClassificationMetricsEnum


@pytest.fixture()
def data_setup():
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]
    train_data_x, test_data_x = split_train_test(predictors)
    train_data_y, test_data_y = split_train_test(response)
    train_data = InputData(features=train_data_x, target=train_data_y,
                           idx=np.arange(0, len(train_data_y)))
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)))
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100))
    return train_data, test_data, data


def test_structural_quality(data_setup):
    _, _, data = data_setup

    metric_functions = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)

    chain = Chain()
    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)

    metric_value = metric_functions(chain)
    assert metric_value == 13


def test_classification_quality_metric(data_setup):
    _, _, data = data_setup

    metric_functions = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = Chain()
    y1 = NodeGenerator.primary_node(model=LogRegression(), input_data=data)
    y2 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y3 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y1])
    y4 = NodeGenerator.secondary_node(model=LogRegression(), nodes_from=[y2, y3])
    chain.add_node(y1)
    chain.add_node(y2)
    chain.add_node(y3)
    chain.add_node(y4)

    metric_value = metric_functions(chain)
    assert 0.0 < abs(metric_value) < 1.0
