import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, split_train_test
from core.repository.model_types_repository import ModelTypesIdsEnum
from core.repository.quality_metrics_repository import (
    MetricsRepository, ComplexityMetricsEnum,
    ClassificationMetricsEnum
)
from core.repository.task_types import MachineLearningTasksEnum


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
                           idx=np.arange(0, len(train_data_y)),
                           task_type=MachineLearningTasksEnum.classification)
    test_data = InputData(features=test_data_x, target=test_data_y,
                          idx=np.arange(0, len(test_data_y)),
                          task_type=MachineLearningTasksEnum.classification)
    return train_data, test_data


def default_valid_chain():
    first = NodeGenerator.primary_node(model_type=ModelTypesIdsEnum.logit)
    second = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                          nodes_from=[first])
    third = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[first])
    final = NodeGenerator.secondary_node(model_type=ModelTypesIdsEnum.logit,
                                         nodes_from=[second, third])

    chain = Chain()
    for node in [first, second, third, final]:
        chain.add_node(node)

    return chain


def test_structural_quality_correct():
    chain = default_valid_chain()
    metric_functions = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)

    expected_metric_value = 13
    actual_metric_value = metric_functions(chain, None)
    assert actual_metric_value == expected_metric_value


def test_classification_quality_metric(data_setup):
    train, _ = data_setup

    metric_functions = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    chain = default_valid_chain()
    chain.fit(input_data=train)
    metric_value = metric_functions(chain=chain, reference_data=train)
    assert 0.0 < abs(metric_value) < 1.0
