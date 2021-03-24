import sys

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    (ClassificationMetricsEnum,
     ComplexityMetricsEnum,
     MetricsRepository,
     RegressionMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum


@pytest.fixture()
def data_setup():
    predictors, response = load_breast_cancer(return_X_y=True)
    np.random.seed(1)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    response = response[:100]
    predictors = predictors[:100]

    # Wrap data into TnputData
    input_data = InputData(features=predictors,
                           target=response,
                           idx=np.arange(0, len(predictors)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    # Train test split
    train_data, test_data = train_test_data_setup(input_data)
    return train_data, test_data


def default_valid_chain():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit', nodes_from=[first])
    third = SecondaryNode(operation_type='logit', nodes_from=[first])
    final = SecondaryNode(operation_type='logit', nodes_from=[second, third])

    chain = Chain(final)

    return chain


def test_structural_quality_correct():
    chain = default_valid_chain()
    metric_function = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)
    expected_metric_value = 13
    actual_metric_value = metric_function(chain)
    assert actual_metric_value <= expected_metric_value


def test_classification_quality_metric(data_setup):
    train, _ = data_setup
    chain = default_valid_chain()
    chain.fit(input_data=train)

    for metric in ClassificationMetricsEnum:
        metric_function = MetricsRepository().metric_by_id(metric)
        metric_value = metric_function(chain=chain, reference_data=train)
        assert 0 < abs(metric_value) < sys.maxsize


def test_regression_quality_metric(data_setup):
    train, _ = data_setup
    chain = default_valid_chain()
    chain.fit(input_data=train)

    for metric in RegressionMetricsEnum:
        metric_function = MetricsRepository().metric_by_id(metric)
        metric_value = metric_function(chain=chain, reference_data=train)
        assert metric_value > 0
