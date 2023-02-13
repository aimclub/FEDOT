import os
import sys

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer

from fedot.core.composer.metrics import QualityMetric, ROCAUC
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
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

    # Wrap data into InputData
    input_data = InputData(features=predictors,
                           target=response,
                           idx=np.arange(0, len(predictors)),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)
    # Train test split
    train_data, test_data = train_test_data_setup(input_data)
    return train_data, test_data


@pytest.fixture()
def multi_target_data_setup():
    test_file_path = str(os.path.dirname(__file__))
    file = '../../data/multi_target_sample.csv'
    path = os.path.join(test_file_path, file)

    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    task = Task(TaskTypesEnum.regression)
    data = InputData.from_csv(path, target_columns=target_columns,
                              index_col=None, columns_to_drop=['date'], task=task)
    train, test = train_test_data_setup(data)
    return train, test


def default_valid_pipeline():
    first = PrimaryNode(operation_type='logit')
    second = SecondaryNode(operation_type='logit', nodes_from=[first])
    third = SecondaryNode(operation_type='logit', nodes_from=[first])
    final = SecondaryNode(operation_type='logit', nodes_from=[second, third])

    pipeline = Pipeline(final)

    return pipeline


def test_structural_quality_correct():
    pipeline = default_valid_pipeline()
    metric_function = MetricsRepository().metric_by_id(ComplexityMetricsEnum.structural)
    expected_metric_value = 13
    actual_metric_value = metric_function(pipeline)
    assert actual_metric_value <= expected_metric_value


def test_classification_quality_metric(data_setup):
    train, _ = data_setup
    pipeline = default_valid_pipeline()
    pipeline.fit(input_data=train)

    for metric in ClassificationMetricsEnum:
        metric_function = MetricsRepository().metric_by_id(metric)
        metric_value = metric_function(pipeline=pipeline, reference_data=train)
        assert 0 < abs(metric_value) < sys.maxsize


def test_regression_quality_metric(data_setup):
    train, _ = data_setup
    pipeline = default_valid_pipeline()
    pipeline.fit(input_data=train)

    for metric in RegressionMetricsEnum:
        metric_function = MetricsRepository().metric_by_id(metric)
        metric_value = metric_function(pipeline=pipeline, reference_data=train)
        assert 0 < abs(metric_value) < sys.maxsize


def test_data_preparation_for_multi_target_correct(multi_target_data_setup):
    train, test = multi_target_data_setup
    simple_pipeline = Pipeline(PrimaryNode('linear'))
    simple_pipeline.fit(input_data=train)

    source_shape = test.target.shape
    # Get converted data
    results, new_test = QualityMetric()._simple_prediction(simple_pipeline, test)
    number_elements = len(new_test.target)
    assert source_shape[0] * source_shape[1] == number_elements


def test_roc_auc_multiclass_correct():
    data = InputData(features=[[1, 2], [2, 3], [3, 4], [4, 1]], target=np.array([['x'], ['y'], ['z'], ['x']]),
                     idx=np.arange(4),
                     task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    prediction = OutputData(features=[[1, 2], [2, 3], [3, 4], [4, 1]], predict=np.array([[0.4, 0.3, 0.3],
                                                                                         [0.2, 0.5, 0.3],
                                                                                         [0.1, 0.2, 0.7],
                                                                                         [0.8, 0.1, 0.1]]),
                            idx=np.arange(4), task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    for i in range(data.num_classes):
        fpr, tpr, threshold = ROCAUC.roc_curve(data.target, prediction.predict[:, i],
                                               pos_label=data.class_labels[i])
        roc_auc = ROCAUC.auc(fpr, tpr)
        assert roc_auc
