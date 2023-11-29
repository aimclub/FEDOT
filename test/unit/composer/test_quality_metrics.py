import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_linnerud, load_wine

from fedot.core.composer.metrics import QualityMetric, ROCAUC
from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import (ClassificationMetricsEnum,
                                                              ComplexityMetricsEnum,
                                                              MetricsRepository,
                                                              RegressionMetricsEnum,
                                                              TimeSeriesForecastingMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


@pytest.fixture(scope='session')
def data_setup(request):
    task_type = request.param
    if task_type == 'binary':
        x, y = load_breast_cancer(return_X_y=True)
        task = Task(TaskTypesEnum.classification)
        data_type = DataTypesEnum.table
    elif task_type == 'multiclass':
        x, y = load_wine(return_X_y=True)
        task = Task(TaskTypesEnum.classification)
        data_type = DataTypesEnum.table
    elif task_type == 'regression':
        x, y = load_diabetes(return_X_y=True)
        task = Task(TaskTypesEnum.regression)
        data_type = DataTypesEnum.table
    elif task_type == 'multitarget':
        x, y = load_linnerud(return_X_y=True)
        task = Task(TaskTypesEnum.regression)
        data_type = DataTypesEnum.table
    elif task_type == 'ts':
        file_path = fedot_project_root() / 'test/data/short_time_series.csv'
        df = pd.read_csv(file_path)
        x = y = df['sea_height'].to_numpy()
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))
        data_type = DataTypesEnum.ts
    elif task_type == 'multits':
        file_path = fedot_project_root() / 'test/data/short_time_series.csv'
        df = pd.read_csv(file_path)
        x = df[['wind_speed', 'sea_height']].to_numpy()
        y = df['sea_height'].to_numpy()
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))
        data_type = DataTypesEnum.multi_ts
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

    x, y = x[:100], y[:100]

    # Wrap data into InputData
    input_data = InputData(features=x,
                           target=y,
                           idx=np.arange(len(x)),
                           task=task,
                           data_type=data_type)
    # Train test split
    train_data, test_data = train_test_data_setup(input_data)
    return train_data, test_data


def get_classification_pipeline():
    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='logit', nodes_from=[first])
    third = PipelineNode(operation_type='logit', nodes_from=[first])
    final = PipelineNode(operation_type='logit', nodes_from=[second, third])

    pipeline = Pipeline(final)

    return pipeline


def get_regression_pipeline():
    first = PipelineNode(operation_type='scaling')
    second = PipelineNode(operation_type='linear', nodes_from=[first])
    third = PipelineNode(operation_type='linear', nodes_from=[first])
    final = PipelineNode(operation_type='linear', nodes_from=[second, third])

    pipeline = Pipeline(final)

    return pipeline


def get_ts_pipeline(window_size):
    """ Function return pipeline with lagged transformation in it """
    node_lagged = PipelineNode('lagged')
    node_lagged.parameters = {'window_size': window_size}

    node_final = PipelineNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)
    return pipeline


@pytest.mark.parametrize('metric', ComplexityMetricsEnum)
@pytest.mark.parametrize('data_setup', ['binary'], indirect=True)
def test_structural_quality_metrics(metric, data_setup):
    metric: ComplexityMetricsEnum
    train, _ = data_setup
    pipeline = get_classification_pipeline()
    pipeline.fit(train)
    metric_function = MetricsRepository().metric_by_id(metric)
    expected_metric_values = {
        ComplexityMetricsEnum.structural: 0.43,
        ComplexityMetricsEnum.node_number: 0.40,
        ComplexityMetricsEnum.computation_time: 0.00,
    }
    actual_metric_value = metric_function(pipeline)
    actual_metric_value = round(actual_metric_value, 2)
    assert actual_metric_value == expected_metric_values[metric]


@pytest.mark.parametrize('metric', ClassificationMetricsEnum)
@pytest.mark.parametrize('data_setup', ['binary', 'multiclass'], indirect=True)
def test_classification_quality_metric(data_setup, metric):
    metric: ClassificationMetricsEnum
    train, _ = data_setup
    pipeline = get_classification_pipeline()
    pipeline.fit(input_data=train)
    metric_function = MetricsRepository().metric_by_id(metric)
    metric_value = metric_function(pipeline=pipeline, reference_data=train)
    assert 0 < abs(metric_value) < sys.maxsize
    assert metric_value != metric_function.__self__().default_value


@pytest.mark.parametrize('metric', RegressionMetricsEnum)
@pytest.mark.parametrize('data_setup', ['regression', 'multitarget'], indirect=True)
def test_regression_quality_metric(data_setup, metric):
    metric: RegressionMetricsEnum
    train, _ = data_setup
    pipeline = get_regression_pipeline()
    pipeline.fit(input_data=train)
    metric_function = MetricsRepository().metric_by_id(metric)
    metric_value = metric_function(pipeline=pipeline, reference_data=train)
    assert 0 < abs(metric_value) < sys.maxsize
    assert metric_value != metric_function.__self__().default_value


@pytest.mark.parametrize('metric', TimeSeriesForecastingMetricsEnum)
@pytest.mark.parametrize('data_setup', ['ts', 'multits'], indirect=True)
def test_ts_quality_metric(data_setup, metric):
    metric: TimeSeriesForecastingMetricsEnum
    train, _ = data_setup
    pipeline = get_ts_pipeline(len(train.features) / 3)
    pipeline.fit(input_data=train)
    metric_function = MetricsRepository().metric_by_id(metric)
    metric_value = metric_function(pipeline=pipeline, reference_data=train, validation_blocks=2)
    assert 0 < abs(metric_value) < sys.maxsize
    assert metric_value != metric_function.__self__().default_value


@pytest.mark.parametrize('data_setup', ['multitarget'], indirect=True)
def test_predict_shape_multi_target(data_setup):
    train, test = data_setup
    simple_pipeline = Pipeline(PipelineNode('linear'))
    simple_pipeline.fit(input_data=train)

    target_shape = test.target.shape
    # Get converted data
    results = QualityMetric()._simple_prediction(simple_pipeline, test)
    predict_shape = results.predict.shape
    assert target_shape == predict_shape


def test_roc_auc_multiclass_correct():
    data = InputData(features=np.array([[1, 2], [2, 3], [3, 4], [4, 1]]),
                     target=np.array([['x'], ['y'], ['z'], ['x']]),
                     idx=np.arange(4),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    prediction = OutputData(features=np.array([[1, 2], [2, 3], [3, 4], [4, 1]]),
                            predict=np.array([[0.4, 0.3, 0.3],
                                              [0.2, 0.5, 0.3],
                                              [0.1, 0.2, 0.7],
                                              [0.8, 0.1, 0.1]]),
                            idx=np.arange(4), task=Task(TaskTypesEnum.classification), data_type=DataTypesEnum.table)
    for i in range(data.num_classes):
        fpr, tpr, threshold = ROCAUC.roc_curve(data.target, prediction.predict[:, i],
                                               pos_label=data.class_labels[i])
        roc_auc = ROCAUC.auc(fpr, tpr)
        assert roc_auc
