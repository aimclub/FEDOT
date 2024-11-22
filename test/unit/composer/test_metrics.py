import json
import sys
from itertools import product
from typing import Callable, Dict, Tuple, Union

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
from fedot.core.repository.metrics_repository import (ClassificationMetricsEnum, ComplexityMetricsEnum,
                                                      MetricsRepository, RegressionMetricsEnum,
                                                      TimeSeriesForecastingMetricsEnum)
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root, set_random_seed


@pytest.fixture(scope='session')
def data_setup(request):
    task_type = request.param
    validation_blocks = None
    if task_type in ('binary', 'complexity'):
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
        validation_blocks = 2
    elif task_type == 'multits':
        file_path = fedot_project_root() / 'test/data/short_time_series.csv'
        df = pd.read_csv(file_path)
        x = df[['sea_height', 'sea_height']].to_numpy()
        y = df['sea_height'].to_numpy()
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))
        data_type = DataTypesEnum.multi_ts
        validation_blocks = 2
    else:
        raise ValueError(f'Unsupported task type: {task_type}')

    x, y = x[:200], y[:200]

    # Wrap data into InputData
    input_data = InputData(features=x,
                           target=y,
                           idx=np.arange(len(x)),
                           task=task,
                           data_type=data_type)
    # Train test split
    train_data, test_data = train_test_data_setup(input_data, validation_blocks=validation_blocks)
    return train_data, test_data, task_type, validation_blocks


def get_classification_pipeline():
    first = PipelineNode(operation_type='logit')
    second = PipelineNode(operation_type='logit', nodes_from=[first])
    third = PipelineNode(operation_type='logit', nodes_from=[first])
    final = PipelineNode(operation_type='logit', nodes_from=[second, third])

    pipeline = Pipeline(final)

    return pipeline


def get_regression_pipeline():
    first = PipelineNode(operation_type='scaling')
    final = PipelineNode(operation_type='linear', nodes_from=[first])

    pipeline = Pipeline(final)

    return pipeline


def get_ts_pipeline(window_size=30):
    """ Function return pipeline with lagged transformation in it """
    node_lagged = PipelineNode('lagged')
    node_lagged.parameters = {'window_size': window_size}

    node_final = PipelineNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_final)
    return pipeline


@pytest.fixture(scope='session')
def expected_values() -> Dict[str, Dict[str, float]]:
    with open(fedot_project_root() / 'test/data/expected_metric_values.json', 'r') as f:
        return json.load(f)


@pytest.mark.parametrize(
    'metric, pipeline_func, data_setup',
    [  # TODO: Add binary classification to the test after completion of https://github.com/aimclub/FEDOT/issues/1221.
        *product(ComplexityMetricsEnum, [get_classification_pipeline], ['complexity']),
        *product(ClassificationMetricsEnum, [get_classification_pipeline], ['multiclass']),
        *product(RegressionMetricsEnum, [get_regression_pipeline], ['regression', 'multitarget']),
        *product(TimeSeriesForecastingMetricsEnum, [get_ts_pipeline], ['ts', 'multits'])
    ],
    indirect=['data_setup']
)
def test_metrics(metric: ClassificationMetricsEnum, pipeline_func: Callable[[], Pipeline],
                 data_setup: Tuple[InputData, InputData, str, Union[int, None]],
                 expected_values: Dict[str, Dict[str, float]]):
    set_random_seed(0)
    update_expected_values: bool = False

    train, test, task_type, validation_blocks = data_setup

    pipeline = pipeline_func()
    pipeline.fit(input_data=train)
    metric_function = MetricsRepository.get_metric(metric)
    metric_class = MetricsRepository.get_metric_class(metric)
    metric_value = metric_function(pipeline=pipeline, reference_data=test, validation_blocks=validation_blocks)

    if not update_expected_values:
        expected_value = expected_values[task_type][str(metric)]

        if isinstance(expected_value, list):
            expression_expected_value = []

            for value in expected_value:
                expression_expected_value.append(np.isclose(metric_value, value, rtol=0.001, atol=0.001))
            assert any(expression_expected_value)

        else:
            assert np.isclose(metric_value, expected_value, rtol=0.001, atol=0.001)

        assert not np.isclose(metric_value, metric_class.default_value, rtol=0.01, atol=0.01)
    else:
        with open(fedot_project_root() / 'test/data/expected_metric_values.json', 'w') as f:
            expected_values[task_type] = expected_values.get(task_type) or {}
            expected_values[task_type][str(metric)] = metric_value
            json.dump(expected_values, f, indent=2)
        raise ValueError('The value of `update_expected_values` should equal to `False` '
                         'in order for this test to pass.')


@pytest.mark.parametrize(
    'metric, pipeline_func, data_setup',
    [
        *product(ClassificationMetricsEnum, [get_classification_pipeline], ['binary']),
    ],
    indirect=['data_setup']
)
def test_binary_classification(metric: ClassificationMetricsEnum, pipeline_func: Callable[[], Pipeline],
                               data_setup: Tuple[InputData, InputData, str, Union[int, None]],
                               expected_values: Dict[str, Dict[str, float]]):
    train, test, task_type, validation_blocks = data_setup

    pipeline = pipeline_func()
    pipeline.fit(input_data=train)
    metric_function = MetricsRepository.get_metric(metric)
    metric_class = MetricsRepository.get_metric_class(metric)
    metric_value = metric_function(pipeline=pipeline, reference_data=test, validation_blocks=validation_blocks)

    assert not np.isclose(metric_value, metric_class.default_value, rtol=0.01, atol=0.01)
    assert 0 < abs(metric_value) < sys.maxsize


@pytest.mark.parametrize(
    'metric, pipeline_func, data_setup, validation_blocks',
    [
        *product(ClassificationMetricsEnum, [get_classification_pipeline], ['binary', 'multiclass'], [None]),
        *product(RegressionMetricsEnum, [get_regression_pipeline], ['regression', 'multitarget'], [None]),
        *product(TimeSeriesForecastingMetricsEnum, [get_ts_pipeline], ['ts', 'multits'], [2]),
    ],
    indirect=['data_setup']
)
def test_ideal_case_metrics(metric: ClassificationMetricsEnum, pipeline_func: Callable[[], Pipeline],
                            validation_blocks: Union[int, None], data_setup: Tuple[InputData, InputData, str],
                            expected_values):
    reference, _, task_type, _ = data_setup
    metric_class = MetricsRepository.get_metric_class(metric)
    predicted = OutputData(idx=reference.idx, task=reference.task, data_type=reference.data_type)
    if task_type == 'multiclass' and metric_class.output_mode != 'labels':
        label_vals = np.unique(reference.target)
        predicted.predict = np.identity(len(label_vals))[reference.target]
    else:
        predicted.predict = reference.target
    if task_type == 'multits':
        reference.features = reference.features[:, 0]

    ideal_value = metric_class.metric(reference, predicted)

    assert ideal_value != metric_class.default_value


@pytest.mark.parametrize('data_setup', ['multitarget'], indirect=True)
def test_predict_shape_multi_target(data_setup: Tuple[InputData, InputData, str]):
    train, test, _, _ = data_setup
    simple_pipeline = Pipeline(PipelineNode('linear'))
    simple_pipeline.fit(input_data=train)

    target_shape = test.target.shape
    # Get converted data
    _, results = QualityMetric()._simple_prediction(simple_pipeline, test)
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
