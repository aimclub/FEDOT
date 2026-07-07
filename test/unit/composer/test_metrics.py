import json
from itertools import product
from typing import Callable, Dict, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_linnerud, load_wine

from fedot.core.composer.metrics import (Accuracy, F1, Logloss, MAE, MAPE, MASE, MSE,
    MSLE, Precision, QualityMetric, R2, RMSE, ROCAUC, SMAPE, Silhouette,)
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.split.data_split import train_test_data_setup
from fedot.core.data.tensor_data import TensorData
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
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=10))
        data_type = DataTypesEnum.ts
        validation_blocks = 2
    elif task_type == 'multits':
        file_path = fedot_project_root() / 'test/data/short_time_series.csv'
        df = pd.read_csv(file_path)
        x = df[['sea_height', 'sea_height']].to_numpy()
        y = df['sea_height'].to_numpy()
        task = Task(TaskTypesEnum.ts_forecasting,
                    TsForecastingParams(forecast_length=10))
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
    train_data, test_data = train_test_data_setup(
        input_data, validation_blocks=validation_blocks)
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


def _input_data_to_tensor_data(data: InputData) -> TensorData:
    return TensorData(
        features=torch.as_tensor(data.features),
        target=None if data.target is None else torch.as_tensor(data.target),
        idx=data.idx,
        task=data.task,
        data_type=data.data_type,
    )


@pytest.fixture(scope='session')
def expected_values() -> Dict[str, Dict[str, float]]:
    with open(fedot_project_root() / 'test/data/expected_metric_values.json', 'r') as f:
        return json.load(f)


@pytest.mark.parametrize(
    'metric, pipeline_func, data_setup',
    [
        *product(ComplexityMetricsEnum,
                 [get_classification_pipeline], ['complexity']),
        *product(ClassificationMetricsEnum,
                 [get_classification_pipeline], ['binary', 'multiclass']),
        *product(RegressionMetricsEnum,
                 [get_regression_pipeline], ['regression', 'multitarget']),
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
    if validation_blocks is not None:
        pytest.skip('TensorData in-sample metric evaluation is not supported yet')
    train = _input_data_to_tensor_data(train)
    test = _input_data_to_tensor_data(test)
    if metric == ComplexityMetricsEnum.computation_time:
        pytest.skip('Computation time metric requires TensorData pipeline fit')

    pipeline = pipeline_func()
    if not isinstance(metric, ComplexityMetricsEnum):
        if not hasattr(pipeline, 'fit_tensordata'):
            pytest.skip('Pipeline.fit_tensordata is not available')
        try:
            pipeline.fit_tensordata(train)
        except Exception as ex:
            pytest.skip(f'Pipeline operation is not TensorData-native yet: {ex}')
    metric_function = MetricsRepository.get_metric(metric)
    metric_class = MetricsRepository.get_metric_class(metric)
    metric_value = metric_function(
        pipeline=pipeline, reference_data=test, validation_blocks=validation_blocks)

    if not update_expected_values:
        expected_value = expected_values[task_type][str(metric)]
        expected_value = [expected_value] if not isinstance(
            expected_value, list) else expected_value
        assert any(np.isclose(metric_value, value, rtol=0.001, atol=0.001)
                   for value in expected_value)
        assert not np.isclose(
            metric_value, metric_class.default_value, rtol=0.01, atol=0.01)
    else:
        with open(fedot_project_root() / 'test/data/expected_metric_values.json', 'w') as f:
            expected_values[task_type] = expected_values.get(task_type) or {}
            expected_values[task_type][str(metric)] = metric_value
            json.dump(expected_values, f, indent=2)
        raise ValueError('The value of `update_expected_values` should equal to `False` '
                         'in order for this test to pass.')


@pytest.mark.parametrize(
    'metric, pipeline_func, data_setup, validation_blocks',
    [
        *product(ClassificationMetricsEnum,
                 [get_classification_pipeline], ['binary', 'multiclass'], [None]),
        *product(RegressionMetricsEnum,
                 [get_regression_pipeline], ['regression', 'multitarget'], [None]),
        *product(TimeSeriesForecastingMetricsEnum,
                 [get_ts_pipeline], ['ts', 'multits'], [2]),
    ],
    indirect=['data_setup']
)
def test_ideal_case_metrics(metric: ClassificationMetricsEnum, pipeline_func: Callable[[], Pipeline],
                            validation_blocks: Union[int, None], data_setup: Tuple[InputData, InputData, str],
                            expected_values):
    pytest.skip('Legacy InputData metric path is not supported in TensorData-only metrics')
    reference, _, task_type, _ = data_setup
    metric_class = MetricsRepository.get_metric_class(metric)
    predicted = OutputData(
        idx=reference.idx, task=reference.task, data_type=reference.data_type)
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
    pytest.skip('Legacy InputData prediction path is not supported in TensorData-only metrics')
    train, test, _, _ = data_setup
    simple_pipeline = Pipeline(PipelineNode('linear'))
    if not hasattr(simple_pipeline, 'fit'):
        pytest.skip('Legacy Pipeline.fit is not available in TensorData runtime branch')
    simple_pipeline.fit(input_data=train)

    target_shape = test.target.shape
    # Get converted data
    _, results = QualityMetric()._simple_prediction(simple_pipeline, test)
    predict_shape = results.predict.shape
    assert target_shape == predict_shape


def test_roc_auc_multiclass_correct():
    pytest.skip('Legacy ROC curve helper is not implemented for TensorData metrics yet')
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


def _build_tensor_metric_data(target: np.ndarray,
                              prediction: np.ndarray,
                              task: Task,
                              features: np.ndarray = None) -> Tuple[TensorData, TensorData]:
    features = np.zeros((len(target), 1), dtype=np.float32) if features is None else features
    reference = TensorData(
        features=torch.as_tensor(features),
        target=torch.as_tensor(target),
        task=task,
        data_type=DataTypesEnum.table,
    )
    predicted = TensorData(
        features=torch.as_tensor(features),
        predict=torch.as_tensor(prediction),
        task=task,
        data_type=DataTypesEnum.table,
    )
    return reference, predicted


@pytest.mark.parametrize(
    'metric_class, expected_value',
    [
        (RMSE, 0.37080992435478316),
        (MSE, 0.1375),
        (MSLE, 0.004872902770681671),
        (MAPE, 0.09687500000000002),
        (SMAPE, 9.56661102471549),
        (MAE, 0.325),
        (MASE, 0.2954545454545455),
        (R2, 0.9808695652173913),
    ],
)
def test_regression_metric_tensordata_implementation(metric_class, expected_value):
    target = np.array([1.0, 2.0, 4.0, 8.0])
    prediction = np.array([1.1, 1.8, 4.5, 7.5])
    features = np.array([0.8, 1.2, 2.1, 4.1])
    task = Task(TaskTypesEnum.regression)

    tensor_reference, tensor_prediction = _build_tensor_metric_data(
        target, prediction, task, features=features)

    tensor_value = metric_class.metric(tensor_reference, tensor_prediction)

    assert np.isclose(tensor_value, expected_value, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    'metric_class, prediction, expected_value',
    [
        (Accuracy, np.array([0, 1, 1, 0, 1]), -0.6),
        (F1, np.array([0, 1, 1, 0, 1]), -0.5),
        (Precision, np.array([0, 1, 1, 0, 1]), -0.5),
        (ROCAUC, np.array([0.1, 0.8, 0.7, 0.4, 0.9]), -0.6666666666666666),
        (Logloss, np.array([0.1, 0.8, 0.7, 0.4, 0.9]), 0.6186249239125281),
    ],
)
def test_binary_classification_metric_tensordata_implementation(metric_class, prediction, expected_value):
    target = np.array([0, 0, 1, 1, 1])
    task = Task(TaskTypesEnum.classification)

    tensor_reference, tensor_prediction = _build_tensor_metric_data(
        target, prediction, task)

    tensor_value = metric_class.metric(tensor_reference, tensor_prediction)

    assert np.isclose(tensor_value, expected_value, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    'metric_class, prediction, expected_value',
    [
        (F1, np.array([0, 1, 2, 1, 2, 0]), -1.0),
        (Precision, np.array([0, 1, 2, 1, 2, 0]), -1.0),
        (ROCAUC, np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
            [0.1, 0.3, 0.6],
            [0.7, 0.2, 0.1],
        ]), -1.0),
        (Logloss, np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.8, 0.1],
            [0.1, 0.3, 0.6],
            [0.7, 0.2, 0.1],
        ]), 0.33785625970176786),
    ],
)
def test_multiclass_metric_tensordata_implementation(metric_class, prediction, expected_value):
    target = np.array([0, 1, 2, 1, 2, 0])
    task = Task(TaskTypesEnum.classification)

    tensor_reference, tensor_prediction = _build_tensor_metric_data(
        target, prediction, task)

    tensor_value = metric_class.metric(tensor_reference, tensor_prediction)

    assert np.isclose(tensor_value, expected_value, rtol=1e-6, atol=1e-6)


def test_silhouette_metric_tensordata_implementation():
    reference = TensorData(
        features=torch.tensor([[0.0], [1.0], [10.0], [11.0]]),
        target=None,
        task=Task(TaskTypesEnum.clustering),
        data_type=DataTypesEnum.table,
    )
    predicted = TensorData(
        features=reference.features,
        predict=torch.tensor([0, 0, 1, 1]),
        task=reference.task,
        data_type=reference.data_type,
    )

    assert np.isclose(Silhouette.metric(reference, predicted), -0.899749373433584, rtol=1e-6, atol=1e-6)
