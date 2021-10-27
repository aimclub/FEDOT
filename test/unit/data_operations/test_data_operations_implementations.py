import os
from itertools import product

import numpy as np
import pandas as pd
import pytest

from examples.classification_with_tuning_example import get_classification_dataset
from examples.regression_with_tuning_example import get_regression_dataset
from examples.time_series.ts_gapfilling_example import generate_synthetic_data
from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.data_operations. \
    sklearn_transformations import ImputationImplementation
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root

np.random.seed(2021)


def get_small_regression_dataset():
    """ Function returns features and target for train and test regression models """
    features_options = {'informative': 2, 'bias': 2.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=70,
                                                              features_amount=4)
    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_small_classification_dataset():
    """ Function returns features and target for train and test classification models """
    features_options = {'informative': 1, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(features_options=features_options,
                                                                  samples_amount=70,
                                                                  features_amount=4,
                                                                  classes_amount=2)
    # Define regression task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_time_series():
    """ Function returns time series for time series forecasting task """
    len_forecast = 100
    synthetic_ts = generate_synthetic_data(length=1000)

    train_data = synthetic_ts[:-len_forecast]
    test_data = synthetic_ts[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, test_data


def get_nan_inf_data():
    train_input = InputData(idx=[0, 1, 2, 3],
                            features=np.array([[1, 2, 3, 4],
                                               [2, np.nan, 4, 5],
                                               [3, 4, 5, np.inf],
                                               [-np.inf, 5, 6, 7]]),
                            target=np.array([1, 2, 3, 4]),
                            task=Task(TaskTypesEnum.regression),
                            data_type=DataTypesEnum.table)

    return train_input


def get_single_feature_data(task=None):
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5],
                            features=np.array([[1], [2], [3], [7], [8], [9]]),
                            target=np.array([[0], [0], [0], [1], [1], [1]]),
                            task=task,
                            data_type=DataTypesEnum.table)

    return train_input


def get_one_hot_encoding_data(task=None):
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5],
                            features=np.array([[1, 0, 1],
                                               [2, 1, 0],
                                               [3, 1, 0],
                                               [7, 1, 1],
                                               [8, 1, 1],
                                               [9, 0, 0]]),
                            target=np.array([[0], [0], [0], [1], [1], [1]]),
                            task=task,
                            data_type=DataTypesEnum.table)

    return train_input


def test_regression_data_operations():
    train_input, predict_input, y_test = get_small_regression_dataset()

    operation_names, _ = OperationTypesRepository('data_operation').suitable_operation(
        task_type=TaskTypesEnum.regression)

    for data_operation in operation_names:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('linear', nodes_from=[node_data_operation])
        pipeline = Pipeline(node_final)

        # Fit and predict for pipeline
        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(predict_input)
        predicted = predicted_output.predict

        assert len(predicted) == len(y_test)


def test_classification_data_operations():
    train_input, predict_input, y_test = get_small_classification_dataset()

    operation_names, _ = OperationTypesRepository('data_operation').suitable_operation(
        task_type=TaskTypesEnum.classification)

    for data_operation in operation_names:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('logit', nodes_from=[node_data_operation])
        pipeline = Pipeline(node_final)

        # Fit and predict for pipeline
        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(predict_input)
        predicted = predicted_output.predict

        assert len(predicted) == len(y_test)


def test_ts_forecasting_lagged_data_operation():
    train_input, predict_input, y_test = get_time_series()

    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    pipeline = Pipeline(node_ridge)

    pipeline.fit_from_scratch(train_input)
    predicted_output = pipeline.predict(predict_input)
    predicted = np.ravel(predicted_output.predict)

    assert len(predicted) == len(np.ravel(y_test))


def test_ts_forecasting_smoothing_data_operation():
    train_input, predict_input, y_test = get_time_series()

    model_names, _ = OperationTypesRepository().operations_with_tag(tags=['smoothing'])

    for smoothing_operation in model_names:
        node_smoothing = PrimaryNode(smoothing_operation)
        node_lagged = SecondaryNode('lagged', nodes_from=[node_smoothing])
        node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
        pipeline = Pipeline(node_ridge)

        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(predict_input)
        predicted = np.ravel(predicted_output.predict)

        assert len(predicted) == len(np.ravel(y_test))


def test_inf_and_nan_absence_after_imputation_implementation_fit_transform():
    input_data = get_nan_inf_data()
    output_data = ImputationImplementation().fit_transform(input_data)

    assert np.sum(np.isinf(output_data.predict)) == 0
    assert np.sum(np.isnan(output_data.predict)) == 0


def test_inf_and_nan_absence_after_imputation_implementation_fit_and_transform():
    input_data = get_nan_inf_data()
    imputer = ImputationImplementation()
    imputer.fit(input_data)
    output_data = imputer.transform(input_data)

    assert np.sum(np.isinf(output_data.predict)) == 0
    assert np.sum(np.isnan(output_data.predict)) == 0


def test_inf_and_nan_absence_after_pipeline_fitting_from_scratch():
    train_input = get_nan_inf_data()

    model_names, _ = OperationTypesRepository().suitable_operation(task_type=TaskTypesEnum.regression)

    for data_operation in model_names:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('linear', nodes_from=[node_data_operation])
        pipeline = Pipeline(node_final)

        # Fit and predict for pipeline
        pipeline.fit_from_scratch(train_input)
        predicted_output = pipeline.predict(train_input)
        predicted = predicted_output.predict

        assert np.sum(np.isinf(predicted)) == 0
        assert np.sum(np.isnan(predicted)) == 0


def test_feature_selection_of_single_features():
    for task_type in [TaskTypesEnum.classification, TaskTypesEnum.regression]:
        model_names, _ = OperationTypesRepository(operation_type='data_operation') \
            .suitable_operation(tags=['feature_selection'], task_type=task_type)

        task = Task(task_type)
        data_functions = [get_single_feature_data(task), get_one_hot_encoding_data(task)]
        list_with_operations = list(product(model_names, data_functions))

        for data_operation, train_input in list_with_operations:
            node_data_operation = PrimaryNode(data_operation)

            assert node_data_operation.fitted_operation is None

            # Fit and predict for pipeline
            node_data_operation.fit(train_input)
            predicted_output = node_data_operation.predict(train_input)
            predicted = predicted_output.predict

            assert node_data_operation.fitted_operation is not None
            assert predicted.shape == train_input.features.shape


def test_one_hot_encoding_new_category_in_test():
    file_path_train = 'cases/data/river_levels/station_levels.csv'
    full_path_train = os.path.join(str(fedot_project_root()), file_path_train)
    data = pd.read_csv(full_path_train)
    test_features = data[data['month'] == 'Декабрь']
    test_target = test_features['level_station_2']
    test_features.drop(['level_station_2'], axis=1, inplace=True)

    train_features = data[data['month'] != 'Декабрь']
    train_target = train_features['level_station_2']
    train_features.drop(['level_station_2'], axis=1, inplace=True)

    test_data = InputData(idx=np.arange(0, len(test_features)),
                          features=np.array(test_features), target=np.array(test_target),
                          task=Task(TaskTypesEnum.regression),
                          data_type=DataTypesEnum.table)

    train_data = InputData(idx=np.arange(0, len(train_features)),
                           features=np.array(train_features),
                           target=np.array(train_target),
                           task=Task(TaskTypesEnum.classification),
                           data_type=DataTypesEnum.table)

    pipeline = Pipeline()
    one_hot_node = PrimaryNode('one_hot_encoding')
    final_node = SecondaryNode('dt', nodes_from=[one_hot_node])
    pipeline.add_node(final_node)
    pipeline.fit(train_data)

    with pytest.raises(ValueError):
        pipeline.predict(test_data)
