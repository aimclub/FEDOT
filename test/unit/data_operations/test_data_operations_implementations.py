import numpy as np

from fedot.core.data.data import InputData
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

from examples.regression_with_tuning_example import get_regression_dataset
from examples.classification_with_tuning_example import get_classification_dataset
from examples.ts_gapfilling_example import generate_synthetic_data

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


def test_regression_data_operations():
    train_input, predict_input, y_test = get_small_regression_dataset()

    for data_operation in ['kernel_pca', 'pca', 'scaling', 'normalization',
                           'poly_features', 'ransac_lin_reg', 'ransac_non_lin_reg',
                           'rfe_lin_reg', 'rfe_non_lin_reg', 'simple_imputation']:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('linear', nodes_from=[node_data_operation])
        chain = Chain(node_final)

        # Fit and predict for chain
        chain.fit_from_scratch(train_input)
        predicted_output = chain.predict(predict_input)
        predicted = predicted_output.predict

        assert len(predicted) == len(y_test)


def test_classification_data_operations():
    train_input, predict_input, y_test = get_small_classification_dataset()

    for data_operation in ['kernel_pca', 'pca', 'scaling', 'normalization',
                           'poly_features', 'rfe_lin_class', 'rfe_non_lin_class']:
        node_data_operation = PrimaryNode(data_operation)
        node_final = SecondaryNode('logit', nodes_from=[node_data_operation])
        chain = Chain(node_final)

        # Fit and predict for chain
        chain.fit_from_scratch(train_input)
        predicted_output = chain.predict(predict_input)
        predicted = predicted_output.predict

        assert len(predicted) == len(y_test)


def test_ts_forecasting_lagged_data_operation():
    train_input, predict_input, y_test = get_time_series()

    node_lagged = PrimaryNode('lagged')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
    chain = Chain(node_ridge)

    chain.fit_from_scratch(train_input)
    predicted_output = chain.predict(predict_input)
    predicted = np.ravel(predicted_output.predict)

    assert len(predicted) == len(np.ravel(y_test))


def test_ts_forecasting_smoothing_data_operation():
    train_input, predict_input, y_test = get_time_series()

    for smoothing_operation in ['smoothing', 'gaussian_filter']:
        node_smoothing = PrimaryNode(smoothing_operation)
        node_lagged = SecondaryNode('lagged', nodes_from=[node_smoothing])
        node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])
        chain = Chain(node_ridge)

        chain.fit_from_scratch(train_input)
        predicted_output = chain.predict(predict_input)
        predicted = np.ravel(predicted_output.predict)

        assert len(predicted) == len(np.ravel(y_test))
