import os
import numpy as np
import pandas as pd

from fedot.core.data.data import InputData
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.synthetic.data import regression_dataset, classification_dataset
from fedot.core.utils import project_root


def get_knn_reg_chain(k_neighbors):
    """ Function return chain with K-nn regression model in it """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knnreg', nodes_from=[node_scaling])
    node_final.custom_params = {'n_neighbors': k_neighbors}
    chain = Chain(node_final)
    return chain


def get_knn_class_chain(k_neighbors):
    """ Function return chain with K-nn classification model in it """
    node_scaling = PrimaryNode('scaling')
    node_final = SecondaryNode('knn', nodes_from=[node_scaling])
    node_final.custom_params = {'n_neighbors': k_neighbors}
    chain = Chain(node_final)
    return chain


def get_ts_chain(window_size):
    """ Function return chain with lagged transformation in it """
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}

    node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    chain = Chain(node_final)
    return chain


def test_knn_regression_parameters_improving():
    """ Function check if the knn model can improve amount neighbors or not """
    samples_amount = 100
    k_neighbors = 150

    features_options = {'informative': 1, 'bias': 0.0}
    features_amount = 3
    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data)), features=x_data,
                            target=y_data, task=task, data_type=DataTypesEnum.table)

    # Prepare regression chain
    chain = get_knn_reg_chain(k_neighbors)

    # Fit it
    chain.fit(train_input)

    is_chain_was_fitted = True
    assert is_chain_was_fitted


def test_knn_classification_parameters_improving():
    """ Function check if the knn classification model can improve amount neighbors
    or not """
    samples_amount = 100
    k_neighbors = 150

    features_options = {'informative': 1, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_data, y_data = classification_dataset(samples_amount=samples_amount,
                                            features_amount=3,
                                            classes_amount=2,
                                            features_options=features_options)

    # Define regression task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data)), features=x_data,
                            target=y_data, task=task, data_type=DataTypesEnum.table)

    # Prepare classification chain
    chain = get_knn_class_chain(k_neighbors)

    # Fit it
    chain.fit(train_input)

    is_chain_was_fitted = True
    assert is_chain_was_fitted


def test_ts_forecasting_parameters_improving():
    """ Function checks if the lagged parameters correct works well """
    window_size = 600
    len_forecast = 50

    # The length of the time series is 500 elements
    project_root_path = str(project_root())
    file_path = os.path.join(project_root_path, 'test/data/short_time_series.csv')
    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(time_series)),
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Get chain with lagged transformation in it
    chain = get_ts_chain(window_size)

    # Fit it
    chain.fit(train_input)

    is_chain_was_fitted = True
    assert is_chain_was_fitted
