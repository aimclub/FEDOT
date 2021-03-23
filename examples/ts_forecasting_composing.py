import datetime
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import project_root


def get_source_chain():
    """
    Return chain with the following structure:
    lagged - ridge \
                    -> ridge
    lagged - ridge /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_2 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)

    return chain


def display_validation_metric(predicted, real, actual_values,
                              is_visualise: bool, label: str) -> float:
    """ Function calculate metrics based on predicted and tests data

    :param predicted: predicted values
    :param real: real values
    :param actual_values: source time series
    :param is_visualise: is it needed to show the plots
    :param label: label for display
    """

    # plot results
    if is_visualise:
        plot_results(actual_time_series=actual_values,
                     predicted_values=predicted,
                     len_train_data=len(actual_values)-len(predicted))

    # the quality assessment for the simulation results
    rmse = mse(y_true=real, y_pred=predicted, squared=False)

    print(f'RESULT: {label}, RMSE: {round(rmse, 3)}')

    return rmse


def plot_results(actual_time_series, predicted_values, len_train_data,
                 y_name='Sea surface height, m'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')
    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def run_ts_forecasting_problem(forecast_length=50,
                               with_visualisation=True) -> None:
    """ Function launch time series task with composing

    :param forecast_length: length of the forecast
    :param with_visualisation: is it needed to show the plots
    """
    file_path = '../cases/data/metocean/metocean_data_test.csv'

    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    # Train/test split
    train_part = time_series[:-forecast_length]
    test_part = time_series[-forecast_length:]

    # specify the task to solve
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    # TODO finish this example


if __name__ == '__main__':
    run_ts_forecasting_problem(forecast_length=100,
                               with_visualisation=True)
