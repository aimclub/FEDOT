import os
import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum
from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast(pipeline, train: InputData, predict: InputData,
                  train_exog: InputData, predict_exog: InputData):
    """
    Function for predicting values in a time series

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()

    second_node_name = 'exog_ts'

    if train_exog is None:
        second_node_name = 'data_source_ts/2'
        train_exog = train
        predict_exog = predict

    train_dataset = MultiModalData({
        'data_source_ts/1': train,
        second_node_name: train_exog})

    predict_dataset = MultiModalData({
        'data_source_ts/1': predict,
        second_node_name: predict_exog})

    pipeline.fit_from_scratch(train_dataset)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    # Predict
    predicted_values = pipeline.predict(predict_dataset)
    predicted_values = predicted_values.predict

    return predicted_values


def run_exogenous_experiment(path_to_file, len_forecast=250, with_exog=True,
                             visualization=False) -> None:
    """ Function with example how time series forecasting can be made with using
    exogenous features

    :param path_to_file: path to the csv file with dataframe
    :param len_forecast: forecast length
    :param with_exog: is it needed to make prediction with exogenous time series
    :param visualization: is it needed to make visualizations
    """

    df = pd.read_csv(path_to_file)
    time_series = np.array(df['Level'])
    exog_variable = np.array(df['Neighboring level'])

    # Source time series
    train_input, predict_input = train_test_data_setup(InputData(idx=range(len(time_series)),
                                                                 features=time_series,
                                                                 target=time_series,
                                                                 task=Task(TaskTypesEnum.ts_forecasting,
                                                                           TsForecastingParams(
                                                                               forecast_length=len_forecast)),
                                                                 data_type=DataTypesEnum.ts))

    # Exogenous time series
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=len_forecast))

    predict_input_exog = InputData(idx=np.arange(len(exog_variable)),
                                   features=exog_variable, target=time_series,
                                   task=task, data_type=DataTypesEnum.ts)

    train_input_exog, predict_input_exog = train_test_data_setup(predict_input_exog)

    if with_exog is True:
        # Example with exogenous time series
        node_source = PrimaryNode('data_source_ts/1')
        node_lagged = SecondaryNode('lagged', nodes_from=[node_source])

        node_exog = PrimaryNode('exog_ts')

        node_final = SecondaryNode('ridge', nodes_from=[node_lagged, node_exog])
        pipeline = Pipeline(node_final)
    else:
        # Simple example without exogenous time series
        node_source_1 = PrimaryNode('data_source_ts/1')
        node_source_2 = PrimaryNode('data_source_ts/2')

        node_lagged_1 = SecondaryNode('lagged', nodes_from=[node_source_1])
        node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_source_2])

        node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
        node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])
        node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
        train_input_exog = None
        predict_input_exog = None
        pipeline = Pipeline(node_final)

    predicted = make_forecast(pipeline, train_input, predict_input, train_input_exog, predict_input_exog)

    predicted = np.ravel(np.array(predicted))
    test_data = np.ravel(predict_input.target)

    print(f'Predicted values: {predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')

    if visualization:
        plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
        plt.plot(range(len(train_input.target), len(time_series)), predicted, label='Forecast')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    data_path = os.path.join(f'{fedot_project_root()}', 'examples/data/ts', 'ts_sea_level.csv')
    run_exogenous_experiment(path_to_file=data_path, len_forecast=250, with_exog=True, visualization=True)
