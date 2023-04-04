import os
import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TsForecastingParams, TaskTypesEnum
from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')
np.random.seed(2020)



def plot_results(actual_time_series, predicted_values, len_train_data, y_name='Parameter'):
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
             [min(actual_time_series), max(actual_time_series)], c='black', linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()

def run_exogenous_experiment(path_to_file, len_forecast=250, with_exog=True,
                             visualization=False) -> None:
    """ Function with example how time series forecasting can be made with using
    exogenous features

    :param path_to_file: path to the csv file with dataframe
    :param len_forecast: forecast length
    :param with_exog: is it needed to make prediction with exogenous time series
    :param visualization: is it needed to make visualizations
    """

    # Read the file
    df = pd.read_csv(path_to_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Specify forecast length
    len_forecast = 250

    # Got train, test parts, and the entire data
    true_values = np.array(df['Level'])
    train_array = true_values[:-len_forecast]
    test_array = true_values[-len_forecast:]

    exog_arr = np.array(df['Neighboring level'])
    exog_train = exog_arr[:-len_forecast]
    exog_test = exog_arr[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=250))

    # Data for lagged transformation
    train_lagged = InputData(idx=np.arange(0, len(train_array)),
                             features=train_array,
                             target=train_array,
                             task=task,
                             data_type=DataTypesEnum.ts)
    start_forecast = len(train_array)
    end_forecast = start_forecast + len_forecast
    predict_lagged = InputData(idx=np.arange(start_forecast, end_forecast),
                               features=train_array,
                               target=test_array,
                               task=task,
                               data_type=DataTypesEnum.ts)

    # Data for exog operation
    train_exog = InputData(idx=np.arange(0, len(exog_train)),
                           features=exog_train,
                           target=train_array,
                           task=task,
                           data_type=DataTypesEnum.ts)
    start_forecast = len(exog_train)
    end_forecast = start_forecast + len_forecast
    predict_exog = InputData(idx=np.arange(start_forecast, end_forecast),
                             features=exog_test,
                             target=test_array,
                             task=task,
                             data_type=DataTypesEnum.ts)

    if with_exog:
        train_dataset = MultiModalData({
            'lagged': train_lagged,
            'exog_ts': train_exog
        })

        predict_dataset = MultiModalData({
            'lagged': predict_lagged,
            'exog_ts': predict_exog
        })

        # Create a pipeline with different data sources in th nodes
        node_lagged = PipelineNode('lagged')
        node_exog = PipelineNode('exog_ts')
        node_ridge = PipelineNode('ridge', nodes_from=[node_lagged, node_exog])
        pipeline = Pipeline(node_ridge)
    else:
        train_dataset = train_lagged
        predict_dataset = predict_lagged

        # Simple example without exogenous time series
        node_lagged = PipelineNode('lagged')
        node_final = PipelineNode('ridge', nodes_from=[node_lagged])
        pipeline = Pipeline(node_final)

    # Fit it
    fedot = Fedot(problem='classification', timeout=5, initial_assumption=pipeline)
    fedot.fit(train_dataset)
    pipeline.fit(predict_dataset)

    # Predict
    predicted = pipeline.predict(predict_dataset)
    predicted_values = np.ravel(np.array(predicted.predict))

    if visualization:
        pipeline.show()
        # Plot predictions and true values
        plot_results(actual_time_series=true_values,
                     predicted_values=predicted_values,
                     len_train_data=len(train_array),
                     y_name='Sea level, m')

    # Print MAE metric
    print(f'Mean absolute error: {mean_absolute_error(test_array, predicted_values):.3f}')


if __name__ == '__main__':
    data_path = os.path.join(f'{fedot_project_root()}', 'examples/data/ts', 'ts_sea_level.csv')
    run_exogenous_experiment(path_to_file=data_path, len_forecast=250, with_exog=True, visualization=True)
