import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.chains.ts_chain import TsForecastingChain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from examples.time_series_gapfilling_example import generate_synthetic_data

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

np.random.seed(2020)


def make_forecast_old(chain, train_data, len_forecast: int, max_window_size: int):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_data: one-dimensional numpy array to train chain
    :param len_forecast: amount of values for predictions
    :param max_window_size: moving window size

    :return predicted_values: numpy array, forecast of model
    """

    # Here we define which task should we use, here we also define two main
    # hyperparameters: forecast_length and max_window_size
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast,
                                    max_window_size=max_window_size,
                                    return_all_steps=False,
                                    make_future_prediction=True))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Make a "blank", here we need just help FEDOT understand that the
    # forecast should be made exactly the "len_forecast" length
    predict_input = InputData(idx=np.arange(0, len_forecast),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.forecast(initial_data=train_input,
                                      supplementary_data=predict_input).predict

    return predicted_values


def make_forecast_new(chain, train_data, len_forecast: int, max_window_size: int):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_data: one-dimensional numpy array to train chain
    :param len_forecast: amount of values for predictions
    :param max_window_size: moving window size

    :return predicted_values: numpy array, forecast of model
    """

    # Here we define which task should we use, here we also define two main
    # hyperparameters: forecast_length and max_window_size
    task = Task(TaskTypesEnum.regression,
                TsForecastingParams(forecast_length=len_forecast,
                                    max_window_size=max_window_size,
                                    return_all_steps=False,
                                    make_future_prediction=True))

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(train_data)),
                            features=train_data,
                            target=train_data,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Make a "blank", here we need just help FEDOT understand that the
    # forecast should be made exactly the "len_forecast" length
    predict_input = InputData(idx=np.arange(0, len(train_data)),
                              features=train_data,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)

    return predicted_values


def run_experiment_old(time_series, chain, len_forecast = 150):
    # Let's dividide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    predicted = make_forecast_old(chain, train_data, len_forecast,
                                  max_window_size=2)

    predicted = np.ravel(np.array(predicted))
    test_data = np.ravel(test_data)
    print(f'Predicted values: {predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')
    print(f'RMSE - {mean_squared_error(test_data, predicted, squared=False):.2f}\n')


def run_experiment_new(time_series, chain, len_forecast = 250):
    # Let's dividide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    predicted = make_forecast_new(chain, train_data, len_forecast,
                                  max_window_size=10)

    predicted = np.ravel(np.array(predicted.predict))
    test_data = np.ravel(test_data)
    print(f'Predicted values: {predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')
    print(f'RMSE - {mean_squared_error(test_data, predicted, squared=False):.2f}\n')
    print(f'MAE - {mean_absolute_error(test_data, predicted):.2f}\n')

    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_data), len(time_series)), predicted, label='Forecast')
    plt.legend()
    plt.grid()
    plt.show()


df = pd.read_csv('../notebooks/time_series_forecasting/Sea_level.csv')
if __name__ == '__main__':
    #time_series = generate_synthetic_data()
    time_series = np.array(df['Level'])

    print('New scaling "node-based" preprocessing functionality')
    # Chain looking like this
    # --> --> --> --> --> --> --> --> --> --> --> --> --> -->
    #            / lagged - linear \
    #           /                   \
    # smoothing|                     ridge -> final forecast
    #           \                   /
    #            \      arima      /
    # --> --> --> --> --> --> --> --> --> --> --> --> --> -->

    # node_smooth = PrimaryNode('smoothing')
    # node_smooth.custom_params = {'window_size': 50}
    #
    # node_lagged = SecondaryNode('lagged', nodes_from=[node_smooth])
    # node_lagged.custom_params = {'window_size': 50}
    # node_linear = SecondaryNode('ridge', nodes_from=[node_lagged])
    #
    # node_arima = SecondaryNode('arima', nodes_from=[node_smooth])
    # node_arima.custom_params = {'order': (2, 0, 4)}
    #
    # node_final = SecondaryNode('dtreg', nodes_from=[node_arima, node_linear])
    # chain = Chain(node_final)

    # node_lagged = PrimaryNode('lagged')
    # node_lagged.custom_params = {'window_size': 400}
    # node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
    # chain = Chain(node_final)

    node_final = PrimaryNode('arima')
    node_final.custom_params = {'order': (10, 0, 4)}
    chain = Chain(node_final)
    run_experiment_new(time_series, chain)
