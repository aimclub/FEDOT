import datetime
import warnings
import timeit

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.operations.tuning.hyperopt_tune.entire_tuning import ChainTuner
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast(chain, train_data, len_forecast: int, max_window_size: int):
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
    start_time = timeit.default_timer()
    chain.fit_from_scratch(train_input)
    amount_of_seconds = timeit.default_timer()-start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')

    # Predict
    predicted_values = chain.predict(predict_input)
    old_predicted_values = predicted_values.predict

    chain_tuner = ChainTuner(chain=chain, task=task,
                             max_lead_time=datetime.timedelta(minutes=2),
                             iterations=20)
    chain = chain_tuner.tune(input_data=train_input)

    print('\nChain parameters after tuning')
    for node in chain.nodes:
        print(f' Operation {node.operation}, - {node.custom_params}')

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values


def run_experiment(time_series, chain, len_forecast=250):
    # Let's dividide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    old_predicted, new_predicted = make_forecast(chain, train_data,
                                                 len_forecast,
                                                 max_window_size=10)

    old_predicted = np.ravel(np.array(old_predicted))
    new_predicted = np.ravel(np.array(new_predicted))
    test_data = np.ravel(test_data)

    print(f'Predicted values before tuning: {old_predicted[:5]}')
    print(f'Predicted values after tuning: {new_predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')

    mse_before = mean_squared_error(test_data, old_predicted, squared=False)
    mae_before = mean_absolute_error(test_data, old_predicted)
    print(f'RMSE before tuning - {mse_before:.4f}')
    print(f'MAE before tuning - {mae_before:.4f}\n')

    mse_after = mean_squared_error(test_data, new_predicted, squared=False)
    mae_after = mean_absolute_error(test_data, new_predicted)
    print(f'RMSE after tuning - {mse_after:.4f}')
    print(f'MAE after tuning - {mae_after:.4f}\n')

    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_data), len(time_series)), old_predicted, label='Forecast before tuning')
    plt.plot(range(len(train_data), len(time_series)), new_predicted, label='Forecast after tuning')
    plt.legend()
    plt.grid()
    plt.show()


df = pd.read_csv('../notebooks/time_series_forecasting/Sea_level.csv')
time_series = np.array(df['Level'])
if __name__ == '__main__':

    # Chain looking like this
    # --> --> --> --> --> --> --> --> --> --> --> --> --> -->
    # lagged - ridge \
    #                 \
    #                  ridge -> final forecast
    #                 /
    # lagged - ridge /
    # --> --> --> --> --> --> --> --> --> --> --> --> --> -->

    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_2 = PrimaryNode('lagged')

    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    chain = Chain(node_final)
    run_experiment(time_series, chain)
