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
from fedot.core.operations.tuning.hyperopt_tune.\
    tuners import SequentialTuner, ChainTuner
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast(chain, train_input, predict_input, task):
    """
    Function for predicting values in a time series

    :param chain: TsForecastingChain object
    :param train_input: InputData for fit
    :param predict_input: InputData for predict
    :param task: Ts_forecasting task

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()
    chain.fit_from_scratch(train_input, verbose=True)
    amount_of_seconds = timeit.default_timer()-start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')

    # Predict
    predicted_values = chain.predict(predict_input)
    old_predicted_values = predicted_values.predict

    chain_tuner = ChainTuner(chain=chain, task=task,
                             iterations=5)
    chain = chain_tuner.tune_chain(input_data=train_input,
                                   loss_function=mean_absolute_error)

    print('\nChain parameters after tuning')
    for node in chain.nodes:
        print(f' Operation {node.operation}, - {node.custom_params}')

    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)
    new_predicted_values = predicted_values.predict

    return old_predicted_values, new_predicted_values


def get_chain():
    """
    Chain looking like this
    lagged - ridge \
                    \
          AR ------> ridge -> final forecast
                    /
        ARIMA      /
    """

    node_arima = PrimaryNode('ar')

    node_lagged = PrimaryNode('lagged')
    node_linear = SecondaryNode('linear', nodes_from=[node_lagged])

    node_final = SecondaryNode('ridge', nodes_from=[node_arima, node_linear])
    chain = Chain(node_arima)

    return chain


def prepare_input_data(len_forecast, train_data_features, train_data_target,
                       test_data_features):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_data_features: time series which can be used as predictors for train
    :param train_data_target: time series which can be used as target for train
    :param test_data_features: time series which can be used as predictors for prediction

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data_features)),
                            features=train_data_features,
                            target=train_data_target,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_data_features)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=test_data_features,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def run_experiment(time_series, len_forecast=250):
    """ Function with example how time series forecasting can be made

    :param time_series: time series for prediction
    :param len_forecast: forecast length
    """

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    chain = get_chain()

    old_predicted, new_predicted = make_forecast(chain, train_input,
                                                 predict_input, task)

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
    run_experiment(time_series, len_forecast=250)
