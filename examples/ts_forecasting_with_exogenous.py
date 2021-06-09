import os
import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.ts_forecasting_tuning import prepare_input_data
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast(chain, train: InputData, predict: InputData,
                  train_exog: InputData, predict_exog: InputData):
    """
    Function for predicting values in a time series

    :param chain: Chain object

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()

    second_node_name = 'exog_ts_data_source'

    if train_exog is None:
        second_node_name = 'lagged/2'
        train_exog = train
        predict_exog = predict

    train_dataset = MultiModalData({
        'lagged/1': train,
        second_node_name: train_exog,
    })

    predict_dataset = MultiModalData({
        'lagged/1': predict,
        second_node_name: predict_exog,
    })

    chain.fit_from_scratch(train_dataset)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')

    # Predict
    predicted_values = chain.predict(predict_dataset)
    predicted_values = predicted_values.predict

    return predicted_values


def run_exogenous_experiment(path_to_file, len_forecast=250, with_exog=True,
                             with_visualisation=True) -> None:
    """ Function with example how time series forecasting can be made with using
    exogenous features

    :param path_to_file: path to the csv file with dataframe
    :param len_forecast: forecast length
    :param with_exog: is it needed to make prediction with exogenous time series
    :param with_visualisation: is it needed to make visualisations
    """

    df = pd.read_csv(path_to_file)
    time_series = np.array(df['Level'])
    exog_variable = np.array(df['Neighboring level'])

    # Let's divide our data on train and test samples
    train_data = time_series[:-len_forecast]
    test_data = time_series[-len_forecast:]

    # Exog feature
    train_data_exog = exog_variable[:-len_forecast]
    test_data_exog = exog_variable[-len_forecast:]

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_data,
                                                          train_data_target=train_data,
                                                          test_data_features=train_data)

    # Exogenous time series
    train_input_exog, predict_input_exog, _ = prepare_input_data(len_forecast=len_forecast,
                                                                 train_data_features=train_data_exog,
                                                                 train_data_target=train_data,
                                                                 test_data_features=test_data_exog)

    if with_exog is True:
        # Example with exogenous time series
        node_lagged_1 = PrimaryNode('lagged/1')
        node_exog = PrimaryNode('exog_ts_data_source')

        node_final = SecondaryNode('ridge', nodes_from=[node_lagged_1, node_exog])
        chain = Chain(node_final)
    else:
        # Simple example without exogenous time series
        node_lagged_1 = PrimaryNode('lagged/1')
        node_lagged_2 = PrimaryNode('lagged/2')
        node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
        node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])
        node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
        train_input_exog = None
        predict_input_exog = None
        chain = Chain(node_final)

    predicted = make_forecast(chain, train_input, predict_input, train_input_exog, predict_input_exog)

    predicted = np.ravel(np.array(predicted))
    test_data = np.ravel(test_data)

    print(f'Predicted values: {predicted[:5]}')
    print(f'Actual values: {test_data[:5]}')

    mse_before = mean_squared_error(test_data, predicted, squared=False)
    mae_before = mean_absolute_error(test_data, predicted)
    print(f'RMSE - {mse_before:.4f}')
    print(f'MAE - {mae_before:.4f}\n')

    if with_visualisation:
        plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
        plt.plot(range(len(train_data), len(time_series)), predicted, label='Forecast')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    data_path = os.path.join(f'{fedot_project_root()}', 'notebooks', 'data', 'ts_sea_level.csv')
    run_exogenous_experiment(path_to_file=data_path, len_forecast=250, with_exog=True)
