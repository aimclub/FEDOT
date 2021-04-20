import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_ts_wrappers import out_of_sample_ts_forecast
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from examples.ts_forecasting_tuning import prepare_input_data

warnings.filterwarnings('ignore')
np.random.seed(2020)


def get_chain():
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_1.custom_params = {'window_size': 120}
    node_lagged_2 = PrimaryNode('lagged')
    node_lagged_2.custom_params = {'window_size': 10}

    node_first = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_second = SecondaryNode('dtreg', nodes_from=[node_lagged_2])
    node_final = SecondaryNode('ridge', nodes_from=[node_first, node_second])
    chain = Chain(node_final)

    return chain


def run_multistep_example(time_series, len_forecast=250, future_steps=1000,
                          vis=True):
    """ Function with example how time series forecasting can be made

    :param time_series: time series for prediction
    :param future_steps: amount of steps to make them in the future
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

    # Get chain with several models
    chain = get_chain()

    # Fit it
    start_time = timeit.default_timer()
    chain.fit_from_scratch(train_input)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')

    # Make forecast
    predicted = out_of_sample_ts_forecast(chain=chain,
                                          input_data=predict_input,
                                          horizon=future_steps)

    if vis:
        plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
        plt.plot(range(len(train_data), len(train_data) + len(predicted)),
                 predicted, label='Forecast')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../notebooks/jupyter_media/time_series_forecasting/sea_level.csv')
    time_series = np.array(df['Level'])

    run_multistep_example(time_series,
                          len_forecast=200,
                          future_steps=2000,
                          vis=True)
