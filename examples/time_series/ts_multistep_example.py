import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

warnings.filterwarnings('ignore')
np.random.seed(2020)


def get_pipeline():
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_1.custom_params = {'window_size': 120}
    node_lagged_2 = PrimaryNode('lagged')
    node_lagged_2.custom_params = {'window_size': 10}

    node_first = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_second = SecondaryNode('dtreg', nodes_from=[node_lagged_2])
    node_final = SecondaryNode('ridge', nodes_from=[node_first, node_second])
    pipeline = Pipeline(node_final)

    return pipeline


def run_multistep_example(time_series, len_forecast=250, future_steps=1000,
                          vis=True):
    """ Function with example how time series forecasting can be made

    :param time_series: time series for prediction
    :param future_steps: amount of steps to make them in the future
    :param len_forecast: forecast length
    """

    # Source time series
    train_input, predict_input = train_test_data_setup(InputData(idx=range(len(time_series)),
                                                                 features=time_series,
                                                                 target=time_series,
                                                                 task=Task(TaskTypesEnum.ts_forecasting,
                                                                           TsForecastingParams(
                                                                               forecast_length=len_forecast)),
                                                                 data_type=DataTypesEnum.ts))

    # Get pipeline with several models
    pipeline = get_pipeline()

    # Fit it
    start_time = timeit.default_timer()
    pipeline.fit_from_scratch(train_input)
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train pipeline\n')

    # Make forecast
    predicted = out_of_sample_ts_forecast(pipeline=pipeline,
                                          input_data=predict_input,
                                          horizon=future_steps)

    if vis:
        plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
        plt.plot(range(len(train_input.target), len(train_input.target) + len(predicted)),
                 predicted, label='Forecast')
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../data/ts_sea_level.csv')
    time_series = np.array(df['Level'])

    run_multistep_example(time_series,
                          len_forecast=200,
                          future_steps=2000,
                          vis=True)
