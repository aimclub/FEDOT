import warnings
from typing import List, Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from examples.simple.time_series_forecasting.api_forecasting import TS_DATASETS
from examples.simple.time_series_forecasting.ts_pipelines import ts_ar_pipeline
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from fedot.core.utils import set_random_seed

warnings.filterwarnings('ignore')


def run_multistep(dataset: str, pipeline: Pipeline, step_forecast: int = 10, future_steps: int = 5,
                  visualisation=False):
    """ Example of out-of-sample ts forecasting using custom pipelines
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param step_forecast: horizon to train model. Real horizon = step_forecast * future_steps
    :param future_steps: number of future steps
    :param visualisation: is visualisation needed
    """
    # show initial pipeline
    pipeline.print_structure()

    horizon = step_forecast * future_steps
    time_series = pd.read_csv(TS_DATASETS[dataset])
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=step_forecast))

    idx = np.arange(len(time_series['idx'].values))
    time_series = time_series['value'].values
    train_input = InputData(idx=idx,
                            features=time_series,
                            target=time_series,
                            task=task,
                            data_type=DataTypesEnum.ts)
    train_data, test_data = train_test_data_setup(train_input)

    pipeline.fit(train_data)

    predict = out_of_sample_ts_forecast(pipeline=pipeline,
                                        input_data=test_data,
                                        horizon=horizon)

    plot_info = [{'idx': idx,
                  'series': time_series,
                  'label': 'Actual time series'},
                 {'idx': np.arange(test_data.idx[0], test_data.idx[0] + predict.shape[0]),
                  'series': predict,
                  'label': 'Forecast'},
                 get_border_line_info(np.arange(test_data.idx[0] + 1)[-1], predict, time_series, 'train|test'),
                 get_border_line_info(np.arange(test_data.idx[-1] + 1)[-1], predict, time_series, 'End of test',
                                      'gray')]

    # plot lines
    if visualisation:
        visualise(plot_info)


def visualise(plot_info: List[dict]):
    """
    Creates a plot based on plot_info

    :param plot_info: list of parameters for plot:
    The possible parameters are:
            'idx' - idx (or x axis data)
            'series' - data to plot (or y axis data)
            'label' - label for legend
            'color' - (optional) color of line
    """
    plt.figure()
    for p in plot_info:
        color = p.get('color')
        plt.plot(p['idx'], p['series'], label=p['label'], color=color)
    plt.legend()
    plt.grid()
    plt.show()


def get_border_line_info(idx: Any, predict: np.array, time_series: np.array, label: str, color: str = 'black') -> dict:
    """
    Return plot_info for border vertical line that divides train and test part of data

    :param idx: idx for vertical line
    :param predict: predictions
    :param time_series: full time series with test_data
    :param label: label for a legend
    :parma color: color of a line
    """
    return {'idx': [idx, idx],
            'series': [min(np.concatenate([np.ravel(time_series), predict])),
                       max(np.concatenate([np.ravel(time_series), predict]))],
            'label': label,
            'color': color}


if __name__ == '__main__':
    set_random_seed(2020)

    run_multistep("m4_monthly", ts_ar_pipeline(), step_forecast=10, visualisation=True)
