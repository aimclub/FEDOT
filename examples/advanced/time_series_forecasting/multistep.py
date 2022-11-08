import os
import warnings

import numpy as np
import pandas as pd

from examples.advanced.time_series_forecasting.composing_pipelines import get_border_line_info
from examples.simple.time_series_forecasting.ts_pipelines import *
from examples.simple.time_series_forecasting.tuning_pipelines import visualise
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline import Pipeline

from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

from fedot.core.utils import fedot_project_root

warnings.filterwarnings('ignore')
np.random.seed(2020)

datasets = {
    'australia': f'{fedot_project_root()}/examples/data/ts/australia.csv',
    'beer': f'{fedot_project_root()}/examples/data/ts/beer.csv',
    'salaries': f'{fedot_project_root()}/examples/data/ts/salaries.csv',
    'stackoverflow': f'{fedot_project_root()}/examples/data/ts/stackoverflow.csv',
    'test_sea': os.path.join(fedot_project_root(), 'test', 'data', 'simple_sea_level.csv')}


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
    time_series = pd.read_csv(datasets[dataset])
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=step_forecast))

    idx = np.arange(len(time_series['idx'].values))
    time_series = time_series['Level'].values
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


if __name__ == '__main__':
    run_multistep("australia", ts_ar_pipeline(), step_forecast=10, visualisation=True)
