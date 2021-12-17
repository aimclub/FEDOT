import warnings

import numpy as np
import pandas as pd

from examples.time_series.pipelines import *
from examples.time_series.tuning_ts_pipelines import visualise
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup

from fedot.core.pipelines.ts_wrappers import out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams

warnings.filterwarnings('ignore')
np.random.seed(2020)

datasets = {
    'australia': '../data/ts/australia.csv',
    'beer': '../data/ts/beer.csv',
    'salaries': '../data/ts/salaries.csv',
    'stackoverflow': '../data/ts/stackoverflow.csv',
}


def run_multistep(dataset, pipeline, step_forecast=10):
    """ Function with example of out-of-sample ts forecasting with different models
    :param dataset: name of dataset
    :param pipeline: pipeline to use
    :param step_forecast: horizon to train model. Real horizon = step_forecast * 5
    """
    horizon = step_forecast * 5
    time_series = pd.read_csv(datasets[dataset])
    # Let's divide our data on train and test samples
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
                  'label': 'Actual time series'
                  },
                 {'idx': np.arange(test_data.idx[0], test_data.idx[0] + predict.shape[0]),
                  'series': predict,
                  'label': 'Forecast'
                  },
                 {'idx': [np.arange(test_data.idx[0] + 1)[-1], np.arange(test_data.idx[0] + 1)[-1]],
                  'series': [
                      min(np.concatenate([np.ravel(time_series), predict])),
                      max(np.concatenate([np.ravel(time_series), predict]))
                  ],
                  'label': 'Train|Test',
                  'color': 'black'
                  },
                 {'idx': [np.arange(test_data.idx[-1] + 1)[-1], np.arange(test_data.idx[-1] + 1)[-1]],
                  'series': [
                      min(np.concatenate([np.ravel(time_series), predict])),
                      max(np.concatenate([np.ravel(time_series), predict]))
                  ],
                  'label': 'End of Test',
                  'color': 'black'
                  }]
    pipeline.print_structure()
    visualise(plot_info)


if __name__ == '__main__':
    run_multistep("australia", glm_ridge_pipeline(), step_forecast=10)
