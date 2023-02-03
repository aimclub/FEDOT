import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.advanced.time_series_forecasting.composing_pipelines import visualise, get_border_line_info
from examples.simple.time_series_forecasting.api_forecasting import get_ts_data
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.pipelines.pipeline_builder import PipelineBuilder
from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_ts_data_long(n_steps=80, forecast_length=5):
    """ Prepare data from csv file with time series and take needed number of
    elements

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    """
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'examples/data/ts/ts_long.csv')
    df = pd.read_csv(file_path)
    df = df[df["series_id"] == "traffic_volume"]
    time_series = np.array(df['value'])[:n_steps]

    time_series = (time_series - np.mean(time_series)) / np.std(time_series)

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data), task


def cgru_forecasting():
    """ Example of cgru pipeline serialization """
    horizon = 2
    window_size = 200
    train_data, test_data = get_ts_data('salaries', horizon)

    pipeline = PipelineBuilder().add_node("lagged", params={'window_size': window_size}).add_node("cgru", params={
        'hidden_size': 300,
        'learning_rate': 0.001,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 50}).build()


    #pipeline=PipelineBuilder().add_node("lagged", params={'window_size': window_size}).add_node('ridge').build()
    pipeline.fit(train_data)
    prediction = pipeline.predict(test_data).predict[0]


    plot_info = [
        {'idx': np.concatenate([train_data.idx, test_data.idx]),
         'series': np.concatenate([test_data.features]),
         'label': 'Actual time series'},
        {'idx': test_data.idx,
         'series': np.ravel(prediction),
         'label': 'prediction'},
        get_border_line_info(test_data.idx[0],
                             prediction,
                             np.ravel(np.concatenate([test_data.features, test_data.target])),
                             'Border line')
    ]

    rmse = mean_squared_error(test_data.target, prediction, squared=False)
    mae = mean_absolute_error(test_data.target, prediction)
    print(f'RMSE - {rmse:.4f}')
    print(f'MAE - {mae:.4f}')

    visualise(plot_info)


if __name__ == '__main__':
    cgru_forecasting()
