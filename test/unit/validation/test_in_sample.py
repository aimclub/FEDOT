import os
import numpy as np
import pandas as pd
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from sklearn.metrics import mean_absolute_error
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

test_file_path = str(os.path.dirname(__file__))
df = pd.read_csv(os.path.join(test_file_path, '../../data/waves_mod.csv'))
test_size = 50
variable = df['Hsig'][:100]
train, test = variable[1: len(variable) - test_size], variable[len(variable) - test_size:]


def get_pipeline_arima():
    node_arima = PrimaryNode('arima')
    return Pipeline(node_arima)


def get_pipeline_ar():
    node_arima = PrimaryNode('ar')
    return Pipeline(node_arima)


def wrap_into_input(forecast_length, time_series):
    time_series = np.array(time_series)
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    input_data = InputData(idx=np.arange(0, len(time_series)),
                           features=time_series, target=time_series,
                           task=task, data_type=DataTypesEnum.ts)
    return input_data


def test_in_sample_ar():
    input_data_short_train = wrap_into_input(forecast_length=2, time_series=train)
    input_data_short_test = wrap_into_input(forecast_length=2, time_series=variable)

    pipeline = get_pipeline_ar()
    pipeline.fit(input_data_short_train)
    short_val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                              input_data=input_data_short_test,
                                              horizon=test_size,
                                              force_refit=True)
    mae_refit = mean_absolute_error(test, short_val_predict)

    pipeline = get_pipeline_ar()
    pipeline.fit(input_data_short_train)
    short_val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                              input_data=input_data_short_test,
                                              horizon=test_size,
                                              force_refit=False)
    mae = mean_absolute_error(test, short_val_predict)
    assert mae_refit < mae


def test_in_sample_arima():
    input_data_short_train = wrap_into_input(forecast_length=2, time_series=train)
    input_data_short_test = wrap_into_input(forecast_length=2, time_series=variable)

    pipeline = get_pipeline_arima()
    pipeline.fit(input_data_short_train)
    short_val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                              input_data=input_data_short_test,
                                              horizon=test_size,
                                              force_refit=True)
    mae_refit = mean_absolute_error(test, short_val_predict)

    pipeline = get_pipeline_arima()
    pipeline.fit(input_data_short_train)
    short_val_predict = in_sample_ts_forecast(pipeline=pipeline,
                                              input_data=input_data_short_test,
                                              horizon=test_size,
                                              force_refit=False)
    mae = mean_absolute_error(test, short_val_predict)
    assert mae_refit < mae
