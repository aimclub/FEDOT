from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.arima import \
    ARIMAImplementation
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations.statsmodels import \
    AutoRegImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast, out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from fedot.utilities.synth_dataset_generator import generate_synthetic_data


def _max_rmse_threshold_by_std(values, is_strict=True):
    tolerance_coeff = 3.0 if is_strict else 5.0
    return np.std(values) * tolerance_coeff


def get_ts_data(n_steps: int = 80, forecast_length: int = 5, validation_blocks: Optional[int] = None):
    """ Prepare data from csv file with time series and take needed number of
    elements

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    :param validation_blocks: number of validation blocks
    """
    file_path = fedot_project_root().joinpath('test/data/simple_time_series.csv')
    df = pd.read_csv(file_path)

    time_series = np.array(df['sea_height'])[:n_steps]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data, validation_blocks=validation_blocks)


def get_ts_data_with_dt_idx(n_steps=80, forecast_length=5):
    """ Prepare data from csv file with time series and take needed number of
    elements with datetime indexes

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    """
    file_path = fedot_project_root().joinpath('test/data/simple_sea_level.csv')
    df = pd.read_csv(file_path)

    time_series = np.array(df.iloc[:n_steps, 1])
    idx = pd.to_datetime(df.iloc[:n_steps, 0]).values
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=idx,
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data)


def get_multiscale_pipeline():
    # First branch
    node_lagged_1 = PipelineNode('lagged')
    node_lagged_1.parameters = {'window_size': 20}
    node_ridge_1 = PipelineNode('ridge', nodes_from=[node_lagged_1])

    # Second branch, which will try to make prediction based on smoothed ts
    node_filtering = PipelineNode('gaussian_filter')
    node_filtering.parameters = {'sigma': 3}
    node_lagged_2 = PipelineNode('lagged', nodes_from=[node_filtering])
    node_lagged_2.parameters = {'window_size': 100}
    node_ridge_2 = PipelineNode('ridge', nodes_from=[node_lagged_2])

    node_final = PipelineNode('linear', nodes_from=[node_ridge_1, node_ridge_2])

    pipeline = Pipeline(node_final)

    return pipeline


def get_simple_ts_pipeline(model_root: str = 'ridge', window_size: int = 20):
    node_lagged = PipelineNode('lagged')
    node_lagged.parameters = {'window_size': window_size}
    node_root = PipelineNode(model_root, nodes_from=[node_lagged])

    pipeline = Pipeline(node_root)

    return pipeline


def get_statsmodels_pipeline():
    node_ar = PipelineNode('ar')
    node_ar.parameters = {'lag_1': 20, 'lag_2': 100}
    pipeline = Pipeline(node_ar)
    return pipeline


def get_multiple_ts_pipeline():
    node_filter_first = PipelineNode('smoothing')
    node_filter_first.parameters = {'window_size': 2}
    node_filter_second = PipelineNode('gaussian_filter')
    node_filter_second.parameters = {'sigma': 2}

    node_lagged = PipelineNode('lagged', nodes_from=[node_filter_first, node_filter_second])
    node_ridge = PipelineNode('ridge', nodes_from=[node_lagged])
    return Pipeline(node_ridge)


def test_arima_pipeline_fit_correct():
    train_data, test_data = get_ts_data(n_steps=300, forecast_length=5)

    pipeline = get_statsmodels_pipeline()

    pipeline.fit(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))

    rmse_test = mean_squared_error(test_target, test_pred, squared=False)

    rmse_threshold = _max_rmse_threshold_by_std(test_data.target)

    assert rmse_test < rmse_threshold


def test_arima_inverse_box_cox_correct():
    """Tests if negative values after box-cox transformation are correct (not nan) after inverse box-cox"""
    ts = np.random.uniform(0, 100, 1000)
    input_ts_len = len(ts)
    arima = ARIMAImplementation()

    _, arima.lambda_value = stats.boxcox(ts)

    nan_inds = np.random.randint(1, 999, size=10)
    nan_inds = np.append(nan_inds, [0, int(len(ts) - 1)])
    ts[nan_inds] = -10
    ts = arima._inverse_boxcox(ts, arima.lambda_value)

    assert len(np.ravel(np.argwhere(np.isnan(ts)))) == 0
    assert input_ts_len == len(ts)


def test_simple_pipeline_forecast_correct():
    train_data, test_data = get_ts_data(forecast_length=5)

    pipeline = get_simple_ts_pipeline()

    pipeline.fit(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))
    rmse_test = mean_squared_error(test_target, test_pred, squared=False)

    rmse_threshold = _max_rmse_threshold_by_std(test_data.target, is_strict=True)

    assert rmse_test < rmse_threshold


def test_ar_do_correct_lags():
    train_data, test_data = get_ts_data(n_steps=80)
    ar = AutoRegImplementation(OperationParameters(**{'lag_1': 70, 'lag_2': 80}))
    params = ar.get_params()
    old_params = deepcopy(params)
    ar.fit(train_data)
    new_params = ar.get_params()
    for lag in old_params.keys():
        assert lag in new_params.changed_parameters.keys()
        assert old_params.get(lag) != new_params.get(lag)


def test_regression_multiscale_pipeline_forecast_correct():
    train_data, test_data = get_ts_data(forecast_length=5)

    pipeline = get_multiscale_pipeline()

    pipeline.fit(input_data=train_data)
    test_pred = pipeline.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))
    rmse_test = mean_squared_error(test_target, test_pred, squared=False)

    rmse_threshold = _max_rmse_threshold_by_std(test_data.target,
                                                is_strict=True)

    assert rmse_test < rmse_threshold


def test_ts_single_pipeline_model_without_multioutput_support():
    time_series = generate_synthetic_data(20)
    len_forecast = 2
    train_part = time_series[:-len_forecast]
    test_part = time_series[-len_forecast:]

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_data = InputData(idx=np.arange(0, len(train_part)),
                           features=train_part,
                           target=train_part,
                           task=task,
                           data_type=DataTypesEnum.ts)

    start_forecast = len(train_part)
    end_forecast = start_forecast + len_forecast
    idx_for_predict = np.arange(start_forecast, end_forecast)

    # Data for making prediction for a specific length
    test_data = InputData(idx=idx_for_predict,
                          features=train_part,
                          target=test_part,
                          task=task,
                          data_type=DataTypesEnum.ts)

    for model_id in ['rfr', 'gbr', 'adareg', 'svr', 'sgdr']:
        pipeline = get_simple_ts_pipeline(model_root=model_id, window_size=2)

        # making predictions for the missing part in the time series
        pipeline.fit_from_scratch(train_data)
        predicted_values = pipeline.predict(test_data)
        pipeline_forecast = np.ravel(np.array(predicted_values.predict))

        test_part = np.ravel(np.array(test_part))
        mae = mean_absolute_error(test_part, pipeline_forecast)
        assert mae < 50


def test_exception_if_incorrect_forecast_length():
    with pytest.raises(ValueError) as exc:
        _, _ = get_ts_data(forecast_length=0)
    assert str(exc.value) == 'Forecast length should be more then 0'


def test_multistep_out_of_sample_forecasting():
    horizon = 12
    train_data, test_data = get_ts_data(forecast_length=5)

    pipeline = get_multiscale_pipeline()

    # Fit pipeline to make forecasts 5 elements above
    pipeline.fit(input_data=train_data)

    # Make prediction for 12 elements
    predicted = out_of_sample_ts_forecast(pipeline=pipeline,
                                          input_data=test_data,
                                          horizon=horizon)

    assert len(predicted) == horizon


def test_multistep_in_sample_forecasting():
    horizon = 12
    train_data, test_data = get_ts_data(n_steps=200, forecast_length=5)

    pipeline = get_multiscale_pipeline()

    # Fit pipeline to make forecasts 5 elements above
    pipeline.fit(input_data=train_data)

    # Make prediction for 12 elements
    predicted = in_sample_ts_forecast(pipeline=pipeline,
                                      input_data=test_data,
                                      horizon=horizon)

    assert len(predicted) == horizon


def test_ts_forecasting_with_multiple_series_in_lagged():
    """ Test pipeline predict correctly when lagged operation get several time series """
    horizon = 3
    n_steps = 50
    train_data, test_data = get_ts_data(n_steps=n_steps + horizon, forecast_length=horizon)

    pipeline = get_multiple_ts_pipeline()
    pipeline.fit(train_data)
    predict_output = pipeline.predict(test_data)

    assert len(np.ravel(predict_output.predict)) == horizon
