from random import seed

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

from fedot.core.data.data import InputData
from fedot.core.pipelines.ts_wrappers import in_sample_ts_forecast, out_of_sample_ts_forecast
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.operations.evaluation.operation_implementations.models.ts_implementations import ARIMAImplementation
from data.data_manager import generate_synthetic_data, get_ts_data
from data.pipeline_manager import get_statsmodels_pipeline, get_simple_ts_pipeline, get_multiscale_pipeline

np.random.seed(42)
seed(42)


def _max_rmse_threshold_by_std(values, is_strict=True):
    tolerance_coeff = 3.0 if is_strict else 5.0
    return np.std(values) * tolerance_coeff


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


def test_ts_single_pipeline_model_without_multiotput_support():
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

    for model_id in ['xgbreg', 'gbr', 'adareg', 'svr', 'sgdr']:
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
    assert str(exc.value) == f'Forecast length should be more then 0'


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
