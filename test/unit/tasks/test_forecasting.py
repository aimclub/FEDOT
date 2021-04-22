from random import seed

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_process import ArmaProcess

from fedot.utilities.synthetic.data import generate_synthetic_data
from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData, train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

np.random.seed(42)
seed(42)


def _max_rmse_threshold_by_std(values, is_strict=True):
    tolerance_coeff = 3.0 if is_strict else 5.0
    return np.std(values) * tolerance_coeff


def get_synthetic_ts_data_period(n_steps=1000, forecast_length=5):
    simulated_data = ArmaProcess().generate_sample(nsample=n_steps)
    x1 = np.arange(0, n_steps)
    x2 = np.arange(0, n_steps) + 1

    simulated_data = simulated_data + x1 * 0.0005 - x2 * 0.0001

    periodicity = np.sin(x1 / 50)

    simulated_data = simulated_data + periodicity

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, n_steps),
                     features=simulated_data,
                     target=simulated_data,
                     task=task,
                     data_type=DataTypesEnum.ts)
    a, b = train_test_data_setup(data)
    return train_test_data_setup(data)


def get_multiscale_chain():
    # First branch
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_1.custom_params = {'window_size': 20}
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])

    # Second branch, which will try to make prediction based on smoothed ts
    node_filtering = PrimaryNode('gaussian_filter')
    node_filtering.custom_params = {'sigma': 3}
    node_lagged_2 = SecondaryNode('lagged', nodes_from=[node_filtering])
    node_lagged_2.custom_params = {'window_size': 100}
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    node_final = SecondaryNode('linear', nodes_from=[node_ridge_1, node_ridge_2])

    chain = Chain(node_final)

    return chain


def get_simple_ts_chain(model_root: str = 'ridge', window_size: int = 20):
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': window_size}
    node_root = SecondaryNode(model_root, nodes_from=[node_lagged])

    chain = Chain(node_root)

    return chain


def get_statsmodels_chain():
    node_ar = PrimaryNode('ar')
    node_ar.custom_params = {'lag_1': 20, 'lag_2': 100}
    chain = Chain(node_ar)
    return chain


def test_arima_chain_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=12)

    chain = get_statsmodels_chain()

    chain.fit(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))

    rmse_test = mean_squared_error(test_target, test_pred, squared=False)

    rmse_threshold = _max_rmse_threshold_by_std(test_data.target)

    assert rmse_test < rmse_threshold


def test_simple_chain_forecast_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=5)

    chain = get_simple_ts_chain()

    chain.fit(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))
    rmse_test = mean_squared_error(test_target, test_pred, squared=False)

    rmse_threshold = _max_rmse_threshold_by_std(test_data.target, is_strict=True)

    assert rmse_test < rmse_threshold


def test_regression_multiscale_chain_forecast_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=5)

    chain = get_multiscale_chain()

    chain.fit(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)

    # Calculate metric
    test_pred = np.ravel(np.array(test_pred.predict))
    test_target = np.ravel(np.array(test_data.target))
    rmse_test = mean_squared_error(test_target, test_pred, squared=False)

    rmse_threshold = _max_rmse_threshold_by_std(test_data.target,
                                                is_strict=True)

    assert rmse_test < rmse_threshold


def test_ts_single_chain_model_without_multiotput_support():
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
        chain = get_simple_ts_chain(model_root=model_id, window_size=2)

        # making predictions for the missing part in the time series
        chain.fit_from_scratch(train_data)
        predicted_values = chain.predict(test_data)
        chain_forecast = np.ravel(np.array(predicted_values.predict))

        test_part = np.ravel(np.array(test_part))
        mae = mean_absolute_error(test_part, chain_forecast)
        assert mae < 50


def test_exception_if_incorrect_forecast_length():
    with pytest.raises(ValueError) as exc:
        _, _ = get_synthetic_ts_data_period(forecast_length=0)
    assert str(exc.value) == f'Forecast length should be more then 0'
