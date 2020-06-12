import numpy as np
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima_process import ArmaProcess

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def get_synthetic_ts_data(n_steps=10000) -> InputData:
    simulated_data = ArmaProcess().generate_sample(nsample=n_steps)
    x1 = np.arange(0, n_steps)
    x2 = np.arange(0, n_steps) + 1

    simulated_data = simulated_data + x1 * 0.0005 - x2 * 0.0001

    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=1, max_window_size=2))

    input_data = InputData(idx=np.arange(0, n_steps),
                           features=np.asarray([x1, x2]).T,
                           target=simulated_data,
                           task=task,
                           data_type=DataTypesEnum.ts)
    return input_data


def get_rmse_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)
    rmse_value_test = mse(y_true=test_data.target[0:len(test_pred.predict)], y_pred=test_pred.predict, squared=False)
    rmse_value_train = mse(y_true=train_data.target[0:len(train_pred.predict)], y_pred=train_pred.predict,
                           squared=False)
    return rmse_value_train, rmse_value_test


def get_decomposed_chain():
    chain = Chain()
    node_trend = NodeGenerator.primary_node('trend_data_model')
    node_lstm_trend = NodeGenerator.secondary_node('lstm', nodes_from=[node_trend])

    # decrase the number of parameters
    node_lstm_trend.model.external_params = {'epochs': 1}

    node_residual = NodeGenerator.primary_node('residual_data_model')
    node_ridge_residual = NodeGenerator.secondary_node('ridge', nodes_from=[node_residual])

    node_final = NodeGenerator.secondary_node('additive_data_model',
                                              nodes_from=[node_ridge_residual, node_lstm_trend])
    chain.add_node(node_final)
    return chain


def test_arima_chain_fit_correct():
    data = get_synthetic_ts_data()

    chain = Chain()
    node_arima = NodeGenerator.primary_node('arima')
    chain.add_node(node_arima)

    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(data.target)

    assert rmse_on_test < rmse_threshold


def test_regression_chain_fit_correct():
    data = get_synthetic_ts_data()

    chain = Chain()
    node_rfr = NodeGenerator.primary_node('rfr')
    chain.add_node(node_rfr)

    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(data.target) * 1.5

    assert rmse_on_test < rmse_threshold


def test_composite_lstm_chain_fit_correct():
    data = get_synthetic_ts_data()

    chain = get_decomposed_chain()

    train_data, test_data = train_test_data_setup(data)

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(data.target)
    assert rmse_on_test < rmse_threshold
