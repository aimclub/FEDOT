import numpy as np
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.arima_process import ArmaProcess

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def get_synthetic_ts_data_period(n_steps=1000, forecast_length=1, max_window_size=50):
    simulated_data = ArmaProcess().generate_sample(nsample=n_steps)
    x1 = np.arange(0, n_steps)
    x2 = np.arange(0, n_steps) + 1

    simulated_data = simulated_data + x1 * 0.0005 - x2 * 0.0001

    periodicity = np.sin(x1 / 50)

    simulated_data = simulated_data + periodicity

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size,
                                    return_all_steps=False))

    data = InputData(idx=np.arange(0, n_steps),
                     features=np.asarray([x1, x2]).T,
                     target=simulated_data,
                     task=task,
                     data_type=DataTypesEnum.ts)

    return train_test_data_setup(data)


def get_synthetic_ts_data_linear(n_steps=1000, forecast_length=1, max_window_size=50):
    simulated_data = np.asarray([float(_) for _ in (np.arange(0, n_steps))])

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size,
                                    return_all_steps=False,
                                    make_future_prediction=False))

    data = InputData(idx=np.arange(0, n_steps),
                     features=simulated_data,
                     target=simulated_data,
                     task=task,
                     data_type=DataTypesEnum.ts)

    return train_test_data_setup(data)


def get_rmse_value(chain: Chain, train_data: InputData, test_data: InputData) -> (float, float):
    train_pred = chain.predict(input_data=train_data)
    test_pred = chain.predict(input_data=test_data)

    rmse_value_test = mse(y_true=test_data.target[~np.isnan(test_pred.predict)],
                          y_pred=test_pred.predict[~np.isnan(test_pred.predict)],
                          squared=False)
    rmse_value_train = mse(y_true=train_data.target[~np.isnan(train_pred.predict)],
                           y_pred=train_pred.predict[~np.isnan(train_pred.predict)],
                           squared=False)

    return rmse_value_train, rmse_value_test


def get_decomposed_chain(model_trend='lstm', model_residual='ridge'):
    chain = Chain()
    node_trend = PrimaryNode('trend_data_model')
    node_first_trend = SecondaryNode(model_trend,
                                     nodes_from=[node_trend])

    if model_trend == 'lstm':
        # decrease the number of epochs to fit
        node_first_trend.model.params = {'epochs': 1}

    node_residual = PrimaryNode('residual_data_model')
    node_model_residual = SecondaryNode(model_residual,
                                        nodes_from=[node_residual])

    node_final = SecondaryNode('additive_data_model',
                               nodes_from=[node_model_residual,
                                           node_first_trend])

    chain.add_node(node_final)
    return chain


def test_arima_chain_fit_correct():
    train_data, test_data = get_synthetic_ts_data_linear(forecast_length=12)

    chain = Chain(PrimaryNode('arima'))

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test < rmse_threshold


def test_regression_chain_forecast_onestep_correct():
    train_data, test_data = get_synthetic_ts_data_linear(forecast_length=1, max_window_size=10)

    chain = Chain(PrimaryNode('ridge'))

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test < rmse_threshold


def test_regression_chain_forecast_multistep_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=20, max_window_size=30)

    chain = Chain(PrimaryNode('ridge'))

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test < rmse_threshold


def test_regression_chain_linear_forecast_multistep_correct():
    train_data, test_data = get_synthetic_ts_data_linear(forecast_length=20, max_window_size=30)

    chain = Chain(PrimaryNode('linear'))

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = 0.01
    assert rmse_on_test < rmse_threshold


def test_forecasting_regression_composite_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=10, max_window_size=10)

    chain = get_decomposed_chain(model_trend='linear',
                                 model_residual='linear')

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)

    assert rmse_on_test < rmse_threshold


def test_forecasting_composite_lstm_chain_fit_correct():
    train_data, test_data = get_synthetic_ts_data_period(forecast_length=10, max_window_size=10)

    chain = get_decomposed_chain()

    chain.fit(input_data=train_data)
    _, rmse_on_test = get_rmse_value(chain, train_data, test_data)

    rmse_threshold = np.std(test_data.target)
    assert rmse_on_test < rmse_threshold
