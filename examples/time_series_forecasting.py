from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from core.composer.node import PrimaryNode, SecondaryNode
from core.composer.ts_chain import TsForecastingChain
from core.models.data import InputData, train_test_data_setup
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


def get_synthetic_ts_data_period(n_steps=6000, forecast_length=1, max_window_size=50,
                                 with_exog: bool = True) -> InputData:
    x1 = np.arange(0, n_steps) / 10
    x2 = np.arange(0, n_steps) + 1

    simulated_data = x1 * 0.005 - x2 * 0.001
    periodicity = np.sin(x1 * 2)
    random = np.random.normal(0, 0.1, n_steps)
    simulated_data = simulated_data + periodicity + random

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size,
                                    return_all_steps=True,
                                    make_future_prediction=True))

    exog_features = np.asarray([x1, x2]).T
    if not with_exog:
        # move target to features
        exog_features = None
    input_data = InputData(idx=np.arange(0, n_steps),
                           features=exog_features,
                           target=simulated_data,
                           task=task,
                           data_type=DataTypesEnum.ts)
    return input_data


def plot_prediction(prediction, data: InputData, desc) -> (float, float):
    plt.plot(prediction, label='predict')
    plt.plot(data.target, label='target')
    plt.legend()
    plt.title(desc)
    plt.show()


def run_forecasting(chain: TsForecastingChain, data: InputData, is_visualise: bool, desc: str):
    train_data, test_data = train_test_data_setup(data)
    chain.fit_from_scratch(train_data)

    test_data_for_pred = copy(test_data)
    # to avoid data leak
    test_data_for_pred.target = None

    full_prediction = chain.forecast(initial_data=train_data, supplementary_data=test_data_for_pred).predict
    if is_visualise:
        plot_prediction(full_prediction, test_data, desc)


def run_onestep_linear_example(n_steps=6000, is_visualise: bool = True):
    window_size = 16

    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=1,
                                           max_window_size=window_size,
                                           with_exog=True)
    # regression forecasting
    chain = TsForecastingChain(PrimaryNode('linear'))

    # one step regression
    run_forecasting(chain=chain, data=dataset, is_visualise=is_visualise,
                    desc=f'Linear model, {dataset.task.task_params.forecast_length} step prediction with exog')

    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=1,
                                           max_window_size=window_size,
                                           with_exog=False)

    run_forecasting(chain=chain, data=dataset, is_visualise=is_visualise,
                    desc=f'Linear model, {dataset.task.task_params.forecast_length} step prediction without exog')


def run_multistep_linear_example(n_steps=6000, is_visualise: bool = True):
    chain = TsForecastingChain(PrimaryNode('ridge'))

    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=64,
                                           max_window_size=64,
                                           with_exog=True)
    # multi step regression
    run_forecasting(chain=chain, data=dataset, is_visualise=is_visualise,
                    desc=f'Linear model, {dataset.task.task_params.forecast_length} step prediction with exog')

    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=64,
                                           max_window_size=64,
                                           with_exog=False)
    run_forecasting(chain=chain, data=dataset, is_visualise=is_visualise,
                    desc=f'Linear model, {dataset.task.task_params.forecast_length} step prediction without exog')


def run_multistep_multiscale_example(n_steps=6000, is_visualise: bool = True):
    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=64,
                                           max_window_size=64,
                                           with_exog=False)

    # composite forecasting with decomposition
    node_first = PrimaryNode('trend_data_model')
    node_second = PrimaryNode('residual_data_model')
    node_trend_model = SecondaryNode('linear', nodes_from=[node_first])
    node_residual_model = SecondaryNode('linear', nodes_from=[node_second])

    node_final = SecondaryNode('additive_data_model', nodes_from=[node_trend_model, node_residual_model])

    chain = TsForecastingChain(node_final)

    run_forecasting(chain=chain, data=dataset,
                    is_visualise=is_visualise,
                    desc=f'Multiscale model, {dataset.task.task_params.forecast_length} step prediction with exog')


def run_onestep_composite_example(n_steps=6000, is_visualise: bool = True):
    # composite forecasting with ensemble
    node_first = PrimaryNode('lasso')
    node_second = PrimaryNode('ridge')
    node_final = SecondaryNode('linear', nodes_from=[node_first, node_second])

    chain = TsForecastingChain(node_final)

    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=1,
                                           max_window_size=64,
                                           with_exog=True)

    run_forecasting(chain=chain, data=dataset,
                    is_visualise=is_visualise,
                    desc=f'Composite model, {dataset.task.task_params.forecast_length} step prediction with exog')


def run_multistep_lstm_example(n_steps=6000, is_visualise: bool = True):
    # lstm forecasting
    dataset = get_synthetic_ts_data_period(n_steps=n_steps,
                                           forecast_length=16,
                                           max_window_size=16,
                                           with_exog=True)

    chain = TsForecastingChain(PrimaryNode('lstm'))
    run_forecasting(chain=chain, data=dataset,
                    is_visualise=is_visualise,
                    desc=f'LSTM model, {dataset.task.task_params.forecast_length} step prediction with exog')

    return True


if __name__ == '__main__':
    run_onestep_linear_example()
    run_multistep_linear_example()
    run_multistep_multiscale_example()
    run_onestep_composite_example()
    run_multistep_lstm_example()
