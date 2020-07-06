import os

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.node import PrimaryNode, SecondaryNode
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root


def get_composite_lstm_chain():
    chain = Chain()
    node_trend = PrimaryNode('trend_data_model')
    node_lstm_trend = SecondaryNode('lasso', nodes_from=[node_trend])

    node_residual = PrimaryNode('residual_data_model')
    node_ridge_residual = SecondaryNode('ridge', nodes_from=[node_residual])

    node_final = SecondaryNode('additive_data_model',
                                              nodes_from=[node_ridge_residual, node_lstm_trend])
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, valid: InputData,
                                name: str, is_visualise=False) -> float:
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict
    real = valid.target[len(valid.target) - len(predicted):]

    # plot results
    if is_visualise:
        compare_plot(predicted, real,
                     forecast_length=forecast_length,
                     model_name=name)

    # the quality assessment for the simulation results
    rmse = mse(y_true=real, y_pred=predicted, squared=False)

    return rmse


def compare_plot(predicted, real, forecast_length, model_name):
    plt.clf()
    _, ax = plt.subplots()
    plt.plot(real, linewidth=1, label="Observed", alpha=0.4)
    plt.plot(predicted, linewidth=1, label="Predicted", alpha=0.6)
    ax.legend()
    plt.xlabel('Time, h')
    plt.ylabel('SSH, cm')
    plt.title(f'Sea surface height forecast for {forecast_length} hours with {model_name}')
    plt.show()


def run_metocean_forecasting_problem(train_file_path, test_file_path,
                                     forecast_length=1, max_window_size=64,
                                     is_visualise=False):
    # specify the task to solve
    task_to_solve = Task(TaskTypesEnum.ts_forecasting,
                         TsForecastingParams(forecast_length=forecast_length,
                                             max_window_size=max_window_size))

    full_path_train = os.path.join(str(project_root()), train_file_path)
    dataset_to_train = InputData.from_csv(
        full_path_train, task=task_to_solve, data_type=DataTypesEnum.ts)

    # a dataset for a final validation of the composed model
    full_path_test = os.path.join(str(project_root()), test_file_path)
    dataset_to_validate = InputData.from_csv(
        full_path_test, task=task_to_solve, data_type=DataTypesEnum.ts)

    chain = get_composite_lstm_chain()

    chain_simple = Chain()
    node_single = PrimaryNode('ridge')
    chain_simple.add_node(node_single)

    chain_lstm = Chain()
    node_lstm = PrimaryNode('lstm')
    chain_lstm.add_node(node_lstm)

    chain.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid = calculate_validation_metric(
        chain.predict(dataset_to_validate), dataset_to_validate,
        f'full-composite_{forecast_length}',
        is_visualise)

    chain_lstm.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid_lstm_only = calculate_validation_metric(
        chain_lstm.predict(dataset_to_validate), dataset_to_validate,
        f'full-lstm-only_{forecast_length}',
        is_visualise)

    chain_simple.fit(input_data=dataset_to_train, verbose=False)
    rmse_on_valid_simple = calculate_validation_metric(
        chain_simple.predict(dataset_to_validate), dataset_to_validate,
        f'full-simple_{forecast_length}',
        is_visualise)

    print(f'RMSE composite: {rmse_on_valid}')
    print(f'RMSE simple: {rmse_on_valid_simple}')
    print(f'RMSE LSTM only: {rmse_on_valid_lstm_only}')

    return rmse_on_valid_simple


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test,
                                     forecast_length=1, is_visualise=True)
