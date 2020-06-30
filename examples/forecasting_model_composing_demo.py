import os

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from core.composer.chain import Chain
from core.composer.node import NodeGenerator
from core.composer.visualisation import ComposerVisualiser
from core.models.data import InputData, OutputData
from core.repository.dataset_types import DataTypesEnum
from core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from core.utils import project_root


def get_composite_lstm_chain():
    chain = Chain()
    node_trend = NodeGenerator.primary_node('trend_data_model')
    node_trend.labels = ["fixed"]
    node_lstm_trend = NodeGenerator.secondary_node('linear', nodes_from=[node_trend])
    node_trend.labels = ["fixed"]
    node_residual = NodeGenerator.primary_node('residual_data_model')
    node_ridge_residual = NodeGenerator.secondary_node('rfr', nodes_from=[node_residual])

    node_final = NodeGenerator.secondary_node('additive_data_model',
                                              nodes_from=[node_ridge_residual, node_lstm_trend])
    node_final.labels = ["fixed"]
    chain.add_node(node_final)
    return chain


def calculate_validation_metric(pred: OutputData, valid: InputData, name: str, is_visualise: bool) -> float:
    window_size = valid.task.task_params.max_window_size
    forecast_length = valid.task.task_params.forecast_length

    # skip initial part of time series
    predicted = pred.predict[window_size + forecast_length:]
    real = valid.target[window_size + forecast_length:]

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


def run_metocean_forecasting_problem(train_file_path, test_file_path, forecast_length=1,
                                     max_window_size=64, with_visualisation=True):
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

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)

    available_model_types_primary = ['trend_data_model',
                                     'residual_data_model']

    available_model_types_secondary = ['rfr', 'linear',
                                       'ridge', 'lasso',
                                       'lasso', 'gbr',
                                       'additive_data_model']

    history = []

    chain = Chain()
    chain.add_node(NodeGenerator.primary_node('linear'))
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    chain.add_node(NodeGenerator.primary_node('rfr'))
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    chain.add_node(NodeGenerator.primary_node('ridge'))
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    node1 = NodeGenerator.primary_node('rfr')
    node2 = NodeGenerator.primary_node('dtreg')
    chain.add_node(NodeGenerator.secondary_node('linear', nodes_from=[node1, node2]))
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    node1 = NodeGenerator.primary_node('rfr')
    node2 = NodeGenerator.primary_node('dtreg')
    node3 = NodeGenerator.primary_node('lasso')
    chain.add_node(NodeGenerator.secondary_node('linear', nodes_from=[node1, node2, node3]))
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    node_trend = NodeGenerator.primary_node('trend_data_model')
    node_trend2 = NodeGenerator.secondary_node('linear', nodes_from=[node_trend])
    chain.add_node(node_trend2)
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    node_trend = NodeGenerator.primary_node('residual_data_model')
    node_trend2 = NodeGenerator.secondary_node('dtreg', nodes_from=[node_trend])
    chain.add_node(node_trend2)
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = Chain()
    node1 = NodeGenerator.primary_node('rfr')
    node2 = NodeGenerator.primary_node('dtreg')
    chain.add_node(NodeGenerator.secondary_node('ridge', nodes_from=[node1, node2]))
    chain.fit(dataset_to_train)
    history.append(chain)

    chain = get_composite_lstm_chain()
    chain.fit(dataset_to_train)
    history.append(chain)

    historical_fitness = [0.31, 0.3, 0.32, 0.2, 0.23, 0.5, 0.46, 0.15, 0.14]

    ComposerVisualiser.visualise_history_ts(history, historical_fitness,
                                            dataset_to_validate)


if __name__ == '__main__':
    # the dataset was obtained from NEMO model simulation for sea surface height

    # a dataset that will be used as a train and test set during composition
    file_path_train = 'cases/data/metocean/metocean_data_train.csv'
    full_path_train = os.path.join(str(project_root()), file_path_train)

    # a dataset for a final validation of the composed model
    file_path_test = 'cases/data/metocean/metocean_data_test.csv'
    full_path_test = os.path.join(str(project_root()), file_path_test)

    run_metocean_forecasting_problem(full_path_train, full_path_test, forecast_length=1)
