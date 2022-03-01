import os
import datetime
from copy import deepcopy
from typing import Any, List

from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.simple.time_series_forecasting.ts_pipelines import *
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.api.main import Fedot


def create_complex_train(points_list, forecast_length):
    complex_train = pd.DataFrame()
    for name in points_list:
        ts = pd.read_csv(f'data/arctic/{name}_topaz.csv')['ssh']
        train_ts = ts[:-forecast_length]
        complex_train[name] = train_ts
    complex_train = complex_train.to_numpy()
    return complex_train


def initial_pipeline():
    """
        Return pipeline with the following structure:
        lagged - ridge \
                        -> ridge -> final forecast
        lagged - ridge /
        """
    node_gaus = PrimaryNode("smoothing")
    node_lagged_1 = SecondaryNode("lagged", nodes_from=[node_gaus])
    node_lagged_1.custom_params = {'window_size': 50}
    node_lagged_2 = PrimaryNode("lagged")
    node_lagged_2.custom_params = {'window_size': 30}

    node_ridge_1 = SecondaryNode("ridge", nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode("ridge", nodes_from=[node_lagged_2])

    node_final = SecondaryNode("ridge", nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'gaussian_filter', 'diff_filter']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'knnreg', 'linear']
    return primary_operations, secondary_operations


def run_multiple_ts_forecasting(forecast_length):

    # points prefixes
    points = ['61_91', '56_86', '61_86', '66_86']

    # target point
    time_series = pd.read_csv('data/arctic/61_91_topaz.csv')['ssh'].values
    x_test = time_series[:-forecast_length]
    y_test = time_series[-forecast_length:]

    x_train = create_complex_train(points, forecast_length)

    # indices preparation
    idx = np.arange(len(time_series))
    idx_train = idx[:-forecast_length]
    idx_test = idx[-forecast_length:]

    # Prepare data to train the operation
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    train_data = InputData(idx=idx_train, features=x_train, target=x_train,
                           task=task, data_type=DataTypesEnum.multi_ts)
    test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                          task=task, data_type=DataTypesEnum.multi_ts)

    pipeline = initial_pipeline()
    train_data_copied = deepcopy(train_data)
    pipeline.fit(train_data)
    prediction_before = np.ravel(np.array(pipeline.predict(test_data).predict))

    rmse_before = mean_squared_error(test_data.target, prediction_before, squared=False)
    mae_before = mean_absolute_error(test_data.target, prediction_before)

    # pipeline structure optimization
    primary_operations, secondary_operations = get_available_operations()
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations, max_arity=3,
        max_depth=4, pop_size=10, num_of_generations=30,
        crossover_prob=0.8, mutation_prob=0.8,
        timeout=datetime.timedelta(minutes=10),
        validation_blocks=1)
    mutation_types = [parameter_change_mutation, MutationTypesEnum.single_change, MutationTypesEnum.single_drop,
                      MutationTypesEnum.single_add]
    optimiser_parameters = GPGraphOptimiserParameters(mutation_types=mutation_types)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
    builder = ComposerBuilder(task=task). \
        with_optimiser(parameters=optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_pipelines([pipeline])
    composer = builder.build()

    obtained_pipeline = composer.compose_pipeline(data=train_data_copied)
    obtained_pipeline.show()
    obtained_pipeline.fit_from_scratch(train_data_copied)
    prediction_after = obtained_pipeline.predict(test_data)
    predict_after = np.ravel(np.array(prediction_after.predict))

    plt.plot(np.ravel(test_data.idx), y_test, label='test')
    plt.plot(np.ravel(train_data.idx), np.ravel(train_data.target[:, 0]), label='history')
    plt.plot(np.ravel(test_data.idx), prediction_before, label='prediction')
    plt.plot(np.ravel(test_data.idx), predict_after, label='prediction_after')
    plt.legend()
    plt.show()


def run_via_api():
    forecast_length = 60
    # points prefixes
    points = ['61_91', '56_86', '61_86', '66_86']

    # target point
    time_series = pd.read_csv('data/arctic/61_91_topaz.csv')['ssh'].values
    x_test = time_series[:-forecast_length]
    y_test = time_series[-forecast_length:]

    x_train = create_complex_train(points, forecast_length)

    # indices preparation
    idx = np.arange(len(time_series))
    idx_train = idx[:-forecast_length]
    idx_test = idx[-forecast_length:]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    train_data = InputData(idx=idx_train, features=x_train, target=x_train,
                           task=task, data_type=DataTypesEnum.multi_ts)
    test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                          task=task, data_type=DataTypesEnum.multi_ts)

    # init model for the time series forecasting
    model = Fedot(problem='ts_forecasting',
                  task_params=task.task_params,
                  timeout=10,
                  preset='fast_train')

    # run AutoML model design in the same way
    pipeline = model.fit(train_data)
    pipeline.show()

    # use model to obtain forecast
    forecast = model.predict(test_data)
    target = np.ravel(test_data.target)
    print(model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=target))

if __name__ == '__main__':
    #run_via_api()
    run_multiple_ts_forecasting(forecast_length=60)

