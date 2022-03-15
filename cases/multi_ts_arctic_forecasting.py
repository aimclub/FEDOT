import datetime
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from examples.simple.time_series_forecasting.ts_pipelines import *
from fedot.core.composer.composer_builder import ComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


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
    node_lagged_1 = PrimaryNode("lagged")
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
    primary_operations = ['lagged', 'smoothing', 'diff_filter', 'gaussian_filter']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'linear']
    return primary_operations, secondary_operations


def compose_pipeline(pipeline, train_data, task):
    # pipeline structure optimization
    primary_operations, secondary_operations = get_available_operations()
    composer_requirements = PipelineComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations, max_arity=3,
        max_depth=5, pop_size=15, num_of_generations=30,
        crossover_prob=0.8, mutation_prob=0.8,
        timeout=datetime.timedelta(minutes=10))
    mutation_types = [parameter_change_mutation,
                      MutationTypesEnum.single_change,
                      MutationTypesEnum.single_drop,
                      MutationTypesEnum.single_add]
    optimiser_parameters = GPGraphOptimiserParameters(mutation_types=mutation_types)
    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.MAE)
    builder = ComposerBuilder(task=task). \
        with_optimiser(parameters=optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_pipelines([pipeline])
    composer = builder.build()
    obtained_pipeline = composer.compose_pipeline(data=train_data)
    obtained_pipeline.show()
    return obtained_pipeline


def prepare_data(forecast_length, multi_ts):
    # points prefixes
    points = ['61_91', '56_86', '61_86', '66_86']

    # target point
    time_series = pd.read_csv('data/arctic/61_91_topaz.csv')['ssh'].values
    x_test = time_series[:-forecast_length]
    y_test = time_series[-forecast_length:]

    if multi_ts:
        x_train = create_complex_train(points, forecast_length)
        data_type = DataTypesEnum.multi_ts
    else:
        x_train = time_series[:-forecast_length]
        data_type = DataTypesEnum.ts

    # indices preparation
    idx = np.arange(len(time_series))
    idx_train = idx[:-forecast_length]
    idx_test = idx[-forecast_length:]

    # Prepare data to train the operation
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))
    train_data = InputData(idx=idx_train, features=x_train, target=x_train,
                           task=task, data_type=data_type)
    test_data = InputData(idx=idx_test, features=x_test, target=y_test,
                          task=task, data_type=data_type)
    return train_data, test_data, task, x_test, y_test


def run_multiple_ts_forecasting(forecast_length, multi_ts):
    # separate data on test/train
    train_data, test_data, task, x_test, y_test = prepare_data(forecast_length, multi_ts=multi_ts)
    # pipeline initialization
    pipeline = initial_pipeline()
    # pipeline fit and predict
    pipeline.fit(train_data)
    prediction_before = np.ravel(np.array(pipeline.predict(test_data).predict))
    # metrics evaluation
    rmse = mean_squared_error(test_data.target, prediction_before, squared=False)
    mae = mean_absolute_error(test_data.target, prediction_before)

    # compose pipeline with initial approximation
    obtained_pipeline = compose_pipeline(pipeline, train_data, task)
    # composed pipeline fit and predict
    obtained_pipeline_copy = deepcopy(obtained_pipeline)
    obtained_pipeline.fit_from_scratch(train_data)
    prediction_after = obtained_pipeline.predict(test_data)
    predict_after = np.ravel(np.array(prediction_after.predict))
    # metrics evaluation
    rmse_composing = mean_squared_error(test_data.target, predict_after, squared=True)
    mae_composing = mean_absolute_error(test_data.target, predict_after)

    # tuning composed pipeline
    obtained_pipeline_copy.fine_tune_all_nodes(input_data=train_data,
                                               loss_function=mean_absolute_error,
                                               iterations=50)
    obtained_pipeline_copy.print_structure()
    # tuned pipeline fit and predict
    obtained_pipeline_copy.fit_from_scratch(train_data)
    prediction_after_tuning = obtained_pipeline_copy.predict(test_data)
    predict_after_tuning = np.ravel(np.array(prediction_after_tuning.predict))
    # metrics evaluation
    rmse_tuning = mean_squared_error(test_data.target, predict_after_tuning, squared=True)
    mae_tuning = mean_absolute_error(test_data.target, predict_after_tuning)

    # visualization of results
    if multi_ts:
        history = np.ravel(train_data.target[:, 0])
    else:
        history = np.ravel(train_data.target)
    plt.plot(np.ravel(test_data.idx), y_test, label='test')
    plt.plot(np.ravel(train_data.idx), history, label='history')
    plt.plot(np.ravel(test_data.idx), prediction_before, label='prediction')
    plt.plot(np.ravel(test_data.idx), predict_after, label='prediction_after_composing')
    plt.plot(np.ravel(test_data.idx), predict_after_tuning, label='prediction_after_tuning')
    plt.legend()
    plt.show()

    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'RMSE after composing: {rmse_composing}')
    print(f'MAE after composing: {mae_composing}')
    print(f'RMSE after tuning: {rmse_tuning}')
    print(f'MAE after tuning: {mae_tuning}')


if __name__ == '__main__':
    run_multiple_ts_forecasting(forecast_length=60, multi_ts=True)
