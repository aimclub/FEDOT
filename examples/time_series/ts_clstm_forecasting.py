import datetime
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from examples.time_series.ts_forecasting_composing import prepare_train_test_input, fit_predict_for_pipeline, \
    display_validation_metric, get_available_operations, plot_results
from fedot.core.composer.gp_composer.fixed_structure_composer import GPComposerRequirements
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root


def get_ts_data_long(n_steps=80, forecast_length=5):
    """ Prepare data from csv file with time series and take needed number of
    elements

    :param n_steps: number of elements in time series to take
    :param forecast_length: the length of forecast
    """
    project_root_path = str(fedot_project_root())
    file_path = os.path.join(project_root_path, 'examples/data/ts_long.csv')
    df = pd.read_csv(file_path)
    df = df[df["series_id"] == "traffic_volume"]
    time_series = np.array(df['value'])[:n_steps]
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length))

    data = InputData(idx=np.arange(0, len(time_series)),
                     features=time_series,
                     target=time_series,
                     task=task,
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data), task


def clstm_forecasting():
    horizon = 24*2
    window_size = 29
    n_steps = 100
    (train_data, test_data), _ = get_ts_data_long(n_steps=n_steps + horizon, forecast_length=horizon)

    node_root = PrimaryNode("clstm")
    node_root.custom_params = {
        'input_size': 1,
        'window_size': window_size,
        'hidden_size': 135.66211398383396,
        'learning_rate': 0.00041403016307329,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 50
    }

    pipeline = Pipeline(node_root)
    pipeline.fit(train_data)
    prediction_before_export = pipeline.predict(test_data).predict

    display_validation_metric(
        np.ravel(prediction_before_export),
        test_data.target, np.concatenate([test_data.features[-window_size:], test_data.target]),
        True)


def get_source_pipeline_clstm():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge
    clstm - - - - /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_clstm = PrimaryNode('clstm')
    node_clstm.custom_params = {
        'input_size': 1,
        'window_size': 29.3418382487487,
        'hidden_size': 135.66211398383396,
        'learning_rate': 0.004641403016307329,
        'cnn1_kernel_size': 5,
        'cnn1_output_size': 32,
        'cnn2_kernel_size': 4,
        'cnn2_output_size': 32,
        'batch_size': 64,
        'num_epochs': 10
    }
    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_clstm])
    pipeline = Pipeline(node_final)

    return pipeline


def display_validation_metric(predicted, real, actual_values,
                              is_visualise: bool) -> None:
    """ Function calculate metrics based on predicted and tests data

    :param predicted: predicted values
    :param real: real values
    :param actual_values: source time series
    :param is_visualise: is it needed to show the plots
    """
    rmse_value = mean_squared_error(real, predicted, squared=False)
    mae_value = mean_absolute_error(real, predicted)
    print(f'RMSE - {rmse_value:.2f}')
    print(f'MAE - {mae_value:.2f}\n')
    if is_visualise:
        plot_results(actual_time_series=actual_values,
                     predicted_values=predicted,
                     len_train_data=len(actual_values) - len(predicted))


def run_ts_forecasting_problem(forecast_length=50,
                               with_visualisation=True,
                               cv_folds=None) -> None:
    """ Function launch time series task with composing

    :param forecast_length: length of the forecast
    :param with_visualisation: is it needed to show the plots
    :param cv_folds: is it needed apply cross validation and what number
    of folds to use
    """
    file_path = '../../cases/data/metocean/metocean_data_test.csv'

    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    # Train/test split
    train_part = time_series[len(time_series)-200:-forecast_length]
    test_part = time_series[-forecast_length:]

    # Prepare data for train and test
    train_input, predict_input, task = prepare_train_test_input(train_part,
                                                                forecast_length)

    # Get pipeline with pre-defined structure
    init_pipeline = get_source_pipeline_clstm()

    # Init check
    preds = fit_predict_for_pipeline(pipeline=init_pipeline,
                                     train_input=train_input,
                                     predict_input=predict_input)

    display_validation_metric(predicted=preds,
                              real=test_part,
                              actual_values=time_series[-100:],
                              is_visualise=with_visualisation)

    # Get available_operations type
    primary_operations, secondary_operations = get_available_operations()

    # Composer parameters
    composer_requirements = GPComposerRequirements(
        primary=primary_operations,
        secondary=secondary_operations, max_arity=3,
        max_depth=8, pop_size=10, num_of_generations=10,
        crossover_prob=0.8, mutation_prob=0.8,
        timeout=datetime.timedelta(minutes=10),
        cv_folds=cv_folds, validation_blocks=3)

    mutation_types = [parameter_change_mutation, MutationTypesEnum.growth, MutationTypesEnum.reduce,
                      MutationTypesEnum.simple]
    optimiser_parameters = GPGraphOptimiserParameters(mutation_types=mutation_types)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.RMSE)
    builder = GPComposerBuilder(task=task). \
        with_optimiser_parameters(optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_pipeline(init_pipeline)
    composer = builder.build()

    obtained_pipeline = composer.compose_pipeline(data=train_input, is_visualise=False)

    ###################################
    # Obtained pipeline visualisation #
    ###################################
    if with_visualisation:
        obtained_pipeline.show()

    preds = fit_predict_for_pipeline(pipeline=obtained_pipeline,
                                     train_input=train_input,
                                     predict_input=predict_input)

    display_validation_metric(predicted=preds,
                              real=test_part,
                              actual_values=time_series,
                              is_visualise=with_visualisation)

    obtained_pipeline.print_structure()


if __name__ == '__main__':
    clstm_forecasting()
    # run_ts_forecasting_problem(forecast_length=50,
    #                            with_visualisation=True,
    #                            cv_folds=2)
