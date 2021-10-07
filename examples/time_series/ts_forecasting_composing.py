import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.gp_composer.specific_operators import parameter_change_mutation
from fedot.core.data.data import InputData
from fedot.core.optimisers.gp_comp.gp_optimiser import GPGraphOptimiserParameters
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

warnings.filterwarnings('ignore')


def get_source_pipeline():
    """
    Return pipeline with the following structure:
    lagged - ridge \
                    -> ridge
    lagged - ridge /
    """

    # First level
    node_lagged_1 = PrimaryNode('lagged')
    node_lagged_2 = PrimaryNode('lagged')

    # Second level
    node_ridge_1 = SecondaryNode('ridge', nodes_from=[node_lagged_1])
    node_ridge_2 = SecondaryNode('ridge', nodes_from=[node_lagged_2])

    # Third level - root node
    node_final = SecondaryNode('ridge', nodes_from=[node_ridge_1, node_ridge_2])
    pipeline = Pipeline(node_final)

    return pipeline


def get_available_operations():
    """ Function returns available operations for primary and secondary nodes """
    primary_operations = ['lagged', 'smoothing', 'gaussian_filter', 'ar']
    secondary_operations = ['lagged', 'ridge', 'lasso', 'knnreg', 'linear',
                            'scaling', 'ransac_lin_reg', 'rfe_lin_reg']
    return primary_operations, secondary_operations


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


def plot_results(actual_time_series, predicted_values, len_train_data,
                 y_name='Sea surface height, m'):
    """
    Function for drawing plot with predictions

    :param actual_time_series: the entire array with one-dimensional data
    :param predicted_values: array with predicted values
    :param len_train_data: number of elements in the training sample
    :param y_name: name of the y axis
    """

    plt.plot(np.arange(0, len(actual_time_series)),
             actual_time_series, label='Actual values', c='green')
    plt.plot(np.arange(len_train_data, len_train_data + len(predicted_values)),
             predicted_values, label='Predicted', c='blue')

    # Plot black line which divide our array into train and test
    plt.plot([len_train_data, len_train_data],
             [min(actual_time_series), max(actual_time_series)], c='black',
             linewidth=1)
    plt.ylabel(y_name, fontsize=15)
    plt.xlabel('Time index', fontsize=15)
    plt.legend(fontsize=15)
    plt.grid()
    plt.show()


def prepare_train_test_input(train_part, len_forecast):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_part: time series which can be used as predictors for train

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    # Specify the task to solve
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_part)),
                            features=train_part,
                            target=train_part,
                            task=task,
                            data_type=DataTypesEnum.ts)

    start_forecast = len(train_part)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=train_part,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def fit_predict_for_pipeline(pipeline, train_input, predict_input):
    """ Function apply fit and predict operations

    :param pipeline: pipeline to process
    :param train_input: InputData for fit
    :param predict_input: InputData for predict

    :return preds: prediction of the pipeline
    """
    # Fit it
    pipeline.fit_from_scratch(train_input)

    # Predict
    predicted_values = pipeline.predict(predict_input)
    # Convert to one dimensional array
    preds = np.ravel(np.array(predicted_values.predict))

    return preds


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
    train_part = time_series[:-forecast_length]
    test_part = time_series[-forecast_length:]

    # Prepare data for train and test
    train_input, predict_input, task = prepare_train_test_input(train_part,
                                                                forecast_length)

    # Get pipeline with pre-defined structure
    init_pipeline = get_source_pipeline()

    # Init check
    preds = fit_predict_for_pipeline(pipeline=init_pipeline,
                                     train_input=train_input,
                                     predict_input=predict_input)
    display_validation_metric(predicted=preds,
                              real=test_part,
                              actual_values=time_series,
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
    run_ts_forecasting_problem(forecast_length=100,
                               with_visualisation=True,
                               cv_folds=2)
