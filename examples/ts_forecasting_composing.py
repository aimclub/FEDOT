import datetime
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.composer.visualisation import ChainVisualiser
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

warnings.filterwarnings('ignore')


def get_source_chain():
    """
    Return chain with the following structure:
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
    chain = Chain(node_final)

    return chain


def display_chain_info(chain):
    """ Function print info about chain """

    print('\nObtained chain:')
    for node in chain.nodes:
        print(f'{node.operation.operation_type}, params: {node.custom_params}')
    depth = int(chain.depth)
    print(f'Chain depth {depth}\n')


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


def fit_predict_for_chain(chain, train_input, predict_input):
    """ Function apply fit and predict operations

    :param chain: chain to process
    :param train_input: InputData for fit
    :param predict_input: InputData for predict

    :return preds: prediction of the chain
    """
    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)
    # Convert to one dimensional array
    preds = np.ravel(np.array(predicted_values.predict))

    return preds


def run_ts_forecasting_problem(forecast_length=50,
                               with_visualisation=True) -> None:
    """ Function launch time series task with composing

    :param forecast_length: length of the forecast
    :param with_visualisation: is it needed to show the plots
    """
    file_path = '../cases/data/metocean/metocean_data_test.csv'

    df = pd.read_csv(file_path)
    time_series = np.array(df['sea_height'])

    # Train/test split
    train_part = time_series[:-forecast_length]
    test_part = time_series[-forecast_length:]

    # Prepare data for train and test
    train_input, predict_input, task = prepare_train_test_input(train_part,
                                                                forecast_length)

    # Get chain with pre-defined structure
    init_chain = get_source_chain()

    # Init check
    preds = fit_predict_for_chain(chain=init_chain,
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
        max_depth=8, pop_size=10, num_of_generations=15,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=10),
        allow_single_operations=False)

    mutation_types = [MutationTypesEnum.parameter_change, MutationTypesEnum.simple,
                      MutationTypesEnum.reduce]
    optimiser_parameters = GPChainOptimiserParameters(mutation_types=mutation_types)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.MAE)
    builder = GPComposerBuilder(task=task). \
        with_optimiser_parameters(optimiser_parameters).\
        with_requirements(composer_requirements).\
        with_metrics(metric_function).with_initial_chain(init_chain)
    composer = builder.build()

    obtained_chain = composer.compose_chain(data=train_input, is_visualise=False)

    ################################
    # Obtained chain visualisation #
    ################################
    if with_visualisation:
        visualiser = ChainVisualiser()
        visualiser.visualise(obtained_chain)

    preds = fit_predict_for_chain(chain=obtained_chain,
                                  train_input=train_input,
                                  predict_input=predict_input)

    display_validation_metric(predicted=preds,
                              real=test_part,
                              actual_values=time_series,
                              is_visualise=with_visualisation)

    display_chain_info(obtained_chain)


if __name__ == '__main__':
    run_ts_forecasting_problem(forecast_length=100,
                               with_visualisation=True)
