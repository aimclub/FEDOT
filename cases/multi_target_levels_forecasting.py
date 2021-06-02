import datetime
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.composer.optimisers.gp_comp.gp_optimiser import GPChainOptimiserParameters
from fedot.core.composer.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data_split import train_test_data_setup
from cases.river_levels_prediction.river_level_case_composer import get_chain_info

warnings.filterwarnings('ignore')


def dataframe_into_inputs(dataframe):
    """ Function converts pandas DataFrame into InputData FEDOT format

    :param dataframe: pandas Dataframe with data
    """
    target_columns = ['1_day', '2_day', '3_day', '4_day', '5_day', '6_day', '7_day']
    features_columns = ['stage_max_amplitude', 'stage_max_mean', 'snow_coverage_station_amplitude',
                        'snow_height_mean', 'snow_height_amplitude', 'water_hazard_sum']

    # Get features and targets arrays
    targets = np.array(dataframe[target_columns])
    features = np.array(dataframe[features_columns])

    task = Task(TaskTypesEnum.regression)
    input_data = InputData(idx=np.arange(0, len(features)), features=features,
                           target=targets, task=task, data_type=DataTypesEnum.table)

    # Split into train InputData and test one
    train_data, test_data = train_test_data_setup(input_data)

    return train_data, test_data


def plot_predictions(predicted_output, test_output):
    """ Function plot the predictions of the algorithm """
    predicted_columns = np.array(predicted_output.predict)
    actual_columns = np.array(test_output.target)

    # Take mean value for columns
    predicted = predicted_columns.mean(axis=1)
    actual = actual_columns.mean(axis=1)

    plt.plot(actual, label='7-day moving average actual')
    plt.plot(predicted, label='7-day moving average forecast')
    plt.ylabel('River level', fontsize=14)
    plt.xlabel('Time index', fontsize=14)
    plt.grid()
    plt.legend(fontsize=12)
    plt.show()


def run_multi_output_case(path, vis=False):
    """ Function launch case for river levels prediction on Lena river as
    multi-output regression task

    :param path: path to the file with table
    :param vis: is it needed to visualise chain and predictions
    """

    # Targets: 1_day, 2_day, 3_day, 4_day, 5_day, 6_day, 7_day
    dataframe = pd.read_csv(path, parse_dates=['date'])

    # Wrap dataframe into output
    train_data, test_data = dataframe_into_inputs(dataframe)

    # Create simple chain
    node_scaling = PrimaryNode('scaling')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    init_chain = Chain(node_ridge)

    available_operations_types = ['ridge', 'lasso', 'dtreg',
                                  'xgbreg', 'adareg', 'rfr',
                                  'linear', 'svr', 'poly_features',
                                  'scaling', 'ransac_lin_reg', 'rfe_lin_reg',
                                  'pca', 'ransac_non_lin_reg',
                                  'rfe_non_lin_reg', 'normalization']
    composer_requirements = GPComposerRequirements(
        primary=available_operations_types,
        secondary=available_operations_types, max_arity=3,
        max_depth=8, pop_size=10, num_of_generations=25,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=5),
        allow_single_operations=False)
    mutation_types = [MutationTypesEnum.parameter_change, MutationTypesEnum.simple,
                      MutationTypesEnum.reduce]
    optimiser_parameters = GPChainOptimiserParameters(mutation_types=mutation_types)

    metric_function = MetricsRepository().metric_by_id(RegressionMetricsEnum.MAE)
    builder = GPComposerBuilder(task=train_data.task). \
        with_optimiser_parameters(optimiser_parameters). \
        with_requirements(composer_requirements). \
        with_metrics(metric_function).with_initial_chain(init_chain)
    composer = builder.build()
    obtained_chain = composer.compose_chain(data=train_data, is_visualise=False)

    get_chain_info(obtained_chain)
    if vis:
        obtained_chain.show()

    # Fit chain after composing
    obtained_chain.fit(train_data)
    predicted_output = obtained_chain.predict(test_data)
    # Convert output into one dimensional array
    forecast = np.ravel(np.array(predicted_output.predict))

    mse_value = mean_squared_error(np.ravel(test_data.target), forecast, squared=False)
    mae_value = mean_absolute_error(np.ravel(test_data.target), forecast)

    print(f'MAE - {mae_value:.2f}')
    print(f'RMSE - {mse_value:.2f}\n')

    if vis:
        plot_predictions(predicted_output, test_data)


if __name__ == '__main__':
    path_file = './data/lena_levels/multi_sample.csv'
    run_multi_output_case(path_file, vis=True)
