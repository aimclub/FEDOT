import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data_split import train_test_data_setup

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


def run_multi_output_case(path):
    """ Function launch case for river levels prediction on Lena river as
    multi-output regression task

    :param path: path to the file with table
    """

    # Targets: 1_day, 2_day, 3_day, 4_day, 5_day, 6_day, 7_day
    dataframe = pd.read_csv(path, parse_dates=['date'])

    # Wrap dataframe into output
    train_data, test_data = dataframe_into_inputs(dataframe)

    # Create simple chain
    node_ridge = PrimaryNode('ridge')
    chain = Chain(node_ridge)

    chain.fit(train_data)
    predicted_output = chain.predict(test_data)
    forecast = predicted_output.predict


if __name__ == '__main__':
    path_file = './data/lena_levels/multi_sample.csv'
    run_multi_output_case(path_file)