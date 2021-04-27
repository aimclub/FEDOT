import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.tuning.unified import ChainTuner
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from examples.ts_forecasting_tuning import prepare_input_data

warnings.filterwarnings('ignore')
np.random.seed(2020)


def get_refinement_chain():
    """ Create a chain like this
                    (Main branch)
            /->     linear  ->    ->    -> ridge -> FINAL FORECAST
           /          |                   /
    lagged            |                  /
           \          V                 /
            \->   decompose -> knnreg /
                    (Side branch)

       1              2          3            4
    """

    # 1
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': 150}

    # 2
    node_linear = SecondaryNode('linear', nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_lagged, node_linear])

    # 3
    node_knnreg = SecondaryNode('knnreg', nodes_from=[node_decompose])
    node_knnreg.custom_params = {'n_neighbors': 5}

    # 4 Combining branches with different targets (T and T_decomposed)
    final_node = SecondaryNode('ridge', nodes_from=[node_linear, node_knnreg])

    chain = Chain(final_node)
    return chain


def get_non_refinement_chain():
    """ Create a chain like this
            /->     linear  ->  ridge -> FINAL FORECAST
           /          |      /
    lagged            |     /
           \          V    /
            \->    knnreg /

       1              2          3
    """

    # 1
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': 150}

    # 2
    node_linear = SecondaryNode('linear', nodes_from=[node_lagged])
    node_knnreg = SecondaryNode('knnreg', nodes_from=[node_lagged, node_linear])
    node_knnreg.custom_params = {'n_neighbors': 5}

    # 3
    final_node = SecondaryNode('ridge', nodes_from=[node_linear, node_knnreg])

    chain = Chain(final_node)
    return chain


def run_refinement_forecast(path_to_file, len_forecast=100, with_decompose=True):
    """ Function launch example with experimental features of the FEDOT framework """

    # Read dataframe
    df = pd.read_csv(path_to_file)
    time_series = np.array(df['value'])
    train_part = time_series[:-len_forecast]
    test_part = time_series[-len_forecast:]

    if with_decompose is True:
        # Get chain with decomposing operation
        chain = get_refinement_chain()
    else:
        # Get simple chain without decomposing operation
        chain = get_non_refinement_chain()

    # Prepare InputData
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_part,
                                                          train_data_target=train_part,
                                                          test_data_features=train_part)
    # Fit chain
    chain.fit(train_input)

    # Make prediction
    predicted_output = chain.predict(predict_input)
    predicted_values = np.ravel(np.array(predicted_output.predict))

    mse = mean_squared_error(test_part, predicted_values, squared=False)
    mae = mean_absolute_error(test_part, predicted_values)
    print(f'RMSE - {mse:.4f}')
    print(f'MAE - {mae:.4f}\n')

    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_part), len(time_series)), predicted_values, label='Forecast')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    path = '../cases/data/ts_with_trend/economic_data.csv'
    run_refinement_forecast(path, with_decompose=True)
