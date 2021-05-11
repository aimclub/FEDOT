import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.chains.chain_ts_wrappers import in_sample_ts_forecast
from examples.ts_forecasting_tuning import prepare_input_data

warnings.filterwarnings('ignore')
np.random.seed(2020)


def get_refinement_chain(lagged, pre_decompose_model: str = 'lasso'):
    """ Create a chain like this
                    (Main branch)
            /-> pre_decompose_model ->    ->    -> ridge -> FINAL FORECAST
           /          |                   /
    lagged            |                  /
           \          V                 /
            \->   decompose ->  dtreg  /
                    (Side branch)

       1              2          3            4
    """

    # 1
    node_lagged = PrimaryNode('lagged')
    node_lagged.custom_params = {'window_size': lagged}

    # 2
    node_pre_decompose_model = SecondaryNode(pre_decompose_model, nodes_from=[node_lagged])
    node_decompose = SecondaryNode('decompose', nodes_from=[node_pre_decompose_model, node_lagged])

    # 3
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])
    node_dtreg.custom_params = {'max_depth': 3}

    # 4 Combining branches with different targets (T and T_decomposed)
    final_node = SecondaryNode('ridge', nodes_from=[node_pre_decompose_model,
                                                    node_dtreg])

    chain = Chain(final_node)
    return chain


def run_refinement_forecast(path_to_file, len_forecast=100, lagged=150,
                            validation_blocks=3, with_tuning=False,
                            pre_decompose_model='lasso'):
    """ Function launch example with experimental features of the FEDOT framework

    :param path_to_file: path to the csv file
    :param len_forecast: forecast length
    :param lagged: window size for lagged transformation
    :param validation_blocks: amount of parts for time series validation
    :param with_tuning: is it need to tune chains or not
    :param pre_decompose_model: model for prediction decompose
    """

    # Read dataframe
    df = pd.read_csv(path_to_file)
    time_series = np.array(df['value'])
    # 3 folds for validation
    horizon = len_forecast*validation_blocks
    train_part = time_series[:-horizon]
    test_part = time_series[-horizon:]

    # Get chain with decomposing operation
    chain = get_refinement_chain(lagged, pre_decompose_model)

    # Prepare InputData
    train_input, _, task = prepare_input_data(len_forecast=len_forecast,
                                              train_data_features=train_part,
                                              train_data_target=train_part,
                                              test_data_features=train_part)
    # Fit chain
    chain.fit(train_input)

    if with_tuning:
        chain.fine_tune_all_nodes(loss_function=mean_absolute_error,
                                  loss_params=None,
                                  input_data=train_input,
                                  iterations=20)

    # Create data for validation
    predict_input = InputData(idx=range(0, len(time_series)),
                              features=time_series,
                              target=time_series,
                              task=task,
                              data_type=DataTypesEnum.ts)

    # Make prediction
    predicted_values = in_sample_ts_forecast(chain=chain,
                                             input_data=predict_input,
                                             horizon=horizon)

    mse = mean_squared_error(test_part, predicted_values, squared=False)
    mae = mean_absolute_error(test_part, predicted_values)
    print(f'RMSE with decomposition - {mse:.4f}')
    print(f'MAE with decomposition - {mae:.4f}\n')

    plt.plot(range(0, len(time_series)), time_series, label='Actual time series')
    plt.plot(range(len(train_part), len(time_series)), predicted_values, label='With decomposition')

    i = len(train_part)
    for _ in range(0, validation_blocks):
        deviation = np.std(predicted_values)
        plt.plot([i, i], [min(predicted_values)-deviation, max(predicted_values)+deviation],
                 c='black', linewidth=1)
        i += len_forecast

    plt.legend()
    plt.close()


if __name__ == '__main__':
    # Good example: economic_data_6.csv
    path = '../cases/data/time_series/economic_data_2.csv'
    for model in ['lasso', 'ridge', 'dtreg', 'knnreg', 'treg', 'rfr', 'svr', 'sgdr', 'linear', 'adareg']:
        print(model)
        run_refinement_forecast(path, len_forecast=10, validation_blocks=5,
                                lagged=50,
                                with_tuning=False,
                                pre_decompose_model=model)
