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
                (Side branch)
    lagged -> ts_decompose -> dtreg
       \                       |
        \                      v
         -----------------> linear -> final forecast
                (Main branch)
    """
    # First transformation
    node_lagged = PrimaryNode('lagged')

    # Make prediction in this node
    node_ridge = SecondaryNode('ridge', nodes_from=[node_lagged])

    # Find difference between predicted values and target
    node_decompose = SecondaryNode('ts_decompose', nodes_from=[node_ridge])

    # Model, which will predict residuals
    node_dtreg = SecondaryNode('dtreg', nodes_from=[node_decompose])

    # Final model, which will combine forecasts
    node_linear = SecondaryNode('linear', nodes_from=[node_ridge, node_dtreg])

    chain = Chain(node_linear)
    return chain


def run_refinement_forecast(path_to_file, len_forecast=100):
    """ Function launch example with experimental features of the FEDOT framework """
    # Read dataframe
    df = pd.read_csv(path_to_file)
    time_series = np.array(df['value'])
    train_part = time_series[:-len_forecast]
    test_part = time_series[-len_forecast:]

    # Get chain
    refinement_chain = get_refinement_chain()

    # Prepare InputData
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_part,
                                                          train_data_target=train_part,
                                                          test_data_features=train_part)
    # Fit chain
    refinement_chain.fit(train_input)

    # Make prediction
    predicted_output = refinement_chain.predict(predict_input)
    predicted_values = np.ravel(np.array(predicted_output.predict))

    print(f'Predicted values: {predicted_values}')


if __name__ == '__main__':
    path = '../cases/data/ts_with_trend/economic_data.csv'
    run_refinement_forecast(path)
