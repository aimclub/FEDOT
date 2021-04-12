import warnings

import pandas as pd
import numpy as np

from copy import copy
from datetime import timedelta

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.chains.tuning.unified import ChainTuner

warnings.filterwarnings('ignore')


def prepare_input_data(features, target):
    """ Function create InputData with features """
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        features,
        target,
        test_size=0.2,
        shuffle=True,
        random_state=10)
    y_data_test = np.ravel(y_data_test)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_data_train)),
                            features=x_data_train,
                            target=y_data_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_data_test)),
                              features=x_data_test,
                              target=y_data_test,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, task


def run_river_experiment(file_path, chain, iterations=20, tuner=None,
                         tuner_iterations=100):
    """ Function launch experiment for river level prediction. Tuner processes
    are available for such experiment.

    :param file_path: path to the file with river level data
    :param chain: chain to fit and make prediction
    :param iterations: amount of iterations to process
    :param tuner: if tuning after composing process is required or not. tuner -
    NodesTuner or ChainTuner.
    :param tuner_iterations: amount of iterations for tuning
    """

    # Read dataframe and prepare train and test data
    df = pd.read_csv(file_path)
    features = np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']])
    target = np.array(df['level_station_2'])

    # Prepare InputData for train and test
    train_input, predict_input, task = prepare_input_data(features, target)
    y_data_test = predict_input.target

    for i in range(0, iterations):
        print(f'Iteration {i}\n')

        # To avoid inplace transformations make copy
        current_chain = copy(chain)

        # Fit it
        current_chain.fit_from_scratch(train_input)

        # Predict
        predicted_values = current_chain.predict(predict_input)
        preds = predicted_values.predict

        y_data_test = np.ravel(y_data_test)
        mse_value = mean_squared_error(y_data_test, preds, squared=False)
        mae_value = mean_absolute_error(y_data_test, preds)

        print(f'Obtained metrics for current iteration {i}:')
        print(f'RMSE - {mse_value:.2f}')
        print(f'MAE - {mae_value:.2f}\n')

        if tuner is not None:
            print(f'Start tuning process ...')
            chain_tuner = tuner(chain=current_chain, task=task,
                                iterations=tuner_iterations, max_lead_time=timedelta(seconds=30))
            tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                                 loss_function=mean_absolute_error)

            # Predict
            predicted_values_tuned = tuned_chain.predict(predict_input)
            preds_tuned = predicted_values_tuned.predict

            mse_value = mean_squared_error(y_data_test, preds_tuned,
                                           squared=False)
            mae_value = mean_absolute_error(y_data_test, preds_tuned)

            print(f'Obtained metrics for current iteration {i} after tuning:')
            print(f'RMSE - {mse_value:.2f}')
            print(f'MAE - {mae_value:.2f}\n')


if __name__ == '__main__':

    node_encoder = PrimaryNode('one_hot_encoding')
    node_scaling = SecondaryNode('scaling', nodes_from=[node_encoder])
    node_ridge = SecondaryNode('ridge', nodes_from=[node_scaling])
    node_lasso = SecondaryNode('lasso', nodes_from=[node_scaling])
    node_final = SecondaryNode('rfr', nodes_from=[node_ridge, node_lasso])

    init_chain = Chain(node_final)

    # Available tuners for application: ChainTuner, NodesTuner
    run_river_experiment(file_path='../data/river_levels/station_levels.csv',
                         chain=init_chain,
                         iterations=20,
                         tuner=ChainTuner)
