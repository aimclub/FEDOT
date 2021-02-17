""" Тестовый пример задачи регрессии на основе данных измерений уровня воды """
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import regression_dataset

np.random.seed(10)


def chain_tuning(nodes_to_tune: str, chain: Chain, train_input: InputData,
                 predict_input: InputData, y_test: np.array, local_iter: int,
                 tuner_iter_num: int = 50) -> (float, list):
    several_iter_scores_test = []

    if nodes_to_tune == 'primary':
        print('primary_node_tuning')
        chain_tune_strategy = chain.fine_tune_primary_nodes
    elif nodes_to_tune == 'root':
        print('root_node_tuning')
        chain_tune_strategy = chain.fine_tune_all_nodes
    else:
        raise ValueError(f'Invalid type of nodes. Nodes must be primary or root')

    for iteration in range(local_iter):
        print(f'current local iteration {iteration}')

        # Chain tuning
        chain_tune_strategy(train_input, iterations=tuner_iter_num)

        # After tuning prediction
        chain.fit(train_input)
        predicted = chain.predict(predict_input)

        # Metrics
        mse_metric = mse(y_test, predicted.predict)
        several_iter_scores_test.append(mse_metric)

    return float(np.mean(several_iter_scores_test)), several_iter_scores_test


def run_experiment(file_path, chain):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        np.array(df[['level_station_1', 'month', 'mean_temp', 'precip']]),
        np.array(df['level_station_2']),
        test_size=0.2,
        shuffle=True,
        random_state=10)

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
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)
    # Fit it
    chain.fit(train_input, verbose=True)
    # Predict
    predicted_values = chain.predict(predict_input)
    before_tuning_predicted = predicted_values.predict

    y_data_test = np.ravel(y_data_test)
    print(f'RMSE before tuning - {mse(y_data_test, before_tuning_predicted, squared=False):.2f}\n')

    # Chain tuning
    local_iter=5
    after_tune_mse, several_iter_scores_test = chain_tuning(nodes_to_tune='primary',
                                                            chain=chain,
                                                            train_input=train_input,
                                                            predict_input=predict_input,
                                                            y_test=y_data_test,
                                                            local_iter=local_iter)

    print(f'Several test scores {several_iter_scores_test}')
    print(f'Mean test score over {local_iter} iterations: {after_tune_mse}')


if __name__ == '__main__':
    node_encoder = PrimaryNode('one_hot_encoding')

    # Chain with custom hyperparameters in nodes
    node_rans = SecondaryNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', nodes_from=[node_rans])
    node_final = SecondaryNode('ridge', nodes_from=[node_scaling])
    chain = Chain(node_final)

    run_experiment('../cases/data/river_levels/station_levels.csv', chain)









