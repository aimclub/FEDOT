""" Тестовый пример задачи регрессии на основе данных измерений уровня воды """
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from fedot.core.data.preprocessing import preprocessing_func_for_data, PreprocessingStrategy, \
    Scaling, Normalization, ImputationStrategy, EmptyStrategy
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.synthetic.data import regression_dataset

np.random.seed(10)


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
    preds = predicted_values.predict

    y_data_test = np.ravel(y_data_test)
    print(f'Predicted values: {preds[:5]}')
    print(f'Actual values: {y_data_test[:5]}')
    print(f'RMSE - {mse(y_data_test, preds, squared=False):.2f}\n')



if __name__ == '__main__':
    node_encoder = PrimaryNode('one_hot_encoding', manual_preprocessing_func=EmptyStrategy)
    node_poly = SecondaryNode('poly_features', manual_preprocessing_func=EmptyStrategy, nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', manual_preprocessing_func=EmptyStrategy, nodes_from=[node_poly])
    node_final = SecondaryNode('rfr', manual_preprocessing_func=EmptyStrategy, nodes_from=[node_encoder])
    chain = Chain(node_final)

    run_experiment('../cases/data/river_levels/station_levels.csv', chain)









