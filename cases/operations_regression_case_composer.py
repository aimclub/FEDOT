""" Тестовый пример задачи регрессии на основе данных измерений уровня воды """
import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from matplotlib import pyplot as plt
import timeit

from pylab import rcParams
rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')
import datetime

from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum


from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

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

    available_model_types_primary = ['one_hot_encoding']

    available_model_types_secondary = ['ridge', 'dtreg', 'poly_features', 'scaling',
                                       'ransac_lin_reg', 'rfe_lin_reg', 'pca']

    composer_requirements = GPComposerRequirements(
        primary=available_model_types_primary,
        secondary=available_model_types_secondary, max_arity=5,
        max_depth=8, pop_size=10, num_of_generations=12,
        crossover_prob=0.8, mutation_prob=0.8,
        max_lead_time=datetime.timedelta(minutes=5),
        add_single_operation_chains=True)

    metric_function = MetricsRepository().metric_by_id(
        RegressionMetricsEnum.RMSE)
    builder = GPComposerBuilder(task=task).with_requirements(
        composer_requirements).with_metrics(metric_function).with_initial_chain(
        chain)
    composer = builder.build()

    obtained_chain = composer.compose_chain(data=train_input, is_visualise=False)

    print('Obtained chain')
    obtained_models = []
    for node in obtained_chain.nodes:
        print(str(node))
        obtained_models.append(str(node))
    depth = int(obtained_chain.depth)
    print(f'Глубина цепочки {depth}')

    # Fit it
    obtained_chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = obtained_chain.predict(predict_input)
    preds = predicted_values.predict

    y_data_test = np.ravel(y_data_test)
    print(f'Predicted values: {preds[:5]}')
    print(f'Actual values: {y_data_test[:5]}')
    print(f'RMSE - {mse(y_data_test, preds, squared=False):.2f}\n')


if __name__ == '__main__':

    node_encoder = PrimaryNode('one_hot_encoding')
    node_rans = SecondaryNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', nodes_from=[node_rans])
    node_final = SecondaryNode('linear', nodes_from=[node_scaling])
    chain = Chain(node_final)

    run_experiment('../cases/data/river_levels/station_levels.csv', chain)









