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


def run_experiment(file_path, chain, file_to_save):
    df = pd.read_csv(file_path)
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        np.array(df[['month_enc_1', 'month_enc_2', 'month_enc_3', 'month_enc_4', 'month_enc_5',
                     'month_enc_6', 'month_enc_7', 'month_enc_8', 'month_enc_9', 'month_enc_10',
                     'month_enc_11', 'month_enc_12', 'level_station_1', 'mean_temp', 'precip']]),
        np.array(df['level_station_2']),
        test_size=0.2,
        shuffle=True,
        random_state=10)

    obt_chains = []
    depths = []
    maes = []
    for i in range(0, 10):
        print(f'Iteration {i}')

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

        available_model_types_primary = ['ridge', 'lasso', 'dtreg',
                                         'xgbreg', 'adareg', 'knnreg',
                                         'linear', 'svr']

        available_model_types_secondary = ['ridge', 'lasso', 'dtreg',
                                           'xgbreg', 'adareg', 'knnreg',
                                           'linear', 'svr']

        composer_requirements = GPComposerRequirements(
            primary=available_model_types_primary,
            secondary=available_model_types_secondary, max_arity=3,
            max_depth=8, pop_size=10, num_of_generations=12,
            crossover_prob=0.8, mutation_prob=0.8,
            max_lead_time=datetime.timedelta(minutes=5),
            add_single_model_chains=True)

        metric_function = MetricsRepository().metric_by_id(
            RegressionMetricsEnum.MAE)
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
        print(f'Chain depth {depth}')

        # Fit it
        obtained_chain.fit_from_scratch(train_input)

        # Predict
        predicted_values = obtained_chain.predict(predict_input)
        preds = predicted_values.predict

        y_data_test = np.ravel(y_data_test)
        mae = mean_absolute_error(y_data_test, preds)

        print(f'RMSE - {mse(y_data_test, preds, squared=False):.2f}')
        print(f'MAE - {mae:.2f}\n')

        obt_chains.append(obtained_models)
        maes.append(mae)
        depths.append(depth)

    report = pd.DataFrame({'Chain': obt_chains,
                           'Depth': depths,
                           'MAE': maes})
    report.to_csv(file_to_save, index=False)


if __name__ == '__main__':

    node_ridge = PrimaryNode('ridge')
    node_reg = PrimaryNode('dtreg')
    node_final = SecondaryNode('linear', nodes_from=[node_ridge, node_reg])
    chain = Chain(node_final)

    run_experiment('../cases/data/river_levels/encoded_data_levels.csv', chain,
                   file_to_save='data/river_levels/old_composer_report.csv')
