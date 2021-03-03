import datetime
import warnings

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from fedot.core.composer.gp_composer.gp_composer import \
    GPComposerBuilder, GPComposerRequirements
from fedot.core.repository.quality_metrics_repository import \
    MetricsRepository, RegressionMetricsEnum
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.operations.tuning.hyperopt_tune.\
    tuners import SequentialTuner, ChainTuner

warnings.filterwarnings('ignore')


def run_experiment(file_path, init_chain, file_to_save,
                   iterations=20, tuner=None):
    """ Function launch experiment for river level prediction. Composing and
    tuner processes are available for such experiment.

    :param file_path: path to the file with river level data
    :param init_chain: chain to start composing process
    :param file_to_save: path to the file and file name to save report
    :param iterations: amount of iterations to process
    :param tuner: if tuning after composing process is required or not. tuner -
    NodesTuner or ChainTuner.
    """

    # Read dataframe and prepare train and test data
    df = pd.read_csv(file_path)
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']]),
        np.array(df['level_station_2']),
        test_size=0.2,
        shuffle=True,
        random_state=10)

    # Report arrays
    obtained_chains = []
    depths = []
    maes = []
    for i in range(0, iterations):
        print(f'Iteration {i}\n')

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

        available_model_types_secondary = ['ridge', 'lasso', 'dtreg',
                                           'xgbreg', 'adareg', 'knnreg',
                                           'linear', 'svr', 'poly_features', 'scaling',
                                           'ransac_lin_reg', 'rfe_lin_reg', 'pca',
                                           'ransac_non_lin_reg', 'rfe_non_lin_reg',
                                           'normalization']

        composer_requirements = GPComposerRequirements(
            primary=['one_hot_encoding'],
            secondary=available_model_types_secondary, max_arity=3,
            max_depth=8, pop_size=10, num_of_generations=12,
            crossover_prob=0.8, mutation_prob=0.8,
            max_lead_time=datetime.timedelta(minutes=5),
            allow_single_operations=True)

        metric_function = MetricsRepository().metric_by_id(
            RegressionMetricsEnum.MAE)
        builder = GPComposerBuilder(task=task).with_requirements(
            composer_requirements).with_metrics(metric_function).with_initial_chain(
            init_chain)
        composer = builder.build()

        obtained_chain = composer.compose_chain(data=train_input, is_visualise=False)

        print('\nObtained chain for current iteration')
        obtained_models = []
        for node in obtained_chain.nodes:
            print(str(node))
            obtained_models.append(str(node))
        depth = int(obtained_chain.depth)
        print(f'Chain depth {depth}\n')

        # Fit it
        obtained_chain.fit_from_scratch(train_input)

        # Predict
        predicted_values = obtained_chain.predict(predict_input)
        preds = predicted_values.predict

        y_data_test = np.ravel(y_data_test)
        mse_value = mean_squared_error(y_data_test, preds, squared=False)
        mae_value = mean_absolute_error(y_data_test, preds)

        print(f'Obtained metrics for current iteration {i}:')
        print(f'RMSE - {mse_value:.2f}')
        print(f'MAE - {mae_value:.2f}\n')

        if tuner is not None:
            print(f'Start tuning process ...')
            chain_tuner = tuner(chain=obtained_chain, task=task,
                                iterations=100)
            tuned_chain = chain_tuner.tune_chain(input_data=train_input,
                                                 loss_function=mean_absolute_error)

            # Fit it
            tuned_chain.fit_from_scratch(train_input)

            # Predict
            predicted_values_tuned = tuned_chain.predict(predict_input)
            preds_tuned = predicted_values_tuned.predict

            mse_value = mean_squared_error(y_data_test, preds_tuned, squared=False)
            mae_value = mean_absolute_error(y_data_test, preds_tuned)

            print(f'Obtained metrics for current iteration {i} after tuning:')
            print(f'RMSE - {mse_value:.2f}')
            print(f'MAE - {mae_value:.2f}\n')

        obtained_chains.append(obtained_models)
        maes.append(mae_value)
        depths.append(depth)

    report = pd.DataFrame({'Chain': obtained_chains,
                           'Depth': depths,
                           'MAE': maes})
    report.to_csv(file_to_save, index=False)


if __name__ == '__main__':

    # Define chain to start composing with it
    node_encoder = PrimaryNode('one_hot_encoding')
    node_rans = SecondaryNode('ransac_lin_reg', nodes_from=[node_encoder])
    node_scaling = SecondaryNode('scaling', nodes_from=[node_rans])
    node_final = SecondaryNode('linear', nodes_from=[node_scaling])

    init_chain = Chain(node_final)

    # Available tuners for application: ChainTuner, NodesTuner
    run_experiment(file_path='../cases/data/river_levels/station_levels.csv',
                   init_chain=init_chain,
                   file_to_save='data/river_levels/old_composer_new_preprocessing_report.csv',
                   iterations=20,
                   tuner=ChainTuner)
