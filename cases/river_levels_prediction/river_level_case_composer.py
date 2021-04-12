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

warnings.filterwarnings('ignore')


def get_chain_info(chain):
    """ Function print info about chain and return operations in it and depth

    :param chain: chain to process
    :return obtained_operations: operations in the nodes
    :return depth: depth of the chain
    """

    print('\nObtained chain for current iteration')
    obtained_operations = []
    for node in chain.nodes:
        print(str(node))
        obtained_operations.append(str(node))
    depth = int(chain.depth)
    print(f'Chain depth {depth}\n')

    return obtained_operations, depth


def fit_predict_for_chain(chain, train_input, predict_input):
    """ Function apply fit and predict operations

    :param chain: chain to process
    :param train_input: InputData for fit
    :param predict_input: InputData for predict

    :return preds: prediction of the chain
    """
    # Fit it
    chain.fit_from_scratch(train_input)

    # Predict
    predicted_values = chain.predict(predict_input)
    preds = predicted_values.predict

    return preds


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


def run_river_composer_experiment(file_path, init_chain, file_to_save,
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
    features = np.array(df[['level_station_1', 'mean_temp', 'month', 'precip']])
    target = np.array(df['level_station_2'])

    # Prepare InputData for train and test
    train_input, predict_input, task = prepare_input_data(features, target)
    y_data_test = predict_input.target

    available_operations_types = ['ridge', 'lasso', 'dtreg',
                                  'xgbreg', 'adareg', 'knnreg',
                                  'linear', 'svr', 'poly_features',
                                  'scaling', 'ransac_lin_reg', 'rfe_lin_reg',
                                  'pca', 'ransac_non_lin_reg',
                                  'rfe_non_lin_reg', 'normalization']

    # Report arrays
    obtained_chains = []
    depths = []
    maes = []
    for i in range(0, iterations):
        print(f'Iteration {i}\n')

        composer_requirements = GPComposerRequirements(
            primary=['one_hot_encoding'],
            secondary=available_operations_types, max_arity=3,
            max_depth=8, pop_size=10, num_of_generations=5,
            crossover_prob=0.8, mutation_prob=0.8,
            max_lead_time=datetime.timedelta(minutes=5),
            allow_single_operations=False)

        metric_function = MetricsRepository().metric_by_id(
            RegressionMetricsEnum.MAE)
        builder = GPComposerBuilder(task=task).\
            with_requirements(composer_requirements).\
            with_metrics(metric_function).with_initial_chain(init_chain)
        composer = builder.build()

        obtained_chain = composer.compose_chain(data=train_input, is_visualise=False)

        # Display info about obtained chain
        obtained_models, depth = get_chain_info(chain=obtained_chain)

        preds = fit_predict_for_chain(chain=obtained_chain,
                                      train_input=train_input,
                                      predict_input=predict_input)

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

            preds_tuned = fit_predict_for_chain(chain=tuned_chain,
                                                train_input=train_input,
                                                predict_input=predict_input)

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
    run_river_composer_experiment(file_path='../data/river_levels/station_levels.csv',
                                  init_chain=init_chain,
                                  file_to_save='data/river_levels/old_composer_new_preprocessing_report.csv',
                                  iterations=20,
                                  tuner=ChainTuner)
