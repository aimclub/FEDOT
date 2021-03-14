import os
import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import roc_auc_score as roc_auc
from matplotlib import pyplot as plt
import timeit
from copy import deepcopy

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
from fedot.utilities.synthetic.data import regression_dataset, classification_dataset

np.random.seed(1)


def get_regression_dataset(features_options, samples_amount=250,
                           features_amount=5):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - bias: bias term in the underlying linear model;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = regression_dataset(samples_amount=samples_amount,
                                        features_amount=features_amount,
                                        features_options=features_options,
                                        n_targets=1,
                                        noise=0.0, shuffle=True)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1, 100, features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                        test_size=0.3)

    return x_train, y_train, x_test, y_test


def get_classification_dataset(features_options, samples_amount=250,
                               features_amount=5, classes_amount=2):
    """
    Prepares four numpy arrays with different scale features and target
    :param samples_amount: Total amount of samples in the resulted dataset.
    :param features_amount: Total amount of features per sample.
    :param classes_amount: The amount of classes in the dataset.
    :param features_options: The dictionary containing features options in key-value
    format:
        - informative: the amount of informative features;
        - redundant: the amount of redundant features;
        - repeated: the amount of features that repeat the informative features;
        - clusters_per_class: the amount of clusters for each class;
    :return x_data_train: features to train
    :return y_data_train: target to train
    :return x_data_test: features to test
    :return y_data_test: target to test
    """

    x_data, y_data = classification_dataset(samples_amount=samples_amount,
                                            features_amount=features_amount,
                                            classes_amount=classes_amount,
                                            features_options=features_options)

    # Changing the scale of the data
    for i, coeff in zip(range(0, features_amount),
                        np.random.randint(1, 100, features_amount)):
        # Get column
        feature = np.array(x_data[:, i])

        # Change scale for this feature
        rescaled = feature * coeff
        x_data[:, i] = rescaled

    # Train and test split
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data,
                                                                            test_size=0.3)

    return x_data_train, y_data_train, x_data_test, y_data_test


def reg_chain_1():
    """ Return chain with the following structure:
    knnreg \
            lasso ->
    ridge  |
    """
    node_knnreg = PrimaryNode('knnreg')
    node_ridge = PrimaryNode('ridge')
    node_lasso = SecondaryNode('lasso', nodes_from=[node_knnreg, node_ridge])

    chain = Chain(node_lasso)

    return chain


def reg_chain_2():
    """ Return chain with the following structure:
    svr   \
           xgbreg ->
    ridge |
    """
    node_svr = PrimaryNode('svr')
    node_ridge = PrimaryNode('ridge')
    node_xgbreg = SecondaryNode('xgbreg', nodes_from=[node_svr, node_ridge])

    chain = Chain(node_xgbreg)

    return chain


def reg_chain_3():
    """ Return chain with the following structure:
    svr    ->  rfr  ->   ridge  -\
                     |            \
    lasso  ->  rfr  ->    svr   -> xgbreg
                    |             |
    ridge  -> dtreg ->  ridge   -|
    * dtreg from second line has connection with ridge from third line
    """
    # First line
    node_1_svr = PrimaryNode('svr')
    node_1_lasso = PrimaryNode('lasso')
    node_1_ridge = PrimaryNode('ridge')

    # Second line
    node_2_rfr_0 = SecondaryNode('rfr', nodes_from=[node_1_svr])
    node_2_rfr_1 = SecondaryNode('rfr', nodes_from=[node_1_lasso])
    node_2_dtreg = SecondaryNode('dtreg', nodes_from=[node_1_ridge])

    # Third line
    node_3_ridge_0 = SecondaryNode('ridge', nodes_from=[node_2_rfr_0, node_2_dtreg])
    node_3_svr = SecondaryNode('svr', nodes_from=[node_2_rfr_1])
    node_3_ridge_1 = SecondaryNode('ridge', nodes_from=[node_2_dtreg])

    # Root node
    node_xgbreg = SecondaryNode('xgbreg', nodes_from=[node_3_ridge_0, node_3_svr, node_3_ridge_1])
    chain = Chain(node_xgbreg)

    return chain


def class_chain_1():
    """ Return chain with the following structure:
    knn   \
            logit ->
    logit |
    """
    node_knn = PrimaryNode('knn')
    node_logit_1 = PrimaryNode('logit')
    node_logit_2 = SecondaryNode('logit', nodes_from=[node_knn, node_logit_1])

    chain = Chain(node_logit_2)

    return chain


def class_chain_2():
    """ Return chain with the following structure:
    dt    \
            xgboost ->
    logit |
    """
    node_dt = PrimaryNode('dt')
    node_logit_1 = PrimaryNode('logit')
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_dt, node_logit_1])

    chain = Chain(node_xgboost)

    return chain


def class_chain_3():
    """ Return chain with the following structure:
    logit ->    dt    ->   xgboost -\
                                     \
    dt    ->    knn   ->    logit   -> xgboost
                       |             |
    rf    ->  xgboost ->     dt    -|
    * xgboost from second line has connection with logit from third line
    """
    # First line
    node_1_logit = PrimaryNode('logit')
    node_1_dt = PrimaryNode('dt')
    node_1_rf = PrimaryNode('rf')

    # Second line
    node_2_dt = SecondaryNode('dt', nodes_from=[node_1_logit])
    node_2_knn = SecondaryNode('knn', nodes_from=[node_1_dt])
    node_2_xgboost = SecondaryNode('xgboost', nodes_from=[node_1_rf])

    # Third line
    node_3_xgboost = SecondaryNode('xgboost', nodes_from=[node_2_dt])
    node_3_logit = SecondaryNode('logit', nodes_from=[node_2_knn, node_2_xgboost])
    node_3_dt = SecondaryNode('dt', nodes_from=[node_2_xgboost])

    # Root node
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_3_xgboost, node_3_logit, node_3_dt])
    chain = Chain(node_xgboost)

    return chain


def get_real_case_regression_dataset():
    """ Function returns InputData for algorithm launch """
    file_path = '../cases/data/river_levels/encoded_data_levels.csv'
    df = pd.read_csv(file_path)
    features = np.array(df[['month_enc_1', 'month_enc_2', 'month_enc_3', 'month_enc_4', 'month_enc_5',
                            'month_enc_6', 'month_enc_7', 'month_enc_8', 'month_enc_9', 'month_enc_10',
                            'month_enc_11', 'month_enc_12', 'level_station_1', 'mean_temp', 'precip']])
    target = np.array(df['level_station_2'])
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_synthetic_case_regression_dataset():
    features_options = {'informative': 1, 'bias': 0.0}
    x_train, y_train, x_test, y_test = get_regression_dataset(features_options=features_options,
                                                              samples_amount=250,
                                                              features_amount=2)

    # Define regression task
    task = Task(TaskTypesEnum.regression)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_synthetic_case_classification_dataset():
    features_options = {'informative': 1, 'redundant': 0,
                        'repeated': 0, 'clusters_per_class': 1}
    x_train, y_train, x_test, y_test = get_classification_dataset(features_options=features_options,
                                                                  samples_amount=250,
                                                                  features_amount=2,
                                                                  classes_amount=2)

    # Define regression task
    task = Task(TaskTypesEnum.classification)

    # Prepare data to train the model
    train_input = InputData(idx=np.arange(0, len(x_train)),
                            features=x_train,
                            target=y_train,
                            task=task,
                            data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)),
                              features=x_test,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def run_rivers_case_regression(chain, iterations, tuner_function):
    """ Function start real case regression example

    :param chain: chain to process
    :param iterations: amount of iterations to repeat
    :param tuner_function: function which use chain and give tuned chain after
    """

    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = get_real_case_regression_dataset()
    y_test = np.ravel(y_test)

    maes_before_tuning = []
    maes_after_tuning = []
    ids = []
    chain_structures = []
    for i in range(0, iterations):
        print(f'Iteration {i}')

        chain.fit_from_scratch(train_input)
        # Predict
        predicted_values = chain.predict(predict_input)
        predictions_before_tuning = predicted_values.predict

        mae_before_tuning = mean_absolute_error(y_test, predictions_before_tuning)
        print(f'MAE before tuning - {mae_before_tuning:.3f}')

        # Tuning the chain
        tuned_chain = tuner_function(deepcopy(chain), train_input)

        # Predictions after tuning
        predicted_values = tuned_chain.predict(predict_input)
        predictions_after_tuning = predicted_values.predict

        mae_after_tuning = mean_absolute_error(y_test, predictions_after_tuning)
        print(f'MAE after tuning - {mae_after_tuning:.3f}\n')

        maes_before_tuning.append(mae_before_tuning)
        maes_after_tuning.append(mae_after_tuning)
        ids.append(i)
        chain_structures.append(obtained_operations)

    result = pd.DataFrame({'Chain': chain_structures,
                           'Iteration': ids,
                           'MAE before tuning': maes_before_tuning,
                           'MAE after tuning': maes_after_tuning})

    return result


def run_synthetic_case_regression(chain, iterations, tuner_function):
    """ Function start synthetic case regression example

    :param chain: chain to process
    :param iterations: amount of iterations to repeat
    :param tuner_function: function which use chain and give tuned chain after
    """

    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = get_synthetic_case_regression_dataset()
    y_test = np.ravel(y_test)

    maes_before_tuning = []
    maes_after_tuning = []
    ids = []
    chain_structures = []
    for i in range(0, iterations):
        print(f'Iteration {i}')

        chain.fit_from_scratch(train_input)
        # Predict
        predicted_values = chain.predict(predict_input)
        predictions_before_tuning = predicted_values.predict

        mae_before_tuning = mean_absolute_error(y_test, predictions_before_tuning)
        print(f'MAE before tuning - {mae_before_tuning:.3f}')

        # Tuning the chain
        tuned_chain = tuner_function(deepcopy(chain), train_input)

        # Predictions after tuning
        predicted_values = tuned_chain.predict(predict_input)
        predictions_after_tuning = predicted_values.predict

        mae_after_tuning = mean_absolute_error(y_test, predictions_after_tuning)
        print(f'MAE after tuning - {mae_after_tuning:.3f}\n')

        maes_before_tuning.append(mae_before_tuning)
        maes_after_tuning.append(mae_after_tuning)
        ids.append(i)
        chain_structures.append(obtained_operations)

    result = pd.DataFrame({'Chain': chain_structures,
                           'Iteration': ids,
                           'MAE before tuning': maes_before_tuning,
                           'MAE after tuning': maes_after_tuning})

    return result


def run_synthetic_case_classification(chain, iterations, tuner_function):
    """ Function start synthetic case classification example

    :param chain: chain to process
    :param iterations: amount of iterations to repeat
    :param tuner_function: function which use chain and give tuned chain after
    """

    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = get_synthetic_case_classification_dataset()
    y_test = np.ravel(y_test)

    rocs_before_tuning = []
    rocs_after_tuning = []
    ids = []
    chain_structures = []
    for i in range(0, iterations):
        print(f'Iteration {i}')

        chain.fit_from_scratch(train_input)
        # Predict
        predicted_values = chain.predict(predict_input)
        predictions_before_tuning = predicted_values.predict

        roc_before_tuning = roc_auc(y_test, predictions_before_tuning)
        print(f'ROC AUC before tuning - {roc_before_tuning:.3f}')

        # Tuning the chain
        tuned_chain = tuner_function(deepcopy(chain), train_input)

        # Predictions after tuning
        predicted_values = tuned_chain.predict(predict_input)
        predictions_after_tuning = predicted_values.predict

        roc_after_tuning = roc_auc(y_test, predictions_after_tuning)
        print(f'ROC AUC after tuning - {roc_after_tuning:.3f}\n')

        rocs_before_tuning.append(roc_before_tuning)
        rocs_after_tuning.append(roc_after_tuning)
        ids.append(i)
        chain_structures.append(obtained_operations)

    result = pd.DataFrame({'Chain': chain_structures,
                           'Iteration': ids,
                           'ROC AUC before tuning': rocs_before_tuning,
                           'ROC AUC after tuning': rocs_after_tuning})

    return result
