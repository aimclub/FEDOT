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
            rf ->
    knn   |
    """
    node_knn_1 = PrimaryNode('knn')
    node_knn_2 = PrimaryNode('knn')
    node_final = SecondaryNode('rf', nodes_from=[node_knn_1, node_knn_2])

    chain = Chain(node_final)

    return chain


def class_chain_2():
    """ Return chain with the following structure:
    dt    \
            xgboost ->
    knn   |
    """
    node_dt = PrimaryNode('dt')
    node_knn = PrimaryNode('knn')
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_dt, node_knn])

    chain = Chain(node_xgboost)

    return chain


def class_chain_3():
    """ Return chain with the following structure:
    knn   ->    dt    ->   xgboost -\
                                     \
    dt    ->    knn   ->    knn     -> xgboost
                       |             |
    rf    ->  xgboost ->     dt    -|
    * xgboost from second line has connection with knn from third line
    """
    # First line
    node_1_knn = PrimaryNode('knn')
    node_1_dt = PrimaryNode('dt')
    node_1_rf = PrimaryNode('rf')

    # Second line
    node_2_dt = SecondaryNode('dt', nodes_from=[node_1_knn])
    node_2_knn = SecondaryNode('knn', nodes_from=[node_1_dt])
    node_2_xgboost = SecondaryNode('xgboost', nodes_from=[node_1_rf])

    # Third line
    node_3_xgboost = SecondaryNode('xgboost', nodes_from=[node_2_dt])
    node_3_knn = SecondaryNode('knn', nodes_from=[node_2_knn, node_2_xgboost])
    node_3_dt = SecondaryNode('dt', nodes_from=[node_2_xgboost])

    # Root node
    node_xgboost = SecondaryNode('xgboost', nodes_from=[node_3_xgboost, node_3_knn, node_3_dt])
    chain = Chain(node_xgboost)

    return chain


def get_pnn_1_regression_dataset():
    """ Function returns InputData for algorithm launch """
    file_path = '../cases/data/pnn_ml/529_pollen.csv'
    df = pd.read_csv(file_path)
    features = np.array(df[['RIDGE', 'NUB', 'CRACK', 'WEIGHT']])
    target = np.array(df['target'])
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


def get_pnn_2_regression_dataset():
    # TODO to implement
    pass


def get_pnn_3_regression_dataset():
    # TODO to implement
    pass


def get_pnn_1_classification_dataset():
    """ Function returns InputData for algorithm launch """
    file_path = '../cases/data/pnn_ml/wine_quality_red.csv'
    df = pd.read_csv(file_path)
    features = np.array(df[['fixed acidity', 'volatile acidity', 'citric acid',
                            'residual sugar', 'chlorides', 'free sulfur dioxide',
                            'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']])
    target = np.array(df['target'])
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)

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


def get_pnn_2_classification_dataset():
    # TODO to implement
    pass


def get_pnn_3_classification_dataset():
    # TODO to implement
    pass


def run_pnn_1_regression(chain, iterations, tuner_function):
    """ Function start pnn regression case

    :param chain: chain to process
    :param iterations: amount of iterations to repeat
    :param tuner_function: function which use chain and give tuned chain after
    """

    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = get_pnn_1_regression_dataset()
    y_test = np.ravel(y_test)

    maes_before_tuning = []
    maes_after_tuning = []
    ids = []
    chain_structures = []
    times = []
    for i in range(0, iterations):
        print(f'Iteration {i}')

        chain.fit_from_scratch(train_input)
        # Predict
        predicted_values = chain.predict(predict_input)
        predictions_before_tuning = predicted_values.predict

        mae_before_tuning = mean_absolute_error(y_test,
                                                predictions_before_tuning)
        print(f'MAE before tuning - {mae_before_tuning:.3f}')

        # Tuning the chain
        start = timeit.default_timer()
        tuned_chain = tuner_function(deepcopy(chain), train_input)

        # Predictions after tuning
        predicted_values = tuned_chain.predict(predict_input)
        predictions_after_tuning = predicted_values.predict

        launch_time = timeit.default_timer() - start
        mae_after_tuning = mean_absolute_error(y_test, predictions_after_tuning)
        print(f'MAE after tuning - {mae_after_tuning:.3f}\n')

        maes_before_tuning.append(mae_before_tuning)
        maes_after_tuning.append(mae_after_tuning)
        ids.append(i)
        chain_structures.append(obtained_operations)
        times.append(launch_time)

    result = pd.DataFrame({'Chain': chain_structures,
                           'Iteration': ids,
                           'MAE before tuning': maes_before_tuning,
                           'MAE after tuning': maes_after_tuning,
                           'Time': times})

    return result


def run_pnn_2_regression(chain, iterations, tuner_function):
    # TODO to implement
    pass


def run_pnn_3_regression(chain, iterations, tuner_function):
    # TODO to implement
    pass


def run_pnn_1_classification(chain, iterations, tuner_function):
    """ Function start pnn classification case

    :param chain: chain to process
    :param iterations: amount of iterations to repeat
    :param tuner_function: function which use chain and give tuned chain after
    """

    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = get_pnn_1_classification_dataset()
    y_test = np.ravel(y_test)

    rocs_before_tuning = []
    rocs_after_tuning = []
    ids = []
    chain_structures = []
    times = []
    for i in range(0, iterations):
        print(f'Iteration {i}')

        chain.fit_from_scratch(train_input)
        # Predict
        predicted_values = chain.predict(predict_input)
        predictions_before_tuning = predicted_values.predict

        roc_before_tuning = roc_auc(y_test, predictions_before_tuning,
                                    multi_class='ovr')
        print(f'ROC AUC before tuning - {roc_before_tuning:.3f}')

        # Tuning the chain
        start = timeit.default_timer()
        tuned_chain = tuner_function(deepcopy(chain), train_input)

        # Predictions after tuning
        predicted_values = tuned_chain.predict(predict_input)
        predictions_after_tuning = predicted_values.predict

        launch_time = timeit.default_timer() - start
        roc_after_tuning = roc_auc(y_test, predictions_after_tuning,
                                   multi_class='ovr')
        print(f'ROC AUC after tuning - {roc_after_tuning:.3f}\n')

        rocs_before_tuning.append(roc_before_tuning)
        rocs_after_tuning.append(roc_after_tuning)
        ids.append(i)
        chain_structures.append(obtained_operations)
        times.append(launch_time)

    result = pd.DataFrame({'Chain': chain_structures,
                           'Iteration': ids,
                           'ROC AUC before tuning': rocs_before_tuning,
                           'ROC AUC after tuning': rocs_after_tuning,
                           'Time': times})

    return result


def run_pnn_2_classification(chain, iterations, tuner_function):
    # TODO to implement
    pass


def run_pnn_3_classification(chain, iterations, tuner_function):
    # TODO to implement
    pass
