import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as roc_auc
import timeit
from copy import deepcopy

from pylab import rcParams

from fedot.core.utils import project_root

rcParams['figure.figsize'] = 18, 7
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.chains.chain import Chain
from fedot.core.data.data import InputData, OutputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

np.random.seed(1)


def smape_metric(y_true: np.array, y_pred: np.array) -> float:
    """ Symmetric mean absolute percentage error """

    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)
    result = numerator / denominator
    result[np.isnan(result)] = 0.0
    return float(np.mean(100 * result))


def reg_chain_1():
    """ Return chain with the following structure:
    knnreg -> ridge
    """
    node_knnreg = PrimaryNode('knnreg')
    node_ridge = SecondaryNode('ridge', nodes_from=[node_knnreg])
    chain = Chain(node_ridge)

    return chain


def reg_chain_2():
    """ Return chain with the following structure:
    svr   \
           rfr ->
    ridge |
    """
    node_svr = PrimaryNode('svr')
    node_ridge = PrimaryNode('ridge')
    node_rfr = SecondaryNode('rfr', nodes_from=[node_svr, node_ridge])

    chain = Chain(node_rfr)

    return chain


def reg_chain_3():
    """ Return chain with the following structure:
    svr    ->  rfr  ->   ridge  -\
                     |            \
    lasso  ->  rfr  ->    svr   -> rfr
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
    node_rfr = SecondaryNode('rfr', nodes_from=[node_3_ridge_0, node_3_svr, node_3_ridge_1])
    chain = Chain(node_rfr)

    return chain


def class_chain_1():
    """ Return chain with the following structure:
    knn -> rf
    """
    node_knn = PrimaryNode('knn')
    node_final = SecondaryNode('rf', nodes_from=[node_knn])
    chain = Chain(node_final)

    return chain


def class_chain_2():
    """ Return chain with the following structure:
    dt    \
            rf ->
    knn   |
    """
    node_dt = PrimaryNode('dt')
    node_knn = PrimaryNode('knn')
    node_rf = SecondaryNode('rf', nodes_from=[node_dt, node_knn])

    chain = Chain(node_rf)

    return chain


def class_chain_3():
    """ Return chain with the following structure:
    knn   ->    dt    ->   rf -\
                                     \
    dt    ->    knn   ->    knn     -> rf
                       |             |
    rf    ->  rf ->     dt    -|
    * xgboost from second line has connection with knn from third line
    """
    # First line
    node_1_knn = PrimaryNode('knn')
    node_1_dt = PrimaryNode('dt')
    node_1_rf = PrimaryNode('rf')

    # Second line
    node_2_dt = SecondaryNode('dt', nodes_from=[node_1_knn])
    node_2_knn = SecondaryNode('knn', nodes_from=[node_1_dt])
    node_2_xgboost = SecondaryNode('rf', nodes_from=[node_1_rf])

    # Third line
    node_3_xgboost = SecondaryNode('rf', nodes_from=[node_2_dt])
    node_3_knn = SecondaryNode('knn', nodes_from=[node_2_knn, node_2_xgboost])
    node_3_dt = SecondaryNode('dt', nodes_from=[node_2_xgboost])

    # Root node
    node_xgboost = SecondaryNode('rf', nodes_from=[node_3_xgboost, node_3_knn, node_3_dt])
    chain = Chain(node_xgboost)

    return chain


def get_cal_housing():
    """ Function returns InputData for algorithm launch """
    file_path = os.path.join(project_root(), 'cases', 'data', 'tuning_test', 'reg_cal_housing.csv')
    df = pd.read_csv(file_path)

    features_names = list(df.columns[:-1])
    features = np.array(df[features_names])
    target = np.array(df['target'])
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)
    task = Task(TaskTypesEnum.regression)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                            target=y_train, task=task, data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                              target=None, task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_delta_ailerons():
    file_path = os.path.join(project_root(), 'cases', 'data', 'tuning_test', 'reg_delta_ailerons.csv')
    df = pd.read_csv(file_path)

    features_names = list(df.columns[:-1])
    features = np.array(df[features_names])
    target = np.array(df['target'])
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)
    task = Task(TaskTypesEnum.regression)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                            target=y_train, task=task, data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                              target=None, task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_pol():
    file_path = os.path.join(project_root(), 'cases', 'data', 'tuning_test', 'reg_pol.csv')
    df = pd.read_csv(file_path)

    features_names = list(df.columns[:-1])
    features = np.array(df[features_names])
    target = np.array(df['target'])
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)
    task = Task(TaskTypesEnum.regression)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                            target=y_train, task=task, data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                              target=None, task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_amazon_employee_access():
    """ Function returns InputData for algorithm launch """
    file_path = os.path.join(project_root(), 'cases', 'data', 'tuning_test', 'class_Amazon_employee_access.csv')
    df = pd.read_csv(file_path)
    features_names = list(df.columns[:-1])
    features = np.array(df[features_names])
    target = np.array(df['target'])

    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)
    task = Task(TaskTypesEnum.classification)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                            target=y_train, task=task, data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                              target=None, task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_cnae9():
    file_path = os.path.join(project_root(), 'cases', 'data', 'tuning_test', 'class_cnae-9.csv')
    df = pd.read_csv(file_path)
    features_names = list(df.columns[:-1])
    features = np.array(df[features_names])
    target = np.array(df['target'])

    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)
    task = Task(TaskTypesEnum.classification)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                            target=y_train, task=task, data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                              target=None, task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def get_volkert_small():
    file_path = os.path.join(project_root(), 'cases', 'data', 'tuning_test', 'class_volkert_small.csv')
    df = pd.read_csv(file_path)
    features_names = list(df.columns[:-1])
    features = np.array(df[features_names])
    target = np.array(df['target'])

    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=10)
    task = Task(TaskTypesEnum.classification)
    train_input = InputData(idx=np.arange(0, len(x_train)), features=x_train,
                            target=y_train, task=task, data_type=DataTypesEnum.table)

    predict_input = InputData(idx=np.arange(0, len(x_test)), features=x_test,
                              target=None, task=task, data_type=DataTypesEnum.table)

    return train_input, predict_input, y_test


def run_reg_cal_housing(chain, iterations, tuner_function):
    """ Function start pnn regression case

    :param chain: chain to process
    :param iterations: amount of iterations to repeat
    :param tuner_function: function which use chain and give tuned chain after
    """
    return _regression_run(chain, iterations, tuner_function, get_cal_housing)


def run_reg_delta_ailerons(chain, iterations, tuner_function):
    return _regression_run(chain, iterations, tuner_function, get_delta_ailerons)


def run_reg_pol(chain, iterations, tuner_function):
    return _regression_run(chain, iterations, tuner_function, get_pol)


def run_class_amazon_employee_access(chain, iterations, tuner_function):
    return _classification_run(chain, iterations, tuner_function, get_amazon_employee_access)


def run_class_cnae9(chain, iterations, tuner_function):
    return _classification_run(chain, iterations, tuner_function, get_cnae9)


def run_class_volkert_small(chain, iterations, tuner_function):
    return _classification_run(chain, iterations, tuner_function, get_volkert_small)


def create_folder(save_path):
    """ Create folder for files """
    save_path = os.path.abspath(save_path)
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)


def _regression_run(chain, iterations, tuner_function, data_generator):
    """ Function start regression case check """

    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = data_generator()
    y_test = np.ravel(y_test)

    smapes_before_tuning = []
    smapes_after_tuning = []
    ids = []
    chain_structures = []
    times = []
    for i in range(0, iterations):
        print(f'Iteration {i}')

        chain.fit_from_scratch(train_input)
        # Predict
        predicted_values = chain.predict(predict_input)
        predictions_before_tuning = predicted_values.predict

        smape_before_tuning = smape_metric(y_test,
                                           predictions_before_tuning)
        print(f'SMAPE before tuning - {smape_before_tuning:.3f}')

        # Tuning the chain
        start = timeit.default_timer()
        tuned_chain = tuner_function(deepcopy(chain), train_input)

        # Predictions after tuning
        predicted_values = tuned_chain.predict(predict_input)
        predictions_after_tuning = predicted_values.predict

        launch_time = timeit.default_timer() - start
        smape_after_tuning = smape_metric(y_test, predictions_after_tuning)
        print(f'SMAPE after tuning - {smape_after_tuning:.3f}\n')

        smapes_before_tuning.append(smape_before_tuning)
        smapes_after_tuning.append(smape_after_tuning)
        ids.append(i)
        chain_structures.append(obtained_operations)
        times.append(launch_time)

    result = pd.DataFrame({'Pipeline': chain_structures,
                           'Iteration': ids,
                           'SMAPE before tuning': smapes_before_tuning,
                           'SMAPE after tuning': smapes_after_tuning,
                           'Time, sec.': times})

    return result


def _classification_run(chain, iterations, tuner_function, data_generator):
    # Get structure of the chain
    obtained_operations = []
    for node in chain.nodes:
        obtained_operations.append(str(node))

    train_input, predict_input, y_test = data_generator()
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

    result = pd.DataFrame({'Pipeline': chain_structures,
                           'Iteration': ids,
                           'ROC AUC before tuning': rocs_before_tuning,
                           'ROC AUC after tuning': rocs_after_tuning,
                           'Time, sec.': times})

    return result
