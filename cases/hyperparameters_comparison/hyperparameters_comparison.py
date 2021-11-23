import os
import random
from time import time
import tracemalloc

from sklearn.metrics import mean_squared_error
from functools import partial
import numpy as np
import pandas as pd

from hyperopt import rand, tpe, hp

from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.tuning.unified import PipelineTuner
from fedot.core.utils import fedot_project_root
from fedot.core.repository.tasks import TaskTypesEnum
from fedot.core.pipelines.tuning.search_space import SearchSpace

from multiprocessing import Pool, freeze_support


def pool_map(func,
             iterable,
             n_threads=1):
    if n_threads <= 1:
        res = [func(i) for i in iterable]
    else:
        p = Pool(n_threads)
        res = p.map(func=func,
                    iterable=iterable)
        p.close()
    return res


def get_data(file_path,
             task=None):
    full_path = os.path.join(str(fedot_project_root()), file_path)
    data = InputData.from_csv(full_path)
    if task == task:
        data.task.task_type = task
    return data


def make_measurement(func,
                     train_file_path,
                     test_file_path=None,
                     task=TaskTypesEnum.regression,
                     metrics=mean_squared_error,
                     params={}):
    if params is None:
        params = {}
    train = get_data(train_file_path, task)
    if test_file_path == test_file_path:
        test = get_data(test_file_path, task)
    else:
        test = train

    tracemalloc.start()
    tracemalloc.stop()
    tracemalloc.start()

    start_time = time()

    pipeline = func(train, **params)

    memory_spent = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    time_spent = time() - start_time
    tracemalloc.stop()

    output_train = pipeline.predict(train)
    output_test = pipeline.predict(test)

    try:
        score_train = metrics(output_train.target, output_train.predict)
        score_test = metrics(output_test.target, output_test.predict)
    except Exception:
        if output_test.predict.shape[1] == 1:
            score_train = metrics(output_train.target, output_train.predict.astype(int))
            score_test = metrics(output_test.target, output_test.predict.astype(int))
        else:
            score_train = metrics(output_train.target, output_train.predict.argmax(axis=1))
            score_test = metrics(output_test.target, output_test.predict.argmax(axis=1))

    return score_train, score_test, time_spent, memory_spent, pipeline.root_node.descriptive_id + ''


def get_fixed_pipeline(fitted_operation='xgbreg'):
    imputation_node = PrimaryNode(operation_type='simple_imputation')
    onehot_node = SecondaryNode(operation_type='one_hot_encoding',
                                nodes_from=[imputation_node])
    scaling_node = SecondaryNode(operation_type='scaling',
                                 nodes_from=[onehot_node])
    final_node = SecondaryNode(operation_type=fitted_operation,
                               nodes_from=[scaling_node])

    return final_node


def fixed_structure_with_default_params(train,
                                        fitted_operation='xgbreg'):
    fixed_pipeline = get_fixed_pipeline(fitted_operation)

    pipeline_ = Pipeline(fixed_pipeline)
    pipeline_.fit(train)

    return pipeline_


def fixed_structure_with_params_optimization(train,
                                             fitted_operation='xgbreg',
                                             iterations=100,
                                             timeout=timedelta(minutes=5),
                                             algo=rand.suggest,
                                             search_space=SearchSpace(),
                                             metrics=mean_squared_error,
                                             cv_folds=3):
    fixed_pipeline = get_fixed_pipeline(fitted_operation)

    pipeline_ = Pipeline(fixed_pipeline)

    pipeline_tuner = PipelineTuner(pipeline_,
                                   iterations=iterations,
                                   timeout=timeout,
                                   task=train.task.task_type,
                                   search_space=search_space,
                                   algo=algo)

    tuned_pipeline = pipeline_tuner.tune_pipeline(train,
                                                  loss_function=metrics,
                                                  cv_folds=cv_folds)

    return tuned_pipeline


def save_comparison_results(key):
    d, m, fo, algo, i, experiment = key
    filename = d + '_' + m + '_' + fo + '_' + algo + '_' + str(i) + '_' + str(experiment)
    if filename not in os.listdir('comparison_results'):
        if algo == 'default':
            score_train, score_test, time_spent, memory_spent, pipeline = make_measurement(
                func=fixed_structure_with_default_params,
                train_file_path=datasets[d]['train_file'],
                test_file_path=datasets[d]['test_file'],
                task=datasets[d]['task'],
                metrics=datasets[d]['metrics'][m],
                params={
                    'fitted_operation': fo
                }
            )
        else:
            score_train, score_test, time_spent, memory_spent, pipeline = make_measurement(
                func=fixed_structure_with_params_optimization,
                train_file_path=datasets[d]['train_file'],
                test_file_path=datasets[d]['test_file'],
                task=datasets[d]['task'],
                metrics=datasets[d]['metrics'][m],
                params={
                    'fitted_operation': fo,
                    'algo': rand.suggest if algo == 'random' else tpe.suggest,
                    'iterations': i,
                    'timeout': timeout,
                    'search_space': ss,
                    'metrics': datasets[d]['metrics'][m]
                }
            )
        txt = ','.join([str(score_train), str(score_test), str(time_spent), str(memory_spent), pipeline])
        open('comparison_results/' + filename, 'w').write(txt)


fitted_operations_for_classification = ['logit', 'lgbm']
fitted_operations_for_regression = ['ridge', 'lgbmreg']
metrics_for_binary_classification = {'f1': f1_score, 'auc': roc_auc_score}
metrics_for_multi_classification = {'f1': f1_score, 'accuracy': accuracy_score}
metrics_for_regression = {'mse': mean_squared_error, 'mae': mean_absolute_error}

overview = pd.read_csv('data_for_comparison/overview.csv')

datasets = {}
for idx, row in overview.iterrows():
    datasets[row['name']] = {
        'train_file': 'cases/hyperparameters_comparison/data_for_comparison/train_datasets/' + row['name'] + '.csv',
        'test_file': 'cases/hyperparameters_comparison/data_for_comparison/test_datasets/' + row['name'] + '.csv',
        'task': TaskTypesEnum.regression if row['task'] == 'regression'
        else TaskTypesEnum.classification,
        'fitted_operations': fitted_operations_for_regression if row['task'] == 'regression'
        else fitted_operations_for_classification,
        'metrics': metrics_for_regression if row['task'] == 'regression'
        else metrics_for_binary_classification if row['num_classes'] == 2
        else metrics_for_multi_classification
    }

params = {
    'ridge': {
        'alpha': (hp.loguniform, [np.log(1e-9), np.log(1e4)])
    },
    'logit': {
        'C': (hp.loguniform, [np.log(1e-9), np.log(1e4)])
    },
    'knn': {
        'n_neighbors': (hp.randint, [1, 100]),
        'metric': (hp.choice, [['euclidean', 'manhattan', 'chebyshev', 'cosine']]),
        'weights': (hp.choice, [['uniform', 'distance']])
    },
    'knnreg': {
        'n_neighbors': (hp.randint, [1, 100]),
        'metric': (hp.choice, [['euclidean', 'manhattan', 'chebyshev', 'cosine']]),
        'weights': (hp.choice, [['uniform', 'distance']])
    },
    'svr': {
        'C': (hp.loguniform, [np.log(1e-9), np.log(1e4)]),
        'kernel': (hp.choice, [['linear', 'poly', 'rbf', 'sigmoid']]),
        'degree': (hp.randint, [2, 4])
    },
    'dt': {
        'max_depth': (hp.randint, [2, 10]),
        'min_samples_leaf': (hp.randint, [1, 10000])
    },
    'dtreg': {
        'max_depth': (hp.randint, [2, 10]),
        'min_samples_leaf': (hp.randint, [1, 10000])
    },
    'rf': {
        'n_estimators': (hp.randint, [10, 1000]),
        'max_features': (hp.uniform, [0.05, 1]),
        'bootstrap': (hp.choice, [[True, False]]),
        'max_depth': (hp.randint, [2, 10]),
        'min_samples_leaf': (hp.randint, [1, 10000])
    },
    'rfr': {
        'n_estimators': (hp.randint, [10, 1000]),
        'max_features': (hp.uniform, [0.05, 1]),
        'bootstrap': (hp.choice, [[True, False]]),
        'max_depth': (hp.randint, [2, 10]),
        'min_samples_leaf': (hp.randint, [1, 10000])
    },
    'xgbreg': {
        'n_estimators': (hp.randint, [10, 1000]),
        'learning_rate': (hp.loguniform, [np.log(1e-6), np.log(1e1)]),
        'max_depth': (hp.randint, [2, 10]),
        'min_child_weight': (hp.randint, [1, 10000]),
        'cosample_bytree': (hp.uniform, [0.05, 1]),
        'subsample': (hp.uniform, [0.05, 1]),
        'alpha': (hp.loguniform, [np.log(1e-5), np.log(1e2)]),
        'lambda': (hp.loguniform, [np.log(1e-5), np.log(1e2)])
    },
    'lgbm': {
        'n_estimators': (hp.randint, [10, 1000]),
        'early_stopping_rounds': (hp.randint, [10, 100]),
        'learning_rate': (hp.loguniform, [np.log(1e-6), np.log(1e1)]),
        'max_depth': (hp.randint, [2, 10]),
        'min_data_in_leaf': (hp.randint, [1, 10000]),
        'feature_fraction': (hp.uniform, [0.05, 1]),
        'bagging_fraction': (hp.uniform, [0.05, 1]),
        'lambda_l1': (hp.loguniform, [np.log(1e-5), np.log(1e2)]),
        'lambda_l2': (hp.loguniform, [np.log(1e-5), np.log(1e2)])
    },
    'lgbmreg': {
        'n_estimators': (hp.randint, [10, 1000]),
        'early_stopping_rounds': (hp.randint, [10, 100]),
        'learning_rate': (hp.loguniform, [np.log(1e-6), np.log(1e1)]),
        'max_depth': (hp.randint, [2, 10]),
        'min_data_in_leaf': (hp.randint, [1, 10000]),
        'feature_fraction': (hp.uniform, [0.05, 1]),
        'bagging_fraction': (hp.uniform, [0.05, 1]),
        'lambda_l1': (hp.loguniform, [np.log(1e-5), np.log(1e2)]),
        'lambda_l2': (hp.loguniform, [np.log(1e-5), np.log(1e2)])
    }
}

ss = SearchSpace(params, True)
n_threads = 1
n_experiments = 3
n_iter = [50, 200]
timeout = timedelta(minutes=30)

keys = []

for experiment in range(n_experiments):
    for d in datasets:
        for m in datasets[d]['metrics']:
            for fo in datasets[d]['fitted_operations']:
                keys += [(d, m, fo, 'default', 0, experiment)]
                for i in n_iter:
                    keys += [(d, m, fo, 'random', i, experiment)]
                    keys += [(d, m, fo, 'tpe', i, experiment)]

if __name__ == '__main__':
    freeze_support()
    pool_map(func=save_comparison_results,
             iterable=keys,
             n_threads=n_threads)
