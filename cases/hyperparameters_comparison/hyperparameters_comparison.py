import os
from time import time
from datetime import timedelta
import tracemalloc

from sklearn.metrics import mean_squared_error, roc_auc_score
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

    tracemalloc.stop()
    tracemalloc.start()

    start_time = time()

    pipeline = func(train, **params)

    memory_spent = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    time_spent = time() - start_time
    tracemalloc.stop()

    output_train = pipeline.predict(train)
    score_train = metrics(output_train.target, output_train.predict)

    output_test = pipeline.predict(test)
    score_test = metrics(output_test.target, output_test.predict)

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


score_train, score_test, time_spent, memory_spent, pipeline = {}, {}, {}, {}, {}

datasets = {
    # 'elo': {'train_file': 'cases/data/elo/train_elo_split.csv',
    #         'test_file': 'cases/data/elo/test_elo_split.csv',
    #         'task': TaskTypesEnum.regression,
    #         'fitted_operations': ['xgbreg'],
    #         'metrics': partial(mean_squared_error, squared=False)},
    'scoring': {'train_file': 'cases/data/scoring/scoring_train.csv',
                'test_file': 'cases/data/scoring/scoring_test.csv',
                'task': TaskTypesEnum.classification,
                'fitted_operations': ['logit', 'lgbm'],
                'metrics': roc_auc_score}
}

params = {
    'xgbreg': {
        'n_estimators': (hp.choice, [[10 ** n for n in range(1, 10)]]),
        'max_depth': (hp.randint, [1, 10]),
        'learning_rate': (hp.loguniform, [np.log(0.01), np.log(1)]),
        'lambda': (hp.choice, [[0] + [1.21 ** i for i in range(0, 7)]]),
        'alpha': (hp.choice, [[0] + [1.2 ** i for i in range(0, 7)]]),
        'max_bin': (hp.choice, [[2 ** i - 1 for i in range(1, 8)]]),
        'cosample_bytree': (hp.uniform, [0.01, 1]),
        'subsample': (hp.uniform, [0.01, 1]),
        'min_child_weight': (hp.choice, [[2 ** i - 1 for i in range(3, 8)]])
    },
    'logit': {
        'C': (hp.loguniform, [np.log(1e-9), np.log(1e4)])
    },
    'lgbm': {
        'n_estimators': (hp.randint, [1, 1000]),
        'learning_rate': (hp.loguniform, [np.log(1e-6), np.log(1e2)]),
        'max_depth': (hp.randint, [1, 10]),
        'num_leaves': (hp.randint, [1, 10000]),
        'min_data_in_leaf': (hp.choice, [[2 ** i - 1 for i in range(3, 8)]]),
        'feature_fraction': (hp.uniform, [0, 1]),
        'bagging_fraction': (hp.uniform, [0, 1]),
        'lambda_l1': (hp.loguniform, [np.log(1e-4), np.log(1e4)]),
        'lambda_l2': (hp.loguniform, [np.log(1e-4), np.log(1e4)])
    }
}

ss = SearchSpace(params, True)
n_iter = [50, 100, 200, 500]
timeout = timedelta(minutes=40)

for d in datasets:
    for fo in datasets[d]['fitted_operations']:
        key = (d, fo, 'default', 0)
        score_train[key], score_test[key], time_spent[key], memory_spent[key], \
        pipeline[key] = make_measurement(
            func=fixed_structure_with_default_params,
            train_file_path=datasets[d]['train_file'],
            test_file_path=datasets[d]['test_file'],
            task=datasets[d]['task'],
            metrics=datasets[d]['metrics'],
            params={
                'fitted_operation': fo
            }
        )

        for i in n_iter:
            key = (d, fo, 'random', i)
            score_train[key], score_test[key], time_spent[key], memory_spent[key], \
            pipeline[key] = make_measurement(
                func=fixed_structure_with_params_optimization,
                train_file_path=datasets[d]['train_file'],
                test_file_path=datasets[d]['test_file'],
                task=datasets[d]['task'],
                metrics=datasets[d]['metrics'],
                params={
                    'fitted_operation': fo,
                    'algo': rand.suggest,
                    'iterations': i,
                    'timeout': timeout,
                    'search_space': ss,
                    'metrics': datasets[d]['metrics']
                }
            )

            key = (d, fo, 'tpe', i)
            score_train[key], score_test[key], time_spent[key], memory_spent[key], \
            pipeline[key] = make_measurement(
                func=fixed_structure_with_params_optimization,
                train_file_path=datasets[d]['train_file'],
                test_file_path=datasets[d]['test_file'],
                task=datasets[d]['task'],
                metrics=datasets[d]['metrics'],
                params={
                    'fitted_operation': fo,
                    'algo': tpe.suggest,
                    'iterations': i,
                    'timeout': timeout,
                    'search_space': ss,
                    'metrics': datasets[d]['metrics']
                }
            )

pd.set_option('max_column', 10)
long_comparison_table = pd.concat([pd.Series(score_train),
                                   pd.Series(score_test),
                                   pd.Series(time_spent),
                                   pd.Series(memory_spent),
                                   pd.Series(pipeline)],
                                  axis=1).reset_index()
long_comparison_table.columns = ['dataset', 'fitted_operation', 'approach', 'iterations',
                                 'score_train', 'score_test', 'time_spent', 'memory_spent', 'pipeline']
print(long_comparison_table)
long_comparison_table.to_csv('long_comparison.csv', index=False)
