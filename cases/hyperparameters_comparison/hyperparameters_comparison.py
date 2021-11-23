import os
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

    start_time = time()
    tracemalloc.start()

    pipeline = func(train, **params)

    memory_spent = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    time_spent = time() - start_time
    tracemalloc.stop()

    output = pipeline.predict(test)

    score = metrics(output.target, output.predict)

    return score, time_spent, memory_spent, pipeline.root_node.descriptive_id


def get_fixed_pipeline(task_type):
    imputation_node = PrimaryNode(operation_type='simple_imputation')
    onehot_node = SecondaryNode(operation_type='one_hot_encoding',
                                nodes_from=[imputation_node])
    scaling_node = SecondaryNode(operation_type='scaling',
                                 nodes_from=[onehot_node])
    if task_type == TaskTypesEnum.regression:
        xgboost_node = SecondaryNode(operation_type='xgbreg',
                                     nodes_from=[scaling_node])
    else:
        xgboost_node = SecondaryNode(operation_type='xgboost',
                                     nodes_from=[scaling_node])

    return xgboost_node


def fixed_structure_with_default_params(train):
    fixed_pipeline = get_fixed_pipeline(train.task.task_type)

    pipeline_ = Pipeline(fixed_pipeline)
    pipeline_.fit(train)

    return pipeline_


def fixed_structure_with_random_search(train, search_space=SearchSpace()):
    fixed_pipeline = get_fixed_pipeline(train.task.task_type)

    pipeline_ = Pipeline(fixed_pipeline)

    pipeline_tuner = PipelineTuner(pipeline_, task=train.task.task_type, search_space=search_space, algo=rand.suggest)
    tuned_pipeline = pipeline_tuner.tune_pipeline(train, loss_function=mean_squared_error)

    return tuned_pipeline


def fixed_structure_with_bayesian_optimization(train, search_space=SearchSpace()):
    fixed_pipeline = get_fixed_pipeline(train.task.task_type)

    pipeline_ = Pipeline(fixed_pipeline)

    pipeline_tuner = PipelineTuner(pipeline_, task=train.task.task_type, search_space=search_space, algo=tpe.suggest)
    tuned_pipeline = pipeline_tuner.tune_pipeline(train, loss_function=mean_squared_error)

    return tuned_pipeline


score, time_spent, memory_spent, pipeline = {}, {}, {}, {}

datasets = {
    'elo': {'train_file': 'cases/data/elo/train_elo_split.csv',
            'test_file': 'cases/data/elo/test_elo_split.csv',
            'task': TaskTypesEnum.regression,
            'metrics': partial(mean_squared_error, squared=False)}
}

approaches = {
    'default': {
        'function': fixed_structure_with_default_params,
        'params': {}
    }
}

params = [
    {
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
        }
    }
]

for p in params:
    approaches['rand___' + str(p)] = {'function': fixed_structure_with_random_search,
                                      'params': {'search_space': SearchSpace(p, True)}}
    approaches['tpe___' + str(p)] = {'function': fixed_structure_with_bayesian_optimization,
                                     'params': {'search_space': SearchSpace(p, True)}}

for d in datasets:
    for a in approaches:
        score[(d, a)], time_spent[(d, a)], memory_spent[(d, a)], pipeline[(d, a)] = make_measurement(
            approaches[a]['function'],
            datasets[d]['train_file'],
            datasets[d]['test_file'],
            datasets[d]['task'],
            datasets[d]['metrics'],
            approaches[a]['params']
        )

pd.set_option('max_column', 6)
long_comparison_table = pd.concat([pd.Series(score),
                                   pd.Series(time_spent),
                                   pd.Series(memory_spent),
                                   pd.Series(pipeline)],
                                  axis=1).reset_index()
long_comparison_table.columns = ['dataset', 'approach', 'score', 'time_spent', 'memory_spent', 'pipeline']
print(long_comparison_table)
