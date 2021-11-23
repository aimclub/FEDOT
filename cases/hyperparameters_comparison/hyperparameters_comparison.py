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
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.quality_metrics_repository import RegressionMetricsEnum
from fedot.core.composer.gp_composer.gp_composer import GPComposer, GPComposerBuilder, GPComposerRequirements, \
    GPGraphOptimiser, GPGraphOptimiserParameters
from fedot.api.api_utils.api_composer import ApiComposer
from fedot.core.composer.gp_composer.specific_operators import boosting_mutation, parameter_change_mutation


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


# not ready

"""
def variable_structure_with_mutual_bayesian_optimization(train,
                                                         max_arity=None,
                                                         max_depth=None,
                                                         pop_size=None,
                                                         num_of_generations=None,
                                                         crossover_prob=None,
                                                         mutation_prob=None):
    fixed_pipeline = get_fixed_pipeline(train.task.task_type)

    pipeline_ = Pipeline(fixed_pipeline)

    available_model_types, _ = OperationTypesRepository().suitable_operation(task_type=train.task.task_type)

    optimizer_parameters = GPGraphOptimiserParameters(genetic_scheme_type=genetic_scheme_type,
                                                      mutation_types=[boosting_mutation, parameter_change_mutation,
                                                                      MutationTypesEnum.single_edge,
                                                                      MutationTypesEnum.single_change,
                                                                      MutationTypesEnum.single_drop,
                                                                      MutationTypesEnum.single_add],
                                                      crossover_types=[CrossoverTypesEnum.one_point,
                                                                       CrossoverTypesEnum.subtree],
                                                      history_folder=composer_params.get('history_folder'))

    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 max_arity=5 if max_arity is None else max_arity,
                                 max_depth=5 if max_depth is None else max_depth,
                                 pop_size=10 if pop_size is None else pop_size,
                                 num_of_generations=5 if num_of_generations is None else num_of_generations,
                                 crossover_prob=0.4 if crossover_prob is None else crossover_prob,
                                 mutation_prob=0.5 if mutation_prob is None else mutation_prob,)

    builder = ApiComposer.get_gp_composer_builder(task=api_params['task'],
                                           metric_function=metric_function,
                                           composer_requirements=composer_requirements,
                                           optimizer_parameters=optimizer_parameters,
                                           data=api_params['train_data'],
                                           initial_pipeline=api_params['initial_pipeline'],
                                           logger=api_params['logger'])

    builder = GPComposerBuilder(train.task).with_requirements(req).with_metrics(RegressionMetricsEnum.RMSE)
    print(1)

    gp_composer = builder.build()
    print(2)
    pipeline_gp_composed = gp_composer.compose_pipeline(data=train)
    print(3)

    pipeline_gp_composed.fit_from_scratch(input_data=train)
    print(4)

    pipeline_tuner = PipelineTuner(pipeline_gp_composed, task=train.task.task_type, algo=tpe.suggest)
    print(5)
    tuned_pipeline = pipeline_tuner.tune_pipeline(train, loss_function=mean_squared_error)
    print(6)

    return tuned_pipeline
"""

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
