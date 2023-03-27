import logging
import os
import timeit
from pathlib import Path
from typing import Type

import pandas as pd
from golem.core.log import Log
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.tuner_interface import BaseTuner
from hyperopt import hp

from cases.tuner_comparison.regression_test_pipelines import get_pipelines_for_regression
from cases.tuner_comparison.test_pipelines_clssification import get_pipelines_for_classification
from fedot.core.composer.metrics import ROCAUC, SMAPE
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.quality_metrics_repository import MetricType

search_space_dict = \
    {'knn': {
        'n_neighbors': {
            'hyperopt-dist': hp.uniformint,
            'sampling-scope': [1, 50],
            'type': 'discrete'}
    },
        'logit': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-2, 10.0],
                'type': 'continuous'}
        },
        'rf': {
            'max_features': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 10],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 15],
                'type': 'discrete'}
        },
        'dt': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'}
        },
        'knnreg': {
                'n_neighbors': {
                    'hyperopt-dist': hp.uniformint,
                    'sampling-scope': [1, 50],
                    'type': 'discrete'}
            },
        'svr': {
            'C': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-4, 25.0],
                'type': 'continuous'},
            'epsilon': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [1e-4, 1],
                'type': 'continuous'},
            'tol': {
                'hyperopt-dist': hp.loguniform,
                'sampling-scope': [1e-5, 1e-1],
                'type': 'continuous'}
        },
        'rfr': {
            'max_features': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.05, 1.0],
                'type': 'continuous'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 15],
                'type': 'discrete'}
        },
        'dtreg': {
            'max_depth': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 11],
                'type': 'discrete'},
            'min_samples_split': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [2, 21],
                'type': 'discrete'},
            'min_samples_leaf': {
                'hyperopt-dist': hp.uniformint,
                'sampling-scope': [1, 21],
                'type': 'discrete'}
        },
        'ridge': {
            'alpha': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 10.0],
                'type': 'continuous'}
        },
        'lasso': {
            'alpha': {
                'hyperopt-dist': hp.uniform,
                'sampling-scope': [0.01, 10.0],
                'type': 'continuous'}
        }
    }


def get_data_for_experiment(data_path):
    data = InputData.from_csv(file_path=data_path)
    train_data, test_data = train_test_data_setup(data)
    return train_data, test_data


def get_objective_evaluate(metric: MetricType, data: InputData, n_jobs: int = -1):
    objective = MetricsObjective(metric)
    data_split = DataSourceSplitter(cv_folds=3).build(data)
    objective_eval = PipelineObjectiveEvaluate(objective, data_split, eval_n_jobs=n_jobs)
    return objective_eval


def create_folder(save_path):
    """ Create folder for files """
    save_path = os.path.abspath(save_path)
    if os.path.isdir(save_path) is False:
        os.makedirs(save_path)


def get_pipelines_for_task(task: str):
    if task == 'classification':
        return get_pipelines_for_classification()
    elif task == 'regression':
        return get_pipelines_for_regression()


def get_metric_for_task(task: str):
    metric_by_task = {'classification': ROCAUC.get_value, 'regression': SMAPE.get_value}
    return metric_by_task[task]


def run_experiment(task: str, data_path: os.PathLike, tuner_cls: Type[BaseTuner], iterations: int, launch_num: int):
    n_jobs = -1
    pipelines = get_pipelines_for_task(task)
    train_data, test_data = get_data_for_experiment(data_path)
    metric = get_metric_for_task(task)
    objective_eval = get_objective_evaluate(metric, train_data, n_jobs)
    adapter = PipelineAdapter()
    search_space = PipelineSearchSpace(custom_search_space=search_space_dict, replace_default_search_space=True)

    column_names = ['pipeline_type', 'init_metric', 'final_metric', 'tuning_time', 'iter_num', 'dataset']
    df = pd.DataFrame(columns=column_names)

    dir_to_save = os.path.join(task, f'{tuner_cls.__name__}_{iterations}')
    create_folder(dir_to_save)
    dataset_name = os.path.basename(data_path)
    path_to_save = f'{dir_to_save}/{dataset_name}'

    tuner = tuner_cls(objective_eval, search_space, adapter, iterations, n_jobs)

    for pipeline_type, pipeline in pipelines.items():

        pipeline.fit(train_data)
        init_metric = abs(metric(pipeline, test_data))

        for i in range(launch_num):
            print(f'\nLaunch: {i+1}/{launch_num}\n'
                  f'On dataset: {dataset_name}\n'
                  f'Pipeline: {pipeline_type}\n'
                  f'Tuner {tuner_cls.__name__} with {iterations} iterations\n')

            start = timeit.default_timer()
            tuned_pipeline = tuner.tune(pipeline, show_progress=False)
            launch_time = timeit.default_timer() - start

            tuned_pipeline.fit(train_data)
            final_metric = abs(metric(tuned_pipeline, test_data))

            print(f'\nMetric before tuning: {init_metric}')
            print(f'Metric after tuning: {final_metric}\n')

            launch_result = pd.DataFrame(data=[[pipeline_type, init_metric, final_metric,
                                                launch_time, iterations, dataset_name]],
                                         columns=column_names)
            df = pd.concat([df, launch_result], ignore_index=True, axis=0)

        df.to_csv(path_to_save)

    return df


if __name__ == '__main__':
    task = 'classification'
    datasets = os.listdir(f'{task}_data')
    tuners = [IOptTuner, SimultaneousTuner]
    iters_num = [20, 100]
    Log().reset_logging_level(45)
    for dataset in datasets:
        for tuner in tuners:
            for iter_num in iters_num:
                path = Path('{task}_data', dataset)
                dataframe = run_experiment(task, path, tuner, iterations=iter_num, launch_num=30)
