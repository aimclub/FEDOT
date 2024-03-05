import os
import sys
import timeit
from datetime import timedelta
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
from golem.core.log import Log
from golem.core.tuning.iopt_tuner import IOptTuner
from golem.core.tuning.optuna_tuner import OptunaTuner
from golem.core.tuning.simultaneous import SimultaneousTuner
from golem.core.tuning.tuner_interface import BaseTuner

from cases.tuner_comparison.classification_test_pipelines import get_pipelines_for_classification
from cases.tuner_comparison.forecasting_test_pipelines import get_pipelines_for_forecasting
from cases.tuner_comparison.regression_test_pipelines import get_pipelines_for_regression
from fedot.core.composer.metrics import ROCAUC, SMAPE
from fedot.core.constants import DEFAULT_TUNING_ITERATIONS_NUMBER
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.objective import MetricsObjective, PipelineObjectiveEvaluate
from fedot.core.optimisers.objective.data_source_splitter import DataSourceSplitter
from fedot.core.pipelines.adapters import PipelineAdapter
from fedot.core.pipelines.tuning.search_space import PipelineSearchSpace
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import set_random_seed


def get_data_for_experiment(data_path, task, forecast_length, validation_blocks):
    if task == 'forecasting':
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length))
        data = InputData.from_csv_time_series(file_path=data_path, task=task)
    else:
        data = InputData.from_csv(file_path=data_path, task=task)
    train_data, test_data = train_test_data_setup(data, validation_blocks=validation_blocks)
    return train_data, test_data


def get_objective_evaluate(metric, data: InputData, validation_blocks: int, n_jobs: int = -1):
    objective = MetricsObjective(metric)
    data_split = DataSourceSplitter(cv_folds=3, validation_blocks=validation_blocks).build(data)
    objective_eval = PipelineObjectiveEvaluate(objective,
                                               data_split,
                                               eval_n_jobs=n_jobs,
                                               validation_blocks=validation_blocks)
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
    elif task == 'forecasting':
        return get_pipelines_for_forecasting()


def get_metric_for_task(task: str):
    metric_by_task = {'classification': ROCAUC.get_value, 'regression': SMAPE.get_value, 'forecasting': SMAPE.get_value}
    return metric_by_task[task]


def run_experiment(task: str,
                   data_path: os.PathLike,
                   tuner_cls: Type[BaseTuner],
                   iterations: int,
                   launch_num: int,
                   with_timeout: bool = False,
                   forecast_length: int = 10,
                   validation_blocks: int = None):
    n_jobs = -1
    pipelines = get_pipelines_for_task(task)
    train_data, test_data = get_data_for_experiment(data_path, task, forecast_length, validation_blocks)
    metric = get_metric_for_task(task)
    objective_eval = get_objective_evaluate(metric, train_data, validation_blocks, n_jobs)
    adapter = PipelineAdapter()
    search_space = PipelineSearchSpace()

    column_names = ['pipeline_type', 'init_metric', 'final_metric', 'tuning_time', 'iter_num', 'dataset']
    df = pd.DataFrame(columns=column_names)

    dir_to_save = os.path.join(task, f'{tuner_cls.__name__}_{iterations}')
    create_folder(dir_to_save)
    dataset_name = os.path.basename(data_path)
    path_to_save = f'{dir_to_save}/{dataset_name}'
    if with_timeout:
        path_to_save = f'{dir_to_save}/mean_time_{dataset_name}'
    num_iter = iterations

    for pipeline_type, pipeline in pipelines.items():
        print('Initial pipeline fit started')
        pipeline.fit(train_data)
        init_metric = abs(metric(pipeline, test_data, validation_blocks=validation_blocks))

        for i in range(launch_num):
            timeout = timedelta(minutes=360)
            if with_timeout:
                mean_time = pd.read_csv(Path(f'{task}', 'mean_time.csv'))
                seconds = mean_time[(mean_time.pipeline_type == pipeline_type)
                                    & (mean_time.iter_num == num_iter)
                                    & (mean_time.dataset == dataset_name)]['IOpt time mean'].values[0]
                timeout = timedelta(seconds=seconds)
                iterations = DEFAULT_TUNING_ITERATIONS_NUMBER

            tuner = tuner_cls(objective_eval,
                              search_space,
                              adapter,
                              iterations=iterations,
                              early_stopping_rounds=iterations,
                              timeout=timeout,
                              n_jobs=n_jobs)

            print(f'\nLaunch: {i + 1}/{launch_num}\n'
                  f'On dataset: {dataset_name}\n'
                  f'Pipeline: {pipeline_type}\n'
                  f'Tuner {tuner_cls.__name__} with {iterations} iterations\n')

            start = timeit.default_timer()
            tuned_pipeline = tuner.tune(pipeline, show_progress=False)
            launch_time = timeit.default_timer() - start

            tuned_pipeline.fit(train_data)
            final_metric = abs(metric(tuned_pipeline, test_data, validation_blocks=validation_blocks))

            print(f'\nMetric before tuning: {init_metric}')
            print(f'Metric after tuning: {final_metric}\n')

            launch_result = pd.DataFrame(data=[[pipeline_type, init_metric, final_metric,
                                                launch_time, iterations, dataset_name]],
                                         columns=column_names)
            df = pd.concat([df, launch_result], ignore_index=True, axis=0)

            df.to_csv(path_to_save)

    return df


if __name__ == '__main__':
    task = 'regression'
    datasets = os.listdir(f'{task}_data')
    tuners = [IOptTuner, SimultaneousTuner, OptunaTuner]
    iters_num = [20, 100]
    Log().reset_logging_level(20)
    set_random_seed(42)
    for dataset in datasets:
        for tuner in tuners:
            for iter_num in iters_num:
                path = Path(f'{task}_data', dataset)
                dataframe = run_experiment(task, path, tuner, iterations=iter_num, launch_num=30, with_timeout=False)
