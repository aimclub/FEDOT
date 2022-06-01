import collections
import operator
import timeit
from collections import defaultdict
from functools import reduce
from statistics import mean
from timeit import repeat
from typing import TYPE_CHECKING, Optional

import pandas as pd
from matplotlib import colors, pyplot as plt

from fedot.api.main import Fedot
from fedot.core.log import SingletonMeta
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root
from fedot.preprocessing.cache import PreprocessingCache
from test.unit.api.test_main_api import get_dataset

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


def _count_pipelines(opt_history: Optional[OptHistory]) -> int:
    if opt_history is not None:
        return reduce(operator.add, map(len, opt_history.individuals), 0)
    return 0


def _show_performance_plot(title: str, x: list, pipelines_count: dict, times: dict, plot_labels: dict):
    plt.title(title)
    plt.xlabel('timeout in minutes')
    plt.ylabel('correctly evaluated pipelines')

    c_norm = colors.Normalize(vmin=max(min(x) - 0.5, 0), vmax=max(x) + 0.5)
    cm = plt.cm.get_cmap('cool')
    for arg in pipelines_count:
        plt.plot(x, pipelines_count[arg], label=plot_labels[arg], zorder=1)
        plt.scatter(x, pipelines_count[arg], c=times[arg], cmap=cm, norm=c_norm, zorder=2)

    smp = plt.cm.ScalarMappable(norm=c_norm, cmap=cm)
    smp.set_array([])  # noqa Just to handle 'You must first set_array for mappable' problem
    cb = plt.colorbar(smp)
    cb.ax.set_ylabel('actual time for optimization in minutes', rotation=90)

    plt.legend()
    plt.grid()
    plt.show()


def dummy_time_check():
    composer_params = {
        'with_tuning': False,
        'validation_blocks': 1,
        'cv_folds': None,

        'max_depth': 4, 'max_arity': 2, 'pop_size': 3,
        'timeout': None, 'num_of_generations': 5
    }

    for use_cache in [False, True]:
        print(f'Using cache mode: {use_cache}')
        for task_type in ['ts_forecasting', 'regression', 'classification']:
            preset = 'best_quality'
            fedot_input = {'problem': task_type, 'seed': 42, 'preset': preset, 'verbose_level': -1,
                           'timeout': composer_params['timeout'], 'use_cache': use_cache,
                           'composer_params': composer_params}
            if task_type == 'ts_forecasting':
                fedot_input['task_params'] = TsForecastingParams(forecast_length=30)
            train_data, test_data, _ = get_dataset(task_type)

            def check():
                Fedot(**fedot_input).fit(features=train_data, target='target')

            print(f"task_type={task_type}, mean_time={mean(repeat(check, repeat=15, number=1))}")


class PreprocessingCacheMock(metaclass=SingletonMeta):
    def try_find_preprocessor(self, pipeline: 'Pipeline', input_data):
        return pipeline.preprocessor

    def add_preprocessor(self, pipeline: 'Pipeline', input_data):
        ...


def use_cache_check(n_jobs: int = 1, test_preprocessing: bool = False):
    """
    Performs experiment to show how caching pipelines operations helps in fitting FEDOT model
    """
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    train_data = pd.read_csv(train_data_path)[:4000]
    test_data = pd.read_csv(test_data_path)[:4000]

    pipelines_count, times = [{False: [], True: []} for _ in range(2)]
    plot_labels = {False: 'without cache', True: 'with cache'}
    composer_params = {'with_tuning': False}
    base_fedot_params = {
        'problem': problem, 'seed': 42,
        'composer_params': composer_params, 'preset': 'fast_train',
        'verbose_level': 0, 'n_jobs': n_jobs
    }
    timeouts = [1, 2, 3, 4, 5]
    preproc_orig_find = PreprocessingCache.try_find_preprocessor.__code__
    preproc_orig_add = PreprocessingCache.add_preprocessor.__code__
    for use_cache in [False, True]:
        if test_preprocessing and not use_cache:
            PreprocessingCache.try_find_preprocessor.__code__ = PreprocessingCacheMock.try_find_preprocessor.__code__
            PreprocessingCache.add_preprocessor.__code__ = PreprocessingCacheMock.add_preprocessor.__code__
        print(f'Using cache: {use_cache}')
        basic_cache_usage = use_cache if not test_preprocessing else False
        for timeout in timeouts:
            c_pipelines = 0.
            time = 0.
            mean_range = 3
            cache_effectiveness = collections.Counter()
            for i in range(mean_range):
                train_data_tmp = train_data.copy()
                test_data_tmp = test_data.copy()

                start_time = timeit.default_timer()
                auto_model = Fedot(**base_fedot_params, timeout=timeout, use_cache=basic_cache_usage)
                auto_model.fit(features=train_data_tmp, target='target')
                auto_model.predict_proba(features=test_data_tmp)
                c_pipelines += _count_pipelines(auto_model.history)
                time += (timeit.default_timer() - start_time) / 60
                cache_effectiveness += auto_model.api_composer.cache.effectiveness_ratio if basic_cache_usage else {}

            time /= mean_range
            c_pipelines //= mean_range
            times[use_cache].append(time)
            pipelines_count[use_cache].append(c_pipelines)
            cache_effectiveness = {k: v / mean_range for k, v in cache_effectiveness.items()}

            print((
                f'\tTimeout: {timeout}'
                f', number of pipelines: {c_pipelines}, elapsed time: {time:.3f}'
                f', cache effectiveness: {cache_effectiveness}'
            ))
        if test_preprocessing and not use_cache:
            PreprocessingCache.try_find_preprocessor.__code__ = preproc_orig_find
            PreprocessingCache.add_preprocessor.__code__ = preproc_orig_add

    _show_performance_plot(f'Cache performance with n_jobs={n_jobs}', timeouts, pipelines_count, times, plot_labels)


def compare_one_process_to_many(n_jobs: int = -1, test_preprocessing: bool = False):
    """
    Performs experiment to show how one-process FEDOT cacher compares to the multiprocessed
    """
    assert n_jobs != 1, 'This test uses multiprocessing, so you should have > 1 processors'
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    train_data = pd.read_csv(train_data_path)[:4000]
    test_data = pd.read_csv(test_data_path)[:4000]

    pipelines_count, times = [{1: [], n_jobs: []} for _ in range(2)]
    plot_labels = {1: 'one process', n_jobs: f'{n_jobs} processes'}
    composer_params = {'with_tuning': False}
    if test_preprocessing:
        composer_params.update({'cv_folds': None, 'pop_size': 10})
    base_fedot_params = {
        'problem': problem, 'seed': 42,
        'composer_params': composer_params, 'preset': 'fast_train',
        'verbose_level': 0
    }
    timeouts = [1, 2, 3, 4, 5]
    preproc_orig_find = PreprocessingCache.try_find_preprocessor.__code__
    preproc_orig_add = PreprocessingCache.add_preprocessor.__code__
    for _n_jobs in [1, n_jobs]:
        if test_preprocessing:
            PreprocessingCache.try_find_preprocessor.__code__ = PreprocessingCacheMock.try_find_preprocessor.__code__
            PreprocessingCache.add_preprocessor.__code__ = PreprocessingCacheMock.add_preprocessor.__code__
        print(f'Processes used: {_n_jobs}')
        basic_cache_usage = True if not test_preprocessing else False
        for timeout in timeouts:
            c_pipelines = 0.
            time = 0.
            mean_range = 3
            cache_effectiveness = collections.Counter()
            for i in range(mean_range):
                train_data_tmp = train_data.copy()
                test_data_tmp = test_data.copy()

                start_time = timeit.default_timer()
                auto_model = Fedot(**base_fedot_params, use_cache=basic_cache_usage, timeout=timeout, n_jobs=_n_jobs)
                auto_model.fit(features=train_data_tmp, target='target')
                auto_model.predict_proba(test_data_tmp)
                c_pipelines += _count_pipelines(auto_model.history)
                time += (timeit.default_timer() - start_time) / 60
                cache_effectiveness += auto_model.api_composer.cache.effectiveness_ratio

            time /= mean_range
            c_pipelines //= mean_range
            times[_n_jobs].append(time)
            pipelines_count[_n_jobs].append(c_pipelines)
            cache_effectiveness = {k: v / mean_range for k, v in cache_effectiveness.items()}

            print((
                f'\tTimeout: {timeout}'
                f', number of pipelines: {c_pipelines}, elapsed time: {time:.3f}'
                f', cache effectiveness: {cache_effectiveness}'
            ))
        if test_preprocessing:
            PreprocessingCache.try_find_preprocessor.__code__ = preproc_orig_find
            PreprocessingCache.add_preprocessor.__code__ = preproc_orig_add

    _show_performance_plot(f'Cache performance comparison between one process and {n_jobs}', timeouts, pipelines_count,
                           times, plot_labels)


if __name__ == "__main__":
    examples_dct = defaultdict(lambda: (lambda: print('Wrong example number option'),))
    examples_dct.update({
        1: (dummy_time_check,),
        2: (use_cache_check, 1, False),
        3: (compare_one_process_to_many, -1, False)
    })
    benchmark_number = 2
    func, *args = examples_dct[benchmark_number]
    func(*args)
