import operator
import timeit
from collections import defaultdict
from functools import reduce
from statistics import mean
from timeit import repeat
from typing import Optional

import pandas as pd
from matplotlib import cm, colors, pyplot as plt

from fedot.api.main import Fedot
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.repository.tasks import TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.unit.api.test_main_api import get_dataset


def _count_pipelines(opt_history: Optional[OptHistory]) -> int:
    if opt_history is not None:
        return reduce(operator.add, map(len, opt_history.individuals), 0)
    return 0


def _show_performance_plot(x: list, pipelines_count: dict, times: dict, plot_labels: dict):
    plt.title('Cache performance')
    plt.xlabel('timeout in minutes')
    plt.ylabel('correctly evaluated pipelines')

    c_norm = colors.Normalize(vmin=min(x) - 0.5, vmax=max(x) + 0.5)
    for arg in pipelines_count:
        plt.plot(x, pipelines_count[arg], label=plot_labels[arg], zorder=1)
        plt.scatter(x, pipelines_count[arg], c=times[arg],
                    cmap=cm.get_cmap('cool'), norm=c_norm, zorder=2)

    cb = plt.colorbar(cm.ScalarMappable(norm=c_norm, cmap=cm.get_cmap('cool')))
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

    test1 = {
        **composer_params
    }
    test2 = {
        **composer_params,
        'with_tuning': True
    }
    test3 = {
        **composer_params,
        'cv_folds': 4
    }
    test4 = {
        **composer_params,
        'cv_folds': 4,
        'with_tuning': True
    }
    for task_type in ['ts_forecasting']:  # , 'regression', 'classification']:
        for params, feature in [(test1,
                                 'basic')]:
            # , (test2, 'with_tuning'), (test3, 'with_cv_folds'), (test4, 'with_tuning_and_cv_folds')]:
            preset = 'best_quality'
            fedot_input = {'problem': task_type, 'seed': 42, 'preset': preset, 'verbose_level': -1,
                           'timeout': params['timeout'], 'use_cache': True,
                           'composer_params': params}
            if task_type == 'ts_forecasting':
                fedot_input['task_params'] = TsForecastingParams(forecast_length=30)
            train_data, test_data, _ = get_dataset(task_type)

            def check():
                Fedot(**fedot_input).fit(features=train_data, target='target')

            print(f"task_type={task_type}, feature={feature}, mean_time={mean(repeat(check, repeat=15, number=1))}")


def use_cache_check():
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
    base_fedot_params = {
        'problem': problem, 'seed': 42,
        'composer_params': {'with_tuning': False}, 'preset': 'fast_train',
        'verbose_level': -1
    }
    timeouts = [1, 2, 3, 4, 5]
    for use_cache in [False, True]:
        print(f'Using cache mode: {use_cache}')
        for timeout in timeouts:
            train_data_tmp = train_data.copy()
            test_data_tmp = test_data.copy()

            start_time = timeit.default_timer()
            auto_model = Fedot(**base_fedot_params, timeout=timeout, use_cache=use_cache)
            auto_model.fit(features=train_data_tmp, target='target')
            auto_model.predict_proba(features=test_data_tmp)
            times[use_cache].append((timeit.default_timer() - start_time) / 60)
            c_pipelines = _count_pipelines(auto_model.history)
            pipelines_count[use_cache].append(c_pipelines)

            cache_ef = str(auto_model.api_composer.cache.effectiveness_ratio) if auto_model.api_composer.cache else ''
            print((
                f'\tTimeout: {timeout}'
                f', number of pipelines: {c_pipelines}, elapsed time: {times[use_cache][-1]:.3f}'
                f'{", cache effectiveness: " + cache_ef}'
            ))

    _show_performance_plot(timeouts, pipelines_count, times, plot_labels)


def multiprocessing_check(n_jobs: int = -1):
    """
    Performs experiment to show how pipelines cacher works whilst multiprocessing is enabled
    """
    assert n_jobs != 1, 'This test uses multiprocessing, so you should have > 1 processors'
    train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
    test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

    problem = 'classification'

    train_data = pd.read_csv(train_data_path)[:4000]
    test_data = pd.read_csv(test_data_path)[:4000]

    pipelines_count, times = [{1: [], n_jobs: []} for _ in range(2)]
    plot_labels = {1: 'one process', n_jobs: f'{n_jobs} processes'}
    base_fedot_params = {
        'problem': problem, 'seed': 42,
        'composer_params': {'with_tuning': False}, 'preset': 'fast_train',
        'verbose_level': -1, 'use_cache': True
    }
    timeouts = [1, 2, 3, 4, 5]
    for _n_jobs in [1, n_jobs]:
        print(f'Processes used: {_n_jobs}')
        for timeout in timeouts:
            train_data_tmp = train_data.copy()
            test_data_tmp = test_data.copy()

            start_time = timeit.default_timer()
            auto_model = Fedot(**base_fedot_params, timeout=timeout, n_jobs=_n_jobs)
            auto_model.fit(features=train_data_tmp, target='target')
            auto_model.predict_proba(test_data_tmp)
            times[_n_jobs].append((timeit.default_timer() - start_time) / 60)
            c_pipelines = _count_pipelines(auto_model.history)
            pipelines_count[_n_jobs].append(c_pipelines)

            print((
                f'\tTimeout: {timeout}'
                f', number of pipelines: {c_pipelines}, elapsed time: {times[_n_jobs][-1]:.3f}'
                f', cache effectiveness: {auto_model.api_composer.cache.effectiveness_ratio}'
            ))

    _show_performance_plot(timeouts, pipelines_count, times, plot_labels)


if __name__ == "__main__":
    examples_dct = defaultdict(lambda: (lambda: print('Wrong example number option'),))
    examples_dct.update({
        1: (dummy_time_check,),
        2: (use_cache_check,),
        3: (multiprocessing_check, -1)
    })
    benchmark_number = 2
    func, *args = examples_dct[benchmark_number]
    func(*args)
