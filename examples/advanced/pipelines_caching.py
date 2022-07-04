import collections
import logging
import operator
import timeit
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from statistics import mean
from timeit import repeat
from typing import List, Optional

from matplotlib import colors, pyplot as plt

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.core.utils import fedot_project_root
from test.unit.api.test_main_api import get_dataset


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
            fedot_input = {'problem': task_type, 'seed': 42, 'preset': preset, 'verbose_level': logging.NOTSET,
                           'timeout': composer_params['timeout'],
                           'use_pipelines_cache': use_cache, 'use_preprocessing_cache': use_cache,
                           **composer_params}
            if task_type == 'ts_forecasting':
                fedot_input['task_params'] = TsForecastingParams(forecast_length=30)
            train_data, test_data, _ = get_dataset(task_type)

            def check():
                Fedot(**fedot_input).fit(features=train_data, target='target')

            print(f"task_type={task_type}, mean_time={mean(repeat(check, repeat=15, number=1))}")


def _run(timeouts: List[int], train_data: InputData, test_data: InputData, base_fedot_params: dict,
         use_pipelines_cache: bool = False, use_preprocessing_cache: bool = False):
    times = []
    pipelines_count = []
    for timeout in timeouts:
        c_pipelines = 0.
        time = 0.
        mean_range = 3
        cache_effectiveness = collections.Counter()
        for _ in range(mean_range):
            train_data_tmp = deepcopy(train_data)

            start_time = timeit.default_timer()
            auto_model = Fedot(**base_fedot_params, timeout=timeout, use_pipelines_cache=use_pipelines_cache,
                               use_preprocessing_cache=use_preprocessing_cache)
            auto_model.fit(features=train_data_tmp)
            time += (timeit.default_timer() - start_time) / 60
            c_pipelines += _count_pipelines(auto_model.history)
            if use_pipelines_cache and auto_model.api_composer.pipelines_cache.effectiveness_ratio:
                cache_effectiveness += auto_model.api_composer.pipelines_cache.effectiveness_ratio
            elif use_preprocessing_cache and auto_model.api_composer.preprocessing_cache.effectiveness_ratio:
                cache_effectiveness += auto_model.api_composer.preprocessing_cache.effectiveness_ratio

        time /= mean_range
        c_pipelines /= mean_range
        times.append(time)
        pipelines_count.append(c_pipelines)
        cache_effectiveness = {k: v / mean_range for k, v in cache_effectiveness.items()}

        print((
            f'\tTimeout: {timeout}'
            f', number of pipelines: {c_pipelines}, elapsed time: {time:.3f}'
            f', cache effectiveness: {cache_effectiveness}'
        ))
    return times, pipelines_count


def use_cache_check(problem: str, train_data, test_data, n_jobs: int = 1, test_preprocessing: bool = False):
    """
    Performs experiment to show how caching pipelines operations helps in fitting FEDOT model
    """
    pipelines_count, times = [{False: [], True: []} for _ in range(2)]
    plot_labels = {False: 'without cache', True: 'with cache'}
    composer_params = {'with_tuning': False}
    preset = 'fast_train'
    base_fedot_params = {
        'problem': problem, 'seed': 42,
        'preset': preset, 'verbose_level': logging.NOTSET, 'n_jobs': n_jobs,
        **composer_params
    }
    timeouts = [1, 2, 3, 4, 5]
    for use_cache in [True, False]:
        print(f'Using cache: {use_cache}')
        use_pipelines_cache = use_cache if not test_preprocessing else False
        use_preprocessing_cache = False if not test_preprocessing else use_cache
        _times, _pipelines_count = _run(timeouts, train_data, test_data, base_fedot_params, use_pipelines_cache,
                                        use_preprocessing_cache)
        times[use_cache] = _times
        pipelines_count[use_cache] = _pipelines_count
    _show_performance_plot(f'Cache performance with n_jobs={n_jobs}', timeouts, pipelines_count, times, plot_labels)


def compare_one_process_to_many(problem: str, train_data, test_data, n_jobs: int = -1,
                                test_preprocessing: bool = False):
    """
    Performs experiment to show how one-process FEDOT cacher compares to the multiprocessed
    """
    assert n_jobs != 1, 'This test uses multiprocessing, so you should have > 1 processors'
    pipelines_count, times = [{1: [], n_jobs: []} for _ in range(2)]
    plot_labels = {1: 'one process', n_jobs: f'{n_jobs} processes'}
    composer_params = {'with_tuning': False}
    base_fedot_params = {
        'problem': problem, 'seed': 42,
        'preset': 'fast_train', 'verbose_level': logging.NOTSET,
        **composer_params
    }
    timeouts = [1, 2, 3, 4, 5]
    for _n_jobs in [1, n_jobs]:
        print(f'Processes used: {_n_jobs}')
        use_pipelines_cache = True if not test_preprocessing else False
        use_preprocessing_cache = False if not test_preprocessing else True
        base_fedot_params['n_jobs'] = _n_jobs
        _times, _pipelines_count = _run(timeouts, train_data, test_data, base_fedot_params, use_pipelines_cache,
                                        use_preprocessing_cache)
        times[_n_jobs] = _times
        pipelines_count[_n_jobs] = _pipelines_count
    _show_performance_plot(f'Cache performance comparison between one process and {n_jobs}', timeouts, pipelines_count,
                           times, plot_labels)


if __name__ == "__main__":
    problem = 'classification'
    if problem == 'classification':
        train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
        test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'
        train_data = InputData.from_csv(train_data_path,
                                        task=Task(TaskTypesEnum.classification)).subset_range(0, 4000)
        test_data = InputData.from_csv(test_data_path,
                                       task=Task(TaskTypesEnum.classification)).subset_range(0, 4000)
    elif problem == 'regression':
        data_path = f'{fedot_project_root()}/cases/data/cholesterol/cholesterol.csv'
        data = InputData.from_csv(data_path,
                                  task=Task(TaskTypesEnum.regression))
        train_data, test_data = train_test_data_setup(data)

    examples_dct = defaultdict(lambda: (lambda: print('Wrong example number option'),))
    examples_dct.update({
        1: (dummy_time_check,),
        2: (use_cache_check, problem, train_data, test_data, 1, True),
        3: (compare_one_process_to_many, problem, train_data, test_data, -1, True)
    })
    benchmark_number = 2
    func, *args = examples_dct[benchmark_number]

    func(*args)
