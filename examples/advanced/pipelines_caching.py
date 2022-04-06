import operator
from functools import reduce

from fedot.core.optimisers.opt_history import OptHistory


def _count_pipelines(opt_history: OptHistory):
    return reduce(
        operator.add,
        map(len, opt_history.individuals),
        0
    )


def dummy_time_check():
    from statistics import mean
    from test.unit.api.test_main_api import get_dataset
    from timeit import repeat

    from fedot.api.main import Fedot
    from fedot.core.repository.tasks import TsForecastingParams

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
                           'timeout': params['timeout'],
                           'composer_params': params}
            if task_type == 'ts_forecasting':
                fedot_input['task_params'] = TsForecastingParams(forecast_length=30)
            train_data, test_data, _ = get_dataset(task_type)

            def check():
                Fedot(**fedot_input).fit(features=train_data, target='target')

            print(f"task_type={task_type}, feature={feature}, mean_time={mean(repeat(check, repeat=15, number=1))}")


def correct_pipelines_cnt_check():
    import timeit

    from typing import Optional

    import pandas as pd

    from fedot.api.main import Fedot
    from fedot.core.composer.cache import OperationsCache
    from fedot.core.pipelines.pipeline import Pipeline
    from fedot.core.utils import fedot_project_root
    from matplotlib import cm, colors, pyplot as plt

    def test_caching(timeout: float = 2., partitions_n=2):
        """
        Performs experiment to show how caching pipelines operations helps in fitting FEDOT model

        :param timeout: timeout for optimization in minutes
        :param partitions_n: on how many folds you want. f.e. if dataset contains 20000 rows, partitions_n=5 will create
            such folds: [4000 rows, 8000 rows, 12000 rows, 16000 rows, 20000 rows]
        """
        train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
        test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

        problem = 'classification'

        train_data = pd.read_csv(train_data_path)
        test_data = pd.read_csv(test_data_path)

        data_len = len(train_data)

        partitions = []
        for i in range(1, partitions_n + 1):
            partitions.append(int(data_len * (i / partitions_n)))

        pipelines_count, times = [{0: [], 1: []} for _ in range(2)]

        def fit_from_cache_mock(self, cache: OperationsCache, fold_num: Optional[int] = None):
            return False

        pipeline_fit_from_cache_orig = Pipeline.fit_from_cache.__code__
        for enable_caching in [0, 1]:
            if not enable_caching:
                Pipeline.fit_from_cache.__code__ = fit_from_cache_mock.__code__
            print(f'Using cache mode: {bool(enable_caching)}')
            for partition in partitions:
                train_data_tmp = train_data.iloc[:partition].copy()
                test_data_tmp = test_data.iloc[:partition].copy()

                start_time = timeit.default_timer()
                auto_model = Fedot(problem=problem, seed=42, timeout=timeout,
                                   composer_params={'with_tuning': False}, preset='fast_train',
                                   verbose_level=-1)
                auto_model.fit(features=train_data_tmp, target='target')
                auto_model.predict_proba(features=test_data_tmp)
                times[enable_caching].append((timeit.default_timer() - start_time) / 60)
                c_pipelines = _count_pipelines(auto_model.history) if auto_model.history else None
                pipelines_count[enable_caching].append(c_pipelines)

                print((
                    f'\tDataset length: {partition}'
                    f', number of pipelines: {c_pipelines}, elapsed time: {times[enable_caching][-1]:.3f}'
                    f', cache effectiveness: {auto_model.api_composer.cache.effectiveness}'
                ))
            if not enable_caching:
                Pipeline.fit_from_cache.__code__ = pipeline_fit_from_cache_orig

        plt.title('Cache performance')
        plt.xlabel('rows in train dataset')
        plt.ylabel('Num of pipelines that were evaluated correctly')
        c_norm = colors.Normalize(vmin=timeout - timeout / 2, vmax=timeout + timeout / 2)

        plt.plot(partitions, pipelines_count[1], label='with caching', zorder=1)
        plt.scatter(partitions, pipelines_count[1], c=times[1],
                    cmap=cm.get_cmap('cool'), norm=c_norm, zorder=2)

        plt.plot(partitions, pipelines_count[0], label=f'without caching', zorder=1)
        plt.scatter(partitions, pipelines_count[0], c=times[0],
                    cmap=cm.get_cmap('cool'), norm=c_norm, zorder=2)
        cb = plt.colorbar(cm.ScalarMappable(norm=c_norm, cmap=cm.get_cmap('cool')))
        cb.ax.set_ylabel('time for optimization in minutes', rotation=90)
        plt.legend()
        plt.grid()
        plt.show()

    test_caching(2, 2)


def multiprocessing_check():
    import timeit

    import pandas as pd

    from fedot.api.main import Fedot
    from fedot.core.utils import fedot_project_root
    from matplotlib import cm, pyplot as plt

    def test_caching(n_jobs=-1):
        """
        Performs experiment to show how pipelines cacher works whilst multiprocessing is enabled
        """
        train_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_train.csv'
        test_data_path = f'{fedot_project_root()}/cases/data/scoring/scoring_test.csv'

        problem = 'classification'

        train_data = pd.read_csv(train_data_path)[:8000]
        test_data = pd.read_csv(test_data_path)[:8000]

        pipelines_count, times = [{1: [], n_jobs: []} for _ in range(2)]
        base_fedot_params = {
            'problem': problem, 'seed': 42, 'composer_params': {'with_tuning': False}, 'preset': 'fast_train',
            'verbose_level': -1
        }
        timeouts = [1, 2, 3, 4, 5]
        for _n_jobs in [1, n_jobs]:
            print(f'Processes used: {_n_jobs}')
            for timeout in timeouts:
                train_data_tmp = train_data.copy()
                test_data_tmp = test_data.copy()

                start_time = timeit.default_timer()
                auto_model = Fedot(**base_fedot_params, n_jobs=_n_jobs, timeout=timeout)
                auto_model.fit(features=train_data_tmp, target='target')
                auto_model.predict_proba(test_data_tmp)
                times[_n_jobs].append((timeit.default_timer() - start_time) / 60)
                c_pipelines = _count_pipelines(auto_model.history) if auto_model.history else 0
                pipelines_count[_n_jobs].append(c_pipelines)

                print((
                    f'\tTimeout: {timeout}'
                    f', number of pipelines: {c_pipelines}, elapsed time: {times[_n_jobs][-1]:.3f}'
                    f', cache effectiveness: {auto_model.api_composer.cache.effectiveness}'
                ))
                # manual cache cleanup, TODO: make it more laconic
                auto_model.api_composer.cache.reset()

        plt.title('Cache performance')
        plt.xlabel('rows in train dataset')
        plt.ylabel('Num of pipelines that were evaluated correctly')

        plt.plot(timeouts, pipelines_count[1], label='with caching', zorder=1)
        plt.scatter(timeouts, pipelines_count[1], c=times[1],
                    cmap=cm.get_cmap('cool'), zorder=2)

        plt.plot(timeouts, pipelines_count[_n_jobs], label=f'without caching', zorder=1)
        plt.scatter(timeouts, pipelines_count[_n_jobs], c=times[_n_jobs],
                    cmap=cm.get_cmap('cool'), zorder=2)
        cb = plt.colorbar(cm.ScalarMappable(cmap=cm.get_cmap('cool')))
        cb.ax.set_ylabel('time for optimization in minutes', rotation=90)
        plt.legend()
        plt.grid()
        plt.show()

    test_caching()


if __name__ == "__main__":
    from collections import defaultdict

    checks_dct = defaultdict(lambda: (lambda: print('Wrong option')))
    checks_dct.update({
        '1': dummy_time_check,
        '2': correct_pipelines_cnt_check,
        '3': multiprocessing_check
    })
    choice_msg = (
        'Type in benchmark option number:\n'
        '[1] dummy fitting time check\n'
        '[2] correct fitted pipelines number check\n'
        '[3] caching whilst multiprocessing check\n'
    )
    checks_dct[input(choice_msg)]()
