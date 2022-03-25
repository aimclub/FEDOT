from statistics import mean
from test.unit.api.test_main_api import get_dataset
from timeit import repeat

from fedot.api.main import Fedot
from fedot.core.repository.tasks import TsForecastingParams


def check_caching():
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
                                 'basic')]:  # , (test2, 'with_tuning'), (test3, 'with_cv_folds'), (test4, 'with_tuning_and_cv_folds')]:
            preset = 'best_quality'
            fedot_input = {'problem': task_type, 'seed': 42, 'preset': preset, 'verbose_level': -1,
                           'timeout': params['timeout'],
                           'composer_params': params}
            if task_type == 'ts_forecasting':
                fedot_input['task_params'] = TsForecastingParams(forecast_length=30)
            train_data, test_data, _ = get_dataset(task_type)

            def check():
                Fedot(**fedot_input).fit(features=train_data, target='target')

            print(f"{task_type=}, {feature=}, mean_time={mean(repeat(check, repeat=15, number=1))}")


if __name__ == "__main__":
    check_caching()
