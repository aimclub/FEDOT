import pytest

from fedot.api.main import Fedot
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.repository.tasks import TsForecastingParams
from .test_main_api import get_dataset
from .dataclasses.api_params_dataclasses import TimeoutParams

TIMEOUT_CASES = [
    TimeoutParams(
        test_input={'timeout': -1, 'num_of_generations': 1},
        test_answer=lambda hist: len(hist.individuals) == 1
    ),
    TimeoutParams(
        test_input={'timeout': None, 'num_of_generations': 1},
        test_answer=lambda hist: len(hist.individuals) == 1
    ),
    TimeoutParams(
        test_input={'timeout': 1, 'num_of_generations': 15},
        test_answer=lambda hist: len(hist.individuals) < 15
    ),
    TimeoutParams(
        test_input={'timeout': -2, 'num_of_generations': 15},
        test_answer=ValueError()
    ),
    TimeoutParams(
        test_input={'timeout': -1, 'num_of_generations': 4},
        test_answer=lambda hist: len(hist.individuals) == 4
    )
]

@pytest.mark.parametrize('case', TIMEOUT_CASES)
def test_timeout(case: TimeoutParams):
    composer_params = {
        'max_depth': 1,
        'max_arity': 1,
        **case.test_input
    }

    task_type = 'ts_forecasting'
    preset = 'fast_train'

    train_data, test_data, _ = get_dataset(task_type)
    if isinstance(case.test_answer, ValueError):
        with pytest.raises(ValueError):
            Fedot(problem=task_type, seed=42, preset=preset, verbose_level=4,
                       timeout=composer_params['timeout'],
                       composer_params=composer_params, task_params=TsForecastingParams(forecast_length=1))
    else:
        auto_model = Fedot(problem=task_type, seed=42, preset=preset, verbose_level=4,
                        timeout=composer_params['timeout'],
                        composer_params=composer_params, task_params=TsForecastingParams(forecast_length=1))
        auto_model.fit(features=train_data, target='target')
        history: OptHistory = auto_model.history

        print(len(history.individuals), history.individuals)

        assert case.test_answer(history)
