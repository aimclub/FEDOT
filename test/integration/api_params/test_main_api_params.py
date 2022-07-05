import logging
from dataclasses import dataclass
from typing import Callable, Union

import pytest

from fedot.api.main import Fedot
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.repository.tasks import TsForecastingParams
from test.unit.api.test_main_api import get_dataset


@dataclass
class TimeoutParams:
    test_input: dict
    test_answer: Union[Callable[[OptHistory], bool], ValueError]


TIMEOUT_CASES = [
    TimeoutParams(
        test_input={'timeout': -1, 'num_of_generations': 1},
        test_answer=lambda hist: len(hist.individuals) == 2
    ),
    TimeoutParams(
        test_input={'timeout': None, 'num_of_generations': 1},
        test_answer=lambda hist: len(hist.individuals) == 2
    ),
    TimeoutParams(
        test_input={'timeout': 0.1, 'num_of_generations': 15},
        test_answer=lambda hist: len(hist.individuals) < 15
    ),
    TimeoutParams(
        test_input={'timeout': -2, 'num_of_generations': 15},
        test_answer=ValueError()
    ),
    TimeoutParams(
        test_input={'timeout': -1, 'num_of_generations': 3},
        test_answer=lambda hist: len(hist.individuals) == 4
    )
]


@pytest.mark.parametrize('case', TIMEOUT_CASES)
def test_timeout(case: TimeoutParams):
    composer_params = {
        'max_depth': 1,
        'max_arity': 1,
        'pop_size': 1,
        'with_tuning': False,
        'validation_blocks': 1,
        'genetic_scheme': GeneticSchemeTypesEnum.generational,
        'num_of_generations': case.test_input['num_of_generations']
    }

    task_type = 'ts_forecasting'
    fedot_input = {'problem': task_type, 'seed': 42, 'preset': 'fast_train',
                   'logging_level': logging.DEBUG,
                   'timeout': case.test_input['timeout'],
                   'task_params': TsForecastingParams(forecast_length=1),
                   **composer_params}

    train_data, test_data, _ = get_dataset(task_type)
    if isinstance(case.test_answer, ValueError):
        with pytest.raises(ValueError):
            Fedot(**fedot_input)
    else:
        auto_model = Fedot(**fedot_input)
        auto_model.fit(features=train_data, target='target')
        history: OptHistory = auto_model.history
        assert case.test_answer(history)
