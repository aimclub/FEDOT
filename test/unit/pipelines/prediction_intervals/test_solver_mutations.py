import platform

import pytest
import pickle
import numpy as np

from golem.core.log import default_log, Log
from fedot.core.utils import fedot_project_root
from fedot.core.data.data import InputData
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams
from fedot.core.pipelines.prediction_intervals.solvers.mutation_of_best_pipeline import solver_mutation_of_best_pipeline
from fedot.core.pipelines.prediction_intervals.utils import get_last_generations

import pathlib

plt = platform.system()
if plt == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath


@pytest.fixture
def params():
    with open(f'{fedot_project_root()}'
              f'/test/unit/pipelines/prediction_intervals/data/pred_ints_model_test.pickle', 'rb') as f:
        model = pickle.load(f)
    ts_train = np.genfromtxt(f'{fedot_project_root()}/test/unit/pipelines/prediction_intervals/data/train_ts.csv')
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=20))
    idx = np.arange(len(ts_train))
    train_input = InputData(idx=idx,
                            features=ts_train,
                            target=ts_train,
                            task=task,
                            data_type=DataTypesEnum.ts)

    Log().reset_logging_level(10)
    logger = default_log(prefix='Testing solver_mutation_of_best_pipeline')
    forecast = model.forecast()
    ind = get_last_generations(model)['final_choice']
    return {'train_input': train_input,
            'ind': ind,
            'forecast': forecast,
            'logger': logger,
            'operations': PredictionIntervalsParams().mutations_operations}


def test_solver_mutation_of_best_pipeline(params):
    params_default = {'choice': 'different', 'discard': True, 'percentage': 0.66, 'number_mutations': 10,
                      'message': 'default solver_mutation_of_best_pipeline failed.'}
    params_with_replacement = {'choice': 'with_replacement', 'discard': True, 'percentage': 0.5, 'number_mutations': 30,
                               'message': 'solver_mutation_of_best_pipeline with inapropriate pipelines failed.'}
    params_different = {'choice': 'different', 'discard': False, 'percentage': 0.8, 'number_mutations': 10,
                        'mesage': 'solver_mutation_of_best_pipeline failed.'}

    for x in [params_default, params_with_replacement, params_different]:
        res = solver_mutation_of_best_pipeline(train_input=params['train_input'],
                                               ind=params['ind'],
                                               horizon=20,
                                               forecast=params['forecast'],
                                               logger=params['logger'],
                                               number_mutations=x['number_mutations'],
                                               operations=params['operations'],
                                               n_jobs=-1,
                                               show_progress=False,
                                               mutations_choice=x['choice'],
                                               discard_inapropriate_pipelines=x['discard'],
                                               keep_percentage=x['percentage'])

        if x == params_default:
            number_predictions = len(res)
            assert number_predictions <= 10 * x['percentage'] + 1, f"{x['message']} Bad pipelines are not deleted."

        elif x in [params_with_replacement, params_different]:
            prediction_length = len(res[0])
            for y in res:
                assert type(y) == np.ndarray, f"{x['message']} Wrong output of a mutation."
                assert len(y) == prediction_length, f"{x['message']} Wrong prediction length."
