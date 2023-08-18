import pytest
import pickle
import numpy as np

from golem.core.log import default_log, Log
from fedot.core.utils import fedot_project_root
from fedot.core.pipelines.prediction_intervals.solvers.mutation_of_best_pipeline import solver_mutation_of_best_pipeline


@pytest.fixture
def params():

    input_name = f'{fedot_project_root()}/test/unit/data/pred_ints_train_input_test.pickle'
    model_name = f'{fedot_project_root()}/test/unit/data/pred_ints_model_test.pickle'

    with open(input_name, 'rb') as f:
        train_input = pickle.load(f)

    with open(model_name, 'rb') as f:
        model = pickle.load(f)

    Log().reset_logging_level(10)
    logger = default_log(prefix='Testing solver_mutation_of_best_pipeline')
    horizon = model.params.task.task_params.forecast_length
    forecast = model.forecast()

    return {'train_input': train_input,
            'model': model,
            'forecast': forecast,
            'horizon': horizon,
            'logger': logger}


def test_solver_mutation_of_best_pipeline(params):
    params_default = {'choice': 'different', 'discard': True, 'percentage': 0.66, 'number_mutations': 10,
                      'message': 'default solver_mutation_of_best_pipeline failed.'}
    params_with_replacement = {'choice': 'with_replacement', 'discard': True, 'percentage': 0.5, 'number_mutations': 30,
                               'message': 'solver_mutation_of_best_pipeline with inapropriate pipelines failed.'}
    params_different = {'choice': 'different', 'discard': False, 'percentage': 0.8, 'number_mutations': 10,
                        'mesage': 'solver_mutation_of_best_pipeline failed.'}

    for x in [params_default, params_with_replacement, params_different]:
        res = solver_mutation_of_best_pipeline(train_input=params['train_input'],
                                     model=params['model'],
                                     horizon=params['horizon'],
                                     forecast=params['forecast'],
                                     logger=params['logger'],
                                     number_mutations=x['number_mutations'],
                                     n_jobs=-1,
                                     show_progress=True,
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
