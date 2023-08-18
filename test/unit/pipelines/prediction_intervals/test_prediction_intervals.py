import pytest
import pickle
import numpy as np

from fedot.core.utils import fedot_project_root
from fedot.core.pipelines.prediction_intervals.main import PredictionIntervals
from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams
from fedot.core.pipelines.prediction_intervals.metrics import interval_score, picp


def synthetic_series(start, end):

    trend = np.array([5 * np.sin(x / 20) + 0.1 * x - 2 * np.sqrt(x) for x in range(start, end)])
    noise = np.random.normal(loc=0, scale=1, size=end - start)

    return trend + noise


@pytest.fixture
def params():

    input_name = f'{fedot_project_root()}/test/unit/data/pred_ints_train_input_test.pickle'
    model_name = f'{fedot_project_root()}/test/unit/data/pred_ints_model_test.pickle'
    with open(input_name, 'rb') as f:
        train_input = pickle.load(f)

    with open(model_name, 'rb') as f:
        model = pickle.load(f)

    ts_test = synthetic_series(start=200, end=220)

    return {'train_input': train_input, 'model': model, 'ts_test': ts_test}


def test_prediction_intervals(params):
    pred_ints_params = PredictionIntervalsParams(number_mutations=80, bpq_number_models='max', copy_model=True)
    for x in ['mutation_of_best_pipeline', 'best_pipelines_quantiles']:

        pred_ints = PredictionIntervals(model=params['model'], method=x, params=pred_ints_params)
        pred_ints.fit(params['train_input'])
        res = pred_ints.forecast()
        pred_ints.plot()

        int_score = interval_score(params['ts_test'], low=res['low_int'], up=res['up_int'])
        int_picp = picp(params['ts_test'], low=res['low_int'], up=res['up_int'])

        assert int_score <= 100, f"Too big interval_score of built prediction intervals for {x}."
        assert int_picp >= 0.3, f"Too small prediction interval coverage probability of prediction intervals for {x}."
