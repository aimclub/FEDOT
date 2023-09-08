import pickle

import numpy as np
import pytest

from fedot.core.data.data import InputData
from fedot.core.pipelines.prediction_intervals.main import PredictionIntervals
from fedot.core.pipelines.prediction_intervals.metrics import interval_score, picp
from fedot.core.pipelines.prediction_intervals.params import PredictionIntervalsParams
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TsForecastingParams, Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root


@pytest.fixture
def params():

    with open(f'{fedot_project_root()}'
              f'/test/unit/pipelines/prediction_intervals/data/pred_ints_model_test.pickle', 'rb') as f:
        model = pickle.load(f)
    ts_train = np.genfromtxt(f'{fedot_project_root()}/test/unit/pipelines/prediction_intervals/data/train_ts.csv')
    ts_test = np.genfromtxt(f'{fedot_project_root()}/test/unit/pipelines/prediction_intervals/data/test_ts.csv')
    task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=20))
    idx = np.arange(len(ts_train))
    train_input = InputData(idx=idx,
                            features=ts_train,
                            target=ts_train,
                            task=task,
                            data_type=DataTypesEnum.ts)

    return {'train_input': train_input, 'model': model, 'ts_test': ts_test}


def test_prediction_intervals(params):
    pred_ints_params = PredictionIntervalsParams(number_mutations=80, bpq_number_models='max', show_progress=False)
    for x in ['mutation_of_best_pipeline', 'best_pipelines_quantiles']:

        pred_ints = PredictionIntervals(model=params['model'], method=x, horizon=20, params=pred_ints_params)
        pred_ints.fit(params['train_input'])
        res = pred_ints.forecast()
        pred_ints.plot()

        int_score = interval_score(params['ts_test'], low=res['low_int'], up=res['up_int'])
        int_picp = picp(params['ts_test'], low=res['low_int'], up=res['up_int'])

        assert int_score <= 100, f"Too big interval_score of prediction intervals for {x}."
        assert int_picp >= 0.3, f"Too small prediction interval coverage probability of prediction intervals for {x}."
