import numpy as np
import pytest
from fedot.core.data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from fedot_ind.core.models.ts_forecasting.ssa_forecaster import SSAForecasterImplementation

params = OperationParameters(window_size_method="hac",
                             history_lookback=15,
                             mode='one_dimensional')


@pytest.fixture(scope='session')
def time_series_data():
    ts = np.random.rand(100)
    input_data = InputData(idx=np.arange(0, len(ts)),
                           features=ts.reshape(-1, 1),
                           target=ts,
                           task=Task(TaskTypesEnum.ts_forecasting),
                           data_type=DataTypesEnum.ts,
                           )
    return input_data


def test_predict_for_fit(time_series_data):
    forecaster = SSAForecasterImplementation({'mode': 'one_dimensional'})
    forecaster.horizon = 10
    input_data = time_series_data
    forecast = forecaster.predict_for_fit(input_data)

    assert forecast is not None
    assert forecast.shape[1] == 100
