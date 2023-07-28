import numpy as np
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data_split import train_test_data_setup


def get_data(data_length=500, test_length=100):
    garmonics = [(0.1, 0.9), (0.1, 1), (0.1, 1.1), (0.05, 2), (0.05, 5), (1, 0.02)]
    time = np.linspace(0, 100, data_length)
    data = time * 0
    for g in garmonics:
        data += g[0] * np.sin(g[1] * 2 * np.pi / time[-1] * 25 * time)

    data = InputData(idx=np.arange(0, data.shape[0]),
                     features=data,
                     target=data,
                     task=Task(TaskTypesEnum.ts_forecasting,
                               TsForecastingParams(forecast_length=test_length)),
                     data_type=DataTypesEnum.ts)
    return train_test_data_setup(data, split_ratio=(data_length - test_length) /
                                 ((data_length - test_length) + test_length))


def test_result_reproducing():
    train, test = get_data()
    old_fedot = None
    # try in cycle because some problems are random
    for _ in range(4):
        fedot = Fedot(problem='ts_forecasting',
                      task_params=TsForecastingParams(forecast_length=test.idx.shape[0]),
                      seed=0,
                      timeout=None,
                      pop_size=50,
                      num_of_generations=5,
                      )
        fedot.fit(train)

        if old_fedot is not None:
            assert np.allclose(fedot.history.all_historical_fitness, old_fedot.history.all_historical_fitness)
            assert np.isclose(np.sum(np.abs(fedot.forecast(test) - old_fedot.forecast(test))), 0)

        old_fedot = fedot
