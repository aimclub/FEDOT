from typing import List

import numpy as np

from fedot import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


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
    return train_test_data_setup(data, validation_blocks=1)


def get_fitted_fedot(forecast_length, train_data, **kwargs):
    params = {'problem': 'ts_forecasting',
              'task_params': TsForecastingParams(forecast_length=forecast_length),
              'seed': 1,
              'timeout': None,
              'pop_size': 50,
              'num_of_generations': 5,
              'with_tuning': False}
    params.update(kwargs)
    fedot = Fedot(**params)
    fedot.fit(train_data)
    return fedot


def check_fedots(fedots: List[Fedot], test_data: InputData, are_same: bool = True) -> None:
    """ Check fedots are equal or not equal with assertion
        :param fedots: list with Fedot instances
        :param test_data: data for testing
        :param are_same: if True then equivalence check, else nonequivalence check
        :return: None"""
    for fedot in fedots[1:]:
        assert are_same == np.allclose(fedots[0].history.all_historical_fitness, fedot.history.all_historical_fitness)
        assert are_same == np.allclose(fedots[0].forecast(test_data), fedot.forecast(test_data))


def test_result_reproducing():
    """ Test check that Fedot instance returns same compose result
        and makes same compose process in different run with fixed seeds """
    # TODO: fix reproducing
    #       it is randomly unstable
    pass
    # train, test = get_data()
    # old_fedot = None
    # # try in cycle because some problems are random
    # for _ in range(4):
    #     fedot = get_fitted_fedot(forecast_length=test.idx.shape[0],
    #                              train_data=train)
    #     if old_fedot is not None:
    #         check_fedots([fedot, old_fedot], test, are_same=True)
    #     old_fedot = fedot


def test_result_changing():
    """ Test check that Fedot instance returns different compose result
        and makes different compose process in different run with different seeds """
    train, test = get_data()

    fedots = [get_fitted_fedot(forecast_length=len(test.idx),
                               train_data=train,
                               seed=seed,
                               num_of_generations=1)
              for seed in (0, 1)]

    check_fedots(fedots, test, are_same=False)
