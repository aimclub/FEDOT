import numpy as np
import pytest

from core.models.data import InputData
from core.models.transformation import direct, ts_lagged_3d_to_ts, ts_to_lagged_3d, ts_to_lagged_table
from core.repository.dataset_types import DataTypesEnum
from core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

max_window_size = 4
forecast_length = 2


def synthetic_forecasting_problem(forecast_length, max_window_size):
    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=forecast_length,
                                    max_window_size=max_window_size))
    ts_len = 10
    ts = np.asarray(range(ts_len))

    exog_variable = np.asarray(range(ts_len))

    ts_data = InputData(idx=range(len(ts)), features=exog_variable, target=ts,
                        task=task, data_type=DataTypesEnum.table)

    target_3d = np.asarray([[[2], [3], [4], [5]],
                            [[3], [4], [5], [6]],
                            [[4], [5], [6], [7]],
                            [[5], [6], [7], [8]],
                            [[6], [7], [8], [9]]])

    target_lagged = [[4], [5], [5], [6], [6], [7], [7], [8], [8], [9]]

    ts_data_3d = InputData(idx=range(len(ts)), features=exog_variable, target=target_3d,
                           task=task, data_type=DataTypesEnum.ts_lagged_3d)

    ts_data_lagged = InputData(idx=range(len(ts)), features=exog_variable, target=target_lagged,
                               task=task, data_type=DataTypesEnum.ts_lagged_table)

    return task, ts_len, ts_data, ts_data_3d, ts_data_lagged


# TODO add features testing
def test_ts_to_lagged_3d():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_to_lagged_3d(ts_data)

    print(transformed_data.target.shape)
    assert np.equal(transformed_data.target.shape,
                    [ts_len - task.task_params.max_window_size - forecast_length + 1,
                     max_window_size,
                     1]).all()
    assert np.equal(transformed_data.target, ts_data_3d.target).all()
    print(transformed_data.features.shape)


@pytest.mark.skip(reason="synthetic features should be added")
def test_lagged_3d_to_ts():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_lagged_3d_to_ts(ts_data_3d)

    print(transformed_data.target.shape)
    assert np.equal(transformed_data.target.shape,
                    [ts_len - task.task_params.max_window_size - forecast_length + 1,
                     max_window_size,
                     1]).all()
    assert np.equal(transformed_data.target, ts_data.target).all()
    print(transformed_data.features.shape)


# TODO add features testing
def test_ts_to_lagged_table():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = ts_to_lagged_table(ts_data)

    print(transformed_data.target.shape)
    assert np.equal(transformed_data.target.shape,
                    [ts_len, 1]).all()
    assert np.equal(transformed_data.target, ts_data_lagged.target).all()
    print(transformed_data.features.shape)


def test_direct():
    task, ts_len, ts_data, ts_data_3d, ts_data_lagged = \
        synthetic_forecasting_problem(forecast_length=forecast_length,
                                      max_window_size=max_window_size)

    transformed_data = direct(ts_data)

    print(transformed_data.target.shape)
    assert np.equal(transformed_data.target.shape,
                    [ts_len]).all()
    assert np.equal(transformed_data.target, ts_data.target).all()
    print(transformed_data.features.shape)
