from datetime import datetime
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
import pytest

import fedot.api.api_utils.data_definition as fedot_api_api_utils_data_definition
from fedot.api.api_utils.data_definition import PandasStrategy, TupleStrategy, NumpyStrategy, StrategyDefineData
from fedot.core.data.data import InputData
from fedot.core.data.data import np_datetime_to_numeric
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

_DATE = '2000-01-01T10:00:00.100'
_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


@pytest.mark.parametrize('features', [
    np.array([
        [_DATE, datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 54, 54.]
    ]),
    np.array([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 42]
    ], dtype=object),
    np.array([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 54., 54]
    ], dtype=object),
    np.array([
        [*pd.date_range(_DATE, periods=3, freq='D').to_numpy(), 54, 54.]
    ], dtype=object),
    np.array([
        [*pd.date_range(_DATE, periods=3, freq='D')]
    ], dtype=np.datetime64),
    pd.date_range(_DATE, periods=3, freq='D').to_numpy(),
    np.array([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE)]
    ]),
    np.array([
        ['without_datetime', 54, 54.]
    ], dtype=object)
])
def test_datetime_erasure(features: np.ndarray):
    result = np_datetime_to_numeric(features)
    assert 'datetime' not in str(pd.DataFrame(result).infer_objects().dtypes)


def _array_to_input_data(features_array: np.ndarray,
                         target_array: np.ndarray,
                         idx: Optional[np.ndarray] = None,
                         task: Task = Task(TaskTypesEnum.classification),
                         data_type: Optional[DataTypesEnum] = None) -> InputData:
    return np.asarray(features_array), np.asarray(target_array)


@pytest.mark.parametrize('strategy, features, task, target, expected', [
    # None target
    (NumpyStrategy, np.array([[1]]), Task(TaskTypesEnum.regression), None, (np.array([[1]]), np.array([]))),
    (PandasStrategy, pd.DataFrame([[1]]), Task(TaskTypesEnum.regression), None,
     (np.array([[1]]), np.array([]))),

    # Time-series target
    (NumpyStrategy, np.array([[1]]), Task(TaskTypesEnum.ts_forecasting), None, (np.array([[1]]), np.array([1]))),
    (NumpyStrategy, np.array([[1]]), Task(TaskTypesEnum.ts_forecasting), np.array([2]),
     (np.array([[1]]), np.array([[1]]))),

    # Index target
    (NumpyStrategy, np.array([[1, 2]]), Task(TaskTypesEnum.regression), 1, (np.array([[1]]), np.array([2]))),
    # Index target, features are already splitted
    (NumpyStrategy, np.array([[1]]), Task(TaskTypesEnum.regression), 1, (np.array([[1]]), np.array([]))),

    # Str target
    (PandasStrategy, pd.DataFrame([[1, 2]], columns=['0', '1']), Task(TaskTypesEnum.regression), '1',
     (np.array([[1]]), np.array([2]))),
    # Str target, features are already splitted
    (PandasStrategy, pd.DataFrame([[1]], columns=['0']), Task(TaskTypesEnum.regression), '1',
     (np.array([[1]]), np.array([]))),

    # Array target
    (NumpyStrategy, np.array([[1, 2]]), Task(TaskTypesEnum.regression), np.array([0, 1]),
     (np.array([[1, 2]]), np.array([0, 1]))),
    (PandasStrategy, pd.DataFrame([[1, 2]]), Task(TaskTypesEnum.regression), pd.Series([0, 1]),
     (np.array([[1, 2]]), np.array([0, 1]))),

    # Tuple
    (TupleStrategy, ([1], [2]), Task(TaskTypesEnum.regression), None, ([1], [2]))
])
def test_data_strategies(strategy: StrategyDefineData, features: Union[np.ndarray, pd.DataFrame, Tuple],
                         task: Task, target: Union[None, np.ndarray], expected: Tuple[np.ndarray, np.ndarray],
                         monkeypatch):
    monkeypatch.setattr(fedot_api_api_utils_data_definition, 'array_to_input_data', _array_to_input_data)

    obtained_features, obtained_target = strategy().define_data(features=features, task=task, target=target)
    expected_features, expected_target = expected

    assert np.allclose(obtained_features, expected_features)
    assert np.allclose(obtained_target, expected_target)
