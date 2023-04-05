from datetime import datetime
from typing import Optional, Type

import numpy as np
import pandas as pd
import pytest

from fedot.api.api_utils.data_definition import PandasStrategy, NumpyStrategy, StrategyDefineData, FeaturesType
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

_DATE = '2000-01-01T10:00:00.100'
_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


def _array_to_input_data(features_array: np.ndarray,
                         target_array: np.ndarray,
                         idx: Optional[np.ndarray] = None,
                         task: Task = Task(TaskTypesEnum.classification),
                         data_type: Optional[DataTypesEnum] = None):
    return features_array


@pytest.mark.parametrize('strategy, features', [
    [PandasStrategy, pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), _DATE, 42., 42]
    ])],
    [PandasStrategy, pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), _DATE, 42., 42]
    ], dtype=object)],
    [PandasStrategy, pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 42]
    ])],
    [PandasStrategy, pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE)]
    ])],
    [PandasStrategy, pd.DataFrame(pd.date_range(_DATE, periods=3, freq='D'))],

    [NumpyStrategy, np.array([
        [_DATE, datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 54, 54.]
    ], dtype=object)],
    [NumpyStrategy, np.array([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 54]
    ], dtype=object)],
    [NumpyStrategy, np.array([
        [*pd.date_range(_DATE, periods=3, freq='D').to_numpy(), 54, 54.]
    ], dtype=object)],
    [NumpyStrategy, np.array([
        [*pd.date_range(_DATE, periods=3, freq='D').to_numpy()]
    ], dtype=np.datetime64)],
    [NumpyStrategy, pd.date_range(_DATE, periods=3, freq='D').to_numpy().reshape(-1, 1)]
])
def test_pandas_strategy(strategy: Type[StrategyDefineData], features: FeaturesType, monkeypatch):
    monkeypatch.setattr('fedot.api.api_utils.data_definition.array_to_input_data', _array_to_input_data)
    date_features = pd.DataFrame(features).infer_objects().select_dtypes('datetime')

    ## Test these
    defined_data: np.ndarray = strategy.define_data(None, features, Task(TaskTypesEnum.classification))
    assert 'int' in str(pd.Series(defined_data[:, date_features.columns].ravel()).infer_objects().dtype)
