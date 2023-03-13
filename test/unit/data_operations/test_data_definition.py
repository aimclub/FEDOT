import numpy as np
import pandas as pd
from fedot.api.api_utils.data_definition import PandasStrategy, NumpyStrategy
from fedot.core.repository.tasks import Task, TaskTypesEnum 
import pytest
from datetime import datetime

_DATE = '2000-01-01T10:00:00.100'
_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

@pytest.mark.parametrize('features', [
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), _DATE, 42., 42]
    ]),
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 42., 42]
    ]),
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE)]
    ]),
    pd.DataFrame(pd.date_range(_DATE, periods=3, freq='D'))
])
def test_pandas_strategy(features: pd.DataFrame):
    try:
        date_features_indexes = features.columns.get_indexer(features.select_dtypes('datetime').columns)
        defined_data = PandasStrategy.define_data(None, features, Task(TaskTypesEnum.classification))
        assert pd.Series(defined_data.features[:, date_features_indexes].ravel()).infer_objects().dtype == 'float64'
    except TypeError as te:
        assert False, te

## TODO: should it be considered anyway?
# def test_numpy_strategy():
#     features = np.array([
#         *pd.date_range("2018-01-01", periods=3, freq="D"),
#         'string'
#     ], dtype=object)
#     try:
#         defined_data = NumpyStrategy.define_data(None, features, Task(TaskTypesEnum.classification))
#         assert defined_data.features.dtype == 'float64'
#     except TypeError as te:
#         assert False, te