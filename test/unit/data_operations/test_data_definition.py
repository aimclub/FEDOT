from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fedot.core.data.data import np_datetime_to_numeric

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
