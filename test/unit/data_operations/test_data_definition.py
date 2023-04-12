from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from fedot.api.api_utils.data_definition import FeaturesType
from fedot.core.data.data import data_with_datetime_to_numeric_np

_DATE = '2000-01-01T10:00:00.100'
_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'


@pytest.mark.parametrize('features', [
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), _DATE, 42., 42]
    ]),
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), _DATE, 42., 42]
    ], dtype=object),
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 42]
    ]),
    pd.DataFrame([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE)]
    ]),
    pd.DataFrame(pd.date_range(_DATE, periods=3, freq='D')),

    np.array([
        [_DATE, datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 54, 54.]
    ], dtype=object),
    np.array([
        [datetime.strptime(_DATE, _DATE_FORMAT), np.datetime64(_DATE), pd.Timestamp(_DATE), 54]
    ], dtype=object),
    np.array([
        [*pd.date_range(_DATE, periods=3, freq='D').to_numpy(), 54, 54.]
    ], dtype=object),
    np.array([
        [*pd.date_range(_DATE, periods=3, freq='D').to_numpy()]
    ], dtype=np.datetime64),
    pd.date_range(_DATE, periods=3, freq='D').to_numpy()
])
def test_datetime_erasure(features: FeaturesType):
    result = data_with_datetime_to_numeric_np(features)
    assert 'datetime' not in str(pd.DataFrame(result).infer_objects().dtypes)
