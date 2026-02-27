import pytest

import numpy as np
import pandas as pd

from fedot_ind.core.operation.dummy.dummy_operation import init_input_data


@pytest.fixture
def sample_univariate():
    rows, cols = 100, 50
    X = pd.DataFrame(np.random.random((rows, cols)))
    y = np.random.randint(0, 2, rows)
    return X, y


@pytest.fixture
def sample_multivariate():
    rows, cols = 100, 50
    X = pd.DataFrame(np.random.random((rows, cols)))
    X = X.apply(lambda x: pd.Series([x, x]), axis=1)
    y = np.random.randint(0, 2, rows)
    return X, y


def test_init_input_data_uni(sample_univariate):
    x, y = sample_univariate
    input_data = init_input_data(X=x, y=y)

    assert np.all(input_data.features == x.values)
    assert np.all(input_data.target == y.reshape(-1, 1))


def test_init_input_data_multi(sample_multivariate):
    x, y = sample_multivariate
    input_data = init_input_data(X=x, y=y)

    assert input_data.features.shape[0] == x.shape[0]
    assert input_data.features.shape[1] == x.shape[1]
    assert np.all(input_data.target == y.reshape(-1, 1))
