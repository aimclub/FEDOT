import numpy as np
import pytest

from fedot_ind.core.operation.decomposition.matrix_decomposition.method_impl.column_sampling_decomposition import \
    CURDecomposition


@pytest.fixture
def sample_matrix():
    return np.random.rand(10, 10)


@pytest.fixture
def sample_ts():
    return np.random.rand(100)


def test_fit_transform(sample_matrix):
    rank = 1
    cur = CURDecomposition({'rank': rank})
    result = cur.fit_transform(sample_matrix)
    approximated = result[0]
    assert isinstance(result, tuple)
    assert approximated.shape[0] == rank


def test_ts_to_matrix(sample_ts):
    cur = CURDecomposition({'rank': 2})
    matrix = cur.ts_to_matrix(time_series=sample_ts, window=10)
    assert isinstance(matrix, np.ndarray)


def test_matrix_to_ts(sample_matrix):
    cur = CURDecomposition({'rank': 2})
    ts = cur.matrix_to_ts(sample_matrix)
    assert isinstance(ts, np.ndarray)
    assert len(ts.shape) == 1
