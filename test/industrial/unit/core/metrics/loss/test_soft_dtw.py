import numpy as np
import pytest

from fedot_ind.core.metrics.loss.soft_dtw import SoftDTWLoss


@pytest.fixture()
def sample_data():
    X = np.random.randn(10, 1)
    Y = np.random.randn(10, 1)
    return X, Y


def test_sdtw(sample_data):
    x, y = sample_data
    metric = SoftDTWLoss(X=x, Y=y)
    v = metric.sdtw(gamma=0.7)
    assert isinstance(v, float)


def test_sdtw_return_all(sample_data):
    x, y = sample_data
    metric = SoftDTWLoss(X=x, Y=y)
    v, p = metric.sdtw(gamma=0.7, return_all=True)
    assert isinstance(v, np.ndarray)
    assert isinstance(p, np.ndarray)
