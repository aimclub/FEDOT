import pytest

from fedot_ind.core.operation.transformation.data.hankel import HankelMatrix
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator

TS_LENGTH = 1000


@pytest.fixture
def ts_data():
    ts_config = {'ts_type': 'random_walk',
                 'length': TS_LENGTH,
                 'start_val': 36.6}
    return TimeSeriesGenerator(ts_config).get_ts()


def test_valid_trajectory_matrix(ts_data, valid_window_size):
    trajectory_matrix = HankelMatrix(
        time_series=ts_data,
        window_size=valid_window_size).trajectory_matrix

    assert trajectory_matrix is not None
    assert trajectory_matrix.shape[0] == valid_window_size


def test_zero_trajectory_matrix(ts_data, zero_window_size):
    trajectory_matrix = HankelMatrix(
        time_series=ts_data,
        window_size=zero_window_size).trajectory_matrix

    made_up_window = int(TS_LENGTH / 3)
    assert trajectory_matrix is not None
    assert trajectory_matrix.shape[0] == made_up_window + 1
    assert trajectory_matrix.shape[1] == TS_LENGTH - made_up_window


@pytest.fixture
def valid_window_size():
    return 100


@pytest.fixture
def zero_window_size():
    return 0
