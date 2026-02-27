from fedot_ind.core.architecture.settings.computational import backend_methods as np
import pytest

from fedot_ind.core.operation.transformation.window_selector import WindowSizeSelector
from fedot_ind.tools.synthetic.ts_generator import TimeSeriesGenerator


@pytest.fixture
def selector_dff():
    return WindowSizeSelector(method='dff')


@pytest.fixture
def selector_ac():
    return WindowSizeSelector(method='hac')


@pytest.fixture
def selector_mwf():
    return WindowSizeSelector(method='mwf')


@pytest.fixture
def selector_sss():
    return WindowSizeSelector(method='sss')


@pytest.fixture
def single_ts_data():
    random_walk_config = {
        'ts_type': 'random_walk',
        'length': 1000,
        'start_val': 36.6}
    rw_ts = TimeSeriesGenerator(random_walk_config).get_ts()
    sin_wave_config = {
        'ts_type': 'sin',
        'length': 1000,
        'amplitude': 10,
        'period': 100}
    sin_ts = TimeSeriesGenerator(sin_wave_config).get_ts()
    return rw_ts + sin_ts


@pytest.fixture
def multiple_ts_data(single_ts_data):
    ts = single_ts_data
    return np.array([ts for i in range(10)])


def test_dominant_fourier_frequency_single(single_ts_data, selector_dff):
    ts = single_ts_data
    selected_window = selector_dff.get_window_size(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_high_autocorrelation_single(single_ts_data, selector_ac):
    ts = single_ts_data
    selected_window = selector_ac.get_window_size(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_mean_waveform_single(single_ts_data, selector_mwf):
    ts = single_ts_data
    selected_window = selector_mwf.get_window_size(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_summary_statistics_subsequence_single(single_ts_data, selector_sss):
    ts = single_ts_data
    selected_window = selector_sss.get_window_size(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_dominant_fourier_frequency_multiple(multiple_ts_data, selector_dff):
    ts = multiple_ts_data
    selected_window = selector_dff.apply(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_high_autocorrelation_multiple(multiple_ts_data, selector_ac):
    ts = multiple_ts_data
    selected_window = selector_ac.apply(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_mean_waveform_multiple(multiple_ts_data, selector_mwf):
    ts = multiple_ts_data
    selected_window = selector_mwf.apply(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100


def test_summary_statistics_subsequence_multiple(
        multiple_ts_data, selector_sss):
    ts = multiple_ts_data
    selected_window = selector_sss.apply(time_series=ts)
    assert selected_window > 0
    assert selected_window < 100
