import pytest

from fedot_ind.core.architecture.settings.computational import backend_methods as np
from fedot_ind.core.operation.transformation.window_cutter import WindowCutter


@pytest.fixture()
def basic_periodic_data():
    size = 300
    x0 = 1 * np.ones(size) + np.random.rand(size) * 1
    x1 = 3 * np.ones(size) + np.random.rand(size) * 2
    x2 = 5 * np.ones(size) + np.random.rand(size) * 1.5
    x = np.hstack([x0, x1, x2])
    x += np.random.rand(x.size)
    return x


def test_WindowCutting(basic_periodic_data):
    test_dict = {
        "ts_1": basic_periodic_data
    }
    cutter = WindowCutter(window_len=100, window_step=10)
    cutter.load_data(test_dict)
    cutter.run()
    windows_list = cutter.get_windows()
    assert len(windows_list) != 0
    assert list(windows_list[0].keys())[0] == "ts_1"
