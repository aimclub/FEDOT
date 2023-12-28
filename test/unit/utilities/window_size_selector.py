from itertools import combinations

import numpy as np
import pytest

from fedot.utilities.window_size_selector import WindowSizeSelector, WindowSizeSelectorMethodsEnum


@pytest.mark.parametrize('method', WindowSizeSelectorMethodsEnum)
@pytest.mark.parametrize(['window_min', 'window_max'],
                         [sorted(x) for x in combinations(map(int, np.random.rand(5) * 100), 2)] +
                         [(1, 2), (98, 99), (1, 99)])
def test_window_size_selector(method, window_min, window_max):
    selector = WindowSizeSelector(method=method, window_range=(window_min, window_max))
    ts = np.random.rand(1000)

    assert window_min <= selector.apply(time_series=ts) <= window_max


@pytest.mark.parametrize(['window_min', 'window_max'],
                         list(combinations(map(int, np.random.rand(10) * 200 - 50), 2)) +
                         [[-1, 10], [10, 5], [95, 105], [-10, -9], [105, 110]])
def test_window_size_selector_with_uncorrect_window_params(window_min, window_max):
    error = window_min < 0
    error |= window_max > 100
    error |= window_min >= window_max
    if error:
        with pytest.raises(ValueError):
            WindowSizeSelector(window_range=(window_min, window_max))
