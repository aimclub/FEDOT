Window Size Selector
=====================

Class to select an appropriate window size to capture periodicity for time series analysis.

There are two groups of algorithms implemented:

Whole-Series-Based (WSB):
    1. WindowSizeSelectorMethodsEnum.HAC - Highest Autocorrelation
    2. WindowSizeSelectorMethodsEnum.DFF - Dominant Fourier Frequency

Subsequence-based (SB):
    1. WindowSizeSelectorMethodsEnum.MWF - Multi-Window Finder
    2. WindowSizeSelectorMethodsEnum.SSS - Summary Statistics Subsequence

Args:
    method (str): The method to use for selecting the window size. Default is 'DFF'.
    window_range (tuple): Percentage range of time series length for selecting the window size. Default is (5, 50).

Attributes:
    length_ts (int): Length of the time series.
    window_max (int): Maximum window size in real values.
    window_min (int): Minimum window size in real values.
    dict_methods (dict): Dictionary with all implemented methods.

Example:
    To find the window size for a single time series:

    .. code-block:: python

        import numpy as np
        from fedot.utilities.window_size_selector import WindowSizeSelector

        ts = np.random.rand(1000)
        ws_selector = WindowSizeSelector(method='HAC')
        window_size = ws_selector.get_window_size(time_series=ts)

    To find the window size for multiple time series:

    .. code-block:: python

        import numpy as np
        from fedot.utilities.window_size_selector import WindowSizeSelector

        ts = np.random.rand(1000, 10)
        ws_selector = WindowSizeSelector(method='HAC')
        window_size = ws_selector.apply(time_series=ts, average='median')

Reference:
    "Windows Size Selection in Unsupervised Time Series Analytics: A Review and Benchmark" by Arik Ermshaus, Patrick Schafer, and Ulf Leser (2022).
