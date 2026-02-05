from abc import ABC, abstractmethod
from datetime import datetime as dt
from math import factorial

import pandas as pd
from matplotlib import pyplot as plt

from fedot.industrial.core.architecture.settings.computational import backend_methods as np


class TimeSeriesGenerator:
    """
    A class for generating various types of time series data.

    Args:
        params (dict): A dictionary containing parameters for generating time series data.

    Attributes:
        ts_type (str): The type of time series to generate.
        ts_length (int): The length of the time series.
        seed (int): The random seed used for data generation.
        ts_types (dict): A dictionary mapping time series types to their corresponding classes.
        params (dict): The input parameters for time series generation.

    """

    def __init__(self, params: dict):
        self.ts_type = params.get('ts_type', 'smooth_normal')
        self.ts_length = params.get('length', 1000)
        self.seed = params.get('seed', self.__define_seed())
        self.__define_randomness()

        self.ts_types = {'sin': SinWave,
                         'random_walk': RandomWalk,
                         'auto_regression': AutoRegression,
                         'smooth_normal': SmoothNormal}
        self.params = params

    def __define_seed(self):
        """
        Define a random seed based on the current second of the system's clock.

        Returns:
            int: The generated random seed.

        """
        self.seed = dt.now().second

    def __define_randomness(self):
        """
        Define randomness by setting the random seed using NumPy.

        """
        np.random.seed(self.seed)

    def get_ts(self):
        """
        Generate and return the time series data.

        Returns:
            np.array: The generated time series data.

        Raises:
            ValueError: If the specified 'ts_type' is not valid.

        """
        if self.ts_type not in self.ts_types.keys():
            raise ValueError(
                'ts_type must be one of the following: "sin", "smooth_normal", "random_walk", '
                '"auto_regression"')
        else:
            ts_class = self.ts_types[self.ts_type](self.params)
            return ts_class.get_ts()


class DefaultTimeSeries(ABC):
    def __init__(self, params: dict):
        self.ts_length = params.get('length', 1000)

    @abstractmethod
    def get_ts(self):
        NotImplementedError()


class SinWave(DefaultTimeSeries):
    def __init__(self, params: dict):
        super().__init__(params)
        self.amplitude = params.get('amplitude', 10)
        self.period = params.get('period', self.ts_length)

    def get_ts(self):
        time_index = np.arange(0, self.ts_length)
        sine_wave = self.amplitude * \
            np.sin(2 * np.pi / self.period * time_index)
        noise = np.random.normal(0, 1, self.ts_length)
        return np.array(sine_wave + noise)

    def get_only_sin_ts(self):
        time_index = np.arange(0, self.ts_length)
        sine_wave = self.amplitude * \
            np.sin(2 * np.pi / self.period * time_index)
        return np.array(sine_wave)


class RandomWalk(DefaultTimeSeries):
    def __init__(self, params: dict):
        super().__init__(params)
        self.start_val = params.get('start_val', 36.6)

    def get_ts(self):
        time_index = pd.Series(np.arange(0, self.ts_length))
        random_walk = pd.Series(np.cumsum(np.random.randn(
            self.ts_length)) + self.start_val, index=time_index)
        noise = np.random.normal(0, 1, self.ts_length)
        return np.array(random_walk + noise)


class AutoRegression(DefaultTimeSeries):
    def __init__(self, params: dict):
        super().__init__(params)
        self.ar_params = params.get('ar_params', [0.5, -0.3, 0.2])
        self.initial_values = params.get('initial_values', None)

    def get_ts(self):
        pd.Series(np.arange(0, self.ts_length))
        ar_process = np.zeros(self.ts_length)

        for i in range(len(ar_process)):
            if i < len(self.ar_params):
                if self.initial_values is None:
                    ar_process[i] = np.random.normal(0, 1, 1)
                else:
                    ar_process[i] = self.initial_values[i]
            else:
                ar_process[i] = np.sum(self.ar_params * ar_process[i - len(self.ar_params):i].ravel(),
                                       axis=0) + np.random.normal(0, 1)
        noise = np.random.normal(0, 1, self.ts_length)
        return np.array(ar_process + noise)


class SmoothNormal(DefaultTimeSeries):
    def __init__(self, params: dict):
        super().__init__(params)
        self.window_size = params.get('window_size', int(self.ts_length / 3))
        self.__check_window_size()

    def __check_window_size(self):
        # window size must be odd for smoothing ts with Savitzky-Golay filter
        if self.window_size % 2 == 0:
            self.window_size += 1

    def get_ts(self):
        normal_ts = np.random.normal(0, 1, self.ts_length) + 100
        noise = np.random.normal(0, 1, self.ts_length)
        return np.array(
            self.savitzky_golay(
                y=normal_ts,
                window_size=self.window_size,
                order=3) + noise)

    @staticmethod
    def savitzky_golay(
            y: np.array,
            window_size: int,
            order: int = 3,
            deriv: int = 0,
            rate: int = 1):
        """
        Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

        Args:
            y: array_like, shape (N,) the values of the time history of the signal.
            window_size: int (odd) the length of the window. Must be an odd integer number.
            order: int (optional) the order of the polynomial used in the filtering.
            deriv: int (optional) the order of the derivative to compute.
            rate: int or float (optional) sampling rate of y. Default is 1.

        Returns:

        Notes:
            Taken from https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
        """
        try:
            window_size = np.abs(int(window_size))
            order = np.abs(order)
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError(
                "window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range]
                    for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')


if __name__ == '__main__':
    sin_config = {
        'ts_type': 'sin',
        'length': 1000,
        'amplitude': 10,
        'period': 500
    }

    random_walk_config = {
        'ts_type': 'random_walk',
        'length': 1000,
        'start_val': 36.6
    }

    auto_regression_config = {
        'ts_type': 'auto_regression',
        'length': 1000,
        'ar_params': [0.5, -0.3, 0.2],
        'initial_values': None
    }

    smooth_normal_config = {
        'ts_type': 'smooth_normal',
        'length': 1000,
        'window_size': 300
    }

    for config in [sin_config, random_walk_config, auto_regression_config, smooth_normal_config]:
        ts_generator = TimeSeriesGenerator(config)
        ts = ts_generator.get_ts()
        plt.plot(ts)

    plt.show()
    _ = 1
