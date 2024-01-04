import math
from enum import Enum, auto

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf


class WindowSizeSelectorMethodsEnum(Enum):
    DFF = auto()
    HAC = auto()
    MWF = auto()
    SSS = auto()


class WindowSizeSelector:
    """Class to select appropriate window size to catch periodicity for time series analysis.
    There are two group of algorithms implemented:
    Whole-Series-Based (WSB):
        1. WindowSizeSelectorMethodsEnum.HAC - highest_autocorrelation
        2. WindowSizeSelectorMethodsEnum.DFF - dominant_fourier_frequency
    Subsequence-based (SB):
        1. WindowSizeSelectorMethodsEnum.MWF - multi_window_finder
        2. WindowSizeSelectorMethodsEnum.SSS - summary_statistics_subsequence
    Args:
        method: by ``default``, it is WindowSizeSelectorMethodsEnum.DFF.
        window_range: % of time series length, by ``default`` it is (5, 50).
    Attributes:
        length_ts(int): length of the time_series.
        window_max(int): maximum window size in real values.
        window_min(int): minimum window size in real values.
        dict_methods(dict): dictionary with all implemented methods.
    Example:
        To find window size for single time series::
            ts = np.random.rand(1000)
            ws_selector = WindowSizeSelector(method='hac')
            window_size = ws_selector.get_window_size(time_series=ts)
        To find window size for multiple time series::
            ts = np.random.rand(1000, 10)
            ws_selector = WindowSizeSelector(method='hac')
            window_size = ws_selector.apply(time_series=ts, average='median')
    Reference:
        (c) "Windows Size Selection in Unsupervised Time Series Analytics: A Review and Benchmark. Arik Ermshaus,
        Patrick Schafer, and Ulf Leser. 2022"
    """

    def __init__(self,
                 method: WindowSizeSelectorMethodsEnum = WindowSizeSelectorMethodsEnum.DFF,
                 window_range: tuple = (5, 50)):

        if window_range[0] >= window_range[1]:
            raise ValueError('Upper bound of window range should be bigger than lower bound')
        if window_range[0] < 0:
            raise ValueError('Lower bound of window range should be bigger or equal to 0')
        if window_range[1] > 100:
            raise ValueError('Upper bound of window range should be lower or equal to 100')

        self.dict_methods = {WindowSizeSelectorMethodsEnum.HAC: self.autocorrelation,
                             WindowSizeSelectorMethodsEnum.DFF: self.dominant_fourier_frequency,
                             WindowSizeSelectorMethodsEnum.MWF: self.mwf,
                             WindowSizeSelectorMethodsEnum.SSS: self.summary_statistics_subsequence}
        self.wss_algorithm = method
        self.window_range = window_range
        self.window_max = None
        self.window_min = None
        self.length_ts = None

    def apply(self, time_series: np.ndarray, average: str = 'median') -> int:
        """Method to run WSS class over selected time series in parallel mode via joblib
        Args:
            time_series: time series to study
            average: 'mean' or 'median' to average window size over all time series
        Returns:
            window_size_selected: value which has been chosen as appropriate window size
        """
        methods = {'mean': np.mean, 'median': np.median}
        if time_series.ndim == 1:
            time_series = time_series.reshape((-1, 1))
        window_list = [self.get_window_size(time_series[:, i].ravel()) for i in range(time_series.shape[1])]
        return round(methods[average](window_list))

    def get_window_size(self, time_series: np.ndarray) -> int:
        """Main function to run WSS class over selected time series
        Note:
            One of the reason of ValueError is that time series size can be equal or smaller than 50.
            In case of it try to initially set window_size min and max.
        Returns:
            window_size_selected: value which has been chosen as appropriate window size
        """
        self.length_ts = len(time_series)

        self.window_max = int(round(self.length_ts * self.window_range[1] / 100))  # in real values
        self.window_min = int(round(self.length_ts * self.window_range[0] / 100))  # in real values

        window_size_selected = self.dict_methods[self.wss_algorithm](time_series=time_series)
        window_size_selected = round(window_size_selected * 100 / self.length_ts)
        window_size_selected = max(self.window_range[0], window_size_selected)
        window_size_selected = min(self.window_range[1], window_size_selected)
        return window_size_selected

    def dominant_fourier_frequency(self, time_series: np.ndarray) -> int:
        """
        Method to find dominant fourier frequency in time series and return appropriate window size. It is based on
        the assumption that the dominant frequency is the one with the highest magnitude in the Fourier transform. The
        window size is then the inverse of the dominant frequency.
        """
        fourier = np.fft.fft(time_series)
        freq = np.fft.fftfreq(time_series.shape[0], 1)

        magnitudes, window_sizes = [], []

        for coef, freq in zip(fourier, freq):
            if coef and freq > 0:
                window_size = int(1 / freq)
                mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

                if self.window_min <= window_size < self.window_max:
                    window_sizes.append(window_size)
                    magnitudes.append(mag)
        if window_sizes and magnitudes:
            return window_sizes[np.argmax(magnitudes)]
        else:
            return self.window_min

    def autocorrelation(self, time_series: np.array) -> int:
        """Method to find the highest autocorrelation in time series and return appropriate window size. It is based on
        the assumption that the lag of highest autocorrelation coefficient corresponds to the window size that best
        captures the periodicity of the time series.
        """
        ts_len = time_series.shape[0]
        acf_values = acf(time_series, fft=True, nlags=int(ts_len / 2))

        peaks, _ = find_peaks(acf_values)
        peaks = peaks[np.logical_and(peaks >= self.window_min, peaks < self.window_max)]
        corrs = acf_values[peaks]

        if peaks.shape[0] == 0:  # if there is no peaks in range (window_min, window_max) return window_min
            return self.window_min
        return peaks[np.argmax(corrs)]

    def mwf(self, time_series: np.array) -> int:
        """ Method to find the window size that minimizes the moving average residual. It is based on the assumption
        that the window size that best captures the periodicity of the time series is the one that minimizes the
        difference between the moving average and the time series.
        """

        all_averages, window_sizes = [], []

        for w in range(self.window_min, self.window_max, 1):
            movingAvg = np.array(self.movmean(time_series, w))
            all_averages.append(movingAvg)
            window_sizes.append(w)

        movingAvgResiduals = []

        for i, w in enumerate(window_sizes):
            moving_avg = all_averages[i][:len(all_averages[-1])]
            movingAvgResidual = np.log(abs(moving_avg - (moving_avg).mean()).sum())
            movingAvgResiduals.append(movingAvgResidual)

        b = (np.diff(np.sign(np.diff(movingAvgResiduals))) > 0).nonzero()[0] + 1  # local min

        if len(b) == 0:
            return self.window_min
        if len(b) < 3:
            return window_sizes[b[0]]

        reswin = np.array([window_sizes[b[i]] / (i + 1) for i in range(3)])
        w = np.mean(reswin)

        return int(w)

    def movmean(self, ts, w):
        """Fast moving average function"""
        moving_avg = np.cumsum(ts, dtype=float)
        moving_avg[w:] = moving_avg[w:] - moving_avg[:-w]
        return moving_avg[w - 1:] / w

    def summary_statistics_subsequence(self, time_series: np.array, threshold=.89) -> int:
        """Method to find the window size that maximizes the subsequence unsupervised similarity score (SUSS). It is
        based on the assumption that the window size that best captures the periodicity of the time series is the one
        that maximizes the similarity between subsequences of the time series.
        """
        # lbound = self.window_min
        time_series = (time_series - time_series.min()) / (time_series.max() - time_series.min())

        ts_mean = np.mean(time_series)
        ts_std = np.std(time_series)
        ts_min_max = np.max(time_series) - np.min(time_series)

        stats = (ts_mean, ts_std, ts_min_max)

        max_score = self.suss_score(time_series=time_series, window_size=1, stats=stats)
        min_score = self.suss_score(time_series=time_series, window_size=time_series.shape[0] - 1, stats=stats)

        exp = 0

        # exponential search (to find window size interval)
        while True:
            window_size = 2 ** exp

            if window_size > self.window_max:
                break

            if window_size < self.window_min:
                exp += 1
                continue

            score = 1 - (self.suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

            if score > threshold:
                break

            exp += 1

        lbound, ubound = max(self.window_min, 2 ** (exp - 1)), 2 ** exp + 1

        # binary search (to find window size in interval)
        while lbound <= ubound:
            window_size = int((lbound + ubound) / 2)
            score = 1 - (self.suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

            if score < threshold:
                lbound = window_size + 1
            elif score > threshold:
                ubound = window_size - 1
            else:
                break

        return 2 * lbound

    def suss_score(self, time_series, window_size, stats):
        roll = pd.Series(time_series).rolling(window_size)
        ts_mean, ts_std, ts_min_max = stats

        roll_mean = roll.mean().to_numpy()[window_size:]
        roll_std = roll.std(ddof=0).to_numpy()[window_size:]
        roll_min = roll.min().to_numpy()[window_size:]
        roll_max = roll.max().to_numpy()[window_size:]

        X = np.array([
            roll_mean - ts_mean,
            roll_std - ts_std,
            (roll_max - roll_min) - ts_min_max
        ])

        X = np.sqrt(np.sum(np.square(X), axis=0)) / np.sqrt(window_size)

        return np.mean(X)
