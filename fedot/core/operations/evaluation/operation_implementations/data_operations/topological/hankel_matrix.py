from typing import Union

import numpy as np
import pandas as pd
from scipy.linalg import hankel


class HankelMatrix:
    """
    This class implements an algorithm for converting an original time series into a Hankel matrix.
    """

    def __init__(self,
                 time_series: Union[pd.DataFrame, pd.Series, np.ndarray, list],
                 window_size: int = None,
                 strides: int = 1):
        self.__time_series = time_series
        self.__convert_ts_to_array()
        self.__window_length = window_size
        self.__strides = strides
        if len(self.__time_series.shape) > 1:
            self.__ts_length = self.__time_series[0].size
        else:
            self.__ts_length = self.__time_series.size

        if self.__window_length is None:
            self.__window_length = round(self.__ts_length * 0.35)
        else:
            self.__window_length = round(self.__window_length)
        self.__subseq_length = self.__ts_length - self.__window_length + 1

        self.__check_windows_length()
        if len(self.__time_series.shape) > 1:
            self.__trajectory_matrix = self.__get_2d_trajectory_matrix()
        else:
            self.__trajectory_matrix = self.__get_1d_trajectory_matrix()

    def __check_windows_length(self):
        if not 2 <= self.__window_length <= self.__ts_length / 2:
            self.__window_length = int(self.__ts_length / 3)

    def __convert_ts_to_array(self):
        if type(self.__time_series) == pd.DataFrame:
            self.__time_series = self.__time_series.values.reshape(-1, 1)
        elif type(self.__time_series) == list:
            self.__time_series = np.array(self.__time_series)
        else:
            self.__time_series = self.__time_series

    def __get_1d_trajectory_matrix(self):
        if self.__strides > 1:
            return self.__strided_trajectory_matrix(self.__time_series)
        else:
            return hankel(self.__time_series[:self.__window_length + 1], self.__time_series[self.__window_length:])

    def __get_2d_trajectory_matrix(self):
        if self.__strides > 1:
            return [self.__strided_trajectory_matrix(time_series) for time_series
                    in self.__time_series]
        else:
            return [hankel(time_series[:self.__window_length + 1], time_series[self.__window_length:]) for time_series
                    in self.__time_series]

    def __strided_trajectory_matrix(self, time_series):
        shape = (time_series.shape[0] - self.__window_length + 1, self.__window_length)
        strides = (time_series.strides[0],) + time_series.strides
        rolled = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)
        return rolled[np.arange(0, shape[0], self.__strides)].T

    @property
    def window_length(self):
        return self.__window_length

    @property
    def time_series(self):
        return self.__time_series

    @property
    def sub_seq_length(self):
        return self.__subseq_length

    @window_length.setter
    def window_length(self, window_length):
        self.__window_length = window_length

    @property
    def trajectory_matrix(self):
        return self.__trajectory_matrix

    @property
    def ts_length(self):
        return self.__ts_length

    @trajectory_matrix.setter
    def trajectory_matrix(self, trajectory_matrix: np.ndarray):
        self.__trajectory_matrix = trajectory_matrix


def get_x_y_pairs(train, train_periods, prediction_periods):
    """
    train_scaled - training sequence
    train_periods - How many data points to use as inputs
    prediction_periods - How many periods to ouput as predictions
    """
    train_scaled = train.T
    r = train_scaled.shape[0] - train_periods - prediction_periods
    x_train = [train_scaled[i:i + train_periods] for i in range(r)]
    y_train = [train_scaled[i + train_periods:i + train_periods + prediction_periods] for i in range(r)]

    # -- use the stack function to convert the list of 1D tensors
    # into a 2D tensor where each element of the list is now a row
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    return x_train, y_train
