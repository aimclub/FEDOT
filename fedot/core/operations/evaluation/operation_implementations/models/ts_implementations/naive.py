from copy import copy
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table, \
    transform_features_and_target_into_lagged
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class RepeatLastValueImplementation(ModelImplementation):
    """
    Repeat last known value of time series to the future -
    LOCF (last observation carried forward)
    """

    def __init__(self, params: OperationParameters):
        super().__init__(params)
        self.elements_to_repeat = None

    @property
    def part_for_repeat(self):
        """Which part of time series will be used for repeating. Vary from 0.01 to 0.5
        If -1 - repeat only last value"""
        return self.params.get('part_for_repeat')

    def fit(self, input_data):
        """ Determine how many elements to repeat during forecasting """
        if self.part_for_repeat == -1:
            self.elements_to_repeat = 1
            return self

        elements_to_repeat = round(len(input_data.features) * self.part_for_repeat)
        if elements_to_repeat < 1:
            # Minimum number of elements is one
            self.elements_to_repeat = 1
        else:
            self.elements_to_repeat = elements_to_repeat

        if self.elements_to_repeat > input_data.task.task_params.forecast_length:
            self.elements_to_repeat = input_data.task.task_params.forecast_length
        return self

    def predict(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        forecast_length = input_data.task.task_params.forecast_length

        # Get last known value from history
        last_observations = input_data.features[-self.elements_to_repeat:].reshape(1, -1)
        forecast = self._generate_repeated_forecast(last_observations, forecast_length)

        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        forecast_length = input_data.task.task_params.forecast_length
        # Transform the predicted time series into a table
        new_idx, transformed_cols, new_target = transform_features_and_target_into_lagged(input_data,
                                                                                          forecast_length,
                                                                                          self.elements_to_repeat)
        input_data.idx = new_idx
        input_data.target = new_target
        forecast = self._generate_repeated_forecast(transformed_cols, forecast_length)
        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def _generate_repeated_forecast(self, transformed_cols: np.array, forecast_length: int):
        if self.elements_to_repeat == 1:
            return np.repeat(transformed_cols, forecast_length, axis=1)
        elif self.elements_to_repeat < forecast_length:
            # Generate pattern
            repeat_number = int(forecast_length / self.elements_to_repeat) + 1
            forecast = np.tile(transformed_cols, repeat_number)
            return forecast[:, :forecast_length]
        else:
            # Number of elements to repeat equal to forecast horizon
            return transformed_cols


class NaiveAverageForecastImplementation(ModelImplementation):
    """ Class for forecasting time series with mean """

    def __init__(self, params: OperationParameters):
        super().__init__(params)

    @property
    def part_for_averaging(self):
        return self.params.get('part_for_averaging')

    def fit(self, input_data):
        """ Such a simple approach does not support fit method """
        pass

    def predict(self, input_data: InputData) -> OutputData:
        """ Get desired part of time series for averaging and calculate mean value """
        forecast_length = input_data.task.task_params.forecast_length

        elements_to_take = self._how_many_elements_use_for_averaging(input_data.features)
        # Prepare single forecast
        mean_value = np.nanmean(input_data.features[-elements_to_take:])
        forecast = np.array([mean_value] * forecast_length).reshape((1, -1))

        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        forecast_length = input_data.task.task_params.forecast_length
        parts = split_rolling_slices(input_data)
        mean_values_for_chunks = self.average_by_axis(parts)
        forecast = np.repeat(mean_values_for_chunks.reshape((-1, 1)), forecast_length, axis=1)
        forecast = forecast[:-forecast_length, :]

        # Update target
        _, transformed_target = ts_to_table(idx=input_data.idx, time_series=input_data.target,
                                            window_size=forecast_length, is_lag=True)
        input_data.target = transformed_target[1:, :]

        # Update indices - there is no forecast for first element and skip last out of boundaries predictions
        last_threshold = forecast_length - 1
        new_idx = input_data.idx[1: -last_threshold]
        input_data.idx = new_idx
        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def average_by_axis(self, parts: np.array):
        """ Perform averaging for each column using last part of it """
        mean_values_for_chunks = np.apply_along_axis(self._average, 1, parts)
        return mean_values_for_chunks

    def _average(self, row: np.array):
        row = row[np.logical_not(np.isnan(row))]
        if len(row) == 1:
            return row

        elements_to_take = self._how_many_elements_use_for_averaging(row)
        return np.mean(row[-elements_to_take:])

    def _how_many_elements_use_for_averaging(self, time_series: np.array):
        elements_to_take = round(len(time_series) * self.part_for_averaging)
        elements_to_take = fix_elements_number(elements_to_take)
        return elements_to_take


def split_rolling_slices(input_data: InputData):
    """ Prepare slices for features series.
    Example of result for time series [0, 1, 2, 3]:
    [[0, nan, nan, nan],
     [0,   1, nan, nan],
     [0,   1,   2, nan],
     [0,   1,   2,   3]]
    """
    nan_mask = np.triu(np.ones_like(input_data.features, dtype=bool), k=1)
    final_matrix = np.tril(input_data.features, k=0)
    final_matrix = np.array(final_matrix, dtype=float)
    final_matrix[nan_mask] = np.nan

    return final_matrix


def fix_elements_number(elements_to_take: int):
    if elements_to_take < 2:
        return 2
    return elements_to_take
