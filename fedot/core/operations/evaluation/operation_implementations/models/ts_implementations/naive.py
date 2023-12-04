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
        input_data = copy(input_data)
        """ Get desired part of time series for averaging and calculate mean value """
        forecast_length = input_data.task.task_params.forecast_length

        window = self._window(input_data.features)
        # Prepare single forecast
        mean_value = np.nanmean(input_data.features[-window:])
        forecast = np.array([mean_value] * forecast_length).reshape((1, -1))

        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        input_data = copy(input_data)
        forecast_length = input_data.task.task_params.forecast_length
        features = input_data.features
        shape = features.shape[0]

        window = self._window(features)
        mean_values = np.array([np.mean(features[-window-shape+i:i+1]) for i in range(shape)])

        forecast = np.repeat(mean_values.reshape((-1, 1)), forecast_length, axis=1)

        # Update target
        new_idx, transformed_target = ts_to_table(idx=input_data.idx, time_series=input_data.target,
                                                  window_size=forecast_length)

        input_data.target = transformed_target
        forecast = forecast[new_idx]
        input_data.idx = new_idx
        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def _window(self, time_series: np.ndarray):
        return max(2, round(time_series.shape[0] * self.part_for_averaging))
