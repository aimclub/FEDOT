from copy import copy

import numpy as np
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.log import Log
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table, \
    prepare_target
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class RepeatLastValueImplementation(ModelImplementation):
    """
    Repeat last known value of time series to the future -
    LOCF (last observation carried forward)
    """

    def __init__(self, log: Optional[Log] = None, **params):
        super().__init__(log)
        self.params = {}

    def fit(self, input_data):
        """ Such a simple approach does not support fit method """
        pass

    def predict(self, input_data: InputData, is_fit_pipeline_stage: bool):
        input_data = copy(input_data)
        forecast_length = input_data.task.task_params.forecast_length

        if is_fit_pipeline_stage:
            # Transform the predicted time series into a table
            old_idx = input_data.idx
            new_idx, transformed_cols = ts_to_table(idx=old_idx,
                                                    time_series=input_data.features,
                                                    window_size=1,
                                                    is_lag=True)
            new_idx, transformed_cols, new_target = prepare_target(all_idx=input_data.idx,
                                                                   idx=new_idx,
                                                                   features_columns=transformed_cols,
                                                                   target=input_data.target,
                                                                   forecast_length=forecast_length)
            input_data.idx = new_idx
            input_data.target = new_target
            forecast = np.repeat(transformed_cols, forecast_length, axis=1)
        else:
            # Get last known value from history
            last_observation = input_data.features[-1]
            forecast = np.array([last_observation] * forecast_length).reshape((1, -1))

        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params


class NaiveAverageForecastImplementation(ModelImplementation):
    """ Class for forecasting time series with mean """

    def __init__(self, log: Optional[Log] = None, **params):
        super().__init__(log)
        self.part_for_averaging = params.get('part_for_averaging')

    def fit(self, input_data):
        """ Such a simple approach does not support fit method """
        pass

    def predict(self, input_data: InputData, is_fit_pipeline_stage: bool):
        """ Get desired part of time series for averaging and calculate mean value """
        forecast_length = input_data.task.task_params.forecast_length
        if is_fit_pipeline_stage:
            parts = split_rolling_slices(input_data)
            mean_values_for_chunks = self.average_by_axis(parts)
            forecast = np.repeat(mean_values_for_chunks.reshape((-1, 1)), forecast_length, axis=1)
        else:
            elements_to_take = self._how_many_elements_use_for_averaging(input_data.features)
            # Prepare single forecast
            mean_value = np.nanmean(input_data.features[-elements_to_take:])
            forecast = np.array([mean_value] * forecast_length).reshape((1, -1))

        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return {'part_for_averaging': self.part_for_averaging}

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
    # Generate two triangle matrices with nan and zeros
    dummy_tril_with_zeros = np.tril(np.full(len(input_data.features), 1), k=0)
    final_matrix = np.tril(np.full(len(input_data.features), 1), k=0)
    final_matrix = np.array(final_matrix, dtype=float)
    final_matrix[final_matrix == 0] = np.nan

    actual_tril = np.tril(input_data.features, k=0)
    actual_tril = np.array(actual_tril, dtype=float)
    final_matrix[dummy_tril_with_zeros != 0] = actual_tril[dummy_tril_with_zeros != 0]
    return final_matrix


def fix_elements_number(elements_to_take: int):
    if elements_to_take < 2:
        return 2
    return elements_to_take
