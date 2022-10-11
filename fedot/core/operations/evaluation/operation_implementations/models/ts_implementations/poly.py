from copy import copy
from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class PolyfitImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.min_degree = 1
        self.max_degree = 5
        self.default_degree = 3
        self._correct_degree()
        self.coefs = None

    def _correct_degree(self):
        degree = self.params.get('degree')
        if not degree or not self.min_degree <= degree <= self.max_degree:
            # default value
            self.log.debug(f"Change invalid parameter degree ({degree}) on default value (3)")
            degree = self.default_degree
            self.params.update(degree=degree)

    @property
    def degree(self):
        return int(self.params.get('degree'))

    def fit(self, input_data):
        f_x = input_data.idx
        f_y = input_data.features
        self.coefs = np.polyfit(f_x, f_y, deg=self.degree)

        return self.coefs

    def predict(self, input_data):
        f_x = input_data.idx
        f_y = np.polyval(self.coefs, f_x)
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx

        start_id = old_idx[-1] - forecast_length + 1
        end_id = old_idx[-1]
        predict = f_y
        predict = np.array(predict).reshape(1, -1)
        new_idx = np.arange(start_id, end_id + 1)

        input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        f_x = input_data.idx
        f_y = np.polyval(self.coefs, f_x)
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        _, predict = ts_to_table(idx=old_idx,
                                 time_series=f_y,
                                 window_size=forecast_length)
        new_idx, target_columns = ts_to_table(idx=old_idx,
                                              time_series=target,
                                              window_size=forecast_length)

        # Update idx and target
        input_data.idx = new_idx
        input_data.target = target_columns
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data
