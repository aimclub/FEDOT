from copy import copy
from typing import Optional

import numpy as np
from fedot.core.log import Log
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import \
    ts_to_table
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.repository.dataset_types import DataTypesEnum


class PolyfitImplementation(ModelImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.min_degree = 1
        self.max_degree = 5
        self.default_degree = 3
        self.parameters_changed = False

        self.params = params
        self.degree = params.get('degree')
        if not self.degree or not self.min_degree <= self.degree <= self.max_degree:
            # default value
            self.log.info(f"Change invalid parameter degree ({self.degree}) on default value (3)")
            self.degree = self.default_degree
            self.parameters_changed = True
        self.degree = int(self.degree)
        self.coefs = None

    def fit(self, input_data):
        f_x = input_data.idx
        f_y = input_data.features
        self.coefs = np.polyfit(f_x, f_y, deg=self.degree)

        return self.coefs

    def predict(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        f_x = input_data.idx
        f_y = np.polyval(self.coefs, f_x)
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target
        if is_fit_pipeline_stage:
            _, predict = ts_to_table(idx=old_idx,
                                     time_series=f_y,
                                     window_size=forecast_length)
            new_idx, target_columns = ts_to_table(idx=old_idx,
                                                  time_series=target,
                                                  window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predict = f_y
            predict = np.array(predict).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        params_dict = {"degree": self.degree}
        if self.parameters_changed is True:
            return tuple([params_dict, ['degree']])
        else:
            return params_dict
