import numpy as np
from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.log import Log
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
        forecast_length = input_data.task.task_params.forecast_length

        # Get last known value from history
        last_observation = input_data.features[-1]
        forecast = np.array([last_observation] * forecast_length)

        output_data = self._convert_to_output(input_data,
                                              predict=forecast,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params
