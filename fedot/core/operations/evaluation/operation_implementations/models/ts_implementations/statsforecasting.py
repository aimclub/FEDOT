import datetime
from copy import copy
from typing import Optional

from statsforecast.models import AutoARIMA, AutoETS, GARCH, AutoTheta, AutoRegressive, AutoCES

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import ts_to_table
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum


class StatsForecastingImplementation(ModelImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = None
        self.fitted = False

    def fit(self, input_data: InputData):
        start_time = datetime.datetime.now()
        self.model.fit(y=input_data.features)
        print(datetime.datetime.now() - start_time)
        return self.model

    def predict(self, input_data: InputData) -> OutputData:
        predict = self.model.predict(h=input_data.task.task_params.forecast_length)['mean']
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def predict_for_fit(self, input_data: InputData):
        input_data = copy(input_data)
        fitted_values = self.model.predict_in_sample()['fitted']

        _, predict = ts_to_table(idx=input_data.idx,
                                 time_series=fitted_values,
                                 window_size=input_data.task.task_params.forecast_length)

        new_idx, target_columns = ts_to_table(idx=input_data.idx,
                                              time_series=input_data.target,
                                              window_size=input_data.task.task_params.forecast_length)
        input_data.idx = new_idx
        input_data.target = target_columns

        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data


class AutoARIMAImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoARIMA(**params.to_dict())


class AutoThetaImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoTheta(**params.to_dict())


class AutoETSImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoETS(**params.to_dict())


class AutoRegImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoRegressive(**params.to_dict())


class GARCHImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = GARCH(**params.to_dict())


class AutoCESImplementation(StatsForecastingImplementation):
    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.model = AutoCES(**params.to_dict())
