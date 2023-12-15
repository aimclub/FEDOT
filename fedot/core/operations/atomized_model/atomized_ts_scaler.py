from typing import Union, Optional, Any, Dict

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams, Task


class AtomizedTimeSeriesScaler(AtomizedModel):
    """ Add bias to data in window """

    def __init__(self, pipeline: Optional['Pipeline'] = None, mode='sparse'):
        if pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        super().__init__(pipeline=pipeline)

        self.mode = mode

    def description(self, operation_params: Optional[dict] = None) -> str:
        return f"{self.__class__}({super().description(operation_params)})"

    def _scale(self, data: InputData, fit_stage: bool):
        new_features = (data.features - data.features[:, :1])[:, 1:]

        target_bias = data.features[:, -1:]
        if fit_stage:
            new_target = data.target - target_bias
        else:
            new_target = data.target

        supplementary_data = data.supplementary_data
        supplementary_data.time_series_bias.append(target_bias)

        new_data = InputData(idx=data.idx,
                             features=new_features,
                             target=new_target,
                             task=data.task,
                             data_type=data.data_type,
                             supplementary_data=supplementary_data)
        return new_data

    def fit(self, params: Optional[Union[OperationParameters, dict]], data: InputData):
        if data.task.task_type is not TaskTypesEnum.ts_forecasting:
            raise ValueError(f"{self.__class__} supports only time series forecasting task")
        return super().fit(params, self._scale(data, fit_stage=True))

    def _sample_predict(self,
                        fitted_operation: 'Pipeline',
                        data: InputData,
                        params: Optional[Union[OperationParameters, Dict[str, Any]]] = None,
                        output_mode: str = 'default') -> OutputData:
        new_data = self._scale(data, fit_stage=False)
        prediction = super().predict(fitted_operation=fitted_operation,
                                      data=new_data,
                                      params=params,
                                      output_mode=output_mode)
        new_predict = prediction.predict.reshape((-1, 1)) + prediction.supplementary_data.time_series_bias.pop()
        new_predict = new_predict.reshape(prediction.predict.shape)
        prediction.predict = new_predict
        return prediction

    def predict(self, *args, **kwargs) -> OutputData:
        return self._sample_predict(*args, **kwargs)

    def predict_for_fit(self, *args, **kwargs) -> OutputData:
        return self._sample_predict(*args, **kwargs)
