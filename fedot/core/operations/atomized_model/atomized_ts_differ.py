from typing import Union, Optional, Any, Dict

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams, Task


class AtomizedTimeSeriesDiffer(AtomizedModel):
    """ Get diff of timeseries, train model/forecast, integrate result """

    operation_type = 'atomized_ts_differ'

    def __init__(self, pipeline: Optional['Pipeline'] = None):
        if pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        super().__init__(pipeline=pipeline)

    def _diff(self, data: InputData, fit_stage: bool):
        new_features = np.diff(data.features, axis=1)
        bias = data.features[:, -1:]

        if fit_stage:
            target = data.target
            if target.ndim == 1:
                target = target.reshape((1, -1))

            new_target = np.diff(np.concatenate([bias, target], axis=1), axis=1)
        else:
            new_target = data.target

        supplementary_data = data.supplementary_data
        supplementary_data.time_series_bias.append(bias)

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
        return super().fit(params, self._diff(data, fit_stage=True))

    def _sample_predict(self,
                        fitted_operation: 'Pipeline',
                        data: InputData,
                        params: Optional[Union[OperationParameters, Dict[str, Any]]] = None,
                        output_mode: str = 'default') -> OutputData:
        new_data = self._diff(data, fit_stage=False)
        prediction = super().predict(fitted_operation=fitted_operation,
                                     data=new_data,
                                     params=params,
                                     output_mode=output_mode)
        bias = prediction.supplementary_data.time_series_bias.pop()
        new_predict = np.cumsum(prediction.predict.reshape((bias.shape[0], -1)), axis=1) + bias
        new_predict = new_predict.reshape(prediction.predict.shape)
        prediction.predict = new_predict
        return prediction

    def predict(self, *args, **kwargs) -> OutputData:
        return self._sample_predict(*args, **kwargs)

    def predict_for_fit(self, *args, **kwargs) -> OutputData:
        return self._sample_predict(*args, **kwargs)
