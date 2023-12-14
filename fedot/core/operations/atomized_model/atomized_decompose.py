from typing import Union, Optional, Any, Dict

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum


class AtomizedTimeSeriesDecompose(AtomizedModel):
    """ Contains pipeline that forecasts previous (in pipeline) model forecast error
        and restore origin time series as sum of forecasted error and previous model forecasting """

    def __init__(self, params_or_pipeline: Optional[Union[OperationParameters, Pipeline]] = None):
        if isinstance(params_or_pipeline, OperationParameters):
            pipeline = Pipeline(PipelineNode(params_or_pipeline.get('initial_model', 'ridge')))
        elif params_or_pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        else:
            pipeline = params_or_pipeline
        super().__init__(pipeline=pipeline)

    def _decompose(self, data: InputData):
        time_series_bias = data.features
        new_time_series = data.target - time_series_bias
        supplementary_data = data.supplementary_data
        supplementary_data.time_series_bias = time_series_bias

        decomposed_input_data = InputData(idx=data.idx,
                                          features=new_time_series,
                                          target=new_time_series,
                                          task=data.task,
                                          data_type=data.data_type,
                                          supplementary_data=supplementary_data)
        return decomposed_input_data

    def fit(self, params: Optional[Union[OperationParameters, dict]], data: InputData):
        if data.task.task_type is not TaskTypesEnum.ts_forecasting:
            raise ValueError(f"{self.__class__} supports only time series forecasting task")

        decomposed_input_data = self._decompose(data)
        return super().fit(params, decomposed_input_data)

    def _decomposed_predict(self,
                            fitted_operation: 'Pipeline',
                            data: InputData,
                            params: Optional[Union[OperationParameters, Dict[str, Any]]] = None,
                            output_mode: str = 'default') -> OutputData:
        decomposed_input_data = self._decompose(data)
        prediction = super().predict(fitted_operation=fitted_operation,
                                     data=decomposed_input_data,
                                     params=params,
                                     output_mode=output_mode)
        prediction.prediction = prediction.prediction + prediction.supplementary_data.time_series_bias
        return prediction

    def predict(self, *args, **kwargs) -> OutputData:
        return self._decomposed_predict(*args, **kwargs)

    def predict_for_fit(self, *args, **kwargs) -> OutputData:
        return self._decomposed_predict(*args, **kwargs)
