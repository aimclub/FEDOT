from typing import Union, Optional, Any, Dict

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory
from fedot.core.pipelines.random_pipeline_factory import RandomPipelineFactory
from fedot.core.repository.pipeline_operation_repository import PipelineOperationRepository
from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams, Task


class AtomizedTimeSeriesDiffer:
    """ Get diff of timeseries, train model/forecast, integrate result """

    def __init__(self, pipeline: Optional['Pipeline'] = None):
        if pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        self.pipeline = pipeline

    @classmethod
    def build_factories(cls, requirements, graph_generation_params):
        graph_model_repository = PipelineOperationRepository(operations_by_keys={'primary': requirements.secondary,
                                                             'secondary': requirements.secondary})
        node_factory = PipelineOptNodeFactory(requirements, graph_generation_params.advisor, graph_model_repository)
        random_pipeline_factory = RandomPipelineFactory(graph_generation_params.verifier, node_factory)
        return node_factory, random_pipeline_factory

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
        # supplementary_data.time_series_bias.append(bias)

        new_data = InputData(idx=data.idx,
                             features=new_features,
                             target=new_target,
                             task=data.task,
                             data_type=data.data_type,
                             supplementary_data=supplementary_data)
        return new_data, bias

    def fit(self, data: InputData):
        # TODO define is there need for unfit
        if data.task.task_type is not TaskTypesEnum.ts_forecasting:
            raise ValueError(f"{self.__class__} supports only time series forecasting task")
        data, _ = self._diff(data, fit_stage=True)
        self.pipeline.fit(data)
        return self

    def predict(self, data: InputData) -> OutputData:
        new_data, bias = self._diff(data, fit_stage=False)
        prediction = self.pipeline.predict(new_data)
        new_predict = np.cumsum(prediction.predict.reshape((bias.shape[0], -1)), axis=1) + bias
        new_predict = new_predict.reshape(prediction.predict.shape)
        prediction.predict = new_predict

        prediction.idx = data.idx
        if prediction.target is not None:
            prediction.predict = np.reshape(prediction.predict, prediction.target.shape)
        return prediction

    def predict_for_fit(self, data: InputData) -> OutputData:
        return self.predict(data)
