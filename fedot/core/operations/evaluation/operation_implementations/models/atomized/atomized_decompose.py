from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_mixins import \
    AtomizedTimeSeriesBuildFactoriesMixin
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline


class AtomizedTimeSeriesDecomposer(AtomizedTimeSeriesBuildFactoriesMixin):
    def __init__(self, pipeline: Optional['Pipeline'] = None):
        if pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        self.pipeline = pipeline

    def _decompose(self, data: InputData, fit_stage: bool):
        # get merged data from lagged and any model
        forecast_length = data.task.task_params.forecast_length
        data_from_lagged = data.features[:, :-forecast_length]
        data_from_model = data.features[:, -forecast_length:]
        new_target = data.target
        if fit_stage:
            new_target -= data_from_model

        new_data = InputData(idx=data.idx,
                             features=data_from_lagged,
                             target=new_target,
                             data_type=data.data_type,
                             task=data.task,
                             supplementary_data=data.supplementary_data)
        return new_data

    def fit(self, data: InputData):
        new_data = self._decompose(data, fit_stage=True)
        self.pipeline.fit(new_data)
        return self

    def predict(self, data: InputData) -> OutputData:
        new_data = self._decompose(data, fit_stage=False)
        return self.pipeline.predict(new_data)

    def predict_for_fit(self, data: InputData) -> OutputData:
        return self.predict(data)
