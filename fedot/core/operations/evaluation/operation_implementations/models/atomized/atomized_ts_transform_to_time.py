from typing import Optional

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.models.atomized.atomized_ts_mixins import \
    AtomizedTimeSeriesBuildFactoriesMixin
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import TaskTypesEnum, Task


class AtomizedTimeSeriesToTime(AtomizedTimeSeriesBuildFactoriesMixin):
    """ Predict based on time not on time series """

    def __init__(self, pipeline: Optional['Pipeline'] = None):
        if pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        self.pipeline = pipeline

    def _convert_task(self, data: InputData):
        # TODO add verification rule
        #      1. can be after ts-to-table model (only one)
        #      2. can be after table-to-table models (any)
        #      3. not any combination of ts-to-table and table-to-table
        target_ts = np.concatenate([data.target[0, :].ravel(), data.target[1:,-1].ravel()])
        dt = data.idx[1] - data.idx[0]
        time = np.concatenate([np.arange(data.idx[0] - (data.features.shape[1] - 1) * dt, data.idx[0], dt),
                               data.idx,
                               np.arange(data.idx[-1], data.idx[-1] + data.target.shape[1] * dt, dt)])

        features = data.features
        if features.shape[1] < data.target.shape[1]:
            # previous model is ts-to-table
            # do not use that data
            previous_forecast = np.zeros(target_ts.shape)
        elif features.shape[1] == data.target.shape[1]:
            # previous model is table-to-table
            previous_forecast = np.concatenate([data.features[0, :].ravel(), data.features[1:, -1].ravel()])
        else:
            features = np.mean(np.reshape(features,
                                          (features.shape[0], data.target.shape[1], -1),
                                          order='F'), axis=2)
            previous_forecast = np.concatenate([features[0, :].ravel(), features[1:, -1].ravel()])

        new_data = InputData(idx=time[-target_ts.shape[0]:],
                             features=time[-target_ts.shape[0]:].reshape((-1, 1)),
                             target=(target_ts - previous_forecast).reshape((-1, 1)),
                             data_type=DataTypesEnum.table, task=Task(TaskTypesEnum.regression))
        return new_data, previous_forecast

    def fit(self, data: InputData):
        new_data, previous_forecast = self._convert_task(data)
        self.pipeline.fit(new_data)
        return self

    def predict(self, data: InputData) -> OutputData:
        new_data, previous_forecast = self._convert_task(data)
        prediction = self.pipeline.predict(new_data)
        predict = prediction.predict.ravel() + previous_forecast

        prediction.idx = data.idx
        prediction.target = data.target
        if prediction.target is not None:
            window = data.target.shape[1]
            predict = np.array([predict[i:window + i] for i in range(predict.shape[0] - window + 1)])
        prediction.predict = predict
        return prediction

    def predict_for_fit(self, data: InputData) -> OutputData:
        return self.predict(data)
