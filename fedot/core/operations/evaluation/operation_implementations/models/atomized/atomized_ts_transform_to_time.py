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
        self._target_shape = None

    def _convert_task(self, data: InputData, fit_stage: bool):
        # TODO add test that model correctly transform data
        if fit_stage:
            target_ts = np.concatenate([data.target[0, :].ravel(), data.target[1:,-1].ravel()])
            self._target_shape = data.target.shape
        else:
            target_ts = None

        dt = data.idx[1] - data.idx[0]
        time = np.concatenate([np.arange(data.idx[0] - (data.features.shape[1] - 1) * dt, data.idx[0], dt),
                               data.idx,
                               np.arange(data.idx[-1], data.idx[-1] + self._target_shape[1] * dt, dt)])
        features = data.features
        if features.shape[1] < self._target_shape[1]:
            # previous model is ts-to-table
            # do not use that data
            previous_forecast = 0
        elif features.shape[1] == self._target_shape[1]:
            # previous model is the only table-to-table
            previous_forecast = np.concatenate([data.features[0, :].ravel(), data.features[1:, -1].ravel()])
        elif features.shape[1] % self._target_shape[1] == 0:
            # previous models are some models of type table-to-table
            features = np.mean(np.reshape(features,
                                          (features.shape[0], self._target_shape[1], -1),
                                          order='F'), axis=2)
            previous_forecast = np.concatenate([features[0, :].ravel(), features[1:, -1].ravel()])
            self._mode = 3
        else:
            raise ValueError('Previous nodes types cannot be defined')

        if fit_stage:
            new_target = (target_ts - previous_forecast).reshape((-1, 1))
        else:
            new_target = None

        predict_length = self._target_shape[1] + features.shape[0] - 1
        new_data = InputData(idx=time[-predict_length:],
                             features=time[-predict_length:].reshape((-1, 1)),
                             target=new_target,
                             data_type=DataTypesEnum.table,
                             task=Task(TaskTypesEnum.regression))
        return new_data, previous_forecast

    def fit(self, data: InputData):
        new_data, previous_forecast = self._convert_task(data, fit_stage=True)
        self.pipeline.fit(new_data)
        return self

    def predict(self, data: InputData) -> OutputData:
        new_data, previous_forecast = self._convert_task(data, fit_stage=False)
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
