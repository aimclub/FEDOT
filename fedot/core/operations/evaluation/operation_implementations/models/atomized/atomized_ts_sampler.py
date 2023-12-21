from typing import Union, Optional, Any, Dict

import numpy as np

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.atomized_model.atomized_model import AtomizedModel
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.node import PipelineNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum, TsForecastingParams, Task


class AtomizedTimeSeriesSampler(AtomizedModel):
    """ Increase data for fitting for short time series """

    operation_type = 'atomized_ts_sampler'

    def __init__(self, pipeline: Optional['Pipeline'] = None, mode='sparse'):
        if pipeline is None:
            pipeline = Pipeline(PipelineNode('ridge'))
        super().__init__(pipeline=pipeline)

        self.mode = mode

    def _sample(self, data: InputData):
        # TODO refactor
        if self.mode == 'sparse':
            features = data.features
            if features.shape[1] % 2 == 1:
                features = features[:, 1:]
            new_features = np.concatenate([features[:, ::2],
                                           features[:, 1::2]], axis=0)

            new_target = data.target
            if new_target is not None:
                if new_target.ndim == 1:
                    target = new_target.reshape(1, -1)
                new_target = np.concatenate([new_target, new_target], axis=0)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        new_data = InputData(idx=np.arange(new_features.shape[0]),
                             features=new_features,
                             target=new_target,
                             task=data.task,
                             data_type=data.data_type,
                             supplementary_data=data.supplementary_data)
        return new_data

    def fit(self, params: Optional[Union[OperationParameters, dict]], data: InputData):
        if data.task.task_type is not TaskTypesEnum.ts_forecasting:
            raise ValueError(f"{self.__class__} supports only time series forecasting task")

        new_data = self._sample(data)
        return super().fit(params, new_data)

    def _sample_predict(self,
                        fitted_operation: 'Pipeline',
                        data: InputData,
                        params: Optional[Union[OperationParameters, Dict[str, Any]]] = None,
                        output_mode: str = 'default') -> OutputData:
        # TODO refactor
        new_data = self._sample(data)
        predictions = list()
        for i in range(new_data.features.shape[0]):
            new_data1 = InputData(idx=new_data.idx,
                                  features=new_data.features[i, :].reshape((1, -1)),
                                  target=new_data.target[i, :] if new_data.target is not None else new_data.target,
                                  task=new_data.task,
                                  data_type=new_data.data_type,
                                  supplementary_data=new_data.supplementary_data)
            prediction1 = super().predict(fitted_operation=fitted_operation,
                                          data=new_data1,
                                          params=params,
                                          output_mode=output_mode)
            predictions.append(prediction1)

        predicts = list()
        limit = int(new_data.features.shape[0] // 2)
        for i in range(limit):
            predicts.append((predictions[i].predict + predictions[i + limit].predict) * 0.5)
        predict = np.concatenate(predicts, axis=0)
        predict = OutputData(idx=data.idx,
                             features=data.features,
                             target=data.target,
                             predict=predict,
                             task=data.task,
                             data_type=data.data_type,
                             supplementary_data=data.supplementary_data)
        return predict

    def predict(self, *args, **kwargs) -> OutputData:
        return self._sample_predict(*args, **kwargs)

    def predict_for_fit(self, *args, **kwargs) -> OutputData:
        return self._sample_predict(*args, **kwargs)
