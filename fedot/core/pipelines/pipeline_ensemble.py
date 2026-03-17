from copy import deepcopy
from typing import List, Optional, Sequence, Union

import numpy as np
from golem.core.log import default_log

from fedot.core.data.data import InputData, InputDataList, OutputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import TaskTypesEnum


class PipelineEnsemble:
    def __init__(self, pipelines: Sequence[Pipeline]):
        if not pipelines:
            raise ValueError('Pipeline ensemble requires at least one pipeline.')
        self.pipelines: List[Pipeline] = list(pipelines)
        self.log = default_log(self)
        self.use_input_preprocessing = self.pipelines[0].use_input_preprocessing
        self.preprocessor = self.pipelines[0].preprocessor

    @property
    def is_fitted(self) -> bool:
        return all(pipeline.is_fitted for pipeline in self.pipelines)

    def fit(self,
            input_data: Union[InputData, InputDataList, MultiModalData],
            time_constraint=None,
            n_jobs: int = 1,
            predictions_cache=None,
            fold_id: Optional[int] = None) -> Optional[OutputData]:
        if isinstance(input_data, list):
            if len(input_data) != len(self.pipelines):
                raise ValueError('InputDataList size must match number of pipelines in the ensemble.')
            outputs = []
            for pipeline, chunk_data in zip(self.pipelines, input_data):
                outputs.append(pipeline.fit(chunk_data,
                                            time_constraint=time_constraint,
                                            n_jobs=n_jobs,
                                            predictions_cache=predictions_cache,
                                            fold_id=fold_id))
        else:
            outputs = []
            for pipeline in self.pipelines:
                outputs.append(pipeline.fit(input_data,
                                            time_constraint=time_constraint,
                                            n_jobs=n_jobs,
                                            predictions_cache=predictions_cache,
                                            fold_id=fold_id))
        return outputs[0] if outputs else None

    def predict(self,
                input_data: Union[InputData, MultiModalData],
                output_mode: str = 'default',
                predictions_cache=None,
                fold_id: Optional[int] = None) -> OutputData:
        if not self.is_fitted:
            raise ValueError('Pipeline ensemble is not fitted yet')

        results: List[OutputData] = []
        predictions = []
        for pipeline in self.pipelines:
            result = pipeline.predict(input_data,
                                      output_mode=output_mode,
                                      predictions_cache=predictions_cache,
                                      fold_id=fold_id)
            results.append(result)
            predictions.append(result.predict)

        aggregated = self._aggregate_predictions(predictions, output_mode, input_data)
        output = deepcopy(results[0])
        output.predict = aggregated
        return output

    def predict_proba(self,
                      input_data: Union[InputData, MultiModalData],
                      output_mode: str = 'probs',
                      predictions_cache=None,
                      fold_id: Optional[int] = None) -> OutputData:
        return self.predict(input_data,
                            output_mode=output_mode,
                            predictions_cache=predictions_cache,
                            fold_id=fold_id)

    def _aggregate_predictions(self,
                               predictions: List[np.ndarray],
                               output_mode: str,
                               input_data: Union[InputData, MultiModalData]) -> np.ndarray:
        task_type = input_data.task.task_type
        prediction_stack = np.asarray(predictions)

        if task_type == TaskTypesEnum.classification:
            if output_mode in ('probs', 'full_probs'):
                return np.mean(prediction_stack, axis=0)
            if output_mode == 'default' and prediction_stack.ndim == 3:
                if prediction_stack.shape[-1] > 1:
                    return np.mean(prediction_stack, axis=0)
                prediction_stack = np.squeeze(prediction_stack, axis=-1)
            if prediction_stack.ndim == 3 and prediction_stack.shape[-1] == 1:
                prediction_stack = np.squeeze(prediction_stack, axis=-1)
            return self._majority_vote(prediction_stack)

        return np.mean(prediction_stack, axis=0)

    @staticmethod
    def _majority_vote(prediction_stack: np.ndarray) -> np.ndarray:
        if prediction_stack.ndim == 1:
            return prediction_stack
        if prediction_stack.ndim == 2:
            stacked = prediction_stack
        else:
            stacked = prediction_stack.reshape(prediction_stack.shape[0], -1)
        votes = []
        for column in stacked.T:
            values, counts = np.unique(column, return_counts=True)
            votes.append(values[np.argmax(counts)])
        return np.asarray(votes)
