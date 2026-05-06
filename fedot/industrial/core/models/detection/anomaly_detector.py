from abc import abstractmethod
from typing import Optional, Union

import pandas as pd
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.utilities.custom_errors import AbstractMethodNotImplementError

from fedot.industrial.core.architecture.settings.computational import backend_methods as np
from fedot.industrial.core.operation.transformation.data.hankel import HankelMatrix


class AnomalyDetector(ModelImplementation):
    """An abstract class for anomaly detectors.

    Args:
        params: additional parameters for an encapsulated model
    """

    def __init__(self, params: Optional[OperationParameters] = None) -> None:
        super().__init__(params)
        self.length_of_detection_window = self.params.get('window_length', 10)
        self.contamination = self.params.get('contamination', 'auto')
        if isinstance(self.contamination, str):
            self.offset = -0.5
        self.transformation_mode = 'lagged'
        self.transformation_type = None

    @property
    def classes_(self) -> int:
        return 1

    def convert_input_data(
            self,
            input_data: InputData,
            fit_stage: bool = True) -> Union[InputData, np.ndarray]:
        if self.transformation_mode == 'lagged':
            feature_matrix = np.concatenate(
                [
                    HankelMatrix(
                        time_series=ts,
                        window_size=self.window_size
                    ).trajectory_matrix.T for ts in input_data.features.T
                ],
                axis=1
            )
            if fit_stage:  # shrink target
                target = input_data.target[:feature_matrix.shape[0]]
            else:  # augmented predict
                target = input_data.target
        elif self.transformation_mode == 'full':
            if self.transformation_type == pd.DataFrame:
                return pd.DataFrame(input_data.features)
            return input_data.features
        elif self.transformation_mode == 'batch':
            feature_matrix, target = input_data.features, input_data.target

        converted_input_data = InputData(
            idx=np.arange(feature_matrix.shape[0]),
            features=feature_matrix,
            target=target,
            task=Task(TaskTypesEnum.classification),
            data_type=DataTypesEnum.table
        )
        converted_input_data.supplementary_data.is_auto_preprocessed = True
        return converted_input_data

    @abstractmethod
    def build_model(self):
        raise AbstractMethodNotImplementError

    def _detect_anomaly_sample(self, score_matrix_row):
        outlier_score = score_matrix_row[0]
        if isinstance(self.contamination, str):
            anomaly_sample = abs(outlier_score) > abs(self.anomaly_threshold) + abs(self.offset)
        else:
            anomaly_sample = outlier_score < 0 and abs(outlier_score) > self.anomaly_threshold
        return anomaly_sample

    def _convert_scores_to_labels(self, prob_matrix_row) -> int:
        anomaly_sample = self._detect_anomaly_sample(prob_matrix_row)
        return 1 if anomaly_sample else 0

    def _convert_scores_to_probs(self, prob_matrix_row) -> int:
        outlier_score = prob_matrix_row[0]
        anomaly_sample = self._detect_anomaly_sample(prob_matrix_row)
        prob = np.array([abs(outlier_score), 0]) if not anomaly_sample else np.array([0, abs(outlier_score)])
        return prob

    def _convert_to_output(self,
                           input_data: InputData,
                           predict: np.array,
                           data_type: DataTypesEnum = DataTypesEnum.table):
        if input_data.features.shape[0] == predict.shape[0]:
            return predict
        else:
            prediction = np.zeros((input_data.features.shape[0], predict.shape[1]))
            start_idx = prediction.shape[0] - predict.shape[0]
            prediction[start_idx:, :] = predict
            return prediction

    def _predict(self, input_data: InputData, output_mode: str = 'default') -> np.ndarray:
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        scores = self.model_impl.score_samples(converted_input_data).reshape(-1, 1)
        if output_mode in ['probs', 'default']:
            return scores
        else:
            prediction = np.apply_along_axis(self._convert_scores_to_labels, 1, scores).reshape(-1, 1)
            prediction = self._convert_to_output(input_data, prediction)
            return prediction

    def fit(self, input_data: InputData) -> None:
        self.model_impl = self.build_model()
        self.window_size = round(
            input_data.features.shape[0] * (self.length_of_detection_window / 100))
        converted_input_data = self.convert_input_data(input_data)
        self.model_impl.fit(converted_input_data)

    def predict(self, input_data: InputData) -> np.ndarray:
        return self._predict(input_data, 'labels')

    def predict_for_fit(self, input_data: InputData):
        return self._predict(input_data)

    def predict_proba(self, input_data: InputData) -> np.ndarray:
        return self.score_samples(input_data)

    def score_samples(self, input_data: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        scores = self.model_impl.score_samples(converted_input_data).reshape(-1, 1)
        self.anomaly_threshold = self.anomaly_threshold if self.anomaly_threshold is not None \
            else abs(np.quantile(scores, 0.01))
        prediction = np.apply_along_axis(self._convert_scores_to_probs, 1, scores)
        prediction = self._convert_to_output(input_data, prediction)
        return prediction
