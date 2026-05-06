from typing import Optional

import numpy as np
from fedot.core.data.input_data.data import InputData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.pipelines.pipeline_builder import PipelineBuilder

from fedot.industrial.core.models.detection.anomaly_detector import AnomalyDetector


class StatisticalDetector(AnomalyDetector):
    """Statistical anomaly detector is build on QuantileExtractor and sklearn.OneClassSVM.

    Args:
        params: additional parameters for a statistical model

            .. details:: Possible parameters:

                    - ``scale_ts`` -> a flag indicating whether to add a scaling node or not
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        self.node_list = self.params.get('node_list', ['quantile_extractor', 'one_class_svm'])
        self.scale_ts = self.params.get('scale', False)
        self.anomaly_threshold = None

    def build_model(self):
        model_impl = PipelineBuilder()
        if self.scale_ts:
            self.node_list.insert(0, 'scaling')
        for node in self.node_list:
            model_impl.add_node(node)
        model_impl = model_impl.build()
        return model_impl

    def score_samples(self, input_data: InputData) -> np.ndarray:
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        features = self.model_impl.nodes[1].fitted_operation.transform(converted_input_data).predict
        scores = self.model_impl.nodes[0].fitted_operation.score_samples(features).reshape(-1, 1)
        self.anomaly_threshold = self.anomaly_threshold if self.anomaly_threshold is not None \
            else abs(np.quantile(scores, 0.01))
        prediction = np.apply_along_axis(self._convert_scores_to_probs, 1, scores)
        prediction = self._convert_to_output(input_data, prediction)
        return prediction

    def _predict(self, input_data: InputData, output_mode: str = 'default') -> np.ndarray:
        converted_input_data = self.convert_input_data(input_data, fit_stage=False)
        features = self.model_impl.nodes[1].fitted_operation.transform(converted_input_data).predict
        scores = self.model_impl.nodes[0].fitted_operation.score_samples(features).reshape(-1, 1)
        if output_mode in ['probs', 'default']:
            return scores
        else:
            prediction = np.apply_along_axis(self._convert_scores_to_labels, 1, scores).reshape(-1, 1)
            prediction = self._convert_to_output(input_data, prediction)
            return prediction
