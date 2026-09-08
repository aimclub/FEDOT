from typing import Union

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy


class CumlClusteringStrategy(CuMLEvaluationStrategy):
    """K-means strategy returning NumPy for InputData and Torch for TensorData."""

    def predict(
        self,
        trained_operation,
        predict_data: Union[InputData, TensorData],
    ) -> Union[OutputData, TensorData]:
        features, runtime_plan = self._features_and_runtime(predict_data)
        prediction = trained_operation.predict(features)
        return self._convert_cuml_output(prediction, predict_data, runtime_plan)
