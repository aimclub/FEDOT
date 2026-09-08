from typing import Union

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.gpu.common import CuMLEvaluationStrategy


class CuMLClassificationStrategy(CuMLEvaluationStrategy):
    """Classification strategy returning NumPy for InputData and Torch for TensorData."""

    def predict(
        self,
        trained_operation,
        predict_data: Union[InputData, TensorData],
    ) -> Union[OutputData, TensorData]:
        features, runtime_plan = self._features_and_runtime(predict_data)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(features)
        elif self.output_mode in ['probs', 'full_probs', 'default', False]:
            if not hasattr(trained_operation, 'predict_proba'):
                if self.output_mode in ['probs', 'full_probs']:
                    raise ValueError(f'cuML {self.operation_type!r} does not provide class probabilities')
                prediction = trained_operation.predict(features)
            else:
                prediction = trained_operation.predict_proba(features)
                n_classes = prediction.shape[1]
                if n_classes < 2:
                    raise ValueError('Data set contains only 1 target class. Please reformat your data.')
                if n_classes == 2 and self.output_mode != 'full_probs':
                    prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output mode {self.output_mode!r} is not supported')
        return self._convert_cuml_output(prediction, predict_data, runtime_plan)
