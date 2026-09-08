from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.torch import (
    TorchLinearClassifier,
    TorchMLPClassifier,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.schemas import validate_classification_output_mode


class SimpleClassificationStrategy(EvaluationStrategy):
    """Lightweight classification models on the TensorData runtime.

    Hosts small differentiable classifiers with a shared fit/predict contract —
    e.g. linear and future MLP heads. Heavier families (boosting, deep architectures)
    get their own strategies.
    """

    _operations_by_types = {
        'torch_linear': TorchLinearClassifier,
        'torch_mlp': TorchMLPClassifier,
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        operation_implementation = self.operation_impl(self.params_for_fit)
        operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        output_mode = validate_classification_output_mode(self.output_mode)
        if output_mode == 'labels':
            prediction = trained_operation.predict_labels(predict_data)
        elif output_mode in ['probs', 'full_probs', 'default', False]:
            prediction = trained_operation.predict_proba(predict_data)
            if prediction.shape[-1] < 2:
                raise ValueError('Data set contains only 1 target class. Please reformat your data.')
            if output_mode != 'full_probs' and prediction.shape[-1] == 2:
                prediction = prediction[:, 1]

        return self._replace_predict_in_tensor_data(prediction, predict_data)
