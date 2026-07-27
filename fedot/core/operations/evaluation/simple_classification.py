from typing import Optional

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.models.torch import TorchLinearClassifier
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
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        operation_implementation = self.operation_impl(self.params_for_fit)
        target = train_data.target

        operation_implementation.fit(
            features=train_data.features,
            target=target,
        )
        return operation_implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        output_mode = validate_classification_output_mode(self.output_mode)
        features = predict_data.features
        if output_mode == 'labels':
            prediction = trained_operation.predict_labels(features)
        elif output_mode in ['probs', 'full_probs', 'default', False]:
            prediction = trained_operation.predict_proba(features)
            if output_mode != 'full_probs' and prediction.shape[-1] == 2:
                prediction = prediction[:, 1]

        predict_data.predict = prediction
        return predict_data
