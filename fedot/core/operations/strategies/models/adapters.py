from typing import Optional
import warnings

from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.implementations.models.estimator_adapters import (
    FedotSVCImplementation,
    KnnClassificationAdapter,
    KnnRegressionAdapter,
    LDAImplementation,
    QDAImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.utilities.random import ImplementationRandomStateHandler


class AdapterModelStrategy(EvaluationStrategy):
    """TensorData strategy for FEDOT wrappers around external estimators."""

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: TensorData):
        implementation = self.operation_impl(self.params_for_fit)
        with warnings.catch_warnings(), ImplementationRandomStateHandler(implementation=implementation):
            warnings.simplefilter('ignore', RuntimeWarning)
            implementation.fit(train_data)
        return implementation

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        prediction = trained_operation.predict(predict_data)
        return self._replace_predict_in_tensor_data(prediction, predict_data)


class AdapterClassificationStrategy(AdapterModelStrategy):
    _operations_by_types = {
        'lda': LDAImplementation,
        'qda': QDAImplementation,
        'svc': FedotSVCImplementation,
        'knn': KnnClassificationAdapter,
    }

    def predict(self, trained_operation, predict_data: TensorData) -> TensorData:
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data)
            if len(trained_operation.classes_) < 2:
                raise ValueError('Data set contain only 1 target class. Please reformat your data.')
            if prediction.shape[-1] == 2 and self.output_mode != 'full_probs':
                prediction = prediction[..., 1]
        else:
            raise ValueError(f'Output mode {self.output_mode!r} is not supported')
        return self._replace_predict_in_tensor_data(prediction, predict_data)


class AdapterRegressionStrategy(AdapterModelStrategy):
    _operations_by_types = {'knnreg': KnnRegressionAdapter}

    def predict_for_fit(self, trained_operation, predict_data: TensorData) -> TensorData:
        prediction = trained_operation.predict_for_fit(predict_data)
        return self._replace_predict_in_tensor_data(prediction, predict_data)
