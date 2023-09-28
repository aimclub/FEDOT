import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, SkLearnEvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose \
    import DecomposerRegImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters \
    import IsolationForestRegImplementation
from fedot.core.operations.evaluation.operation_implementations. \
    data_operations.sklearn_filters import LinearRegRANSACImplementation, NonLinearRegRANSACImplementation
from fedot.core.operations.evaluation.operation_implementations. \
    data_operations.sklearn_selectors import LinearRegFSImplementation, NonLinearRegFSImplementation
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnRegImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for regression task for predict stage
        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return:
        """
        prediction = trained_operation.predict(predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)

        return converted


class FedotRegressionPreprocessingStrategy(EvaluationStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for regression task
    """

    _operations_by_types = {
        'ransac_lin_reg': LinearRegRANSACImplementation,
        'ransac_non_lin_reg': NonLinearRegRANSACImplementation,
        'rfe_lin_reg': LinearRegFSImplementation,
        'rfe_non_lin_reg': NonLinearRegFSImplementation,
        'decompose': DecomposerRegImplementation,
        'isolation_forest_reg': IsolationForestRegImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return:
        """
        prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing for fit stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return:
        """
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


class FedotRegressionStrategy(EvaluationStrategy):
    """
    Strategy for applying custom regression models from FEDOT make predictions
    """

    _operations_by_types = {
        'knnreg': FedotKnnRegImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """ This method is used for operation training """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=operation_implementation):
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        prediction = trained_operation.predict_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
