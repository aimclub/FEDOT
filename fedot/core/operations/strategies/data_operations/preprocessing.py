from typing import Optional
import warnings

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.implementations.data_operations.decompose import (
    DecomposerClassImplementation,
    DecomposerRegImplementation,
)
from fedot.core.operations.implementations.data_operations.resampling import ResampleImplementation
from fedot.core.operations.implementations.data_operations.sklearn_filters import (
    IsolationForestClassImplementation,
    IsolationForestRegImplementation,
    LinearRegRANSACImplementation,
    NonLinearRegRANSACImplementation,
)
from fedot.core.operations.implementations.data_operations.sklearn_selectors import (
    LinearClassFSImplementation,
    LinearRegFSImplementation,
    NonLinearClassFSImplementation,
    NonLinearRegFSImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.utilities.random import ImplementationRandomStateHandler


warnings.filterwarnings('ignore', category=UserWarning)


class ClassificationDataOperationStrategy(EvaluationStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for classification task
    """

    _operations_by_types = {
        'rfe_lin_class': LinearClassFSImplementation,
        'rfe_non_lin_class': NonLinearClassFSImplementation,
        'class_decompose': DecomposerClassImplementation,
        'resample': ResampleImplementation,
        'isolation_forest_class': IsolationForestClassImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

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
        Transform data for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """
        prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform data for fit stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


class RegressionDataOperationStrategy(EvaluationStrategy):
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
