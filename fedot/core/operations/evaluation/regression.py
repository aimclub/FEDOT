import warnings

from typing import Optional

from fedot.core.data.data import InputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, SkLearnEvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.\
    data_operations.sklearn_filters import LinearRegRANSACImplementation, NonLinearRegRANSACImplementation
from fedot.core.operations.evaluation.operation_implementations.\
    data_operations.sklearn_selectors import LinearRegFSImplementation, NonLinearRegFSImplementation
from fedot.core.operations.evaluation.operation_implementations.models.knn import CustomKnnRegImplementation
warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnRegressionStrategy(SkLearnEvaluationStrategy):
    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for regression task
        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """

        prediction = trained_operation.predict(predict_data.features)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)

        return converted


class CustomRegressionPreprocessingStrategy(EvaluationStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for regression task
    """

    __operations_by_types = {
        'ransac_lin_reg': LinearRegRANSACImplementation,
        'ransac_non_lin_reg': NonLinearRegRANSACImplementation,
        'rfe_lin_reg': LinearRegFSImplementation,
        'rfe_non_lin_reg': NonLinearRegFSImplementation,
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Transform method for preprocessing

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """
        prediction = trained_operation.transform(predict_data, is_fit_chain_stage)

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain Custom Regression Preprocessing Strategy for {operation_type}')


class CustomRegressionStrategy(EvaluationStrategy):
    """
    Strategy for applying custom regression models from FEDOT make predictions
    """

    __operations_by_types = {
        'knnreg': CustomKnnRegImplementation
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """ This method is used for operation training """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """ Predict method for regression models """
        prediction = trained_operation.predict(predict_data, is_fit_chain_stage)

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain Custom Regression Strategy for {operation_type}')
