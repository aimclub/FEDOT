import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, SkLearnEvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.decompose \
    import DecomposerClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_filters \
    import IsolationForestClassImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_imbalanced_class import \
    ResampleImplementation
from fedot.core.operations.evaluation.operation_implementations. \
    data_operations.sklearn_selectors import LinearClassFSImplementation, NonLinearClassFSImplementation
from fedot.core.operations.evaluation.operation_implementations.models. \
    discriminant_analysis import LDAImplementation, QDAImplementation
from fedot.core.operations.evaluation.operation_implementations.models. \
    keras import FedotCNNImplementation
from fedot.core.operations.evaluation.operation_implementations.models.knn import FedotKnnClassImplementation
from fedot.core.operations.evaluation.operation_implementations.models.svc import FedotSVCImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utilities.random import RandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnClassificationStrategy(SkLearnEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for classification task for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """

        prediction = self._sklearn_compatible_prediction(trained_operation=trained_operation,
                                                         features=predict_data.features)
        converted = self._convert_to_output(prediction, predict_data)
        return converted


class FedotClassificationStrategy(EvaluationStrategy):
    __operations_by_types = {
        'lda': LDAImplementation,
        'qda': QDAImplementation,
        'svc': FedotSVCImplementation,
        'cnn': FedotCNNImplementation,
        'knn': FedotKnnClassImplementation
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

        with RandomStateHandler():
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Predict method for classification task for predict stage

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :return: prediction target
        """
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs' and len(prediction.shape) > 1:
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain Fedot Classification Strategy for {operation_type}')


class FedotClassificationPreprocessingStrategy(EvaluationStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for classification task
    """

    __operations_by_types = {
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
        with RandomStateHandler():
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

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom classification preprocessing strategy for {operation_type}')
