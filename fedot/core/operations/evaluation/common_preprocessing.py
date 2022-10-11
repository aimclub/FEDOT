import warnings
from typing import Optional

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import \
    OneHotEncodingImplementation, LabelEncodingImplementation
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import \
    ImputationImplementation, KernelPCAImplementation, NormalizationImplementation, PCAImplementation, \
    PolyFeaturesImplementation, ScalingImplementation, FastICAImplementation
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.utilities.random import RandomStateHandler

warnings.filterwarnings("ignore", category=UserWarning)


class FedotPreprocessingStrategy(EvaluationStrategy):
    """
    Args:
        operation_type: ``str`` of the operation defined in operation or data operation repositories

            .. details:: possible operations:

                - ``scaling``-> ScalingImplementation,
                - ``normalization``-> NormalizationImplementation,
                - ``simple_imputation``-> ImputationImplementation,
                - ``pca``-> PCAImplementation,
                - ``kernel_pca``-> KernelPCAImplementation,
                - ``poly_features``-> PolyFeaturesImplementation,
                - ``one_hot_encoding``-> OneHotEncodingImplementation,
                - ``label_encoding``-> LabelEncodingImplementation,
                - ``fast_ica``-> FastICAImplementation

        params: hyperparameters to fit the operation with

    """

    __operations_by_types = {
        'scaling': ScalingImplementation,
        'normalization': NormalizationImplementation,
        'simple_imputation': ImputationImplementation,
        'pca': PCAImplementation,
        'kernel_pca': KernelPCAImplementation,
        'poly_features': PolyFeaturesImplementation,
        'one_hot_encoding': OneHotEncodingImplementation,
        'label_encoding': LabelEncodingImplementation,
        'fast_ica': FastICAImplementation
    }

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """This method is used for operation training with the data provided

        Args:
            train_data: data used for operation training

        Returns:
            trained Sklearn operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        operation_implementation = self.operation_impl(self.params_for_fit)
        with RandomStateHandler():
            operation_implementation.fit(train_data)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """Transform method for preprocessing task

        Args:
            trained_operation: model object
            predict_data: data used for prediction

        Returns:
            prediction
        """
        prediction = trained_operation.transform(predict_data)
        # Convert prediction to output (if it is required)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        Transform method for preprocessing task for fit stage

        Args:
            trained_operation: model object
            predict_data: data used for prediction
        Returns:
            OutputData: 
        """
        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom preprocessing strategy for {operation_type}')
