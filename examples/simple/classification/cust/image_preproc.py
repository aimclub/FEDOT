import warnings
from typing import Optional

from examples.simple.classification.cust.preproc_impls import GammaFiltImplementation
from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler


class ImagePreprocessingStrategy(EvaluationStrategy):
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

    _operations_by_types = {
        'gamma_filt': GammaFiltImplementation,
        'negamma_filt': GammaFiltImplementation,
        'yet_anothe_filt': GammaFiltImplementation

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
        with ImplementationRandomStateHandler(implementation=operation_implementation):
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