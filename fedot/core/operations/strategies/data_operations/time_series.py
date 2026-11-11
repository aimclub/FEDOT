import warnings
from typing import Optional
from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.operations.strategies.base import EvaluationStrategy
from fedot.core.operations.implementations.data_operations.ts_transformations import (
    ExogDataTransformationImplementation,
    GaussianFilterImplementation,
    LaggedTransformationImplementation,
    TsSmoothingImplementation,
    SparseLaggedTransformationImplementation,
    CutImplementation,
    NumericalDerivativeFilterImplementation,
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.utilities.random import ImplementationRandomStateHandler

warnings.filterwarnings('ignore', category=UserWarning)


class TimeSeriesDataOperationStrategy(EvaluationStrategy):
    """
    This class defines the certain data operation implementation for time series
    forecasting

    Args:
        operation_type: str type of the operation defined in operation or
            data operation repositories
        params: hyperparameters to fit the model with
    """

    _operations_by_types = {
        'lagged': LaggedTransformationImplementation,
        'sparse_lagged': SparseLaggedTransformationImplementation,
        'smoothing': TsSmoothingImplementation,
        'exog_ts': ExogDataTransformationImplementation,
        'gaussian_filter': GaussianFilterImplementation,
        'diff_filter': NumericalDerivativeFilterImplementation,
        'cut': CutImplementation}

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        super().__init__(operation_type, params)
        self.operation = self._convert_to_operation(self.operation_type)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained operation (if it is needed for applying)
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        transformation_operation = self.operation(self.params_for_fit)
        with ImplementationRandomStateHandler(implementation=transformation_operation):
            transformation_operation.fit(train_data)
        return transformation_operation

    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during predict stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """
        This method used for prediction of the target data during fit stage.

        :param trained_operation: trained operation object
        :param predict_data: data to predict
        :return OutputData: passed data with new predicted target
        """

        prediction = trained_operation.transform_for_fit(predict_data)
        converted = self._convert_to_output(prediction, predict_data)
        return converted
