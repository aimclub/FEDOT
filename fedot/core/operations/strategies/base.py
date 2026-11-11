from abc import abstractmethod
from dataclasses import replace
from typing import Optional
import warnings

from golem.core.log import default_log

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import get_operation_type_from_id
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


warnings.filterwarnings('ignore', category=UserWarning)


class EvaluationStrategy:
    """Base class to define the evaluation strategy of Operation object:
    the certain sklearn or any other operation with fit/predict methods.

    Args:
        operation_type: ``str`` of the operation defined in operation repository
        params: hyperparameters to fit the operation with
    """

    def __init__(self, operation_type: str, params: Optional[OperationParameters] = None):
        self.params_for_fit = params or OperationParameters()
        self.operation_id = operation_type
        self.output_mode = False

        self.log = default_log(self)

    @property
    def operation_type(self):
        return get_operation_type_from_id(self.operation_id)

    @abstractmethod
    def fit(self, train_data: InputData):
        """Main method to train the operation with the data provided

        Args:
            train_data: data used for operation training

        Returns:

        """
        raise AbstractMethodNotImplementError(self.__class__)

    @abstractmethod
    def predict(self, trained_operation, predict_data: InputData) -> OutputData:
        """Method to predict the target data for predict stage.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict

        Returns:
            passed data with new predicted target
        """
        raise AbstractMethodNotImplementError

    def predict_for_fit(self, trained_operation, predict_data: InputData) -> OutputData:
        """Method to predict the target data for fit stage.
        Allows to implement predict method different from main predict method
        if another behaviour for fit graph stage is needed.

        Args:
            trained_operation: trained operation object
            predict_data: data to predict
        Returns:
            passed data with new predicted target
        """
        return self.predict(trained_operation, predict_data)

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self._operations_by_types:
            return self._operations_by_types[operation_type]
        else:
            raise ValueError(
                f'Impossible to obtain {self.__class__} strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))

    @staticmethod
    def _convert_to_output(prediction, predict_data: InputData,
                           output_data_type: DataTypesEnum = DataTypesEnum.tabular) -> OutputData:
        """Method convert prediction into :obj:`OutputData` if it is not this type yet

        Args:
            prediction: output from model implementation
            predict_data: :obj:`InputData` used for prediction
            output_data_type: :obj:`DataTypesEnum` for output

        Returns: prediction as :obj:`OutputData`
        """

        if isinstance(predict_data, TensorData):
            return EvaluationStrategy._replace_predict_in_tensor_data(prediction, predict_data)

        if not isinstance(prediction, OutputData):
            # Wrap prediction as OutputData
            converted = OutputData(idx=predict_data.idx,
                                   features=predict_data.features,
                                   predict=prediction,
                                   task=predict_data.task,
                                   target=predict_data.target,
                                   data_type=output_data_type,
                                   supplementary_data=predict_data.supplementary_data)
        else:
            converted = prediction

        return converted

    @staticmethod
    def _replace_predict_in_tensor_data(prediction, predict_data: TensorData) -> TensorData:
        """Attach model prediction to TensorData, analogous to OutputData wrapping."""
        return replace(predict_data, predict=prediction)

    @staticmethod
    def _replace_features_in_tensor_data(features, tensor_data: TensorData) -> TensorData:
        """Attach transformed features to TensorData and keep predict empty."""
        return replace(tensor_data, features=features, predict=None)
