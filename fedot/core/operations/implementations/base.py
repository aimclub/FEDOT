from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

from golem.core.log import default_log
import numpy as np

from fedot.core.data.input_data.data import InputData, OutputData
from fedot.core.data.tensor_data.tensor_data import TensorData
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.utilities.custom_errors import AbstractMethodNotImplementError


class DataOperationImplementation(ABC):
    """ Interface for data operations realisations methods
    Contains abstract methods, which should be implemented for applying EA
    optimizer on it
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params or OperationParameters()
        self.log = default_log(self)

    @abstractmethod
    def fit(self, input_data: InputData):
        """ Method fit operation on a dataset

        :param input_data: data with features, target and ids to process
        """
        raise AbstractMethodNotImplementError

    @abstractmethod
    def transform(self, input_data: InputData) -> OutputData:
        """ Method apply transform operation on a dataset for predict stage

        :param input_data: data with features, target and ids to process
        """
        raise AbstractMethodNotImplementError

    def transform_for_fit(self, input_data: InputData) -> OutputData:
        """ Method apply transform operation on a dataset for fit stage.
        Allows to implement transform method different from main transform method
        if another behaviour for fit graph stage is needed.

        :param input_data: data with features, target and ids to process
        """
        return self.transform(input_data)

    def get_params(self) -> OperationParameters:
        """ Method return parameters, which can be optimized for particular
        operation
        """
        return deepcopy(self.params)

    @staticmethod
    def _convert_to_output(input_data: InputData, predict: np.ndarray,
                           data_type: DataTypesEnum = DataTypesEnum.tabular) -> OutputData:
        """ Method prepare prediction of operation as OutputData object """

        converted = _convert_to_output_function(input_data, predict, data_type)
        return converted


class ModelImplementation(ABC):
    """ Interface for models realisations methods
    Contains abstract methods, which should be implemented for applying EA
    optimizer on it
    """

    def __init__(self, params: OperationParameters = None):
        self.log = default_log(self)
        self.params = params or OperationParameters()
        self.is_multi_target: bool = False

    @abstractmethod
    def fit(self, input_data: InputData):
        """ Method fit model on a dataset

        :param input_data: data with features, target and ids to process
        """
        raise AbstractMethodNotImplementError

    @abstractmethod
    def predict(self, input_data: InputData) -> OutputData:
        """ Method make prediction

        :param input_data: data with features, target and ids to process
        """
        raise AbstractMethodNotImplementError

    def predict_for_fit(self, input_data: InputData) -> OutputData:
        """ Method make prediction while graph fitting.
        Allows to implement predict method different from main predict method
        if another behaviour for fit graph stage is needed.

        :param input_data: data with features, target and ids to process
        """
        return self.predict(input_data)

    def get_params(self) -> OperationParameters:
        """ Method return parameters, which can be optimized for particular
        operation
        """
        return deepcopy(self.params)

    @staticmethod
    def _convert_to_output(input_data: InputData, predict: np.array,
                           data_type: DataTypesEnum = DataTypesEnum.tabular):
        """ Method prepare prediction of operation as OutputData object """

        converted = _convert_to_output_function(input_data, predict, data_type)
        return converted


def _convert_to_output_function(input_data: InputData, transformed_features: np.ndarray,
                                data_type: DataTypesEnum = DataTypesEnum.tabular):
    """ Function prepare prediction of operation as OutputData object

    :param input_data: data with features, target and ids to process
    :param transformed_features: transformed features
    :param data_type: type of output data
    """

    # After preprocessing operations by default we get tabular data
    converted = OutputData(idx=input_data.idx,
                           features=input_data.features,
                           predict=transformed_features,
                           task=input_data.task,
                           target=input_data.target,
                           data_type=data_type,
                           numerical_idx=input_data.numerical_idx,
                           categorical_idx=input_data.categorical_idx,
                           encoded_idx=input_data.encoded_idx,
                           categorical_features=input_data.categorical_features,
                           features_names=input_data.features_names,
                           supplementary_data=input_data.supplementary_data)

    return converted


class TensorDataOperationImplementation(ABC):
    """Contract for TensorData transform operations used as pipeline nodes.

    Mirror of :class:`DataOperationImplementation` for the ``TensorData`` path:
    one searchable operation / node with ``fit`` + ``transform``.

    Fit-graph vs predict-graph behaviour is owned by
    :class:`~fedot.core.operations.strategies.base.EvaluationStrategy.predict_for_fit`
    (see :class:`~fedot.core.operations.strategies.data_operations.projection.ProjectionDataOperationStrategy`),
    not by a second method on this contract. Override strategy-level
    ``predict_for_fit`` only when fit-stage transform must differ (leakage-prone
    ops); until then it delegates to ``transform`` via ``predict``.
    """

    def __init__(self, params: Optional[OperationParameters] = None):
        self.params = params or OperationParameters()
        self.log = default_log(self)

    @abstractmethod
    def fit(self, data: TensorData):
        """Fit the operation on ``data``."""
        raise AbstractMethodNotImplementError()

    @abstractmethod
    def transform(self, data: TensorData) -> TensorData:
        """Apply the fitted operation."""
        raise AbstractMethodNotImplementError()

    def get_params(self) -> OperationParameters:
        """Detached params snapshot for node updates (does not share mutable state).

        Uses :meth:`OperationParameters.copy` (shallow value copy + changed-keys).
        Hyperparameters are scalars/strings; nested mutables are not expected.
        """
        return self.params.copy()
