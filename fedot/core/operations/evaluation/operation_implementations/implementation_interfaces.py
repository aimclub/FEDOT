from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
from golem.core.log import default_log

from fedot.core.data.data import InputData, OutputData
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
                           data_type: DataTypesEnum = DataTypesEnum.table) -> OutputData:
        """ Method prepare prediction of operation as OutputData object """

        converted = _convert_to_output_function(input_data, predict, data_type)
        return converted


class EncodedInvariantImplementation(DataOperationImplementation):
    """ Class for processing data without transforming encoded features.
    Encoded features - features after OneHot encoding operation, when one
    feature (with categorical values) can be represented as several boolean
    vectors
    """

    def __init__(self, params: Optional[OperationParameters]):
        super().__init__(params)
        self.operation = None
        self.ids_to_process = None
        self.bool_ids = None

    def fit(self, input_data: InputData):
        """ Method for fit transformer with automatic determination
        of boolean features, with which there is no need to make transformation

        :param input_data: data with features, target and ids to process
        :return operation: trained transformer (optional output)
        """

        if input_data.task.task_type.name == 'ts_forecasting' and input_data.features.ndim == 2:
            features = input_data.features.ravel()
        else:
            features = input_data.features

        # Find boolean columns in features table
        bool_ids, ids_to_process = self._reasonability_check(features)
        self.ids_to_process = ids_to_process
        self.bool_ids = bool_ids
        if len(ids_to_process) > 0:
            if isinstance(features, np.ndarray):
                if input_data.task.task_type.name == 'ts_forecasting' and input_data.features.ndim == 2:
                    features = features.reshape(-1, 1)

                features_to_process = np.array(features[:, ids_to_process]) if features.ndim > 1 else features
            else:
                features_to_process = np.array(features.iloc[:, ids_to_process]) if features.ndim > 1 else features
            self.operation.fit(features_to_process)
        return self.operation

    def transform(self, input_data: InputData) -> OutputData:
        """
        The method that transforms the source features using "operation" for predict stage

        :param input_data: tabular data with features, target and ids to process
        :return output_data: output data with transformed features table
        """
        source_features_shape = input_data.features.shape
        features = input_data.features
        if len(self.ids_to_process) > 0:
            transformed_features = self._make_new_table(features)
        else:
            transformed_features = features

        transformed_features = np.nan_to_num(transformed_features, copy=False, nan=0, posinf=0, neginf=0)

        # Update features and column types
        output_data = self._convert_to_output(input_data, transformed_features)
        self._update_column_types(source_features_shape, output_data)
        return output_data

    def _make_new_table(self, features):
        """
        The method creates a table based on transformed data and source boolean
        features

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """
        if isinstance(features, np.ndarray):
            features_to_process = np.array(features[:, self.ids_to_process]) if features.ndim > 1 else features.copy()
        else:
            features_to_process = np.array(
                features.iloc[:, self.ids_to_process]
            ) if features.ndim > 1 else features.copy()

        transformed_part = self.operation.transform(features_to_process)

        # If there are no binary features in the dataset
        if len(self.bool_ids) == 0:
            transformed_features = transformed_part
        else:
            # Stack transformed features and bool features
            if isinstance(features, np.ndarray):
                bool_features = np.array(features[:, self.bool_ids])
            else:
                bool_features = np.array(features[self.bool_ids])

            frames = (bool_features, transformed_part)
            transformed_features = np.hstack(frames)

        return transformed_features

    def _update_column_types(self, source_features_shape, output_data: OutputData) -> OutputData:
        """
        Update column types after applying operations.
        If new columns added, new type for them are defined
        """
        return output_data

    @staticmethod
    def _reasonability_check(features):
        """
        Method for checking which columns contain boolean data

        :param features: tabular data for check
        :return bool_ids: indices of boolean columns in table
        :return non_bool_ids: indices of non boolean columns in table
        """
        # TODO perhaps there is a more effective way to do this
        source_shape = features.shape

        columns_amount = source_shape[1] if len(source_shape) > 1 else 1

        # Indices of boolean columns in features table
        bool_ids = []
        non_bool_ids = []

        # For every column in table make check
        for column_id in range(columns_amount):
            if isinstance(features, np.ndarray):
                column = features[:, column_id] if columns_amount > 1 else features.copy()
            else:
                column = features.iloc[:, column_id] if columns_amount > 1 else features.copy()

            if (isinstance(column, pd.Series) and len(set(column)) > 2) or \
               (isinstance(column, np.ndarray) and len(np.unique(column)) > 2):
                non_bool_ids.append(column_id)
            else:
                bool_ids.append(column_id)

        return bool_ids, non_bool_ids


class ModelImplementation(ABC):
    """ Interface for models realisations methods
    Contains abstract methods, which should be implemented for applying EA
    optimizer on it
    """

    def __init__(self, params: OperationParameters = None):
        self.log = default_log(self)
        self.params = params or OperationParameters()

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
                           data_type: DataTypesEnum = DataTypesEnum.table):
        """ Method prepare prediction of operation as OutputData object """

        converted = _convert_to_output_function(input_data, predict, data_type)
        return converted


def _convert_to_output_function(input_data: InputData, transformed_features: np.ndarray,
                                data_type: DataTypesEnum = DataTypesEnum.table):
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
