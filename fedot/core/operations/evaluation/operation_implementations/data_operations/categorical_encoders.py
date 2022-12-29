from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from fedot.core.data.data import InputData, OutputData
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    DataOperationImplementation
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.preprocessing.data_types import TYPE_TO_ID


class OneHotEncodingImplementation(DataOperationImplementation):
    """ Class for automatic categorical data detection and one hot encoding """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        default_params = {
            'handle_unknown': 'ignore'
        }
        self.encoder = OneHotEncoder(**{**default_params, **self.params.to_dict()})
        self.categorical_ids = None
        self.non_categorical_ids = None
        self.encoded_ids = None
        self.new_numerical_idx = None

    def fit(self, input_data: InputData):
        """ Method for fit encoder with automatic determination of categorical features

        :param input_data: data with features, target and ids for encoder training
        :return encoder: trained encoder (optional output)
        """
        features = input_data.features
        features_type_ids = input_data.supplementary_data.column_types.get('features')
        categorical_ids, non_categorical_ids = find_categorical_columns(features,
                                                                        features_type_ids)

        # Indices of columns with categorical and non-categorical features
        self.categorical_ids = categorical_ids
        self.non_categorical_ids = non_categorical_ids

        # If there are categorical features - process it
        if self.categorical_ids:
            updated_cat_features = np.array(features[:, self.categorical_ids], dtype=str)
            self.encoder.fit(updated_cat_features)

        return self.encoder

    def transform(self, input_data: InputData) -> OutputData:
        """
        The method that transforms the categorical features in the original
        dataset, but does not affect the rest features. Applicable during predict stage

        :param input_data: data with features, target and ids for transformation
        :return output_data: output data with transformed features table
        """
        copied_data = deepcopy(input_data)

        features = copied_data.features
        if not self.categorical_ids:
            # If there are no categorical features in the table
            transformed_features = features
        else:
            # If categorical features are exists
            transformed_features = self._apply_one_hot_encoding(features)

        # Update features
        output_data = self._convert_to_output(copied_data,
                                              transformed_features)
        self._update_column_types(output_data)
        return output_data

    def _update_column_types(self, output_data: OutputData):
        """ Update column types after encoding. Categorical columns becomes integer with extension """
        if self.categorical_ids:
            # There are categorical features in the table
            col_types = output_data.supplementary_data.column_types['features']
            numerical_columns = [t_name for t_name in col_types if t_name != TYPE_TO_ID[str]]

            # Calculate new binary columns number after encoding
            encoded_columns_number = output_data.predict.shape[1] - len(numerical_columns)
            numerical_columns.extend([TYPE_TO_ID[int]] * encoded_columns_number)

            output_data.encoded_idx = self.encoded_ids
            output_data.supplementary_data.column_types['features'] = numerical_columns

    def _apply_one_hot_encoding(self, features: np.ndarray) -> np.ndarray:
        """
        The method creates a table based on categorical and real features after One Hot Encoding transformation

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """

        categorical_features = np.array(features[:, self.categorical_ids])
        transformed_categorical = self.encoder.transform(categorical_features).toarray()

        # If there are non-categorical features in the data
        if not self.non_categorical_ids:
            transformed_features = transformed_categorical
        else:
            # Stack transformed categorical and non-categorical data
            non_categorical_features = np.array(features[:, self.non_categorical_ids])
            frames = (non_categorical_features, transformed_categorical)
            transformed_features = np.hstack(frames)
            self.encoded_ids = np.array(range(non_categorical_features.shape[1], transformed_features.shape[1]))

        return transformed_features


class LabelEncodingImplementation(DataOperationImplementation):
    """ Class for categorical features encoding based on LabelEncoding """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        # LabelEncoder has no parameters
        self.encoders = {}
        self.categorical_ids = None
        self.non_categorical_ids = None

    def fit(self, input_data: InputData):
        features_types = input_data.supplementary_data.column_types.get('features')
        self.categorical_ids, self.non_categorical_ids = find_categorical_columns(input_data.features,
                                                                                  features_types)

        # If there are categorical features - process it
        if self.categorical_ids:
            # For every categorical feature - perform encoding
            self._fit_label_encoders(input_data)
        return self.encoders

    def transform(self, input_data: InputData) -> OutputData:
        """ Apply LabelEncoder on categorical features and doesn't process float or int ones
        Applicable during predict stage
        """
        copied_data = deepcopy(input_data)
        if self.categorical_ids:
            # If categorical features are exists - transform them inplace in InputData
            for categorical_id in self.categorical_ids:
                categorical_column = input_data.features[:, categorical_id]
                has_nan: np.ndarray = pd.isna(categorical_column)

                transformed = self._apply_label_encoder(categorical_column, categorical_id, has_nan)
                copied_data.features[:, categorical_id] = transformed

        output_data = self._convert_to_output(copied_data,
                                              copied_data.features)

        self._update_column_types(output_data)
        return output_data

    def _update_column_types(self, output_data: OutputData):
        """ Update column types after encoding. Categorical becomes integer """
        if self.categorical_ids:
            # Categorical features were in the dataset
            col_types = output_data.supplementary_data.column_types['features']
            for categorical_id in self.categorical_ids:
                col_types[categorical_id] = TYPE_TO_ID[int]

            output_data.supplementary_data.column_types['features'] = col_types

    def _fit_label_encoders(self, input_data: InputData):
        """ Fit LabelEncoder for every categorical column in the dataset """
        for categorical_id in self.categorical_ids:
            categorical_column = input_data.features[:, categorical_id]
            le = LabelEncoder()
            le.fit(categorical_column)

            self.encoders.update({categorical_id: le})

    def _apply_label_encoder(self, categorical_column: np.ndarray, categorical_id: int,
                             has_nan: np.ndarray) -> np.ndarray:
        """ Apply fitted LabelEncoder for column transformation

        :param categorical_column: numpy array with categorical features
        :param categorical_id: index of current categorical column
        :param has_nan: bool array of gap elements in the ``categorical_column``
        """
        column_encoder = self.encoders[categorical_id]
        column_encoder.classes_ = pd.unique(np.concatenate((column_encoder.classes_, categorical_column)))

        transformed_column = column_encoder.transform(categorical_column)
        if len(has_nan) > 0:
            # Store np.nan values
            transformed_column = transformed_column.astype(object)
            transformed_column[has_nan] = np.nan

        return transformed_column

    def get_params(self) -> OperationParameters:
        """ Due to LabelEncoder has no parameters - return empty set """
        return OperationParameters()
