from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import (
    DataOperationImplementation
)
from fedot.core.operations.operation_parameters import OperationParameters
from fedot.preprocessing.data_types import TYPE_TO_ID
from fedot.utilities.memory import reduce_mem_usage


class OneHotEncodingImplementation(DataOperationImplementation):
    """ Class for automatic categorical data detection and one hot encoding """

    def __init__(self, params: Optional[OperationParameters] = None):
        super().__init__(params)
        default_params = {
            'handle_unknown': 'ignore'
        }
        self.encoder = OneHotEncoder(**{**default_params, **self.params.to_dict()})
        self.categorical_ids: np.ndarray = np.array([])
        self.non_categorical_ids: np.ndarray = np.array([])
        self.encoded_ids: np.ndarray = np.array([])
        self.new_numerical_idx: np.ndarray = np.array([])

    def fit(self, input_data: InputData):
        """ Method for fit encoder with automatic determination of categorical features

        :param input_data: data with features, target and ids for encoder fitting
        :return encoder: encoder (optional output)
        """
        features = input_data.features
        self.categorical_ids, self.non_categorical_ids = input_data.categorical_idx, input_data.numerical_idx

        # If there are categorical features - process it
        if self.categorical_ids.size > 0:
            if isinstance(features, np.ndarray):
                updated_cat_features = features[:, self.categorical_ids].astype(str)
            else:
                updated_cat_features = features.iloc[:, self.categorical_ids].astype(str)

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

        transformed_features = copied_data.features
        if self.categorical_ids.size > 0:
            # If categorical features exist
            transformed_features = self._apply_one_hot_encoding(transformed_features)

        # Update features
        output_data = self._convert_to_output(copied_data, transformed_features)
        self._update_column_types(output_data)

        if isinstance(output_data.features, pd.DataFrame):
            output_data.predict = reduce_mem_usage(
                transformed_features,
                output_data.supplementary_data.col_type_ids['features']
            )

        return output_data

    def _update_column_types(self, output_data: OutputData):
        """ Update column types after encoding. Categorical columns becomes integer with extension """
        if self.categorical_ids.size > 0:
            # There are categorical features in the table
            feature_type_ids = output_data.supplementary_data.col_type_ids['features']
            numerical_columns = feature_type_ids[feature_type_ids != TYPE_TO_ID[str]]

            # Calculate new binary columns number after encoding
            encoded_columns_number = output_data.predict.shape[1] - len(numerical_columns)
            numerical_columns = np.append(numerical_columns, [TYPE_TO_ID[int]] * encoded_columns_number)

            output_data.encoded_idx = self.encoded_ids
            output_data.supplementary_data.col_type_ids['features'] = numerical_columns

    def _apply_one_hot_encoding(self, features: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        The method creates a table based on categorical and real features after One Hot Encoding transformation

        :param features: tabular data for processing
        :return transformed_features: transformed features table
        """
        if isinstance(features, np.ndarray):
            transformed_categorical = self.encoder.transform(features[:, self.categorical_ids]).toarray()
            # Stack transformed categorical and non-categorical data, ignore if none
            non_categorical_features = features[:, self.non_categorical_ids.astype(int)]

        else:
            transformed_categorical = self.encoder.transform(features.iloc[:, self.categorical_ids]).toarray()
            non_categorical_features = features.iloc[:, self.non_categorical_ids.astype(int)].to_numpy()

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
        self.categorical_ids: np.ndarray = np.array([])
        self.non_categorical_ids: np.ndarray = np.array([])

    def fit(self, input_data: InputData):
        self.categorical_ids, self.non_categorical_ids = input_data.categorical_idx, input_data.numerical_idx

        # For every existing categorical feature - perform encoding
        self._fit_label_encoders(input_data.features)
        return self.encoders

    def transform(self, input_data: InputData) -> OutputData:
        """ Apply LabelEncoder on categorical features and doesn't process float or int ones
        Applicable during predict stage
        """
        copied_data = deepcopy(input_data)
        # If categorical features exist - transform them inplace in InputData
        self._apply_label_encoder(copied_data.features)

        output_data = self._convert_to_output(copied_data,
                                              copied_data.features)

        self._update_column_types(output_data)
        return output_data

    def _update_column_types(self, output_data: OutputData):
        """ Update column types after encoding. Categorical becomes integer """
        feature_type_ids = output_data.supplementary_data.col_type_ids['features']
        feature_type_ids[self.categorical_ids] = TYPE_TO_ID[int]

    def _fit_label_encoders(self, data: Union[np.ndarray, pd.DataFrame]):
        """ Fit LabelEncoder for every categorical column in the dataset """
        if isinstance(data, np.ndarray):
            categorical_columns = data[:, self.categorical_ids].astype(str)

            for column_id, column in zip(self.categorical_ids, categorical_columns.T):
                le = LabelEncoder()
                le.fit(column)
                self.encoders[column_id] = le

        else:
            categorical_columns = data.iloc[:, self.categorical_ids].astype(str)

            for column_id in self.categorical_ids:
                le = LabelEncoder()
                le.fit(categorical_columns.iloc[:, column_id])
                self.encoders[column_id] = le

    def _apply_label_encoder(self, data: Union[np.ndarray, pd.DataFrame]):
        """
        Applies fitted LabelEncoder for all categorical features inplace

        Args:
            data: numpy array with all features
        """
        if isinstance(data, np.ndarray):
            categorical_columns = data[:, self.categorical_ids].astype(str)

            for column_id, column in zip(self.categorical_ids, categorical_columns.T):
                column_encoder = self.encoders[column_id]
                column_encoder.classes_ = np.unique(np.concatenate((column_encoder.classes_, column)))

                transformed_column = column_encoder.transform(column)
                nan_indices = np.flatnonzero(column == 'nan')
                if len(nan_indices):
                    # Store np.nan values
                    transformed_column = transformed_column.astype(object)
                    transformed_column[nan_indices] = np.nan

                data[:, column_id] = transformed_column
        else:
            categorical_columns = data.iloc[:, self.categorical_ids].astype(str)

            for column_id in self.categorical_ids:
                column_encoder = self.encoders[column_id]
                column = categorical_columns[column_id]
                column_encoder.classes_ = np.unique(np.concatenate((column_encoder.classes_, column)))

                transformed_column = column_encoder.transform(column)
                nan_indices = np.flatnonzero(column == 'nan')
                if len(nan_indices):
                    # Store np.nan values
                    transformed_column = transformed_column.astype(object)
                    transformed_column[nan_indices] = np.nan

                data.iloc[:, column_id] = transformed_column

    def get_params(self) -> OperationParameters:
        """ Due to LabelEncoder has no parameters - return empty set """
        return OperationParameters()
