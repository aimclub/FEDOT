from copy import deepcopy
from typing import Optional
import bisect

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import str_columns_check
from fedot.core.operations.evaluation.operation_implementations.implementation_interfaces import \
    DataOperationImplementation


class OneHotEncodingImplementation(DataOperationImplementation):
    """ Class for automatic categorical data detection and one hot encoding """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        default_params = {
            'handle_unknown': 'ignore'
        }
        if not params:
            # Default parameters
            self.encoder = OneHotEncoder(**default_params)
        else:
            self.encoder = OneHotEncoder(**{**params, **default_params})
        self.categorical_ids = None
        self.non_categorical_ids = None

    def fit(self, input_data: InputData):
        """ Method for fit encoder with automatic determination of categorical features

        :param input_data: data with features, target and ids for encoder training
        :return encoder: trained encoder (optional output)
        """
        features = input_data.features
        categorical_ids, non_categorical_ids = str_columns_check(features)

        # Indices of columns with categorical and non-categorical features
        self.categorical_ids = categorical_ids
        self.non_categorical_ids = non_categorical_ids

        # If there are categorical features - process it
        if self.categorical_ids:
            updated_cat_features = np.array(features[:, self.categorical_ids], dtype=str)
            self.encoder.fit(updated_cat_features)

        return self.encoder

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        """
        The method that transforms the categorical features in the original
        dataset, but does not affect the rest features

        :param input_data: data with features, target and ids for transformation
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
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
        return output_data

    def _apply_one_hot_encoding(self, features: np.array):
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

        return transformed_features

    def get_params(self):
        return self.encoder.get_params()


class LabelEncodingImplementation(DataOperationImplementation):
    """ Class for categorical features encoding based on LabelEncoding """

    def __init__(self, **params: Optional[dict]):
        super().__init__()
        # LabelEncoder has no parameters
        self.encoders = {}
        self.categorical_ids = None
        self.non_categorical_ids = None

    def fit(self, input_data: InputData):

        self.categorical_ids, self.non_categorical_ids = str_columns_check(input_data.features)

        # If there are categorical features - process it
        if self.categorical_ids:
            # For every categorical feature - perform encoding
            self._fit_label_encoders(input_data)
        return self.encoders

    def transform(self, input_data, is_fit_pipeline_stage: Optional[bool]):
        """ Apply LabelEncoder on categorical features and doesn't process float or int ones """
        copied_data = deepcopy(input_data)
        if self.categorical_ids:
            # If categorical features are exists - transform them inplace in InputData
            for categorical_id in self.categorical_ids:
                self._apply_label_encoder(copied_data, categorical_id)

        # Update features
        output_data = self._convert_to_output(copied_data,
                                              copied_data.features)
        # Store source features values
        output_data.features = input_data.features
        return output_data

    def _fit_label_encoders(self, input_data: InputData):
        """ Fit LabelEncoder for every categorical column in the dataset """
        for categorical_id in self.categorical_ids:
            categorical_column = input_data.features[:, categorical_id]
            le = LabelEncoder()
            le.fit(categorical_column)

            self.encoders.update({categorical_id: le})

    def _apply_label_encoder(self, input_data: InputData, categorical_id: int):
        """ Apply fitted LabelEncoder for column transformation

        :param input_data: data with features to transform
        :param categorical_id: index of current categorical column
        """
        column_encoder = self.encoders[categorical_id]

        # Transform categorical feature into numerical one
        categorical_column = input_data.features[:, categorical_id]
        try:
            input_data.features[:, categorical_id] = column_encoder.transform(categorical_column)
        except ValueError as ex:
            # y contains previously unseen labels
            encoder_classes = list(column_encoder.classes_)
            message = str(ex)
            unseen_label = message.split("\'")[1]

            # Extent encoder classes
            bisect.insort_left(encoder_classes, unseen_label)
            column_encoder.classes_ = encoder_classes
            self._apply_label_encoder(input_data, categorical_id)

    def get_params(self):
        """ Due to LabelEncoder has no parameters - return empty set """
        return {}
