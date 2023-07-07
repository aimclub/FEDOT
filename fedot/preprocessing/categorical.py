from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedot.core.data.data import InputData
from fedot.core.data.data_preprocessing import find_categorical_columns
from fedot.preprocessing.data_types import TYPE_TO_ID, FEDOT_STR_NAN


class BinaryCategoricalPreprocessor:
    """ Class for categories features preprocessing: converting binary string features into integers. """

    def __init__(self):
        self.binary_encoders = {}
        self.binary_ids_to_convert = []

        # List with binary categorical features ids which contain Nans
        self.binary_features_with_nans = []

    def fit(self, input_data: InputData):
        """
        Find indices of columns which are contains categorical values. Binary features and at the same time
        has str objects. If there are such features - convert it into int
        """
        feature_type_ids = input_data.supplementary_data.column_types['features']
        categorical_ids, _ = find_categorical_columns(input_data.features,
                                                      feature_type_ids)

        binary_ids_to_convert = []
        for column_id, column in zip(categorical_ids, input_data.features[:, categorical_ids].T):
            pd_column = pd.Series(column, name=column_id, copy=True)
            is_nan = pd_column.isna()
            column_nuniques = pd_column.nunique(False)
            if is_nan.sum():
                # This categorical column has nans
                pd_column[is_nan] = FEDOT_STR_NAN

                if column_nuniques <= 3:
                    # There is column with binary categories and gaps
                    self.binary_features_with_nans.append(column_id)
                    binary_ids_to_convert.append(column_id)
                    self._train_encoder(pd_column)
            elif column_nuniques <= 2:
                # Column contains binary string feature
                binary_ids_to_convert.append(column_id)
                # Train encoder for current column
                self._train_encoder(pd_column)

        self.binary_ids_to_convert = binary_ids_to_convert
        return self

    def transform(self, input_data: InputData) -> InputData:
        """
        Apply transformation (converting str into integers) for selected (while training) features.
        """
        if len(self.binary_ids_to_convert) == 0:
            # There are no binary categorical features
            return input_data

        copied_data = deepcopy(input_data)
        self._apply_encoder(copied_data.features)

        # Update features types
        feature_types = copied_data.supplementary_data.column_types['features']
        feature_types[self.binary_ids_to_convert] = TYPE_TO_ID[int]
        return copied_data

    def fit_transform(self, input_data: InputData) -> InputData:
        """
        Applies firstly :method:`fit` and then :method:`transform` methods of the class

        Args:
            input_data: to be trained on and transformed

        Returns:
            transformed ``input_data``
        """
        self.fit(input_data)
        return self.transform(input_data)

    def _train_encoder(self, column: pd.Series):
        """ Convert labels in the column from string into int via Label encoding.
        So, Label encoder is fitted to do such transformation.
        """
        encoder = LabelEncoder()
        encoder.fit(column)

        # Store fitted label encoder for transform method
        self.binary_encoders[column.name] = encoder

    def _apply_encoder(self, data: np.ndarray):
        """
        Applies already fitted encoders to all binary features inplace

        Args:
            data: numpy array with all features
        """
        binary_columns = data[:, self.binary_ids_to_convert]
        for column_id, column in zip(self.binary_ids_to_convert, binary_columns.T):
            encoder = self.binary_encoders[column_id]
            nan_idxs = np.flatnonzero(pd.isna(column))
            column[nan_idxs] = FEDOT_STR_NAN
            # Extend encoder classes if the column contains categories not previously encountered
            encoder.classes_ = np.unique(np.concatenate((encoder.classes_, column)))

            converted = encoder.transform(column)
            if len(nan_idxs):
                # Column has nans in its structure - after conversion replace it
                converted = converted.astype(float)
                converted[nan_idxs] = np.nan
            data[:, column_id] = converted
