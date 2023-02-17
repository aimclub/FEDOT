from copy import deepcopy
from typing import Tuple

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
        features_types = input_data.supplementary_data.column_types['features']
        categorical_ids, _ = find_categorical_columns(table=input_data.features,
                                                      column_types=features_types)
        if len(categorical_ids) == 0:
            # There is no need to process categorical features
            return self

        binary_ids_to_convert = []
        for column_id, column in enumerate(input_data.features.T):
            pd_column = pd.Series(column, copy=True)
            is_nan = pd_column.isna()
            column_uniques = pd_column.unique()
            if is_nan.sum() and column_id in categorical_ids:
                # This categorical column has nans
                pd_column[is_nan] = FEDOT_STR_NAN

                if len(column_uniques) <= 3:
                    # There is column with binary categories and gaps
                    self.binary_features_with_nans.append(column_id)
                    binary_ids_to_convert.append(column_id)
                    self._train_encoder(pd_column, column_id)
            else:
                if len(column_uniques) <= 2 and column_id in categorical_ids:
                    # Column contains binary string feature
                    binary_ids_to_convert.append(column_id)
                    # Train encoder for current column
                    self._train_encoder(pd_column, column_id)

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
        for column_id, column in enumerate(copied_data.features.T):
            if column_id in self.binary_ids_to_convert:
                # If column contains nans - replace them with fedot nans special string
                nan_idxs: Tuple[np.ndarray, ...] = pd.isna(column).nonzero()
                column[nan_idxs] = FEDOT_STR_NAN

                # Convert into integers
                column[:] = self._apply_encoder(column, column_id, nan_idxs)

        # Update features types
        features_types = copied_data.supplementary_data.column_types['features']
        for converted_column_id in self.binary_ids_to_convert:
            features_types[converted_column_id] = TYPE_TO_ID[int]
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

    def _train_encoder(self, column: pd.Series, column_id: int):
        """ Convert labels in the column from string into int via Label encoding.
        So, Label encoder is fitted to do such transformation.
        """
        encoder = LabelEncoder()
        encoder.fit(column)

        # Store fitted label encoder for transform method
        self.binary_encoders.update({column_id: encoder})

    def _apply_encoder(self, column: np.ndarray, column_id: int, nan_idxs: Tuple[np.ndarray, ...]) -> np.ndarray:
        """ Apply already fitted encoders """
        encoder = self.binary_encoders[column_id]
        # Extend encoder classes if the column contains categories not previously encountered
        encoder.classes_ = np.unique(np.concatenate((encoder.classes_, column)))

        converted = encoder.transform(column)
        if len(nan_idxs[0]):
            # Column has nans in its structure - after conversion replace it
            converted = converted.astype(float)
            converted[nan_idxs] = np.nan

        return converted
