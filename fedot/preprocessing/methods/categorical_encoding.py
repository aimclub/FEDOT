from typing import Sequence

import numpy as np

from fedot.core.backend.backend import Backend
from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


"""
How to add a new categorical encoder
--------------------------------------
1. Implement the encoder class in this module (e.g. `class MyEncoder:`).
   It must use `backend.xp` for array operations so it works on both CPU (NumPy) and
   GPU (CuPy).
2. Implement the methods expected by the preprocessing pipeline:
   `fit(data, categorical_idx)`, `transform(data)`, and (optionally) `fit_transform(data, categorical_idx)`.
3. To support previously unseen optional-imputation constants, implement
   `encode_imputation_value(column_idx, fill_value)`: encode the category used to
   fill a missing value, using the fitted vocabulary of that source column.
4. Add a new enum value to `EncodingMethodEnum` in `fedot/preprocessing/tools/preprocessor_types.py`.
5. Register the handler in `PREPROCESSING_OBLIGATORY_MAPPING` in
   `fedot/preprocessing/tools/methods_mapping.py`.
"""


class LabelEncoder(AbstractPreprocessingHandler):
    """
    Label encoder for categorical feature columns.

    The encoder builds a per-column mapping from observed category values to
    integer identifiers and then replaces original categorical values with
    numeric labels. Mapping is learned independently for every selected column.

    Behavior details:
    - category vocabulary is inferred from non-missing values during `fit`;
    - transformed known values are dense numeric ids in column-wise order;
    - unseen non-missing values are mapped to a per-column unknown label;
    - missing values are kept as `NaN`.

    The transformed categorical block keeps shape
    `(n_samples, n_selected_categorical_columns)`.
    """

    def __init__(self):
        """Initialize `LabelEncoder`."""
        self.categories_ = {}
        self.unknown_labels_ = {}
        self.categorical_idx_ = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Learn category sets for each categorical column.

        Args:
            data: Feature matrix wrapper.
            features_idx: Indices of categorical columns to encode.

        Returns:
            LabelEncoder: Fitted encoder instance.
        """
        xp = Backend().xp

        features = data.features

        self.categorical_idx_ = list(features_idx)
        self.categories_ = {}
        self.unknown_labels_ = {}

        for idx in self.categorical_idx_:
            column = features[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)

            self.categories_[idx] = categories
            self.unknown_labels_[idx] = float(categories.size)

        return self

    def encode_imputation_value(self, column_idx: int, fill_value) -> np.ndarray:
        """Return the fitted label code for a categorical imputation replacement.

        ``column_idx`` is the column position used when fitting this encoder;
        ``fill_value`` is the original category to put in place of a missing value.
        Return a one-element NumPy array. An unseen category uses the learned
        unknown ID; a missing replacement stays NaN. The vocabulary is unchanged.
        """
        categories = self.categories_[column_idx]
        categories = categories.get() if hasattr(categories, 'get') else categories
        matches = np.flatnonzero(categories == fill_value)
        code = float(matches[0]) if len(matches) else self.unknown_labels_[column_idx]
        return np.array([np.nan if fill_value is None or fill_value != fill_value else code])

    def transform(self, data: PreparedData) -> PreparedData:
        """
        Transform categorical values to label-encoded numeric IDs.

        Args:
            data: Feature matrix wrapper.

        Returns:
            PreparedData: Data with encoded categorical columns.
        """
        xp = Backend().xp

        features = data.features

        n_rows = features.shape[0]
        n_cat = len(self.categorical_idx_)

        encoded = xp.full((n_rows, n_cat), xp.nan, dtype=float)

        for j, idx in enumerate(self.categorical_idx_):
            column = features[:, idx]
            categories = self.categories_[idx]
            unknown_label = self.unknown_labels_[idx]

            nan_mask = column != column
            known_mask = xp.zeros(n_rows, dtype=bool)

            if categories.size > 0:
                matches = column.reshape(-1, 1) == categories.reshape(1, -1)
                known_mask = matches.any(axis=1)

                encoded[known_mask, j] = matches.argmax(
                    axis=1
                )[known_mask].astype(float)

            unknown_mask = ~known_mask & ~nan_mask
            encoded[unknown_mask, j] = unknown_label
            encoded[nan_mask, j] = xp.nan

        features[:, self.categorical_idx_] = encoded

        data.features = features

        return data

    def inverse_transform(self, data: PreparedData) -> PreparedData:
        """
        Inverse transform label-encoded numeric IDs to categorical values.

        Known labels are mapped back with fitted ``categories_``.
        Missing values stay ``NaN``. Unknown labels (unseen at fit) become ``NaN``.

        Args:
            data: Feature matrix wrapper with label-encoded categorical columns.

        Returns:
            PreparedData: Data with decoded categorical columns.
        """
        if self.categorical_idx_ is None:
            raise RuntimeError('LabelEncoder is not fitted yet.')

        xp = Backend().xp
        features = data.features
        n_rows = features.shape[0]

        # Decoded categories may be strings; promote numeric matrices to object.
        if getattr(features.dtype, 'kind', None) in ('f', 'i', 'u', 'b'):
            features = features.astype(object)

        for idx in self.categorical_idx_:
            column = features[:, idx]
            categories = self.categories_[idx]

            nan_mask = column != column
            decoded = xp.full(n_rows, xp.nan, dtype=object)

            for label_id in range(int(categories.size)):
                mask = ~nan_mask & (column == float(label_id))
                decoded[mask] = categories[label_id]

            features[:, idx] = decoded

        data.features = features
        return data


class OneHotEncoder(AbstractPreprocessingHandler):
    """
    One-hot encoder for categorical feature columns.

    The encoder learns unique values for each selected categorical column and
    expands them into binary indicator features. During fitting, it also
    precomputes output slices to place each column's one-hot block into a
    shared output matrix.

    Behavior details:
    - each categorical column is encoded independently;
    - original categorical columns are removed from the feature matrix;
    - one-hot blocks are concatenated to the right side of remaining features;
    - missing values in source categorical columns remain `NaN` in the
      corresponding one-hot block.

    Final encoded block shape is `(n_samples, n_output_features_)`.
    """

    def __init__(self):
        """Initialize `OneHotEncoder`."""
        self.categories_ = {}
        self.categorical_idx_ = None
        self.feature_slices_ = None
        self.n_output_features_ = None
        self.new_cols_dict = {}

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Learn categories and output slices for each categorical column.

        Args:
            data (ArrayType): Feature matrix of shape `(n_samples, n_features)`.
            categorical_idx (IndexType): Indices of categorical columns to encode.

        Returns:
            OneHotEncoder: Fitted encoder instance.
        """

        xp = Backend().xp

        features = data.features

        self.categorical_idx_ = list(features_idx)
        self.categories_ = {}
        self.feature_slices_ = {}

        start = 0
        for idx in self.categorical_idx_:
            column = features[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

            # Keep an all-missing source addressable for later imputation and
            # native categorical models, even without an observed vocabulary.
            self.new_cols_dict[idx] = max(1, int(categories.size))

            end = start + self.new_cols_dict[idx]
            self.feature_slices_[idx] = slice(start, end)
            start = end

        self.n_output_features_ = start
        return self

    def encode_imputation_value(self, column_idx: int, fill_value) -> np.ndarray:
        """Return the fitted one-hot vector for a categorical imputation replacement.

        ``column_idx`` is the column position used when fitting this encoder;
        ``fill_value`` is the original category to put in place of a missing value.
        Return a NumPy vector of the existing block width. An unseen category
        produces zeros; a missing replacement produces NaNs. No columns are added.
        """
        categories = self.categories_[column_idx]
        categories = categories.get() if hasattr(categories, 'get') else categories
        pattern = np.zeros(self.new_cols_dict[column_idx])
        matches = np.flatnonzero(categories == fill_value)
        if fill_value is None or fill_value != fill_value:
            pattern[:] = np.nan
        elif len(matches):
            pattern[matches[0]] = 1
        return pattern

    def transform(self, data: PreparedData) -> PreparedData:
        """
        Transform categorical values to one-hot encoded features.

        Args:
            data (ArrayType): Feature matrix of shape `(n_samples, n_features)`.

        Returns:
            ArrayType: One-hot encoded array of shape
                `(n_samples, self.n_output_features_)`.
        """
        xp = Backend().xp

        features = data.features
        n_rows = features.shape[0]
        encoded = xp.full((n_rows, self.n_output_features_),
                          xp.nan, dtype=float)

        for idx in self.categorical_idx_:
            column = features[:, idx]
            categories = self.categories_[idx]
            feature_slice = self.feature_slices_[idx]

            nan_mask = column != column

            if categories.size == 0:
                encoded[~nan_mask, feature_slice] = 0
                continue

            block = (column.reshape(-1, 1) ==
                     categories.reshape(1, -1)).astype(float)
            block[nan_mask, :] = xp.nan

            encoded[:, feature_slice] = block

        features = xp.delete(features, self.categorical_idx_, axis=1)
        features = xp.hstack((features, encoded))

        data.features = features
        data.new_cols_dict = self.new_cols_dict

        return data
