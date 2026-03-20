from golem.utilities.data_structures import ComparableEnum as Enum
from dataclasses import dataclass
from fedot.core.backend.backend import backend
from typing import Optional, Any
from fedot.core.data.complex_types import ArrayType, IndexType



class LabelEncoder:

    categories_ = {}
    categorical_idx_ = None

    def fit(self, data: ArrayType, categorical_idx: IndexType):
        xp = backend.xp

        self.categorical_idx_ = list(categorical_idx)
        self.categories_ = {}

        for idx in self.categorical_idx_:
            column = data[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

        return self

    def transform(self, data: ArrayType):
        xp = backend.xp

        n_rows = data.shape[0]
        n_cat = len(self.categorical_idx_)

        encoded = xp.full((n_rows, n_cat), xp.nan, dtype=float)

        for j, idx in enumerate(self.categorical_idx_):
            column = data[:, idx]
            categories = self.categories_[idx]

            nan_mask = column != column

            if categories.size > 0:
                matches = column.reshape(-1, 1) == categories.reshape(1, -1)

                matched_rows = matches.any(axis=1)

                encoded[matched_rows, j] = matches.argmax(axis=1)[matched_rows].astype(float)

            encoded[nan_mask, j] = xp.nan

        return encoded

    def fit_transform(self, data: ArrayType, categorical_idx: IndexType):
        return self.fit(data, categorical_idx).transform(data)


class OneHotEncoder:

    categories_ = {}
    categorical_idx_ = None
    feature_slices_ = None
    n_output_features_ = None

    def fit(self, data: ArrayType, categorical_idx: IndexType):
        xp = backend.xp

        self.categorical_idx_ = list(categorical_idx)
        self.categories_ = {}
        self.feature_slices_ = {}

        start = 0
        for idx in self.categorical_idx_:
            column = data[:, idx]

            nan_mask = column != column
            valid_values = column[~nan_mask]

            categories = xp.unique(valid_values)
            self.categories_[idx] = categories

            end = start + int(categories.size)
            self.feature_slices_[idx] = slice(start, end)
            start = end

        self.n_output_features_ = start
        return self

    def transform(self, data: ArrayType):
        xp = backend.xp

        n_rows = data.shape[0]
        encoded = xp.full((n_rows, self.n_output_features_), xp.nan, dtype=float)

        for idx in self.categorical_idx_:
            column = data[:, idx]
            categories = self.categories_[idx]
            feature_slice = self.feature_slices_[idx]

            nan_mask = column != column

            if categories.size == 0:
                continue

            block = (column.reshape(-1, 1) == categories.reshape(1, -1)).astype(float)
            block[nan_mask, :] = xp.nan

            encoded[:, feature_slice] = block

        return encoded

    def fit_transform(self, data: ArrayType, categorical_idx: IndexType):
        return self.fit(data, categorical_idx).transform(data)
