import cloudpickle
import numpy as np
import torch
from typing import Sequence, Optional

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler
from fedot.preprocessing.tools.preprocessor_types import ImputationMethodEnum
from fedot.core.data.tensor_data.contracts import select_rows


class MeanImputation(AbstractPreprocessingHandler):
    """Missing-value imputer that fills NaNs with column means.

    During `fit`, the handler computes a mean value for each selected feature
    column using non-missing observations only. During `transform`, every NaN in
    those columns is replaced with the corresponding learned mean.

    Intended for continuous numeric features where mean-based replacement is an
    acceptable approximation.
    """

    def __init__(self):
        """Initialize `MeanImputation`."""
        self.mean_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute mean values for selected columns (ignoring NaNs).

        Args:
            data: torch.Tensor of shape (n_samples, n_features)
            features_idx: indices of columns to impute

        Returns:
            self
        """
        self.features_idx = features_idx
        selected = data.features[:, self.features_idx]
        self.mean_values = torch.nanmean(selected, dim=0)
        return self

    def transform(self, data: PreparedData):
        """Transform input data with fitted state."""
        if self.mean_values is None or self.features_idx is None:
            raise RuntimeError("MeanImputation is not fitted yet.")

        for i, col_idx in enumerate(self.features_idx):
            column = data.features[:, col_idx]
            data.features[:, col_idx] = torch.where(
                torch.isnan(column),
                self.mean_values[i],
                column
            )

        return data


class MedianImputation(AbstractPreprocessingHandler):
    """Missing-value imputer that fills NaNs with column medians.

    The handler stores per-column medians on `fit` and uses them on `transform`
    to replace missing values in selected columns. Median-based imputation is
    generally more robust to outliers than mean imputation.
    """

    def __init__(self):
        """Initialize `MedianImputation`."""
        self.median_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Fit the handler on input data."""
        self.features_idx = features_idx
        selected = data.features[:, self.features_idx]
        self.median_values = torch.nanquantile(selected, q=0.5, dim=0)
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.median_values is None or self.features_idx is None:
            raise RuntimeError("MedianImputation is not fitted yet.")

        for i, col_idx in enumerate(self.features_idx):
            column = data.features[:, col_idx]
            data.features[:, col_idx] = torch.where(
                torch.isnan(column),
                self.median_values[i],
                column
            )

        return data


class ModeImputation(AbstractPreprocessingHandler):
    """Missing-value imputer that fills NaNs with the most frequent value.

    For each selected column, `fit` finds the mode over non-missing values.
    During `transform`, missing entries are replaced with that mode. If a column
    has no valid values at fit time, its fallback mode remains `NaN`.

    Useful for discrete or category-like numeric columns.
    """

    def __init__(self):
        """Initialize `ModeImputation`."""
        self.mode_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Fit the handler on input data."""
        self.features_idx = features_idx
        features = data.features
        selected = features[:, self.features_idx]

        modes = []
        for i in range(selected.shape[1]):
            column = selected[:, i]
            valid = column[~torch.isnan(column)]

            if valid.numel() == 0:
                modes.append(torch.tensor(
                    float('nan'), device=features.device))
            else:
                values, counts = torch.unique(valid, return_counts=True)
                mode = values[counts.argmax()]
                modes.append(mode)

        self.mode_values = torch.stack(modes)
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.mode_values is None or self.features_idx is None:
            raise RuntimeError("ModeImputation is not fitted yet.")

        for i, col_idx in enumerate(self.features_idx):
            column = data.features[:, col_idx]
            data.features[:, col_idx] = torch.where(
                torch.isnan(column),
                self.mode_values[i],
                column
            )

        return data


class ConstantImputation(AbstractPreprocessingHandler):
    """Missing-value imputer that replaces NaNs with a user constant.

    The handler does not estimate statistics from data; it only stores selected
    feature indices in `fit` and writes the configured constant value to NaN
    positions during `transform`.

    Suitable for deterministic replacement policies and controlled experiments.
    """

    def __init__(self, constant: float = 0.0):
        """Initialize `ConstantImputation`."""
        self.constant = constant
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Fit the handler on input data."""
        self.features_idx = features_idx
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.features_idx is None:
            raise RuntimeError("ConstantImputation is not fitted yet.")

        for col_idx in self.features_idx:
            column = data.features[:, col_idx]
            data.features[:, col_idx] = torch.where(
                torch.isnan(column),
                torch.tensor(self.constant, device=data.features.device,
                             dtype=data.features.dtype),
                column
            )

        return data


class DeleteRawImputation(AbstractPreprocessingHandler):
    """Row-wise missing-value strategy that removes rows containing NaNs.

    Each transform computes its own mask and filters all sample-wise fields.
    Training row positions must never be reused for a different prediction batch.

    Use this strategy when dropping corrupted rows is preferable to value
    imputation.
    """

    def __init__(self):
        """Initialize `DeleteRawImputation`."""
        self.features_idx: Optional[Sequence[int]] = None
        self.valid_rows_mask: Optional[torch.Tensor] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Fit the handler on input data."""
        self.features_idx = features_idx

        nan_mask = torch.isnan(data.features[:, self.features_idx])
        rows_with_nan = nan_mask.any(dim=1)
        self.valid_rows_mask = ~rows_with_nan

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.valid_rows_mask is None:
            raise RuntimeError("DeleteRawImputation is not fitted yet.")

        self.valid_rows_mask = ~torch.isnan(data.features[:, self.features_idx]).any(dim=1)
        data.features = data.features[self.valid_rows_mask]
        if data.target is not None:
            data.target = data.target[self.valid_rows_mask]
        data.categorical_features = select_rows(data.categorical_features, self.valid_rows_mask)
        data.row_idx = select_rows(data.row_idx, self.valid_rows_mask)
        return data


class CategoricalImputation(AbstractPreprocessingHandler):
    """Wrap a numeric imputer for steps that also select stored categories.

    Replacement patterns are taken from the training tensor, preserving earlier
    transformations of its encoded columns. A previously unseen constant is
    encoded with the fitted source encoder, without fitting a new vocabulary.
    """

    def __init__(self, numeric_handler, groups, method, preparation_state, changed_sources):
        self.numeric_handler = numeric_handler
        self.groups = groups
        self.method = method
        self.preparation_state = preparation_state
        self.changed_sources = set(changed_sources)
        self.replacements = []

    def fit(self, data, features_idx):
        if self.method not in (ImputationMethodEnum.mode, ImputationMethodEnum.constant):
            raise ValueError('Original categorical features support mode, constant, or row deletion imputation.')
        constant = self.numeric_handler.constant if self.method == ImputationMethodEnum.constant else None
        categorical_indices = {pos for _, positions in self.groups for pos in positions}
        numeric_indices = [pos for pos in features_idx if pos not in categorical_indices]
        if numeric_indices:
            self.numeric_handler.fit(data, numeric_indices)
        else:
            self.numeric_handler = None
        self.replacements = []
        for column, positions in self.groups:
            values = data.categorical_features[:, column]
            valid = values == values  # Source preparation normalizes missing categories to np.nan.
            if self.method == ImputationMethodEnum.mode:
                categories, counts = np.unique(values[valid], return_counts=True)
                if not len(categories):
                    continue  # An entirely missing training source has no mode.
                value = categories[counts.argmax()]
            else:
                value = np.nan if constant is None else constant
            matches = np.flatnonzero(valid & (values == value))
            if len(matches):
                pattern = data.features[int(matches[0]), positions].clone()
            else:
                source = data.categorical_source_idx[column]
                if source in self.changed_sources:
                    raise ValueError('An unseen categorical constant requires an unmodified encoded source block.')
                pattern = self._encode_constant(source, value, data.features)
                if len(pattern) != len(positions):
                    raise ValueError('An unseen categorical constant requires the complete unmodified encoded block.')
            self.replacements.append((column, positions, value, pattern))
        # Fitted encoders are only needed to prepare replacement patterns.
        self.preparation_state = None
        return self

    def _encode_constant(self, source, value, features):
        if self.preparation_state is None:
            raise ValueError('An unseen categorical constant requires fitted source encoding metadata.')
        fitted = next(step for step in self.preparation_state.encoding_steps if source in step.source_indices)
        handler = cloudpickle.loads(fitted.handler_state)
        column_idx = fitted.indices[fitted.source_indices.index(source)]
        pattern = handler.encode_imputation_value(column_idx, value)
        return torch.as_tensor(pattern, device=features.device, dtype=features.dtype)

    def transform(self, data):
        if self.numeric_handler is not None:
            data = self.numeric_handler.transform(data)
        for column, positions, value, pattern in self.replacements:
            values = data.categorical_features[:, column]
            missing = values != values
            data.categorical_features[missing, column] = value
            rows = torch.as_tensor(np.flatnonzero(missing), device=data.features.device)
            columns = torch.as_tensor(positions, device=data.features.device)
            data.features[rows[:, None], columns] = pattern.to(data.features)
        return data
