import torch
from typing import Sequence, Optional

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


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
                modes.append(torch.tensor(float('nan'), device=features.device))
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
                torch.tensor(self.constant, device=data.features.device, dtype=data.features.dtype),
                column
            )

        return data


class DeleteRawImputation(AbstractPreprocessingHandler):
    """Row-wise missing-value strategy that removes rows containing NaNs.

    The handler builds a row validity mask in `fit` for selected columns and, in
    `transform`, filters both `features` and `target` (if present) by that mask.
    This keeps feature-target alignment while discarding incomplete samples.

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

        data.features = data.features[self.valid_rows_mask]
        if data.target is not None:
            data.target = data.target[self.valid_rows_mask]
        return data
