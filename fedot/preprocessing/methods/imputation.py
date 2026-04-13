import torch
from typing import Sequence, Optional

from fedot.core.data.prepared_data import PreparedData


class MeanImputation:
    def __init__(self):
        self.mean_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Compute mean values for selected columns (ignoring NaNs).

        Args:
            data: torch.Tensor of shape (n_samples, n_features)
            features_idx: indices of columns to impute

        Returns:
            self
        """
        self.features_idx = features_idx
        selected = data[:, self.features_idx]
        self.mean_values = torch.nanmean(selected, dim=0)
        return self

    def transform(self, data: PreparedData):
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

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Fit and transform in one step.
        """
        return self.fit(data.features, features_idx).transform(data)


class MedianImputation:
    def __init__(self):
        self.median_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        self.features_idx = features_idx
        selected = data[:, self.features_idx]
        self.median_values = torch.nanquantile(selected, q=0.5, dim=0)
        return self

    def transform(self, data: PreparedData):
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

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


class ModeImputation:
    def __init__(self):
        self.mode_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        self.features_idx = features_idx
        selected = data[:, self.features_idx]

        modes = []
        for i in range(selected.shape[1]):
            column = selected[:, i]
            valid = column[~torch.isnan(column)]

            if valid.numel() == 0:
                modes.append(torch.tensor(float('nan'), device=data.device))
            else:
                values, counts = torch.unique(valid, return_counts=True)
                mode = values[counts.argmax()]
                modes.append(mode)

        self.mode_values = torch.stack(modes)
        return self

    def transform(self, data: PreparedData):
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

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


class ConstantImputation:
    def __init__(self, constant: float = 0.0):
        self.constant = constant
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        self.features_idx = features_idx
        return self

    def transform(self, data: PreparedData):
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

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


class DeleteRawImputation:
    def __init__(self):
        self.features_idx: Optional[Sequence[int]] = None
        self.valid_rows_mask: Optional[torch.Tensor] = None
    
    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        self.features_idx = features_idx

        nan_mask = torch.isnan(data[:, self.features_idx])
        rows_with_nan = nan_mask.any(dim=1)
        self.valid_rows_mask = ~rows_with_nan

        return self

    def transform(self, data: PreparedData):
        if self.valid_rows_mask is None:
            raise RuntimeError("DeleteRawImputation is not fitted yet.")

        data.features = data.features[self.valid_rows_mask]
        if data.target is not None:
            data.target = data.target[self.valid_rows_mask]
        return data
    
    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)
