from typing import Optional, Sequence
import torch

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class StandartScaling(AbstractPreprocessingHandler):
    """
    Standard score scaling for selected columns with NaN-aware statistics.

    The handler computes per-column mean and standard deviation on `fit`,
    skipping missing values, and applies z-score transformation on `transform`:
    `(x - mean) / std`.

    Configuration flags allow disabling centering and/or scaling to mirror
    standard scaler behavior in different preprocessing scenarios.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """Initialize `StandartScaling`."""
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_values: Optional[torch.Tensor] = None
        self.scale_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute mean and std for selected columns (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data.features[:, self.features_idx]

        if self.with_mean or self.with_std:
            mask = ~torch.isnan(selected)

        if self.with_mean:
            sum_vals = torch.where(mask, selected, torch.zeros_like(selected)).sum(dim=0)
            count = mask.sum(dim=0).clamp(min=1)
            mean = sum_vals / count
            self.mean_values = mean
        else:
            self.mean_values = None

        if self.with_std:
            if self.mean_values is not None:
                mean = self.mean_values
            else:
                sum_vals = torch.where(mask, selected, torch.zeros_like(selected)).sum(dim=0)
                count = mask.sum(dim=0).clamp(min=1)
                mean = sum_vals / count

            diff = selected - mean
            diff = torch.where(mask, diff, torch.zeros_like(diff))

            var = (diff ** 2).sum(dim=0) / mask.sum(dim=0).clamp(min=1)
            std = torch.sqrt(var)

            std = torch.where(std == 0, torch.ones_like(std), std)

            self.scale_values = std
        else:
            self.scale_values = None

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.features_idx is None:
            raise RuntimeError("ScalingImplementation is not fitted yet.")

        selected = data.features[:, self.features_idx]

        if self.with_mean and self.mean_values is not None:
            selected = selected - self.mean_values

        if self.with_std and self.scale_values is not None:
            selected = selected / self.scale_values

        data.features[:, self.features_idx] = selected
        return data


class MinMaxNormalization(AbstractPreprocessingHandler):
    """Min-max normalization for selected feature columns.

    On `fit`, the handler estimates per-column minimum and maximum values
    (ignoring NaNs), then derives scaling coefficients for the configured output
    range. On `transform`, each selected value is linearly mapped to
    `feature_range`.

    Columns with zero data range are handled safely by replacing zero divisors
    with ones.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        """Initialize `MinMaxNormalization`."""
        self.min_range = feature_range[0]
        self.max_range = feature_range[1]

        self.data_min: Optional[torch.Tensor] = None
        self.data_max: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None

        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute min and max for selected columns (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data.features[:, self.features_idx]

        mask = ~torch.isnan(selected)

        pos_inf = torch.full_like(selected, float("inf"))
        masked_min = torch.where(mask, selected, pos_inf)
        self.data_min = masked_min.min(dim=0).values

        neg_inf = torch.full_like(selected, float("-inf"))
        masked_max = torch.where(mask, selected, neg_inf)
        self.data_max = masked_max.max(dim=0).values

        data_range = self.data_max - self.data_min

        data_range = torch.where(data_range == 0, torch.ones_like(data_range), data_range)

        self.scale = (self.max_range - self.min_range) / data_range

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.features_idx is None:
            raise RuntimeError("NormalizationImplementation is not fitted yet.")

        selected = data.features[:, self.features_idx]

        # X_scaled = (X - min) * scale + min_range
        selected = (selected - self.data_min) * self.scale + self.min_range

        data.features[:, self.features_idx] = selected
        return data


class RobustScaling(AbstractPreprocessingHandler):
    """Robust scaler based on median and inter-quantile range (IQR).

    This handler is less sensitive to outliers than standard scaling. During
    `fit`, it computes per-column median (optional) and quantile-based scale
    (`q_max - q_min`) for selected columns while ignoring NaNs. During
    `transform`, it applies centering and scaling according to enabled flags.

    Designed for tabular data where heavy tails or extreme values can distort
    mean/std-based normalization.
    """

    def __init__(
        self,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        """Initialize `RobustScaling`."""
        self.q_min = quantile_range[0] / 100.0
        self.q_max = quantile_range[1] / 100.0
        self.with_centering = with_centering
        self.with_scaling = with_scaling

        self.center_values: Optional[torch.Tensor] = None
        self.scale_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute median and IQR for selected columns (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data.features[:, self.features_idx]

        if self.with_centering:
            self.center_values = torch.nanquantile(selected, 0.5, dim=0)
        else:
            self.center_values = None

        if self.with_scaling:
            q_low = torch.nanquantile(selected, self.q_min, dim=0)
            q_high = torch.nanquantile(selected, self.q_max, dim=0)

            scale = q_high - q_low
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            self.scale_values = scale
        else:
            self.scale_values = None

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Transform input data with fitted state."""
        if self.features_idx is None:
            raise RuntimeError("RobustScaling is not fitted yet.")

        selected = data.features[:, self.features_idx]

        if self.with_centering and self.center_values is not None:
            selected = selected - self.center_values

        if self.with_scaling and self.scale_values is not None:
            selected = selected / self.scale_values

        data.features[:, self.features_idx] = selected
        return data
