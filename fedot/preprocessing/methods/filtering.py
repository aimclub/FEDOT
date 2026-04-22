from typing import Optional, Sequence
import torch

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class QuantileClipping(AbstractPreprocessingHandler):
    def __init__(
        self,
        quantile_range: tuple[float, float] = (1.0, 99.0),
    ):
        self.q_min = quantile_range[0] / 100.0
        self.q_max = quantile_range[1] / 100.0

        self.lower_bounds: Optional[torch.Tensor] = None
        self.upper_bounds: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute clipping bounds based on quantiles (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data.features[:, self.features_idx]

        self.lower_bounds = torch.nanquantile(selected, self.q_min, dim=0)
        self.upper_bounds = torch.nanquantile(selected, self.q_max, dim=0)

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.features_idx is None:
            raise RuntimeError("Clipping is not fitted yet.")

        selected = data.features[:, self.features_idx]
        selected = torch.clamp(selected, self.lower_bounds, self.upper_bounds)
        data.features[:, self.features_idx] = selected
        return data


class VarianceThreshold(AbstractPreprocessingHandler):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

        self.features_idx: Optional[Sequence[int]] = None
        self.variances: Optional[torch.Tensor] = None
        self.support_mask: Optional[torch.Tensor] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute variances for selected columns (ignoring NaNs) and mark
        columns with variance > threshold to keep.

        Args:
            data: torch.Tensor of shape (n_samples, n_features)
            features_idx: indices of columns to evaluate

        Returns:
            self
        """
        self.features_idx = features_idx

        features = data.features
        selected = features[:, self.features_idx]

        mask = ~torch.isnan(selected)
        count = mask.sum(dim=0)

        safe_selected = torch.where(mask, selected, torch.zeros_like(selected))
        sum_values = safe_selected.sum(dim=0)

        mean = sum_values / count.clamp(min=1).to(selected.dtype)

        centered = torch.where(mask, selected - mean, torch.zeros_like(selected))
        var = (centered ** 2).sum(dim=0) / count.clamp(min=1).to(selected.dtype)

        # Если в колонке нет ни одного валидного значения, считаем variance = 0
        var = torch.where(count > 0, var, torch.zeros_like(var))

        self.variances = var
        selected_support = var > self.threshold

        n_features = features.shape[1]
        support_mask = torch.ones(n_features, dtype=torch.bool, device=features.device)
        support_mask[list(self.features_idx)] = selected_support

        self.support_mask = support_mask
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.support_mask is None:
            raise RuntimeError("VarianceThreshold is not fitted yet.")

        data.features = data.features[:, self.support_mask]
        return data
