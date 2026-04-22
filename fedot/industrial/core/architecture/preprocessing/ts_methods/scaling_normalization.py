from typing import Optional, Sequence
import torch

from fedot.core.data.prepared_data import PreparedData
from fedot.industrial.core.architecture.preprocessing.ts_methods.tools import (
    flatten_if_needed, 
    restore_if_needed,
    expand_features_idx_for_flatten)
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class SeasonalNormalization(AbstractPreprocessingHandler):
    def __init__(
        self,
        period: int,
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        if period <= 0:
            raise ValueError("period must be > 0")

        self.period = period
        self.with_centering = with_centering
        self.with_scaling = with_scaling

        self.seasonal_mean: Optional[torch.Tensor] = None
        self.seasonal_std: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None
        self._original_shape: Optional[torch.Size] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute seasonal mean/std for selected columns along sample axis.

        For each seasonal position p in [0, period), statistics are computed
        from rows p, p + period, p + 2 * period, ...
        NaNs are ignored.
        """
        self._original_shape = data.ts_shape
        self.features_idx = expand_features_idx_for_flatten(
            features_idx, self._original_shape)

        flat_data = flatten_if_needed(data.features)
        selected = flat_data[:, self.features_idx]  # shape: [A, F]

        _, n_features = selected.shape
        device = selected.device
        dtype = selected.dtype

        seasonal_mean = torch.zeros(
            (self.period, n_features), dtype=dtype, device=device
        )
        seasonal_std = torch.ones(
            (self.period, n_features), dtype=dtype, device=device
        )

        for p in range(self.period):
            phase_values = selected[p::self.period]  # shape: [A_p, F]
            if phase_values.shape[0] == 0:
                continue

            mask = ~torch.isnan(phase_values)
            count = mask.sum(dim=0)

            safe_values = torch.where(mask, phase_values, torch.zeros_like(phase_values))
            sum_values = safe_values.sum(dim=0)

            mean = torch.where(
                count > 0,
                sum_values / count.clamp(min=1).to(dtype),
                torch.zeros(n_features, dtype=dtype, device=device),
            )

            if self.with_scaling:
                centered = torch.where(mask, phase_values - mean, torch.zeros_like(phase_values))
                var = torch.where(
                    count > 0,
                    (centered ** 2).sum(dim=0) / count.clamp(min=1).to(dtype),
                    torch.zeros(n_features, dtype=dtype, device=device),
                )
                std = torch.sqrt(var)
                std = torch.where(std == 0, torch.ones_like(std), std)
            else:
                std = torch.ones(n_features, dtype=dtype, device=device)

            seasonal_mean[p] = mean
            seasonal_std[p] = std

        self.seasonal_mean = seasonal_mean if self.with_centering else None
        self.seasonal_std = seasonal_std if self.with_scaling else None

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.features_idx is None:
            raise RuntimeError("SeasonalNormalization is not fitted yet.")
        
        flat_features = flatten_if_needed(data.features)

        selected = flat_features[:, self.features_idx]  # [A, F]
        n_samples = selected.shape[0]
        phase_idx = torch.arange(n_samples, device=selected.device) % self.period

        if self.with_centering and self.seasonal_mean is not None:
            selected = selected - self.seasonal_mean[phase_idx]

        if self.with_scaling and self.seasonal_std is not None:
            selected = selected / self.seasonal_std[phase_idx]

        flat_features[:, self.features_idx] = selected
        data.features = restore_if_needed(flat_features, self._original_shape)
        return data


class RollingNormalization(AbstractPreprocessingHandler):
    def __init__(
        self,
        window_size: int,
        with_centering: bool = True,
        with_scaling: bool = True,
        min_periods: int = 1,
    ):
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if min_periods <= 0:
            raise ValueError("min_periods must be > 0")
        if min_periods > window_size:
            raise ValueError("min_periods must be <= window_size")

        self.window_size = window_size
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.min_periods = min_periods

        self.features_idx: Optional[Sequence[int]] = None
        self._original_shape: Optional[torch.Size] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Rolling normalization does not learn global statistics.
        It only stores the target feature indices.
        """
        self._original_shape = data.ts_shape
        self.features_idx = expand_features_idx_for_flatten(
            features_idx, self._original_shape
        )
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.features_idx is None:
            raise RuntimeError("RollingNormalization is not fitted yet.")

        flat_features = flatten_if_needed(data.features)

        x = flat_features[:, self.features_idx]  # [A, F]
        device = x.device
        dtype = x.dtype
        n_samples, n_features = x.shape

        mask = ~torch.isnan(x)
        valid = mask.to(dtype)

        x_safe = torch.where(mask, x, torch.zeros_like(x))
        x2_safe = x_safe ** 2

        csum = torch.cumsum(x_safe, dim=0)
        csum2 = torch.cumsum(x2_safe, dim=0)
        ccount = torch.cumsum(valid, dim=0)

        zero_row = torch.zeros((1, n_features), dtype=dtype, device=device)
        csum = torch.cat([zero_row, csum], dim=0)
        csum2 = torch.cat([zero_row, csum2], dim=0)
        ccount = torch.cat([zero_row, ccount], dim=0)

        end_idx = torch.arange(1, n_samples + 1, device=device)
        start_idx = torch.clamp(end_idx - self.window_size, min=0)

        window_sum = csum[end_idx] - csum[start_idx]
        window_sum2 = csum2[end_idx] - csum2[start_idx]
        window_count = ccount[end_idx] - ccount[start_idx]

        enough = window_count >= self.min_periods
        safe_count = window_count.clamp(min=1.0)

        mean = window_sum / safe_count

        var = window_sum2 / safe_count - mean ** 2
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)

        result = x
        if self.with_centering:
            result = result - mean
        if self.with_scaling:
            result = result / std

        result = torch.where(enough, result, torch.full_like(result, float("nan")))

        flat_features[:, self.features_idx] = result
        data.features = restore_if_needed(flat_features, self._original_shape)
        return data


class PerChannelNormalization(AbstractPreprocessingHandler):
    # as StandartScaler, but for multiple channels
    def __init__(
        self,
        channels_idx: Sequence[int],
        with_centering: bool = True,
        with_scaling: bool = True
    ):
        if channels_idx is None:
            raise ValueError("channels_idx must be not None")
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.channels_idx = channels_idx

        self.mean_values: Optional[torch.Tensor] = None
        self.scale_values: Optional[torch.Tensor] = None
        self.is_fitted: bool = False

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Compute mean and std for selected channels.

        Args:
            data: torch.Tensor of shape (n_samples, n_features, n_channels)
            channels_idx: indices of channels to normalize

        Returns:
            self
        """
        if data.ts_shape is None or len(data.ts_shape) != 3:
            raise ValueError(
                f"Data must be in 3D tensor with shape (n_samples, n_features, n_channels), got shape={tuple(data.ts_shape)}" \
                "It is not possible to apply PerChannelNormalization"
            )
        selected = data.features[:, :, self.channels_idx]

        mask = ~torch.isnan(selected)
        count = mask.sum(dim=(0, 1)).clamp(min=1)

        safe_selected = torch.where(mask, selected, torch.zeros_like(selected))

        if self.with_centering or self.with_scaling:
            sum_values = safe_selected.sum(dim=(0, 1))
            mean = sum_values / count.to(selected.dtype)
        else:
            mean = None

        if self.with_centering:
            self.mean_values = mean
        else:
            self.mean_values = None

        if self.with_scaling:
            centered = torch.where(mask, selected - mean, torch.zeros_like(selected))
            var = (centered ** 2).sum(dim=(0, 1)) / count.to(selected.dtype)
            std = torch.sqrt(var)
            std = torch.where(std == 0, torch.ones_like(std), std)
            self.scale_values = std
        else:
            self.scale_values = None
        
        self.is_fitted = True

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if not self.is_fitted:
            raise RuntimeError("PerChannelNormalization is not fitted yet.")

        selected = data.features[:, :, self.channels_idx]

        if self.with_centering and self.mean_values is not None:
            selected = selected - self.mean_values.view(1, 1, -1)

        if self.with_scaling and self.scale_values is not None:
            selected = selected / self.scale_values.view(1, 1, -1)

        data.features[:, :, self.channels_idx] = selected
        return data
