from typing import Optional, Sequence
import torch

from fedot.core.data.prepared_data import PreparedData


class StandartScaling:
    """
    Scaling with nan
    """
    def __init__(self, with_mean: bool = True, with_std: bool = True):

        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_values: Optional[torch.Tensor] = None
        self.scale_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Compute mean and std for selected columns (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data[:, self.features_idx]

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

    def transform(self, data: PreparedData):
        if self.features_idx is None:
            raise RuntimeError("ScalingImplementation is not fitted yet.")

        selected = data.features[:, self.features_idx]

        if self.with_mean and self.mean_values is not None:
            selected = selected - self.mean_values

        if self.with_std and self.scale_values is not None:
            selected = selected / self.scale_values

        data.features[:, self.features_idx] = selected
        return data

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


class MinMaxNormalization:
    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        self.min_range = feature_range[0]
        self.max_range = feature_range[1]

        self.data_min: Optional[torch.Tensor] = None
        self.data_max: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None

        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Compute min and max for selected columns (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data[:, self.features_idx]

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

    def transform(self, data: PreparedData):
        if self.features_idx is None:
            raise RuntimeError("NormalizationImplementation is not fitted yet.")

        selected = data.features[:, self.features_idx]

        # X_scaled = (X - min) * scale + min_range
        selected = (selected - self.data_min) * self.scale + self.min_range

        data.features[:, self.features_idx] = selected
        return data

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


class RobustScaling:
    def __init__(
        self,
        quantile_range: tuple[float, float] = (25.0, 75.0),
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        self.q_min = quantile_range[0] / 100.0
        self.q_max = quantile_range[1] / 100.0
        self.with_centering = with_centering
        self.with_scaling = with_scaling

        self.center_values: Optional[torch.Tensor] = None
        self.scale_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Compute median and IQR for selected columns (ignoring NaNs).
        """
        self.features_idx = features_idx
        selected = data[:, self.features_idx]

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

    def transform(self, data: PreparedData):
        if self.features_idx is None:
            raise RuntimeError("RobustScaling is not fitted yet.")

        selected = data.features[:, self.features_idx]

        if self.with_centering and self.center_values is not None:
            selected = selected - self.center_values

        if self.with_scaling and self.scale_values is not None:
            selected = selected / self.scale_values

        data.features[:, self.features_idx] = selected
        return data

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


# TODO: move to industrial preprocessing
class SeasonalNormalization:
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

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Compute seasonal mean/std for selected columns along sample axis.

        For each seasonal position p in [0, period), statistics are computed
        from rows p, p + period, p + 2 * period, ...
        NaNs are ignored.
        """
        self.features_idx = features_idx
        selected = data[:, self.features_idx]  # shape: [A, F]

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

    def transform(self, data: PreparedData):
        if self.features_idx is None:
            raise RuntimeError("SeasonalNormalization is not fitted yet.")

        selected = data.features[:, self.features_idx]  # [A, F]
        n_samples = selected.shape[0]
        phase_idx = torch.arange(n_samples, device=selected.device) % self.period

        if self.with_centering and self.seasonal_mean is not None:
            selected = selected - self.seasonal_mean[phase_idx]

        if self.with_scaling and self.seasonal_std is not None:
            selected = selected / self.seasonal_std[phase_idx]

        data.features[:, self.features_idx] = selected
        return data

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


class RollingNormalization:
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

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Rolling normalization does not learn global statistics.
        It only stores the target feature indices.
        """
        self.features_idx = features_idx
        return self

    def transform(self, data: PreparedData):
        if self.features_idx is None:
            raise RuntimeError("RollingNormalization is not fitted yet.")

        x = data.features[:, self.features_idx]  # [A, F]
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

        # prefix pad for simpler window slicing
        zero_row = torch.zeros((1, n_features), dtype=dtype, device=device)
        csum = torch.cat([zero_row, csum], dim=0)     # [A+1, F]
        csum2 = torch.cat([zero_row, csum2], dim=0)   # [A+1, F]
        ccount = torch.cat([zero_row, ccount], dim=0) # [A+1, F]

        end_idx = torch.arange(1, n_samples + 1, device=device)  # [A]
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

        data.features[:, self.features_idx] = result
        return data

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)


# TODO: in ts preprocessing make dependency on PreparedData.ts_shape
class PerChannelNormalization:
    # as StandartScaler, but for multiple channels
    def __init__(
        self,
        with_centering: bool = True,
        with_scaling: bool = True,
        channels_idx: Optional[Sequence[int]] = None,
    ):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.channels_idx = channels_idx

        self.mean_values: Optional[torch.Tensor] = None
        self.scale_values: Optional[torch.Tensor] = None

    def fit(self, data: torch.Tensor, features_idx: Sequence[int]):
        """
        Compute mean and std for selected channels.

        Args:
            data: torch.Tensor of shape (n_samples, n_features, n_channels)
            channels_idx: indices of channels to normalize

        Returns:
            self
        """
        if self.channels_idx is None:
            self.channels_idx = list(range(data.ts_init_shape[2]))

        selected = data[:, :, self.channels_idx]

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

        return self

    def transform(self, data: PreparedData):
        if (self.mean_values is None) and (self.scale_values is None):
            raise RuntimeError("PerChannelNormalization is not fitted yet.")

        selected = data.features[:, :, self.channels_idx]

        if self.with_centering and self.mean_values is not None:
            selected = selected - self.mean_values.view(1, 1, -1)

        if self.with_scaling and self.scale_values is not None:
            selected = selected / self.scale_values.view(1, 1, -1)

        data.features[:, :, self.channels_idx] = selected
        return data

    def fit_transform(self, data: PreparedData, features_idx: Sequence[int]):
        return self.fit(data.features, features_idx).transform(data)
