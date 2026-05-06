from typing import Optional, Sequence
import torch

from fedot.core.data.prepared_data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler
from fedot.industrial.core.architecture.preprocessing.ts_methods.tools import (
    flatten_if_needed,
    restore_if_needed,
    expand_features_idx_for_flatten
)


class TSMeanImputation(AbstractPreprocessingHandler):
    """Time-series imputer that replaces NaNs with per-feature mean values.

    The handler supports both 2D and 3D tensors. For 3D inputs, feature-channel
    axes are temporarily flattened, means are computed on selected columns while
    ignoring NaNs, then data is restored to original shape.
    """

    def __init__(self):
        """Initialize class instance."""
        self.mean_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        x_flat = flatten_if_needed(x)
        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        selected = x_flat[:, self.features_idx]
        self.mean_values = torch.nanmean(selected, dim=0)

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.mean_values is None or self.features_idx is None:
            raise RuntimeError("MeanImputation is not fitted yet.")

        x = data.features
        x_flat = flatten_if_needed(x)

        for i, col_idx in enumerate(self.features_idx):
            column = x_flat[:, col_idx]
            x_flat[:, col_idx] = torch.where(
                torch.isnan(column),
                self.mean_values[i],
                column
            )

        data.features = restore_if_needed(x_flat, self.original_shape)
        return data


class TSMedianImputation(AbstractPreprocessingHandler):
    """Time-series imputer that fills NaNs with per-feature median values.

    This strategy is robust to outliers compared to mean imputation. It uses
    NaN-aware median estimation on selected columns and keeps tensor layout
    unchanged after restore from flattened representation.
    """

    def __init__(self):
        """Initialize class instance."""
        self.median_values: Optional[torch.Tensor] = None
        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        x_flat = flatten_if_needed(x)
        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        selected = x_flat[:, self.features_idx]
        self.median_values = torch.nanquantile(selected, q=0.5, dim=0)

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.median_values is None or self.features_idx is None:
            raise RuntimeError("MedianImputation is not fitted yet.")

        x = data.features
        x_flat = flatten_if_needed(x)

        for i, col_idx in enumerate(self.features_idx):
            column = x_flat[:, col_idx]
            x_flat[:, col_idx] = torch.where(
                torch.isnan(column),
                self.median_values[i],
                column
            )

        data.features = restore_if_needed(x_flat, self.original_shape)
        return data


class TSConstantImputation(AbstractPreprocessingHandler):
    """Time-series imputer that replaces NaNs with a fixed constant.

    Useful when a deterministic fill policy is required (for example, zero
    imputation). Selected features are processed after flattening when needed,
    then restored back to the original tensor shape.
    """

    def __init__(self, constant: float = 0.0):
        """Initialize class instance."""
        self.constant = constant
        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.features_idx is None:
            raise RuntimeError("ConstantImputation is not fitted yet.")

        x = data.features
        x_flat = flatten_if_needed(x)

        const = torch.tensor(
            self.constant,
            device=x_flat.device,
            dtype=x_flat.dtype
        )

        for col_idx in self.features_idx:
            column = x_flat[:, col_idx]
            x_flat[:, col_idx] = torch.where(
                torch.isnan(column),
                const,
                column
            )

        data.features = restore_if_needed(x_flat, self.original_shape)
        return data


class TSFillImputation(AbstractPreprocessingHandler):
    """Directional fill imputation for time-series gaps.

    Supports forward-fill and backward-fill modes. For each selected feature,
    missing values are propagated from nearest valid observations along the time
    axis, preserving existing non-missing values.
    """

    def __init__(self, direction: str = "forward"):
        """Initialize class instance."""
        if direction not in {"forward", "backward"}:
            raise ValueError("direction must be either 'forward' or 'backward'")

        self.direction = direction
        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        self.features_idx = expand_features_idx_for_flatten(
            features_idx,
            self.original_shape
        )

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.features_idx is None or self.original_shape is None:
            raise RuntimeError("TSFillImputation is not fitted yet.")

        x = data.features
        x_flat = flatten_if_needed(x)

        selected = x_flat[:, self.features_idx]
        mask = torch.isnan(selected)

        if not mask.any():
            data.features = restore_if_needed(x_flat, self.original_shape)
            return data

        n_samples = selected.shape[0]
        row_idx = torch.arange(n_samples, device=selected.device).unsqueeze(1).expand_as(selected)

        if self.direction == "forward":
            valid_idx = torch.where(~mask, row_idx, torch.zeros_like(row_idx))
            last_valid_idx = torch.cummax(valid_idx, dim=0).values

            has_prev = (~mask).cumsum(dim=0) > 0
            gathered = torch.gather(selected, dim=0, index=last_valid_idx)
            selected = torch.where(mask & has_prev, gathered, selected)

        else:  # backward
            rev_selected = torch.flip(selected, dims=[0])
            rev_mask = torch.isnan(rev_selected)

            valid_idx = torch.where(
                ~rev_mask,
                row_idx,
                torch.zeros_like(row_idx)
            )
            next_valid_idx = torch.cummax(valid_idx, dim=0).values

            has_next = (~rev_mask).cumsum(dim=0) > 0
            gathered = torch.gather(rev_selected, dim=0, index=next_valid_idx)
            rev_selected = torch.where(rev_mask & has_next, gathered, rev_selected)

            selected = torch.flip(rev_selected, dims=[0])

        x_flat[:, self.features_idx] = selected
        data.features = restore_if_needed(x_flat, self.original_shape)
        return data


class TSRollingImputation(AbstractPreprocessingHandler):
    """Rolling-window imputation for time-series data.

    Missing values are replaced using rolling statistics computed in a local
    window (`mean` or `median`). The window can be centered or causal, which
    makes the method suitable for both offline and streaming-like scenarios.
    """

    def __init__(
        self,
        window_size: int = 5,
        method: str = "mean",
        center: bool = True,
    ):
        """Initialize class instance."""
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if method not in {"mean", "median"}:
            raise ValueError("method must be either 'mean' or 'median'")

        self.window_size = window_size
        self.method = method
        self.center = center

        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        self.features_idx = expand_features_idx_for_flatten(
            features_idx,
            self.original_shape
        )

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.features_idx is None or self.original_shape is None:
            raise RuntimeError("TSRollingImputation is not fitted yet.")

        x = data.features
        x_flat = flatten_if_needed(x)

        selected = x_flat[:, self.features_idx].clone()
        n_samples = selected.shape[0]

        for i in range(n_samples):
            nan_mask = torch.isnan(selected[i])
            if not nan_mask.any():
                continue

            if self.center:
                half = self.window_size // 2
                left = max(0, i - half)
                right = min(n_samples, i + half + 1)
            else:
                left = max(0, i - self.window_size + 1)
                right = i + 1

            window = selected[left:right]

            if self.method == "mean":
                fill_values = torch.nanmean(window, dim=0)
            else:
                fill_values = torch.nanquantile(window, q=0.5, dim=0)

            valid_fill_mask = ~torch.isnan(fill_values)
            apply_mask = nan_mask & valid_fill_mask

            selected[i] = torch.where(
                apply_mask,
                fill_values,
                selected[i]
            )

        x_flat[:, self.features_idx] = selected
        data.features = restore_if_needed(x_flat, self.original_shape)
        return data


class TSKalmanImputation(AbstractPreprocessingHandler):
    """Kalman filter-based imputation for sequential signals.

    The method models each selected feature as a local linear trend process and
    performs prediction-update steps over time. Missing observations are filled
    with filtered state estimates, while observed values are preserved.
    """

    def __init__(self):
        """Initialize class instance."""
        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

        self.init_level: Optional[torch.Tensor] = None
        self.init_trend: Optional[torch.Tensor] = None

        self.q_level: Optional[torch.Tensor] = None
        self.q_trend: Optional[torch.Tensor] = None
        self.r: Optional[torch.Tensor] = None

        """Internal helper for `nanmean` logic."""
    @staticmethod
    def _nanmean(values: torch.Tensor, dim: int) -> torch.Tensor:
        """Internal helper for `nanmean` logic."""
        mask = ~torch.isnan(values)
        safe_values = torch.where(mask, values, torch.zeros_like(values))
        count = mask.sum(dim=dim).clamp_min(1)
        return safe_values.sum(dim=dim) / count

        """Internal helper for `nanvar` logic."""
    @staticmethod
    def _nanvar(values: torch.Tensor, dim: int) -> torch.Tensor:
        """Internal helper for `nanvar` logic."""
        mean = TSKalmanImputation._nanmean(values, dim=dim)
        mask = ~torch.isnan(values)
        centered = torch.where(mask, values - mean.unsqueeze(dim), torch.zeros_like(values))
        count = mask.sum(dim=dim).clamp_min(1)
        return centered.pow(2).sum(dim=dim) / count

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        x_flat = flatten_if_needed(x)
        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        selected = x_flat[:, self.features_idx]
        valid_mask = ~torch.isnan(selected)

        has_valid = valid_mask.any(dim=0)
        first_valid_idx = valid_mask.long().argmax(dim=0)
        col_idx = torch.arange(selected.shape[1], device=selected.device)

        first_values = selected[first_valid_idx, col_idx]
        first_values = torch.where(has_valid, first_values, torch.zeros_like(first_values))

        diffs = selected[1:] - selected[:-1]
        diff_mask = valid_mask[1:] & valid_mask[:-1]
        safe_diffs = torch.where(diff_mask, diffs, torch.full_like(diffs, float("nan")))

        mean_diff = self._nanmean(safe_diffs, dim=0)
        mean_diff = torch.where(diff_mask.any(dim=0), mean_diff, torch.zeros_like(mean_diff))

        series_var = self._nanvar(selected, dim=0)
        diff_var = self._nanvar(safe_diffs, dim=0)

        eps = torch.finfo(selected.dtype).eps

        self.init_level = first_values
        self.init_trend = mean_diff

        self.q_level = torch.clamp(diff_var * 0.25, min=eps)
        self.q_trend = torch.clamp(diff_var * 0.05, min=eps)
        self.r = torch.clamp(series_var * 0.05 + diff_var * 0.10, min=eps)

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if (
            self.features_idx is None
            or self.original_shape is None
            or self.init_level is None
            or self.init_trend is None
            or self.q_level is None
            or self.q_trend is None
            or self.r is None
        ):
            raise RuntimeError("KalmanImputation is not fitted yet.")

        x = data.features

        if x.dim() not in (2, 3):
            raise ValueError("KalmanImputation supports only [time, features] or [time, features, channels].")

        if not torch.is_floating_point(x):
            raise TypeError("KalmanImputation expects floating point tensor.")

        x_flat = flatten_if_needed(x)
        selected = x_flat[:, self.features_idx]

        level = self.init_level.to(device=selected.device, dtype=selected.dtype)
        trend = self.init_trend.to(device=selected.device, dtype=selected.dtype)

        q_level = self.q_level.to(device=selected.device, dtype=selected.dtype)
        q_trend = self.q_trend.to(device=selected.device, dtype=selected.dtype)
        r = self.r.to(device=selected.device, dtype=selected.dtype)

        p00 = r + q_level
        p01 = torch.zeros_like(p00)
        p10 = torch.zeros_like(p00)
        p11 = r + q_trend

        result = selected.clone()

        one = torch.tensor(1.0, device=selected.device, dtype=selected.dtype)

        for t in range(selected.shape[0]):
            level_pred = level + trend
            trend_pred = trend

            p00_pred = p00 + p01 + p10 + p11 + q_level
            p01_pred = p01 + p11
            p10_pred = p10 + p11
            p11_pred = p11 + q_trend

            y_t = selected[t]
            obs_mask = ~torch.isnan(y_t)

            s = p00_pred + r
            inv_s = torch.where(obs_mask, one / s, torch.zeros_like(s))

            k0 = p00_pred * inv_s
            k1 = p10_pred * inv_s

            residual = torch.where(obs_mask, y_t - level_pred, torch.zeros_like(y_t))

            level = level_pred + k0 * residual
            trend = trend_pred + k1 * residual

            p00 = torch.where(obs_mask, (one - k0) * p00_pred, p00_pred)
            p01 = torch.where(obs_mask, (one - k0) * p01_pred, p01_pred)
            p10 = torch.where(obs_mask, p10_pred - k1 * p00_pred, p10_pred)
            p11 = torch.where(obs_mask, p11_pred - k1 * p01_pred, p11_pred)

            result[t] = torch.where(obs_mask, y_t, level)

        x_flat[:, self.features_idx] = result
        data.features = restore_if_needed(x_flat, self.original_shape)

        return data


class TSLinearInterpolation(AbstractPreprocessingHandler):
    """Linear interpolation imputer for missing time points.

    Each NaN is replaced by a linear value between nearest previous and next
    valid observations. Edge gaps are handled via one-sided propagation when one
    of the neighbors is absent.
    """

    def __init__(self):
        """Initialize class instance."""
        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        return self

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.features_idx is None or self.original_shape is None:
            raise RuntimeError("LinearInterpolation is not fitted yet.")

        x = data.features

        x_flat = flatten_if_needed(x)
        selected = x_flat[:, self.features_idx]

        T, N = selected.shape
        device = selected.device
        dtype = selected.dtype

        result = selected.clone()

        time_idx = torch.arange(T, device=device, dtype=dtype).unsqueeze(1).expand(T, N)

        valid_mask = ~torch.isnan(selected)

        # forward fill indices
        last_valid_idx = torch.where(valid_mask, time_idx, torch.zeros_like(time_idx))
        last_valid_idx = torch.cummax(last_valid_idx, dim=0)[0]

        # backward fill indices
        next_valid_idx = torch.where(valid_mask, time_idx, torch.full_like(time_idx, T - 1))
        next_valid_idx = torch.flip(
            torch.cummin(torch.flip(next_valid_idx, dims=[0]), dim=0)[0],
            dims=[0]
        )

        col_idx = torch.arange(N, device=device)

        prev_vals = selected[last_valid_idx.long(), col_idx]
        next_vals = selected[next_valid_idx.long(), col_idx]

        denom = (next_valid_idx - last_valid_idx)
        denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)

        weight = (time_idx - last_valid_idx) / denom_safe

        interp = prev_vals + weight * (next_vals - prev_vals)

        # if previous is missing — use next (backward fill)
        interp = torch.where(last_valid_idx == 0, next_vals, interp)
        # if next is missing — use previous (forward fill)
        interp = torch.where(next_valid_idx == T - 1, prev_vals, interp)

        result = torch.where(valid_mask, selected, interp)

        x_flat[:, self.features_idx] = result
        data.features = restore_if_needed(x_flat, self.original_shape)

        return data


class TSPolynomialInterpolation(AbstractPreprocessingHandler):
    """Windowed polynomial interpolation for time-series gaps.

    For each feature and window, the method fits a polynomial of configurable
    degree on valid points and predicts missing ones. If valid points are
    insufficient, it falls back to linear interpolation.
    """

    def __init__(self, degree: int = 2, window_size: int = 64):
        """Initialize class instance."""
        self.degree = degree
        self.window_size = window_size

        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        return self

        """Internal helper for `linear_fallback` logic."""
    @staticmethod
    def _linear_fallback(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Internal helper for `linear_fallback` logic."""
        T = y.shape[0]
        device = y.device
        dtype = y.dtype

        time_idx = torch.arange(T, device=device, dtype=dtype)
        idx = torch.arange(T, device=device)

        last = torch.where(mask, idx, torch.zeros_like(idx))
        last = torch.cummax(last, dim=0)[0]

        nxt = torch.where(mask, idx, torch.full_like(idx, T - 1))
        nxt = torch.flip(torch.cummin(torch.flip(nxt, dims=[0]), dim=0)[0], dims=[0])

        prev_vals = y[last]
        next_vals = y[nxt]

        prev_exists = mask[last]
        next_exists = mask[nxt]

        last_t = last.to(dtype=dtype)
        next_t = nxt.to(dtype=dtype)

        denom = next_t - last_t
        denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)

        weight = (time_idx - last_t) / denom_safe
        interp = prev_vals + weight * (next_vals - prev_vals)

        interp = torch.where(prev_exists & next_exists, interp, prev_vals)
        interp = torch.where(~prev_exists & next_exists, next_vals, interp)
        interp = torch.where(prev_exists & ~next_exists, prev_vals, interp)

        return torch.where(mask, y, interp)

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.features_idx is None or self.original_shape is None:
            raise RuntimeError("PolynomialInterpolation is not fitted yet.")

        x = data.features

        x_flat = flatten_if_needed(x)
        selected = x_flat[:, self.features_idx]

        T, N = selected.shape
        device = selected.device
        dtype = selected.dtype

        result = selected.clone()

        deg = self.degree
        min_points = deg + 1
        window_size = self.window_size

        for j in range(N):
            y = selected[:, j]
            y_result = y.clone()

            for start in range(0, T, window_size):
                end = min(start + window_size, T)

                y_win = y[start:end]
                mask_win = ~torch.isnan(y_win)
                n_valid = int(mask_win.sum().item())

                if n_valid == 0:
                    continue

                if n_valid < min_points:
                    y_result[start:end] = self._linear_fallback(y_win, mask_win)
                    continue

                t_win = torch.arange(end - start, device=device, dtype=dtype)
                t_valid = t_win[mask_win]
                y_valid = y_win[mask_win]

                t_mean = t_valid.mean()
                t_std = t_valid.std().clamp_min(torch.finfo(dtype).eps)
                t_valid_norm = (t_valid - t_mean) / t_std

                X = torch.stack([t_valid_norm.pow(k) for k in range(deg + 1)], dim=1)
                coef = torch.linalg.lstsq(X, y_valid.unsqueeze(1)).solution.squeeze(1)

                t_all_norm = (t_win - t_mean) / t_std
                X_all = torch.stack([t_all_norm.pow(k) for k in range(deg + 1)], dim=1)
                y_pred = X_all @ coef

                y_result[start:end] = torch.where(mask_win, y_win, y_pred)

            result[:, j] = y_result

        x_flat[:, self.features_idx] = result
        data.features = restore_if_needed(x_flat, self.original_shape)

        return data


class TSSplineInterpolation(AbstractPreprocessingHandler):
    """Windowed spline interpolation for smooth gap reconstruction.

    The method uses natural cubic spline interpolation inside each window when
    enough valid points are available; otherwise it falls back to linear
    interpolation. This provides smoother trajectories than linear filling.
    """

    def __init__(self, window_size: int = 64):
        """Initialize class instance."""
        self.window_size = window_size

        self.features_idx: Optional[Sequence[int]] = None
        self.original_shape: Optional[tuple] = None

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """Run `fit` routine."""
        x = data.features
        self.original_shape = x.shape

        self.features_idx = expand_features_idx_for_flatten(features_idx, self.original_shape)

        return self

        """Internal helper for `linear_fallback` logic."""
    @staticmethod
    def _linear_fallback(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Internal helper for `linear_fallback` logic."""
        T = y.shape[0]
        device = y.device
        dtype = y.dtype

        time_idx = torch.arange(T, device=device, dtype=dtype)
        idx = torch.arange(T, device=device)

        last = torch.where(mask, idx, torch.zeros_like(idx))
        last = torch.cummax(last, dim=0)[0]

        nxt = torch.where(mask, idx, torch.full_like(idx, T - 1))
        nxt = torch.flip(torch.cummin(torch.flip(nxt, dims=[0]), dim=0)[0], dims=[0])

        prev_vals = y[last]
        next_vals = y[nxt]

        prev_exists = mask[last]
        next_exists = mask[nxt]

        last_t = last.to(dtype=dtype)
        next_t = nxt.to(dtype=dtype)

        denom = next_t - last_t
        denom_safe = torch.where(denom == 0, torch.ones_like(denom), denom)

        weight = (time_idx - last_t) / denom_safe
        interp = prev_vals + weight * (next_vals - prev_vals)

        interp = torch.where(prev_exists & next_exists, interp, prev_vals)
        interp = torch.where(~prev_exists & next_exists, next_vals, interp)
        interp = torch.where(prev_exists & ~next_exists, prev_vals, interp)

        return torch.where(mask, y, interp)

    def transform(self, data: PreparedData) -> PreparedData:
        """Run `transform` routine."""
        if self.features_idx is None or self.original_shape is None:
            raise RuntimeError("SplineInterpolation is not fitted yet.")

        x = data.features

        x_flat = flatten_if_needed(x)
        selected = x_flat[:, self.features_idx]

        T, N = selected.shape
        device = selected.device
        dtype = selected.dtype

        result = selected.clone()
        window_size = self.window_size

        for j in range(N):
            y = selected[:, j]
            y_result = y.clone()

            for start in range(0, T, window_size):
                end = min(start + window_size, T)

                y_win = y[start:end]
                mask_win = ~torch.isnan(y_win)
                n_valid = int(mask_win.sum().item())

                if n_valid == 0:
                    continue

                if n_valid < 3:
                    y_result[start:end] = self._linear_fallback(y_win, mask_win)
                    continue

                t_win = torch.arange(end - start, device=device, dtype=dtype)
                t_valid = t_win[mask_win]
                y_valid = y_win[mask_win]

                h = t_valid[1:] - t_valid[:-1]
                if torch.any(h <= 0):
                    y_result[start:end] = self._linear_fallback(y_win, mask_win)
                    continue

                if n_valid == 3:
                    X = torch.stack(
                        [torch.ones_like(t_valid), t_valid, t_valid.pow(2)],
                        dim=1
                    )
                    coef = torch.linalg.lstsq(X, y_valid.unsqueeze(1)).solution.squeeze(1)

                    X_all = torch.stack(
                        [torch.ones_like(t_win), t_win, t_win.pow(2)],
                        dim=1
                    )
                    y_pred = X_all @ coef
                    y_result[start:end] = torch.where(mask_win, y_win, y_pred)
                    continue

                m = n_valid - 2
                A = torch.zeros((m, m), device=device, dtype=dtype)
                rhs = torch.zeros(m, device=device, dtype=dtype)

                h_prev = h[:-1]
                h_next = h[1:]

                diag = 2 * (h_prev + h_next)
                upper = h_next[:-1]
                lower = h_prev[1:]

                idx = torch.arange(m, device=device)
                A[idx, idx] = diag

                if m > 1:
                    A[torch.arange(m - 1, device=device), torch.arange(1, m, device=device)] = upper
                    A[torch.arange(1, m, device=device), torch.arange(m - 1, device=device)] = lower

                rhs[:] = 6 * (
                    (y_valid[2:] - y_valid[1:-1]) / h_next
                    - (y_valid[1:-1] - y_valid[:-2]) / h_prev
                )

                m_inner = torch.linalg.solve(A, rhs)

                second = torch.zeros(n_valid, device=device, dtype=dtype)
                second[1:-1] = m_inner

                y_pred = y_win.clone()
                missing_idx = torch.where(~mask_win)[0]

                if missing_idx.numel() > 0:
                    interval_idx = torch.searchsorted(t_valid, t_win[missing_idx], right=True) - 1
                    interval_idx = interval_idx.clamp(0, n_valid - 2)

                    x0 = t_valid[interval_idx]
                    x1 = t_valid[interval_idx + 1]
                    y0 = y_valid[interval_idx]
                    y1 = y_valid[interval_idx + 1]
                    m0 = second[interval_idx]
                    m1 = second[interval_idx + 1]
                    h_seg = x1 - x0
                    xq = t_win[missing_idx]

                    left = x1 - xq
                    right = xq - x0

                    spline_vals = (
                        m0 * left.pow(3) / (6 * h_seg)
                        + m1 * right.pow(3) / (6 * h_seg)
                        + (y0 - m0 * h_seg.pow(2) / 6) * (left / h_seg)
                        + (y1 - m1 * h_seg.pow(2) / 6) * (right / h_seg)
                    )

                    y_pred[missing_idx] = spline_vals

                y_result[start:end] = y_pred

            result[:, j] = y_result

        x_flat[:, self.features_idx] = result
        data.features = restore_if_needed(x_flat, self.original_shape)

        return data
