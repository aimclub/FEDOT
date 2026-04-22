from typing import Optional, Sequence, Tuple
import torch

from fedot.core.data.prepared_data import PreparedData
from fedot.preprocessing.methods.abstract import AbstractPreprocessingHandler


class ContrastEqualization(AbstractPreprocessingHandler):
    def __init__(
        self,
        output_range: tuple[float, float] = (0.0, 1.0),
        channels_idx: Optional[Tuple[int]] = None,
    ):
        self.min_range = output_range[0]
        self.max_range = output_range[1]
        self.channels_idx = channels_idx

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        """
        Contrast equalization does not learn global statistics.
        It only stores selected channel indices.
        """
        if self.channels_idx is None:
            self.channels_idx = list(range(data.ts_init_shape[2]))
        return self

    def transform(self, data: PreparedData) -> PreparedData:

        if self.channels_idx is None:
            raise RuntimeError("ContrastEqualization is not fitted yet.")

        x = data.features[:, :, self.channels_idx]
        result = x.clone()

        n_samples, n_features, n_channels = x.shape

        for s in range(n_samples):
            for c in range(n_channels):
                values = x[s, :, c]  # [F]
                mask = ~torch.isnan(values)

                valid = values[mask]
                n_valid = valid.numel()

                if n_valid == 0:
                    continue

                if n_valid == 1:
                    out = torch.full_like(valid, self.min_range)
                else:
                    order = torch.argsort(valid)
                    ranks = torch.empty_like(valid, dtype=values.dtype)
                    ranks[order] = torch.arange(
                        n_valid,
                        device=values.device,
                        dtype=values.dtype,
                    )
                    out = ranks / (n_valid - 1)
                    out = out * (self.max_range - self.min_range) + self.min_range

                updated = values.clone()
                updated[mask] = out
                result[s, :, c] = updated

        data.features[:, :, self.channels_idx] = result
        return data


class ContrastStretching(AbstractPreprocessingHandler):
    def __init__(
        self,
        quantile_range: tuple[float, float] = (2.0, 98.0),
        output_range: tuple[float, float] = (0.0, 1.0),
        channels_idx: Optional[Sequence[int]] = None,
    ):
        self.q_min = quantile_range[0] / 100.0
        self.q_max = quantile_range[1] / 100.0

        self.min_range = output_range[0]
        self.max_range = output_range[1]

        self.channels_idx = channels_idx

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        if self.channels_idx is None:
            self.channels_idx = list(range(data.ts_init_shape[2]))
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.channels_idx is None:
            raise RuntimeError("ContrastStretching is not fitted yet.")

        x = data.features[:, :, self.channels_idx]
        result = x.clone()

        n_samples, _, n_channels = x.shape

        for s in range(n_samples):
            for c in range(n_channels):
                values = x[s, :, c]
                mask = ~torch.isnan(values)

                valid = values[mask]
                if valid.numel() == 0:
                    continue

                low = torch.quantile(valid, self.q_min)
                high = torch.quantile(valid, self.q_max)

                scale = high - low
                if scale == 0:
                    scale = torch.tensor(1.0, device=values.device)

                out = (valid - low) / scale
                out = out * (self.max_range - self.min_range) + self.min_range

                out = torch.clamp(out, self.min_range, self.max_range)

                updated = values.clone()
                updated[mask] = out
                result[s, :, c] = updated

        data.features[:, :, self.channels_idx] = result
        return data


class GammaCorrection(AbstractPreprocessingHandler):
    def __init__(
        self,
        gamma: float = 1.0,
        channels_idx: Optional[Sequence[int]] = None,
    ):
        self.gamma = gamma
        self.channels_idx = channels_idx

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        if self.channels_idx is None:
            self.channels_idx = list(range(data.ts_init_shape[2]))
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.channels_idx is None:
            raise RuntimeError("GammaCorrection is not fitted yet.")

        x = data.features[:, :, self.channels_idx]
        result = x.clone()

        mask = ~torch.isnan(x)

        safe = torch.where(mask, x, torch.zeros_like(x))
        transformed = torch.pow(safe, self.gamma)

        result[mask] = transformed[mask]

        data.features[:, :, self.channels_idx] = result
        return data


class LogTransform(AbstractPreprocessingHandler):
    def __init__(
        self,
        eps: float = 1e-6,
        channels_idx: Optional[Sequence[int]] = None,
    ):
        self.eps = eps
        self.channels_idx = channels_idx

    def fit(self, data: PreparedData, features_idx: Sequence[int]):
        if self.channels_idx is None:
            self.channels_idx = list(range(data.ts_init_shape[2]))
        return self

    def transform(self, data: PreparedData) -> PreparedData:
        if self.channels_idx is None:
            raise RuntimeError("LogTransform is not fitted yet.")

        x = data.features[:, :, self.channels_idx]
        result = x.clone()

        mask = ~torch.isnan(x)

        safe = torch.where(mask, x, torch.zeros_like(x))
        safe = torch.clamp(safe, min=0.0)  # log только для >=0

        transformed = torch.log(safe + self.eps)

        result[mask] = transformed[mask]

        data.features[:, :, self.channels_idx] = result
        return data
